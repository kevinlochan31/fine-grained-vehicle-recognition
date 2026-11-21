"""End-to-end hybrid recognition pipeline used by the app and the evaluation.

    input -> validation -> quality analysis -> YOLO detection (classical fallback)
          -> ROI crop -> adaptive classical enhancement -> CNN (+flip TTA)
          -> late fusion with classical-feature classifier
          -> hierarchical low-confidence handling -> annotated output
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

from . import config as C
from .classical import adaptive_preprocess, classical_roi, extract_features, image_stats
from .data import class_hierarchy, crop_image, load_class_names, split_class_name

DEFAULT_HYBRID = dict(fusion_weight=0.1, conf_threshold=C.CONF_THRESHOLD,
                      make_threshold=C.MAKE_CONF_THRESHOLD, tta=True, enhance=True)


def load_hybrid_config() -> dict:
    p = C.MODELS_DIR / "hybrid_config.json"
    cfg = dict(DEFAULT_HYBRID)
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


class InputError(ValueError):
    pass


def validate_image(img) -> np.ndarray:
    """Normalise arbitrary user input to an RGB uint8 array or raise InputError."""
    if img is None:
        raise InputError("No image received. Please upload a JPG or PNG photo of a car.")
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise InputError(f"Unsupported image shape {img.shape}.")
    h, w = img.shape[:2]
    if min(h, w) < 32:
        raise InputError(f"Image is too small ({w}x{h}). Please upload at least 32x32 pixels.")
    if max(h, w) > 2000:                              # keep latency bounded
        s = 2000 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if img.std() < 2:
        raise InputError("The image appears to be blank (uniform colour).")
    return img


class VehicleRecognizer:
    def __init__(self, backend: str = "torch", device: str | None = None, load_classical: bool = True,
                 use_clip: bool = True):
        from .classifier import FineGrainedClassifier
        from .detector import VehicleDetector
        self.names = load_class_names()
        self.make_idx, self.makes, self.mm_idx, self.mms = class_hierarchy(self.names)
        self.detector = VehicleDetector(device=device)
        self.classifier = FineGrainedClassifier(backend=backend, device=device)
        self.classical = None
        if load_classical and (C.MODELS_DIR / "classical_logreg.joblib").exists():
            import joblib
            self.classical = joblib.load(C.MODELS_DIR / "classical_logreg.joblib")
        self.cfg = load_hybrid_config()
        # Open-vocabulary CLIP (vehicle gate + naming of cars outside the training set)
        self.vision = None
        if use_clip:
            from .openvocab import CarVision
            self.vision = CarVision(device=device)
            # classifier make index -> CLIP catalogue make index (catalogue makes are a superset)
            self._mk_map = np.array([self.vision.makes.index(m) for m in self.makes])
        # World-car layer (extra, never touches the 196-class models): names cars outside the training set.
        self.world = None
        from .worldcars import GALLERY
        import os
        if use_clip and GALLERY.exists() and os.environ.get("CVH_WORLD", "1") != "0":
            from .worldcars import WorldCars
            self.world = WorldCars(device=device)
        self.clip_ft = None
        from .clip_classifier import WEIGHTS
        if use_clip and WEIGHTS.exists():
            from .clip_classifier import ClipFineGrained
            self.clip_ft = ClipFineGrained(device=device)

    # ------------------------------------------------------------------ helpers
    def _group(self, p: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
        return np.bincount(idx, weights=p, minlength=n)

    def describe(self, probs: np.ndarray, conf_t: float, make_t: float) -> dict:
        top5 = np.argsort(-probs)[:5]
        c = int(top5[0]); conf = float(probs[c])
        make, model, year = split_class_name(self.names[c])
        pm = self._group(probs, self.make_idx, len(self.makes))
        pmm = self._group(probs, self.mm_idx, len(self.mms))
        out = dict(class_id=c, label=self.names[c], make=make, model=model, year=year, confidence=conf,
                   top5=[(self.names[i], float(probs[i])) for i in top5],
                   make_confidence=float(pm.max()), best_make=self.makes[int(pm.argmax())],
                   makemodel_confidence=float(pmm.max()), best_makemodel=self.mms[int(pmm.argmax())])
        # Hierarchical back-off: fall back to a coarser but more reliable answer.
        if conf >= conf_t:
            out["status"], out["display"] = "confident", self.names[c]
        elif pmm.max() >= conf_t:
            out["status"] = "year-uncertain"
            out["display"] = f"{self.mms[int(pmm.argmax())]} (year uncertain)"
        elif pm.max() >= make_t:
            out["status"] = "model-uncertain"
            out["display"] = f"{self.makes[int(pm.argmax())]} (model uncertain)"
        else:
            out["status"] = "low-confidence"
            out["display"] = "Uncertain - see top-5 candidates"
        return out

    # ------------------------------------------------------------------ open-set helpers
    def _no_car(self, rgb, stats, gate, timings, mode):
        return dict(mode=mode, image=rgb, box=None, det_conf=0.0, det_source="none", crop=rgb,
                    enhancement_steps=[], stats=stats, timings=timings, class_id=-1, label="No car detected",
                    make="", model="", year="", confidence=1 - gate, car_probability=gate, status="no-car",
                    display="No car detected", top5=[], make_confidence=0.0, best_make="",
                    makemodel_confidence=0.0, best_makemodel="", in_training_set=False)

    def _calibrate(self, p, T):
        if T == 1.0:
            return p
        z = np.log(np.clip(p, 1e-12, 1)) / T
        z = np.exp(z - z.max(-1, keepdims=True))
        return z / z.sum(-1, keepdims=True)

    def analyse(self, image) -> dict:
        """Run hybrid, deep and classical on one image, sharing validation, gate and detection."""
        ctx = {}                            # hybrid runs first so detection + CLIP on the crop are shared
        return {m: self.run(image, m, ctx) for m in ("hybrid", "deep", "classical")
                if m != "classical" or self.classical is not None}

    # ------------------------------------------------------------------ modes
    def run(self, image, mode: str = "hybrid", ctx: dict | None = None) -> dict:
        """mode in {'hybrid', 'deep', 'classical'}. `ctx` shares work between modes of one request."""
        t0 = time.perf_counter()
        ctx = {} if ctx is None else ctx
        if "rgb" not in ctx:
            ctx["rgb"] = validate_image(image)
            ctx["stats"] = image_stats(ctx["rgb"])
        rgb, stats = ctx["rgb"], ctx["stats"]
        timings, cfg = {}, self.cfg

        t = time.perf_counter()
        if "det" not in ctx:                 # also needed by the vehicle gate in classical mode
            det, _ = self.detector.primary(rgb)
            if det["source"] != "yolo":
                # Adaptive retry: enhance the whole image only if YOLO found nothing.
                enh_full, s = adaptive_preprocess(rgb, stats)
                if s:
                    det2, _ = self.detector.primary(enh_full)
                    if det2["source"] == "yolo":
                        det = dict(det2, source="yolo (after enhancement)")
            ctx["det"] = det
        timings["detection_ms"] = 1000 * (time.perf_counter() - t)

        # ---- vehicle gate (CLIP): is there a car at all? Shared by every mode.
        t = time.perf_counter()
        if self.vision is not None and "gate" not in ctx:
            d = ctx.get("det")
            yolo_crop = crop_image(rgb, d["box"]) if d is not None and d["source"].startswith("yolo") else None
            ctx["clip"] = self.vision.analyse(rgb, yolo_crop)
            ctx["gate"] = ctx["clip"]["car_prob"]
        timings["gate_ms"] = 1000 * (time.perf_counter() - t)
        # Accept if CLIP is sure (>= gate_strong), or fairly sure AND the car detector agrees.
        d = ctx.get("det")
        det_ok = d is not None and d["source"].startswith("yolo") and d["conf"] >= cfg.get("gate_det_conf", 0.3)
        is_car = self.vision is None or ctx["gate"] >= cfg.get("gate_strong", 0.9) or (
            ctx["gate"] >= cfg.get("gate_threshold", 0.5) and det_ok)
        if not is_car:
            timings["total_ms"] = 1000 * (time.perf_counter() - t0)
            return self._no_car(rgb, stats, ctx["gate"], timings, mode)

        if mode == "classical":
            box, _ = classical_roi(rgb); det = dict(box=box, conf=0.0, source="classical-roi")
        else:
            det = ctx["det"]
            if not det["source"].startswith("yolo"):
                # A car is present (gate passed) but no box: classify the whole frame.
                h, w = rgb.shape[:2]; det = dict(box=(0, 0, w - 1, h - 1), conf=0.0, source="full-image")

        crop = crop_image(rgb, det["box"])
        steps = []
        t = time.perf_counter()
        if mode == "classical":
            if self.classical is None:
                raise RuntimeError("classical model not trained")
            probs = self.classical.predict_proba(extract_features(crop)[None])[0]
        else:
            views = [crop]
            if mode == "hybrid" and cfg.get("enhance", True):
                enh, steps = adaptive_preprocess(crop)
                if steps:            # only when the quality analysis asks for it
                    views.append(enh)
            # Hybrid: average the CNN over the original and the classically enhanced view
            # (each with flip-TTA). Replacing the original by the enhanced crop was worse.
            raw = self.classifier.predict_proba(views, tta=(mode == "hybrid" and cfg.get("tta", True))).mean(0)
            probs = self._calibrate(raw, cfg.get("temperature", 1.0))   # fitted on validation
            members = dict(convnext_raw=raw, convnext=probs)          # kept for "show the working"
            ctx["members_agree"] = None
            if mode == "hybrid" and self.clip_ft is not None:
                # ensemble with the robust CLIP fine-tune (weights + temperature fitted on validation)
                pc_raw = self.clip_ft.predict_proba(views, tta=True).mean(0)
                pc = self._calibrate(pc_raw, cfg.get("clip_temperature", 1.0))
                w = cfg.get("ensemble_weight", 0.5)
                members.update(clip_raw=pc_raw, clip=pc, ensemble_weight=w)
                ctx["members_agree"] = int(probs.argmax()) == int(pc.argmax())
                probs = w * probs + (1 - w) * pc
            w = cfg.get("fusion_weight", 0.0)
            if mode == "hybrid" and self.classical is not None and w > 0:
                pc = self.classical.predict_proba(extract_features(crop)[None])[0]
                probs = (1 - w) * probs + w * pc
        timings["classification_ms"] = 1000 * (time.perf_counter() - t)

        conf_t = cfg["conf_threshold"] if mode == "hybrid" else 0.0
        make_t = cfg["make_threshold"] if mode == "hybrid" else 0.0
        pred = self.describe(probs, conf_t, make_t)
        pred["in_training_set"] = True
        pred["car_probability"] = ctx.get("gate", 1.0)
        if mode == "hybrid" and mode != "classical":
            pred["members"], pred["final_probs"] = members, probs
        if mode == "hybrid" and "clip" in ctx:
            pred = self._open_set(pred, ctx, crop, cfg, probs)
        timings["total_ms"] = 1000 * (time.perf_counter() - t0)
        return dict(mode=mode, image=rgb, box=det["box"], det_conf=det["conf"], det_source=det["source"],
                    crop=crop, enhancement_steps=steps, stats=stats, timings=timings, **pred)

    def _open_set(self, pred, ctx, crop, cfg, probs):
        pred = self._open_set_decide(pred, ctx, crop, cfg, probs)
        # Whenever no answer is claimed, still say what the world-car layer would pick (clearly as a guess),
        # e.g. "maybe Maruti Suzuki Grand Vitara · 1st gen (50%)" for a car with only 10 reference photos.
        wm = ctx.get("world")
        if wm is not None and pred["status"] in ("unknown", "unknown-model", "not-sure") and wm["top_is_world"] \
                and wm["p"] >= cfg.get("world_guess_floor", 0.2):
            from .worldcars import describe_world
            c = wm["cls"]
            pred["best_guess"] = dict(label=describe_world(c), make=c["make"], model=c["model"], gen=c.get("gen"),
                                      years=c.get("years"), p=float(wm["p"]))
        return pred

    def _open_set_decide(self, pred, ctx, crop, cfg, probs):
        """Decide whether the car is one of the 196 trained classes. If not (or if the classifier is too
        unsure to commit), answer responsibly: name the model only when the evidence is strong, otherwise
        report the make with "model not recognised", or "unknown car" when even the make is unclear."""
        from .openvocab import decide_unfamiliar, same_family
        zs = ctx["clip"]                    # computed on the YOLO crop (or the full frame if no box)
        top = zs["top"]
        known = (top["stanford"] >= 0
                 or same_family(top["make"], top["model"], pred["make"], pred["model"])
                 or pred["confidence"] >= cfg.get("known_conf", 0.9)
                 or zs["outside_mass"] < cfg.get("outside_threshold", 0.5)
                 or zs["top_p"] < cfg.get("open_min_top", 0.0)                       # CLIP itself unsure
                 or (cfg.get("open_require_disagreement", False) and ctx.get("members_agree")))  # both nets agree
        pred["outside_mass"] = zs["outside_mass"]
        pred["open_set_top5"] = [{"label": l, "p": p, "in_training_set": s} for l, p, s in zs["name_top5"]]
        # World-car layer: a reference-photo probe over ~170 modern models + the 196 trained classes.
        wm = None
        if self.world is not None and "feature" in zs:
            if "world" not in ctx:
                f, fz = (zs["feature"], None) if self.world.model is None else self.world.embed_pair(
                    ctx["rgb"], crop if ctx.get("det", {}).get("source", "").startswith("yolo") else None)
                ctx["world"] = self.world.match(f, fz)
            wm = ctx["world"]
        world_ok = wm is not None and wm["top_is_world"]
        # Override "this is a trained class" only on very strong reference-photo evidence (budget: <=0.1% of
        # Stanford val cars), e.g. a 2019 Camry that the classifier calls "Toyota Camry Sedan 2012".
        world_route = world_ok and wm["p"] >= cfg.get("world_route_conf", 1.01)
        if known and not world_route and pred["status"] != "low-confidence":
            # Cross-check: the world head also knows the 196 classes. If much of its belief goes to cars
            # outside them (e.g. a 2020+ Thar that the classifier calls "Jeep Wrangler 2012"), do not
            # present the classifier's answer as confident. Threshold: <=0.5% of Stanford val doubted.
            if wm is not None and wm.get("world_mass", 0.0) >= cfg.get("world_doubt_mass", 1.01):
                return self._not_sure(pred, wm)
            return pred
        pred["closest_trained_class"] = pred["label"]
        if world_ok and wm["p"] >= cfg.get("world_name_conf", 1.01):
            return self._world_answer(pred, wm)
        if wm is not None:
            mt, kt = wm["model_tier"], wm["make_tier"]
            # model level: confident about the model, not about which generation
            if mt is not None and mt["p"] >= cfg.get("world_model_conf", 1.01):
                return self._world_answer(pred, dict(cls=dict(mt["cls"], gen=None, years="", y0=None), p=mt["p"],
                                                     ref=mt["ref"], ref_sim=mt["ref_sim"]), level="model")
        # make level (used below, after the model-naming steps): sure of the make only, closest models as hints
        world_make = wm is not None and wm["make_tier"]["p"] >= cfg.get("world_make_conf", 1.01)
        V = self.vision
        p_make_cls = np.zeros(len(V.makes)); p_make_cls[self._mk_map] = self._group(probs, self.make_idx, len(self.makes))
        make, make_p, named, within = decide_unfamiliar(
            V.makes, p_make_cls, V.entries, V.entry_make, zs["name_probs"],
            cfg.get("unknown_make_weight", 0.5), cfg.get("open_name_conf", 0.3))
        closest = pred["label"]
        if known:
            # Probably a trained class but the classifier is too unsure for any level of its hierarchy:
            # do not present its top class as the answer; it stays visible in the top-5 list.
            pred["possibly_outside"] = True
            if world_make:
                return self._world_make(pred, wm["make_tier"])
            if make_p >= cfg.get("unknown_make_conf", 0.5):
                pred.update(status="unknown-model", make=make, model="Model not recognised", year="unknown",
                            confidence=make_p, display=f"{make}, model not recognised")
            else:
                pred.update(status="unknown", make="", model="Unknown car", year="unknown",
                            display="Unknown car (outside the supported classes?)")
            return pred
        pred["in_training_set"] = False
        if wm is not None:
            # With the world-car layer loaded, a car it cannot name from reference photos is not named from text
            # alone: the CLIP open-set name was wrong for 50 of 101 such answers and the catalogue zero-shot name for
            # 16 of 48 (world test photos, round 3). Show them as unconfirmed guesses instead.
            guess = None
            if named is not None:
                e = V.entries[named]; guess = (e["label"], e["make"], float(zs["name_probs"][named]))
            elif wm["zs_p"] >= cfg.get("world_zs_conf", 1.01) and _mk(wm["zs_make"]) == _mk(make):
                guess = (f"{wm['zs_make']} {wm['zs_model']}", wm["zs_make"], float(wm["zs_p"]))
            sure_make = (wm["make_tier"]["make"] if world_make
                         else make if make_p >= cfg.get("unknown_make_conf", 0.5) else "")
            if guess is not None:
                return self._maybe(pred, guess, sure_make, wm)
            if world_make:
                return self._world_make(pred, wm["make_tier"])
        if named is not None:                                       # strong, make-consistent open-vocab name
            e = V.entries[named]
            pred.update(status="open-set", label=e["label"], make=e["make"], model=e["model"], year="unknown",
                        confidence=float(zs["name_probs"][named]),
                        display=f"{e['label']} (not in the 196 training classes)",
                        top5=[(l, p) for l, p, _ in zs["name_top5"]])
        elif (wm is not None and wm["zs_p"] >= cfg.get("world_zs_conf", 1.01)
              and make_p >= cfg.get("unknown_make_conf", 0.5) and _mk(wm["zs_make"]) == _mk(make)):
            # zero-shot over the ~2,000-model world catalogue, make-consistent: model named, generation not
            pred.update(status="world", label=f"{wm['zs_make']} {wm['zs_model']}", make=wm["zs_make"],
                        model=wm["zs_model"], year="generation not determined", confidence=wm["zs_p"],
                        display=f"{wm['zs_make']} {wm['zs_model']} · generation not determined · zero-shot (no reference photo)",
                        world=dict(make=wm["zs_make"], model=wm["zs_model"], gen=None, years="", y0=None,
                                   source="zero-shot catalogue", p=wm["zs_p"], ref=None),
                        top5=[(l, p) for l, p, _ in zs["name_top5"]])
        elif world_make:
            return self._world_make(pred, wm["make_tier"])
        elif make_p >= cfg.get("unknown_make_conf", 0.5):
            pred.update(status="unknown-model", label=f"{make} (unknown model)", make=make,
                        model="Model not recognised", year="unknown", confidence=make_p,
                        display=f"{make}, model not recognised (outside the supported classes)",
                        top5=[(V.entries[i]["label"], float(zs["name_probs"][i])) for i in within])
        else:
            pred.update(status="unknown", label="Unknown car", make="", model="Unknown car", year="unknown",
                        confidence=make_p, display="Unknown car (outside the supported classes)",
                        top5=[(l, p) for l, p, _ in zs["name_top5"]])
        return pred

    def _not_sure(self, pred, wm):
        """The two models disagree: show no answer, only what each of them thinks (both unconfirmed)."""
        from .worldcars import describe_world
        mass = wm["world_mass"]
        w_label = describe_world(wm["cls"])
        kt = wm["make_tier"]
        same_make = _mk(kt["make"]) == _mk(pred["make"])
        pred.update(status="not-sure", closest_trained_class=pred["label"], in_training_set=False,
                    possibly_outside=True, make=pred["make"] if same_make else "", model="Not sure",
                    year="unknown", confidence=float(pred["confidence"] * (1 - mass)),
                    display=f"Not sure · maybe {w_label} or {pred['label']}",
                    not_sure=dict(kind="disagree", world_mass=float(mass), classifier=pred["label"],
                                  classifier_p=float(pred["confidence"]), world=w_label, world_p=float(wm["p"])),
                    top5=[(f"{w_label} (world-car layer)", float(wm["p"]))] + list(pred["top5"])[:4])
        return pred

    def _maybe(self, pred, guess, sure_make, wm):
        """Only a weak guess from text matching: show it as unconfirmed, keep the make only if it is reliable."""
        from .worldcars import describe_world
        label, g_make, p = guess
        make = sure_make if _mk(sure_make) == _mk(g_make) else ""
        w_label = describe_world(wm["cls"])
        pred.update(status="not-sure", make=make, model="Not sure", year="unknown", possibly_outside=True,
                    confidence=float(p), display=f"Not sure · maybe {label}",
                    not_sure=dict(kind="guess", guess=label, guess_p=float(p), world=w_label, world_p=float(wm["p"]),
                                  classifier=pred["closest_trained_class"], classifier_p=float(pred["confidence"])),
                    top5=[(f"{label} (text match)", float(p)), (f"{w_label} (world-car layer)", float(wm["p"]))]
                         + list(pred["top5"])[:3])
        return pred

    def _world_make(self, pred, kt):
        hint = f" · closest: {', '.join(kt['candidates'])}" if kt["candidates"] else ""
        pred.update(status="unknown-model", label=f"{kt['make']} (model not recognised)", make=kt["make"],
                    model="Model not recognised", year="unknown", confidence=kt["p"], in_training_set=False,
                    display=f"{kt['make']}, model not recognised{hint}",
                    world_make=dict(make=kt["make"], p=kt["p"], candidates=kt["candidates"]))
        return pred

    def _world_answer(self, pred, wm, level="generation"):
        from .worldcars import describe_world
        c, r = wm["cls"], wm["ref"]
        label = describe_world(c)
        extra = " · generation not certain" if level == "model" else ""
        self_learnt = str(r.get("author", "")).startswith("self-learnt")
        src = "your photos" if (c.get("user") or (r.get("user") and not self_learnt)) else "reference photos"
        pred.update(status="world", label=label, make=c["make"], model=c["model"],
                    year=c.get("years") or ("generation not certain" if level == "model" else "unknown"),
                    confidence=wm["p"], in_training_set=False,
                    display=f"{label}{extra} · matched from {src}",
                    world=dict(make=c["make"], model=c["model"], gen=c.get("gen"), years=c.get("years"), y0=c.get("y0"),
                               level=level, source=src, p=wm["p"], ref_similarity=wm["ref_sim"],
                               ref=dict(image=r["image"], author=r["author"], license=r["license"],
                                        license_url=r["license_url"], source=r["source"], title=r["title"])))
        return pred


def _mk(m: str) -> str:
    from .worldcars import make_key
    return make_key(m)


def draw_result(res: dict) -> np.ndarray:
    img = res["image"].copy()
    x1, y1, x2, y2 = res["box"]
    colour = {"confident": (40, 200, 70), "year-uncertain": (240, 180, 0),
              "model-uncertain": (255, 120, 0), "world": (150, 210, 40)}.get(res["status"], (230, 40, 40))
    th = max(2, int(min(img.shape[:2]) / 200))
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, th)
    text = f"{res['display']}  {res['confidence'] * 100:.1f}%"
    W = img.shape[1]
    fs = max(0.4, min(img.shape[:2]) / 900)
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1 + (fs > 0.6))
    if tw + 10 > W:                                    # shrink the label so it fits the image width
        fs *= (W - 10) / (tw + 10)
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    lx = int(min(max(0, x1), max(0, W - tw - 10)))     # keep the label inside the frame
    ty = max(y1, tht + 10)
    cv2.rectangle(img, (lx, ty - tht - 10), (lx + tw + 10, ty), colour, -1)
    cv2.putText(img, text, (lx + 5, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                1 + (fs > 0.6), cv2.LINE_AA)
    return img
