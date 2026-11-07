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
            probs = self.classifier.predict_proba(views, tta=(mode == "hybrid" and cfg.get("tta", True))).mean(0)
            probs = self._calibrate(probs, cfg.get("temperature", 1.0))   # fitted on validation
            ctx["members_agree"] = None
            if mode == "hybrid" and self.clip_ft is not None:
                # ensemble with the robust CLIP fine-tune (weights + temperature fitted on validation)
                pc = self.clip_ft.predict_proba(views, tta=True).mean(0)
                pc = self._calibrate(pc, cfg.get("clip_temperature", 1.0))
                w = cfg.get("ensemble_weight", 0.5)
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
        if mode == "hybrid" and "clip" in ctx:
            pred = self._open_set(pred, ctx, crop, cfg, probs)
        timings["total_ms"] = 1000 * (time.perf_counter() - t0)
        return dict(mode=mode, image=rgb, box=det["box"], det_conf=det["conf"], det_source=det["source"],
                    crop=crop, enhancement_steps=steps, stats=stats, timings=timings, **pred)

    def _open_set(self, pred, ctx, crop, cfg, probs):
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
        if known and pred["status"] != "low-confidence":
            return pred
        V = self.vision
        p_make_cls = np.zeros(len(V.makes)); p_make_cls[self._mk_map] = self._group(probs, self.make_idx, len(self.makes))
        make, make_p, named, within = decide_unfamiliar(
            V.makes, p_make_cls, V.entries, V.entry_make, zs["name_probs"],
            cfg.get("unknown_make_weight", 0.5), cfg.get("open_name_conf", 0.3))
        closest = pred["label"]
        pred["closest_trained_class"] = closest
        if known:
            # Probably a trained class but the classifier is too unsure for any level of its hierarchy:
            # do not present its top class as the answer; it stays visible in the top-5 list.
            pred["possibly_outside"] = True
            if make_p >= cfg.get("unknown_make_conf", 0.5):
                pred.update(status="unknown-model", make=make, model="Model not recognised", year="unknown",
                            confidence=make_p, display=f"{make}, model not recognised")
            else:
                pred.update(status="unknown", make="", model="Unknown car", year="unknown",
                            display="Unknown car (outside the supported classes?)")
            return pred
        pred["in_training_set"] = False
        if named is not None:                                       # strong, make-consistent open-vocab name
            e = V.entries[named]
            pred.update(status="open-set", label=e["label"], make=e["make"], model=e["model"], year="unknown",
                        confidence=float(zs["name_probs"][named]),
                        display=f"{e['label']} (not in the 196 training classes)",
                        top5=[(l, p) for l, p, _ in zs["name_top5"]])
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


def draw_result(res: dict) -> np.ndarray:
    img = res["image"].copy()
    x1, y1, x2, y2 = res["box"]
    colour = {"confident": (40, 200, 70), "year-uncertain": (240, 180, 0),
              "model-uncertain": (255, 120, 0)}.get(res["status"], (230, 40, 40))
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
