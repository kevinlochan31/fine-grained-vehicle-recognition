"""World-car layer: names cars outside the 196 Stanford Cars classes, including Indian-market cars, up to 2026.

This is a clearly separated EXTRA layer. It never changes the 196-class models (ConvNeXt-T + fine-tuned CLIP)
or their reported metrics; it only acts on the open-set path of the pipeline.

* World catalogue (data_meta/world_catalogue.json): 4,689 make / model / generation entries (3,668 models).
* Reference gallery (models/world_gallery.npz): SigLIP ViT-B/16 embeddings (mean of frame + YOLO crop) of
  freely licensed Wikimedia Commons photos of 175 modern models, and a temperature-calibrated linear probe
  trained on those references together with the 196 Stanford classes (Stanford train split), so it can tell
  a 2017 Camry from the 2012 Camry the classifier was trained on.
* Zero-shot text naming over the whole catalogue (models/world_text_siglip.npz) for models without photos.
* Teaching (WorldCars.teach): the user adds their own photos of a car. Only that car's row of the probe is
  (re)trained, every other row stays exactly as tuned. User photos and state live in
  WORK_DIR/world_cars/user_teach (outside the repository) and are replayed at start-up.

Answers are hierarchical: make -> model -> generation (years), only when confident; otherwise the existing
"<make>, model not recognised" / "Unknown car" answers are kept. Thresholds live in models/hybrid_config.json
(world_route_conf, world_name_conf, world_zs_conf) and are tuned by scripts/eval_world.py.
"""
from __future__ import annotations

import csv
import json

import numpy as np

from . import config as C

CATALOGUE = C.META_DIR / "world_catalogue.json"
GALLERY = C.MODELS_DIR / "world_gallery.npz"
REFS = C.META_DIR / "world_refs.csv"
TRAIN_DIR = C.WORK_DIR / "world_cars"                     # probe training set (refs + Stanford train embeddings)
USER_DIR = C.WORK_DIR / "world_cars" / "user_teach"       # the user's own photos + taught state (not in the repo)


def load_world_catalogue() -> list[dict]:
    return json.loads(CATALOGUE.read_text(encoding="utf8"))["entries"]


def _softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def make_key(m: str) -> str:
    m = (m or "").lower()
    return "suzuki" if m in ("maruti suzuki", "maruti") else m


class WorldCars:
    def __init__(self, device: str | None = None, _skip_replay: bool = False):
        self._skip_replay = _skip_replay
        z = np.load(GALLERY, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        self.name = meta.get("name", "b16")
        self.model = self.zs_model = None
        if self.name != "b16":
            # A bigger CLIP chosen for the world layer only (the vehicle gate, routing and the 196-class
            # models keep their tuned ViT-B/16). Loaded once; only its image tower is kept.
            import copy
            import open_clip
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            m, _, self.prep = open_clip.create_model_and_transforms(meta["arch"], pretrained=meta["pretrained"])
            self.model = m.visual.eval()
            if meta.get("weights"):
                # Fine-tuned image tower (scripts/train_world.py). Its features no longer line up with the text
                # tower, so the untouched pretrained tower is kept for zero-shot naming of cars without photos.
                self.zs_model = copy.deepcopy(self.model).to(self.device)
                sd = torch.load(C.MODELS_DIR / meta["weights"], map_location="cpu")
                self.model.load_state_dict({k: v.float() for k, v in sd.items()})
            self.model = self.model.to(self.device)
            if self.device == "cuda":
                self.model = self.model.half()
                if self.zs_model is not None:
                    self.zs_model = self.zs_model.half()
            del m
        self.classes = [dict(c) for c in meta["classes"]]
        self.K = meta["n_world"]
        self.W, self.b = z["W"].astype(np.float32), z["b"].astype(np.float32)
        self.ref_emb, self.ref_cls = z["ref_emb"].astype(np.float32), z["ref_cls"].astype(int)
        with open(REFS, encoding="utf8") as f:
            self.refs = list(csv.DictReader(f))
        self.user_X, self.user_y = [], []
        t = np.load(C.MODELS_DIR / f"world_text_{meta.get('zero_shot_backbone', self.name)}.npz", allow_pickle=False)
        self.t_models, self.t_scale = t["t"].astype(np.float32), float(t["scale"])
        self.t_make, self.t_model = [str(x) for x in t["makes"]], [str(x) for x in t["models"]]
        self.stanford_names = meta["stanford_classes"]
        self._regroup()
        self._load_train()
        cat = load_world_catalogue()
        self.gens = {}
        for e in cat:
            self.gens.setdefault((e["make"], e["model"]), []).append(e)
        # identifies this trained gallery: embeddings saved for an older training run are not reused
        import hashlib
        self.version = hashlib.md5(np.ascontiguousarray(z["b"]).tobytes()).hexdigest()[:12]
        self.taught = []                       # user-taught entries (replayed from USER_DIR/state.json)
        self.auto = []                         # self-learnt entries (replayed from USER_DIR/auto_state.json)
        self._base = {}                        # tuned row (and its reference count) of every updated car
        self._replay()

    # ------------------------------------------------------------------ grouping for the middle tiers
    def _regroup(self):
        from .data import split_class_name
        allc = list(self.classes) + [dict(make=m, model=mo, stanford=i) for i, n in enumerate(self.stanford_names)
                                     for (m, mo, _) in [split_class_name(n)]]
        mg, mk = class_groups(allc)
        self.mg_names, self.mk_names = sorted(set(mg)), sorted(set(mk))
        self.mg_idx = np.array([self.mg_names.index(g) for g in mg]); self.mk_idx = np.array([self.mk_names.index(g) for g in mk])
        self.world_groups = {g for g, c in zip(mg, allc) if c.get("stanford", -1) < 0}
        self.make_display = {}
        for c in allc:
            self.make_display.setdefault(make_key(c["make"]), c["make"])

    # ------------------------------------------------------------------ teaching with the user's own photos
    def _load_train(self):
        p = TRAIN_DIR / f"probe_train_{self.name}.npz"
        self.train_X = self.train_y = None
        if p.exists():
            z = np.load(p)
            self.train_X, self.train_y = z["X"].astype(np.float32), z["y"].astype(int)
            self.train_T = float(z["T"]) if "T" in z else 1.0

    @property
    def can_teach(self) -> bool:
        return self.train_X is not None and self.model is not None

    def _find_class(self, make, model, y0):
        for i, c in enumerate(self.classes):
            if make_key(c["make"]) == make_key(make) and c["model"].lower() == model.lower() and c.get("y0") == y0:
                return i
        return None

    def _insert_class(self, c):
        k = self.K                                          # new world rows go before the 196 Stanford rows
        self.classes.append(c)
        self.W = np.insert(self.W, k, self.W[:self.K].mean(0), axis=0)
        self.b = np.insert(self.b, k, self.b[:self.K].min())
        self.train_y = np.where(self.train_y >= k, self.train_y + 1, self.train_y)
        self.user_y = [y + 1 if y >= k else y for y in self.user_y]
        self.K += 1
        self._regroup()
        return k

    def _fit_row(self, k, steps=400):
        """Update row k from the user / self-learnt photos of that car. The shipped head is a cosine
        classifier (row = scale x unit direction), so the row is 'imprinted': its direction moves towards the
        mean embedding of the new photos, weighted by how many there are against the car's reference photos,
        and its length (scale) and bias stay as tuned. A new car gets the mean direction, the median scale and
        the median bias of the world rows. (Re-fitting the row by gradient descent, as the round-1 linear probe
        did, wrecks a fine-tuned row: p(true) on its test photos fell from 0.36 to 0.002 with no new data.)"""
        if getattr(self, "cosine_head", True):
            fs = [x for x, y in zip(self.user_X, self.user_y) if y == k]
            if not fs:
                return
            m = np.mean(fs, 0); m = m / max(1e-6, float(np.linalg.norm(m)))
            norms = np.linalg.norm(self.W[:self.K], axis=1)
            if self.classes[k].get("user") and not self.classes[k].get("_trained"):
                others = [i for i in range(self.K) if i != k]
                self.W[k] = float(np.median(norms[others])) * m
                self.b[k] = float(np.median(self.b[others]))
                self.classes[k]["_trained"] = True
                self._base[k] = (self.W[k].copy(), 0)
                return
            if k not in self._base:              # tuned row + how many reference photos it stands for
                self._base[k] = (self.W[k].copy(), int((self.ref_cls == k).sum()))
            w0, n_ref = self._base[k]
            r = float(np.linalg.norm(w0)); d0 = w0 / r
            a = len(fs) / (len(fs) + max(n_ref, 5))          # 3 new photos vs 20 references -> 13% of the way
            d = (1 - a) * d0 + a * m
            self.W[k] = r * d / float(np.linalg.norm(d))
            return
        self._fit_row_probe(k, steps)

    def _fit_row_probe(self, k, steps=400):
        """Train row k only (all other rows frozen) on base data + every user photo, with balanced weights."""
        # Same recipe as the original probe (scripts/eval_world.fit_probe): raw (pre-temperature) logit space,
        # class-balanced cross-entropy, prototype x 10 initialisation, Adam lr 1e-2 with weight decay 1e-4,
        # 400 steps; then the same temperature T as the tuned gallery.
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        T = self.train_T
        X = self.train_X; y = self.train_y
        if self.user_X:
            X = np.concatenate([X, np.stack(self.user_X)]); y = np.concatenate([y, np.array(self.user_y)])
        Xt = torch.from_numpy(X).to(dev); yt = torch.from_numpy(y).long().to(dev)
        fixed = torch.from_numpy((X @ self.W.T + self.b) * T).to(dev)  # raw logits of the frozen rows
        pos = y == k
        n_cls = self.W.shape[0]
        cnt = torch.bincount(yt, minlength=n_cls).float().clamp(min=1)
        wts = cnt.sum() / (n_cls * cnt)
        if self.classes[k].get("user") and not self.classes[k].get("_trained"):
            proto = X[pos].mean(0); proto = proto / max(1e-6, float(np.linalg.norm(proto)))
            w0, b0 = 10.0 * proto, 0.0                                  # new car: prototype start (as fit_probe)
            self.classes[k]["_trained"] = True
        else:
            w0, b0 = self.W[k] * T, float(self.b[k]) * T                # known car: continue from its tuned row
            steps = steps // 2
        w = torch.nn.Parameter(torch.from_numpy(np.asarray(w0, np.float32)).to(dev))
        bb = torch.nn.Parameter(torch.tensor(float(b0), device=dev))
        opt = torch.optim.Adam([w, bb], lr=1e-2, weight_decay=1e-4)
        onehot = torch.nn.functional.one_hot(torch.tensor(k, device=dev), n_cls).float()
        for _ in range(steps):
            opt.zero_grad()
            logits = fixed * (1 - onehot) + (Xt @ w + bb)[:, None] * onehot
            torch.nn.functional.cross_entropy(logits, yt, weight=wts).backward(); opt.step()
        self.W[k] = w.detach().cpu().numpy() / T; self.b[k] = float(bb.detach().cpu()) / T

    AUTO_MAX_PER_CLASS = 15

    def auto_learn(self, make, model, gen, years, y0, y1, feat, image) -> bool:
        """Self-learning: add a photo the layer itself named with high confidence as an extra reference of that
        car (no user input). Capped per car and skipped for near-duplicates, so repeats of one photo cannot
        drag a row. Returns True when the photo was learnt."""
        if self.train_X is None:
            return False
        k = self._find_class(make, model, y0)
        if k is None:
            return False
        mine = [np.asarray(f, np.float32) for e in self.auto if (e["make"], e["model"], e["y0"]) == (make, model, y0)
                for f in e["feats"]]
        feat = np.asarray(feat, np.float32)
        if len(mine) >= self.AUTO_MAX_PER_CLASS or any(float(feat @ m) > 0.97 for m in mine):
            return False
        self.teach(make, model, gen, years, y0, y1, [feat], [image], save=False, author="self-learnt by AutoLens")
        self.auto.append(dict(make=make, model=model, gen=gen, years=years, y0=y0, y1=y1, images=[image],
                              feats=[feat.round(6).tolist()]))
        USER_DIR.mkdir(parents=True, exist_ok=True)
        (USER_DIR / "auto_state.json").write_text(json.dumps(dict(version=self.version, auto=self.auto)), encoding="utf8")
        return True

    def teach(self, make, model, gen, years, y0, y1, feats, images, save=True, author="you (taught in AutoLens)") -> dict:
        """Add the user's photos (feats: L2-normalised embeddings from self.embed) as references of a car and
        retrain only that car's row. Returns a summary. images: list of (thumbnail path relative to USER_DIR)."""
        if self.train_X is None:
            raise RuntimeError("teaching needs the probe training set (run scripts/eval_world.py build)")
        k = self._find_class(make, model, y0)
        new = k is None
        if new:
            k = self._insert_class(dict(key=f"user_{make_key(make)}_{model}_{y0}".replace(" ", "_").lower(),
                                        make=make, model=model, gen=gen, years=years, y0=y0, y1=y1, user=True))
        # user labels created before an insertion keep pointing at the right class
        for f, im in zip(feats, images):
            self.user_X.append(np.asarray(f, np.float32)); self.user_y.append(k)
            self.ref_emb = np.concatenate([self.ref_emb, np.asarray(f, np.float32)[None]])
            self.ref_cls = np.append(self.ref_cls, k)
            self.refs.append(dict(image=f"user_refs/{im}", author=author, license="your photo",
                                  license_url="", source="", title=im, user="1"))
        self._fit_row(k)
        n_refs = int((self.ref_cls == k).sum())
        entry = dict(make=make, model=model, gen=gen, years=years, y0=y0, y1=y1, images=list(images),
                     feats=[np.asarray(f, np.float32).round(6).tolist() for f in feats])
        if save:
            self.taught.append(entry); self._save()
        return dict(make=make, model=model, gen=gen, years=years, new_class=new, photos_added=len(feats), references=n_refs)

    def forget(self, index: int):
        """Remove one taught entry: reload the tuned gallery and replay the others."""
        rest = [e for i, e in enumerate(self.taught) if i != index]
        removed = self.taught[index]
        for im in removed["images"]:
            try:
                (USER_DIR / "images" / im).unlink()
            except OSError:
                pass
        self.__init__(device=getattr(self, "device", None), _skip_replay=True)
        self.taught = []
        for e in rest:
            self.teach(e["make"], e["model"], e["gen"], e["years"], e["y0"], e["y1"], e["feats"], e["images"])
        self._save()

    def _save(self):
        USER_DIR.mkdir(parents=True, exist_ok=True)
        (USER_DIR / "state.json").write_text(json.dumps(dict(backbone=self.name, version=self.version,
                                                             taught=self.taught)), encoding="utf8")

    def _replay(self):
        if getattr(self, "_skip_replay", False) or self.train_X is None:
            return
        p = USER_DIR / "state.json"
        if p.exists():
            st = json.loads(p.read_text(encoding="utf8"))
            # taught with another backbone or an older training run: embeddings incompatible
            if st.get("backbone") == self.name and st.get("version", self.version) == self.version:
                for e in st.get("taught", []):
                    self.teach(e["make"], e["model"], e["gen"], e["years"], e["y0"], e["y1"], e["feats"], e["images"],
                               save=False)
                    self.taught.append(e)
        p = USER_DIR / "auto_state.json"
        if p.exists():
            st = json.loads(p.read_text(encoding="utf8"))
            if st.get("version") == self.version:
                for e in st.get("auto", []):
                    if self._find_class(e["make"], e["model"], e["y0"]) is None:
                        continue
                    self.teach(e["make"], e["model"], e["gen"], e["years"], e["y0"], e["y1"], e["feats"], e["images"],
                               save=False, author="self-learnt by AutoLens")
                    self.auto.append(e)

    def embed(self, full: np.ndarray, crop: np.ndarray | None, model=None) -> np.ndarray:
        """Mean of frame + crop embeddings with the world layer's own backbone (when it is not the live B/16)."""
        import torch
        from PIL import Image
        ims = [full] + ([crop] if crop is not None else [])
        x = torch.stack([self.prep(Image.fromarray(i)) for i in ims]).to(self.device)
        with torch.no_grad():
            f = (model or self.model)(x.half() if self.device == "cuda" else x).float()
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.mean(0)
        return (f / f.norm()).cpu().numpy()

    def embed_pair(self, full: np.ndarray, crop: np.ndarray | None):
        """(feature for the trained head, feature for zero-shot text naming)."""
        f = self.embed(full, crop)
        return f, (self.embed(full, crop, self.zs_model) if self.zs_model is not None else f)

    def match(self, feature: np.ndarray, zs_feature: np.ndarray | None = None) -> dict:
        """feature: L2-normalised embedding (mean of frame + crop) from the world backbone (or the live B/16).
        zs_feature: embedding for zero-shot text naming when it differs (fine-tuned backbone)."""
        f = feature.astype(np.float32)
        fz = f if zs_feature is None else zs_feature.astype(np.float32)
        p = _softmax(self.W @ f + self.b)
        k = int(p.argmax())
        kw = int(p[:self.K].argmax())
        c = self.classes[kw]
        idx = np.where(self.ref_cls == kw)[0]
        sims = self.ref_emb[idx] @ f
        j = int(idx[int(sims.argmax())])
        ref = self.refs[j]
        pz = _softmax(self.t_scale * self.t_models @ fz)
        zi = int(pz.argmax())
        # zero-shot make distribution over the catalogue (for the make-consistency check)
        zmake = {}
        for m, q in zip(self.t_make, pz):
            zmake[make_key(m)] = zmake.get(make_key(m), 0.0) + float(q)
        # middle tiers
        pmg = np.bincount(self.mg_idx, weights=p, minlength=len(self.mg_names))
        pmk = np.bincount(self.mk_idx, weights=p, minlength=len(self.mk_names))
        gi, ki = int(pmg.argmax()), int(pmk.argmax())
        g = self.mg_names[gi]
        # best world class of that model group (for the label and the reference photo)
        in_g = [i for i in range(self.K) if self.mg_idx[i] == gi]
        mc = max(in_g, key=lambda i: p[i]) if in_g else None
        mk_name = self.mk_names[ki]
        cands = sorted([i for i in range(self.K) if self.mk_idx[i] == ki], key=lambda i: -p[i])
        seen, cand_names = set(), []
        for i in cands:
            nm = self.classes[i]["model"]
            if nm not in seen:
                seen.add(nm); cand_names.append(nm)
        model_tier = None
        if mc is not None and g in self.world_groups:
            idx2 = np.where(self.ref_cls == mc)[0]; s2 = self.ref_emb[idx2] @ f
            model_tier = dict(cls=self.classes[mc], p=float(pmg[gi]), ref=self.refs[int(idx2[int(s2.argmax())])], ref_sim=float(s2.max()))
        make_tier = dict(make=self.make_display.get(mk_name, mk_name), p=float(pmk[ki]), candidates=cand_names[:3])
        return dict(model_tier=model_tier, make_tier=make_tier,
                    top_is_world=k < self.K, p_top=float(p[k]), stanford_class=(k - self.K) if k >= self.K else -1,
                    # belief that the car is NOT one of the 196 Stanford classes (used to doubt the classifier)
                    world_mass=float(p[:self.K].sum()),
                    cls=c, p=float(p[kw]), ref=ref, ref_sim=float(sims.max()),
                    zs_make=self.t_make[zi], zs_model=self.t_model[zi], zs_p=float(pz[zi]),
                    zs_make_probs=zmake)

    def generations(self, make: str, model: str) -> list[dict]:
        return self.gens.get((make, model), [])


def class_groups(classes: list[dict]):
    """Per probe class: (model-group key, make key). Generations of one model share a model group; a Stanford
    class joins the group of its family (make + first model word), e.g. 'toyota|camry'."""
    import re
    mg, mk = [], []
    for c in classes:
        m = make_key(c["make"])
        stanford = c.get("stanford", -1) >= 0
        model = re.split(r"[\s\-]+", c["model"].lower())[0] if stanford else c["model"].lower()
        mg.append(f"{m}|{model}"); mk.append(m)
    return mg, mk


def describe_world(c: dict) -> str:
    """'Maruti Suzuki Swift · 3rd gen (2017–2024)'."""
    s = f"{c['make']} {c['model']}"
    if c.get("gen"):
        s += f" · {c['gen']}" + (f" ({c['years']})" if c.get("years") else "")
    elif c.get("years"):
        s += f" ({c['years']})"
    return s
