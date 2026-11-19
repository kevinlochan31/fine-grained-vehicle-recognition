"""World-car layer: names cars outside the 196 Stanford Cars classes, including Indian-market cars, up to 2026.

This is a clearly separated EXTRA layer. It never changes the 196-class models (ConvNeXt-T + fine-tuned CLIP)
or their reported metrics; it only acts on the open-set path of the pipeline.

* World catalogue (data_meta/world_catalogue.json): ~3,000 make / model / generation entries (133 makes).
* Reference gallery (models/world_gallery.npz): CLIP ViT-B/16 embeddings (frame + crop mean, the same
  feature the app already computes for the vehicle gate) of freely licensed Wikimedia Commons photos of
  ~170 popular modern models, and a linear probe trained on those references together with the 196
  Stanford classes (Stanford train split), so it can tell a 2017 Camry from the 2012 Camry it was trained on.
* Zero-shot text naming over the whole catalogue (models/world_text_b16.npz) for models without photos.

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


def load_world_catalogue() -> list[dict]:
    return json.loads(CATALOGUE.read_text(encoding="utf8"))["entries"]


def _softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def make_key(m: str) -> str:
    m = (m or "").lower()
    return "suzuki" if m in ("maruti suzuki", "maruti") else m


class WorldCars:
    def __init__(self, device: str | None = None):
        z = np.load(GALLERY, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        self.name = meta.get("name", "b16")
        self.model = None
        if self.name != "b16":
            # A bigger CLIP chosen for the world layer only (the vehicle gate, routing and the 196-class
            # models keep their tuned ViT-B/16). Loaded once; only its image tower is kept.
            import open_clip
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            m, _, self.prep = open_clip.create_model_and_transforms(meta["arch"], pretrained=meta["pretrained"])
            self.model = m.visual.eval().to(self.device)
            if self.device == "cuda":
                self.model = self.model.half()
            del m
        self.classes = meta["classes"]
        self.K = meta["n_world"]
        self.W, self.b = z["W"].astype(np.float32), z["b"].astype(np.float32)
        self.ref_emb, self.ref_cls = z["ref_emb"].astype(np.float32), z["ref_cls"].astype(int)
        with open(REFS, encoding="utf8") as f:
            self.refs = list(csv.DictReader(f))
        t = np.load(C.MODELS_DIR / f"world_text_{self.name}.npz", allow_pickle=False)
        self.t_models, self.t_scale = t["t"].astype(np.float32), float(t["scale"])
        self.t_make, self.t_model = [str(x) for x in t["makes"]], [str(x) for x in t["models"]]
        # grouping of the probe's classes (world + 196 Stanford) by model and by make, for the middle tiers
        from .data import split_class_name
        allc = list(self.classes) + [dict(make=m, model=mo, stanford=i) for i, n in enumerate(meta["stanford_classes"])
                                     for (m, mo, _) in [split_class_name(n)]]
        mg, mk = class_groups(allc)
        self.mg_names, self.mk_names = sorted(set(mg)), sorted(set(mk))
        self.mg_idx = np.array([self.mg_names.index(g) for g in mg]); self.mk_idx = np.array([self.mk_names.index(g) for g in mk])
        self.world_groups = {g for g, c in zip(mg, allc) if c.get("stanford", -1) < 0}
        self.make_display = {}
        for c in allc:
            self.make_display.setdefault(make_key(c["make"]), c["make"])
        cat = load_world_catalogue()
        self.gens = {}
        for e in cat:
            self.gens.setdefault((e["make"], e["model"]), []).append(e)

    def embed(self, full: np.ndarray, crop: np.ndarray | None) -> np.ndarray:
        """Mean of frame + crop embeddings with the world layer's own backbone (when it is not the live B/16)."""
        import torch
        from PIL import Image
        ims = [full] + ([crop] if crop is not None else [])
        x = torch.stack([self.prep(Image.fromarray(i)) for i in ims]).to(self.device)
        with torch.no_grad():
            f = self.model(x.half() if self.device == "cuda" else x).float()
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.mean(0)
        return (f / f.norm()).cpu().numpy()

    def match(self, feature: np.ndarray) -> dict:
        """feature: L2-normalised CLIP ViT-B/16 embedding (mean of frame + crop) from CarVision.analyse."""
        f = feature.astype(np.float32)
        p = _softmax(self.W @ f + self.b)
        k = int(p.argmax())
        kw = int(p[:self.K].argmax())
        c = self.classes[kw]
        idx = np.where(self.ref_cls == kw)[0]
        sims = self.ref_emb[idx] @ f
        j = int(idx[int(sims.argmax())])
        ref = self.refs[j]
        pz = _softmax(self.t_scale * self.t_models @ f)
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
