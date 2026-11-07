"""Open-vocabulary car vision with CLIP ViT-B/16 (LAION-2B), used by the hybrid pipeline for:

1. Vehicle gate     - "is there a car at all?" (prompt ensemble: car types vs. everyday non-car scenes)
2. Open-set naming  - zero-shot make + model over a catalogue of ~800 models from ~75 makes
                      (data_meta/car_catalogue.json) plus the 196 Stanford classes, so cars that are
                      NOT in the training set still get a sensible make/model answer.
3. Known/unknown    - how much CLIP probability mass falls on catalogue models outside the 196
                      classes: a strong signal that the fine-grained classifier is being asked about an
                      unseen model and must not be trusted blindly.

Only the image tower runs per request; text embeddings are computed once at start-up.
"""
from __future__ import annotations

import json
import re

import cv2
import numpy as np
import torch

from . import config as C
from .data import load_class_names, split_class_name

ARCH, PRETRAINED = "ViT-B-16", "laion2b_s34b_b88k"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

CAR_PROMPTS = [
    "a photo of a car.", "a photo of a vehicle.", "a photo of a sedan.", "a photo of a hatchback.",
    "a photo of a sports car.", "a photo of a supercar.", "a photo of a convertible car.", "a photo of an SUV.",
    "a photo of a pickup truck.", "a photo of a minivan.", "a photo of a passenger van.", "a photo of a station wagon.",
    "a photo of a classic car.", "a photo of an electric car.", "a photo of a car at night.", "a blurry photo of a car.",
    "a photo of the front of a car.", "a photo of the back of a car.", "a photo of a car parked on the street.",
    "a close-up photo of a car's headlights and grille.", "a photo of a car in a showroom.",
]
NOT_CAR_PROMPTS = [
    "a photo of a person.", "a selfie.", "a photo of a group of people.", "a photo of an animal.", "a photo of a dog.",
    "a photo of a cat.", "a photo of a bird.", "a photo of a flower.", "a photo of a plant in a pot.", "a photo of a tree.",
    "a photo of food.", "a photo of a drink.", "a photo of a building.", "a photo of a house.", "a photo of a landscape.",
    "a photo of the sky.", "a photo of the sea.", "a photo of a room.", "a photo of furniture.", "a photo of a kitchen.",
    "a photo of a document.", "a screenshot of a website.", "a photo of text.", "a photo of a phone or laptop.",
    "a photo of a bicycle.", "a photo of a motorcycle.", "a photo of a scooter.", "a photo of a bus.",
    "a photo of a semi truck.", "a photo of a tractor.", "a photo of a train.", "a photo of an airplane.",
    "a photo of a boat.", "a photo of a toy.", "a cartoon drawing.", "a photo of an object on a table.",
    "a photo of clothes.", "a photo of shoes.", "a photo of a sports game.", "an abstract pattern.", "a black image.",
    "a photo of a clock.", "a photo of a watch.", "a photo of a speedometer.", "a photo of a tyre on its own.",
    "a photo of a toy car.", "a photo of a car logo.",
]
NAME_TEMPLATES = ["a photo of a {}, a type of car.", "a photo of the {}.", "a {} car."]


def _family(make: str, model: str) -> str:
    """Normalised 'make + first model token', e.g. ('Audi', 'R8 Coupe') -> 'audi r8'."""
    tok = re.split(r"[\s\-]+", model.strip())[0].lower() if model.strip() else ""
    return f"{make.lower()} {tok}".strip()


class CarVision:
    def __init__(self, device: str | None = None):
        import open_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
        model = model.eval().to(self.device)
        tok = open_clip.get_tokenizer(ARCH)
        self.scale = float(model.logit_scale.exp())

        @torch.no_grad()
        def enc(texts):
            out = []
            for i in range(0, len(texts), 256):
                e = model.encode_text(tok(texts[i:i + 256]).to(self.device)).float()
                out.append(e / e.norm(dim=-1, keepdim=True))
            return torch.cat(out)

        def enc_names(names):
            e = enc([t.format(n) for n in names for t in NAME_TEMPLATES]).view(len(names), len(NAME_TEMPLATES), -1).mean(1)
            return e / e.norm(dim=-1, keepdim=True)

        self.t_car, self.t_not = enc(CAR_PROMPTS), enc(NOT_CAR_PROMPTS)
        # catalogue = 196 Stanford classes (with year) + extra make/model entries not covered by them
        stanford = load_class_names()
        self.entries = [dict(make=m, model=mo, year=y, stanford=i, label=n)
                        for i, n in enumerate(stanford) for (m, mo, y) in [split_class_name(n)]]
        fams = {_family(e["make"], e["model"]) for e in self.entries}
        cat = json.loads((C.META_DIR / "car_catalogue.json").read_text(encoding="utf8"))
        for make, models in cat.items():
            for mo in models:
                if _family(make, mo) in fams and any(e["make"] == make and e["model"].lower().startswith(mo.lower()) for e in self.entries[:196]):
                    continue                        # already a Stanford class (same make + model name)
                self.entries.append(dict(make=make, model=mo, year=None, stanford=-1, label=f"{make} {mo}"))
        self.is_stanford = np.array([e["stanford"] >= 0 for e in self.entries])
        self.t_names = enc_names([e["label"] for e in self.entries])
        self.makes = sorted({e["make"] for e in self.entries})
        self.entry_make = np.array([self.makes.index(e["make"]) for e in self.entries])   # entry -> make index
        self.t_makes = enc([f"a photo of a {m} car." for m in self.makes])
        self.visual = model.visual.half() if self.device == "cuda" else model.visual
        del model.transformer                         # text tower no longer needed
        self.half = self.device == "cuda"

    # ------------------------------------------------------------------ image side
    def _prep(self, imgs, size=224):
        x = np.empty((len(imgs), 3, size, size), np.float32)
        for i, im in enumerate(imgs):
            h, w = im.shape[:2]; s = size / min(h, w)              # resize short side + centre crop
            r = cv2.resize(im, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            y0, x0 = (r.shape[0] - size) // 2, (r.shape[1] - size) // 2
            r = r[y0:y0 + size, x0:x0 + size]
            x[i] = ((r.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        t = torch.from_numpy(x).to(self.device)
        return t.half() if self.half else t

    @torch.no_grad()
    def embed(self, imgs) -> torch.Tensor:
        f = self.visual(self._prep(imgs)).float()
        return f / f.norm(dim=-1, keepdim=True)

    def car_probability(self, f: torch.Tensor) -> np.ndarray:
        logits = self.scale * f @ torch.cat([self.t_car, self.t_not]).T
        p = logits.softmax(-1)
        return p[:, :len(CAR_PROMPTS)].sum(-1).cpu().numpy()

    def name_probs(self, f: torch.Tensor) -> np.ndarray:
        return (self.scale * f @ self.t_names.T).softmax(-1).cpu().numpy()

    def make_probs(self, f: torch.Tensor) -> np.ndarray:
        return (self.scale * f @ self.t_makes.T).softmax(-1).cpu().numpy()

    def analyse(self, full: np.ndarray, crop: np.ndarray | None) -> dict:
        imgs = [full] + ([crop] if crop is not None else [])
        f = self.embed(imgs)
        pc = self.car_probability(f)
        # Two uses, two inputs:
        # * routing ("is this one of the 196 classes?") uses the crop, as tuned in eval_openset; routing on
        #   the frame re-routed more known cars and cost robustness (occlusion 87.5% -> 85.7%).
        # * naming an unfamiliar car uses the mean embedding of frame + crop: on validation it names the make
        #   far better than the crop alone (92.6% vs 86.5%) and is safer than the frame alone with other cars.
        route = self.name_probs(f[-1:])[0]
        fm = f.mean(0, keepdim=True); fm = fm / fm.norm(dim=-1, keepdim=True)
        names = self.name_probs(fm)[0]
        order, norder = np.argsort(-route)[:5], np.argsort(-names)[:5]
        lab = lambda P, o: [(self.entries[int(i)]["label"], float(P[i]), self.entries[int(i)]["stanford"] >= 0) for i in o]
        return dict(
            car_prob=float(pc.max()), car_prob_full=float(pc[0]),
            route_probs=route, outside_mass=float(route[~self.is_stanford].sum()),
            top=self.entries[int(order[0])], top_p=float(route[order[0]]), top5=lab(route, order),
            name_probs=names, name_top5=lab(names, norder),
            feature=fm[0].cpu().numpy(),
        )


def decide_unfamiliar(make_names, p_make_cls, entries, entry_make, name_probs, w, name_conf):
    """Responsible answer for a car that is (probably) outside the supported classes.

    1. Make first: fuse the classifier's make probabilities (summed over its 196 classes) with CLIP's
       (summed over the catalogue), weight w on the classifier.
    2. Name a specific catalogue model only if CLIP's best model is of that make and CLIP is at least
       `name_conf` sure; otherwise report the make with "model not recognised".
    Returns (make, make_p, named_entry_index_or_None, candidate entry indices within the make).
    """
    p_zs = np.bincount(entry_make, weights=name_probs, minlength=len(make_names))
    p = w * p_make_cls + (1 - w) * p_zs
    m = int(p.argmax())
    top = int(name_probs.argmax())
    named = top if (entry_make[top] == m and name_probs[top] >= name_conf) else None
    within = np.where(entry_make == m)[0]
    within = within[np.argsort(-name_probs[within])][:5]
    return make_names[m], float(p[m]), named, within


def same_family(a_make, a_model, b_make, b_model) -> bool:
    return _family(a_make, a_model) == _family(b_make, b_model)
