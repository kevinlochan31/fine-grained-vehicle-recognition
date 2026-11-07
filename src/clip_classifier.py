"""Inference wrapper for the robust fine-tuned CLIP ViT-B/16 classifier (scripts/train_clip.py)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from . import config as C

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
WEIGHTS = C.MODELS_DIR / "clip_ft_fp16.pt"


class ClipFineGrained:
    def __init__(self, weights: str | Path = WEIGHTS, device: str | None = None, size: int = 224):
        import open_clip
        from scripts.train_clip import ClipClassifier
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        base, _, _ = open_clip.create_model_and_transforms(ck["arch"], pretrained=None)
        m = ClipClassifier(base.visual, torch.zeros(C.NUM_CLASSES, base.visual.output_dim))
        m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
        self.model = m.eval().to(self.device)
        self.half = self.device == "cuda"
        if self.half:
            self.model = self.model.half()
        self.size = size

    def _prep(self, crops):
        x = np.empty((len(crops), 3, self.size, self.size), np.float32)
        for i, c in enumerate(crops):
            r = cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_AREA if min(c.shape[:2]) > self.size else cv2.INTER_CUBIC)
            x[i] = ((r.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        t = torch.from_numpy(x).to(self.device)
        return t.half() if self.half else t

    @torch.no_grad()
    def logits(self, crops, tta: bool = True) -> np.ndarray:
        x = self._prep(crops)
        z = self.model(x).float()
        if tta:
            z = (z + self.model(torch.flip(x, dims=[3])).float()) / 2
        return z.cpu().numpy()

    def predict_proba(self, crops, tta: bool = True, batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(crops), batch):
            z = self.logits(crops[i:i + batch], tta)
            z -= z.max(1, keepdims=True); e = np.exp(z); out.append(e / e.sum(1, keepdims=True))
        return np.concatenate(out)
