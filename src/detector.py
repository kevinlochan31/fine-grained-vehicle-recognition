"""Vehicle localisation: YOLO (fine-tuned on Stanford Cars boxes) with classical fallback."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C
from .classical import classical_roi


class VehicleDetector:
    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        from ultralytics import YOLO
        finetuned = C.MODELS_DIR / "yolo_cars.pt"
        if weights is None:
            weights = finetuned if finetuned.exists() else "yolo11n.pt"
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.single_class = len(self.model.names) == 1      # fine-tuned 'car' detector
        self.device = device

    def detect(self, rgb: np.ndarray, conf: float = C.DET_CONF, imgsz: int = 640):
        """Return list of dicts {box:(x1,y1,x2,y2), conf, source} sorted by score."""
        bgr = rgb[..., ::-1]
        res = self.model.predict(bgr, conf=conf, imgsz=imgsz, verbose=False, device=self.device)[0]
        out = []
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                           res.boxes.cls.cpu().numpy().astype(int)):
            if not self.single_class and k not in C.COCO_VEHICLE_IDS:
                continue
            out.append(dict(box=tuple(int(round(v)) for v in b), conf=float(c), source="yolo"))
        h, w = rgb.shape[:2]
        # Rank by confidence weighted by relative area -> the main subject vehicle first.
        out.sort(key=lambda d: d["conf"] * np.sqrt((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) / (w * h)),
                 reverse=True)
        return out

    def primary(self, rgb: np.ndarray, conf: float = C.DET_CONF):
        dets = self.detect(rgb, conf)
        if dets:
            return dets[0], dets
        box, _ = classical_roi(rgb)                       # classical fallback
        return dict(box=box, conf=0.0, source="classical-fallback"), []


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    ua = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter
    return inter / ua if ua > 0 else 0.0
