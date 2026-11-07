"""Evaluation metrics: top-k accuracy, IoU-based AP (COCO 101-point), calibration."""
from __future__ import annotations

import numpy as np


def topk_acc(P: np.ndarray, y: np.ndarray, k: int) -> float:
    return float((np.argsort(-P, 1)[:, :k] == y[:, None]).any(1).mean())


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    ua = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter
    return inter / ua if ua > 0 else 0.0


def average_precision(dets, gts: dict, thr: float = 0.5) -> float:
    """dets: list of (image_id, score, box); gts: {image_id: box} (one GT per image)."""
    if not gts:
        return 0.0
    dets = sorted(dets, key=lambda d: -d[1])
    matched, tp = set(), np.zeros(len(dets))
    for i, (img, _, box) in enumerate(dets):
        g = gts.get(img)
        if g is not None and img not in matched and _iou(box, g) >= thr:
            tp[i] = 1; matched.add(img)
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    rec = ctp / len(gts); prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]            # monotone envelope
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def class_aware_map(pred_cls, scores, ious, y_true, n_classes, thr=0.5) -> float:
    """mAP where a detection is TP only if IoU>=thr AND predicted class is correct.

    One prediction per image (primary vehicle). AP averaged over classes present.
    """
    aps = []
    for c in range(n_classes):
        n_gt = int((y_true == c).sum())
        if n_gt == 0:
            continue
        sel = np.where(pred_cls == c)[0]
        if len(sel) == 0:
            aps.append(0.0); continue
        order = sel[np.argsort(-scores[sel])]
        tp = ((y_true[order] == c) & (ious[order] >= thr)).astype(float)
        ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
        rec = ctp / n_gt; prec = ctp / (ctp + cfp)
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        rs = np.linspace(0, 1, 101)
        idx = np.searchsorted(rec, rs, side="left")
        aps.append(float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx])))
    return float(np.mean(aps))


def expected_calibration_error(conf, correct, bins=15) -> float:
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
