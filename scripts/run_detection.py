"""Run detectors on the unseen test set, compute IoU / mAP, cache detected crops.

Compares: fine-tuned YOLO11 (ours), COCO-pretrained YOLO11n (no fine-tuning),
and the detector-free classical ROI extractor.

Usage:  python -m scripts.run_detection
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config as C
from src.classical import classical_roi
from src.data import cache_path, crop_image, resize_max
from src.detector import VehicleDetector, iou
from src.metrics import average_precision


def run(det: VehicleDetector | None, df: pd.DataFrame, name: str):
    rows, all_dets, times = [], [], []
    for r in tqdm(df.itertuples(), total=len(df), desc=name):
        img = cv2.cvtColor(cv2.imread(r.img_path), cv2.COLOR_BGR2RGB)
        t = time.perf_counter()
        if det is None:
            box, _ = classical_roi(img); dets = [dict(box=box, conf=1.0)]
        else:
            dets = det.detect(img, conf=0.01)
        times.append(time.perf_counter() - t)
        gt = (r.x1, r.y1, r.x2, r.y2)
        for d in dets[:10]:
            all_dets.append((r.uid, d["conf"], d["box"]))
        if dets:
            p = dets[0]
        else:
            p = dict(box=(0, 0, r.width - 1, r.height - 1), conf=0.0)
        rows.append(dict(uid=r.uid, px1=p["box"][0], py1=p["box"][1], px2=p["box"][2], py2=p["box"][3],
                         det_conf=p["conf"], iou=iou(p["box"], gt), n_dets=len(dets)))
    out = pd.DataFrame(rows)
    gts = {r.uid: (r.x1, r.y1, r.x2, r.y2) for r in df.itertuples()}
    ap50 = average_precision(all_dets, gts, 0.5)
    ap = float(np.mean([average_precision(all_dets, gts, t) for t in np.arange(0.5, 0.96, 0.05)]))
    summary = dict(detector=name, mean_iou=float(out.iou.mean()), median_iou=float(out.iou.median()),
                   iou_ge_0_5=float((out.iou >= 0.5).mean()), iou_ge_0_75=float((out.iou >= 0.75).mean()),
                   mAP50=ap50, mAP50_95=ap, ms_per_img=1000 * float(np.mean(times[5:])),
                   missed=int((out.n_dets == 0).sum()))
    print(json.dumps(summary), flush=True)
    return out, summary


def main():
    df = pd.read_csv(C.META_DIR / "annotations.csv")
    te = df[df.split == "test"]
    summaries = []

    ours = VehicleDetector()                                    # fine-tuned
    out, s = run(ours, te, "yolo11s_finetuned"); summaries.append(s)
    out.to_csv(C.RESULTS_DIR / "detections_test.csv", index=False)

    s = run(VehicleDetector("yolo11n.pt"), te, "yolo11n_coco_pretrained")[1]; summaries.append(s)
    s = run(None, te, "classical_edge_contour_roi")[1]; summaries.append(s)
    (C.RESULTS_DIR / "detection.json").write_text(json.dumps(summaries, indent=1))

    # cache crops from OUR detector's boxes -> used for end-to-end classification eval
    (C.CACHE_DIR / "detcrop").mkdir(parents=True, exist_ok=True)
    m = te.merge(out, on="uid")
    for r in tqdm(m.itertuples(), total=len(m), desc="detcrop"):
        img = cv2.imread(r.img_path)
        c = crop_image(img, (r.px1, r.py1, r.px2, r.py2))
        cv2.imwrite(str(cache_path(r.uid, "detcrop")), resize_max(c, C.CACHE_MAX_SIDE),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


if __name__ == "__main__":
    main()
