"""Fine-tune a COCO-pretrained YOLO11 on Stanford Cars bounding boxes (single class 'car').

Usage:  python -m scripts.train_detector --epochs 15
"""
from __future__ import annotations

import argparse
import shutil

from ultralytics import YOLO

from src import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(data=str(C.YOLO_DIR / "cars.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, workers=4, seed=C.SEED, project=str(C.RUNS_DIR), name="det",
                exist_ok=True, cos_lr=True, close_mosaic=3, plots=True, amp=True)
    best = C.RUNS_DIR / "det" / "weights" / "best.pt"
    shutil.copy2(best, C.MODELS_DIR / "yolo_cars.pt")
    print("saved", C.MODELS_DIR / "yolo_cars.pt")


if __name__ == "__main__":
    main()
