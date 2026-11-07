"""Step 1: parse annotations, make train/val/test split, cache crops, build YOLO dataset.

Usage:  python -m scripts.prepare_data
"""
from __future__ import annotations

import os
import shutil

from src import config as C
from src.data import build_cache, build_metadata, load_class_names


def build_yolo_dataset(df):
    """Single-class ('car') YOLO dataset. Images are hard-linked (no extra disk)."""
    for split in ("train", "val", "test"):
        (C.YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (C.YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        dst = C.YOLO_DIR / "images" / r.split / f"{r.uid}.jpg"
        if not dst.exists():
            try:
                os.link(r.img_path, dst)
            except OSError:
                shutil.copy2(r.img_path, dst)
        cx = (r.x1 + r.x2) / 2 / r.width; cy = (r.y1 + r.y2) / 2 / r.height
        bw = (r.x2 - r.x1) / r.width; bh = (r.y2 - r.y1) / r.height
        (C.YOLO_DIR / "labels" / r.split / f"{r.uid}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    (C.YOLO_DIR / "cars.yaml").write_text(
        f"path: {C.YOLO_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: car\n")


def main():
    load_class_names()
    df = build_metadata()
    print(df.split.value_counts())
    build_cache(df)
    build_yolo_dataset(df)
    print("done ->", C.META_DIR, C.CACHE_DIR, C.YOLO_DIR)


if __name__ == "__main__":
    main()
