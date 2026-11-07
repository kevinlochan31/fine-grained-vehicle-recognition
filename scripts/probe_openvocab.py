"""Quick qualitative probe of the CLIP gate / open-set naming on a few images.

Usage:  python -m scripts.probe_openvocab [image paths...]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from skimage import data as skd

from src import config as C
from src.data import read_rgb
from src.openvocab import CarVision


def main():
    cv = CarVision(device=sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("cpu", "cuda") else None)
    print("catalogue entries:", len(cv.entries), "extra (non-Stanford):", int((~cv.is_stanford).sum()), "makes:", len(cv.makes))
    imgs = {}
    for name in ("astronaut", "coffee", "chelsea", "rocket", "hubble_deep_field", "immunohistochemistry", "motorcycle_left"):
        try:
            im = getattr(skd, name)()
            if im.ndim == 2:
                im = np.stack([im] * 3, -1)
            imgs[f"skimage:{name}"] = im[..., :3].astype(np.uint8)
        except Exception:
            pass
    for p in sys.argv[1:]:
        if p not in ("cpu", "cuda"):
            imgs[p] = read_rgb(p)
    df = pd.read_csv(C.META_DIR / "annotations.csv")
    for r in df[df.split == "test"].sample(5, random_state=7).itertuples():
        imgs[f"stanford:{r.class_name}"] = read_rgb(r.img_path)
    for k, im in imgs.items():
        t = time.perf_counter(); a = cv.analyse(im, im); dt = (time.perf_counter() - t) * 1000
        print(f"{k[:60]:60s} car={a['car_prob']:.2f} outside={a['outside_mass']:.2f} "
              f"top={a['top']['label']} ({a['top_p']:.2f}) [{dt:.0f} ms]")


if __name__ == "__main__":
    main()
