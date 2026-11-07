"""End-to-end check of the hybrid pipeline on hand-picked images (problem cases + examples).

Usage:  python -m scripts.probe_pipeline [extra image paths...]
"""
from __future__ import annotations

import glob
import sys
import time

import numpy as np
from skimage import data as skd

from src import config as C
from src.data import read_rgb
from src.pipeline import VehicleRecognizer


def main():
    rec = VehicleRecognizer()
    imgs = {"skimage clock": skd.clock(), "skimage coffee": skd.coffee(), "skimage astronaut": skd.astronaut()}
    for p in [str(C.WORK_DIR / "userimgs" / "tulips.jpg"), str(C.WORK_DIR / "userimgs" / "aston_new.jpg")] + sorted(glob.glob(str(C.PROJECT_ROOT / "examples" / "*.jpg"))) + sys.argv[1:]:
        try:
            imgs[p.split("\\")[-1]] = read_rgb(p)
        except Exception as e:
            print("skip", p, e)
    rec.analyse(skd.coffee())                                     # warm-up
    for k, im in imgs.items():
        if im.ndim == 2:
            im = np.stack([im] * 3, -1)
        t = time.perf_counter(); r = rec.analyse(im)["hybrid"]; dt = 1000 * (time.perf_counter() - t)
        print(f"{k[:28]:28s} -> {r['display'][:62]:62s} conf={r['confidence']:.2f} status={r['status']:15s} "
              f"car={r.get('car_probability', 1):.2f} [{dt:.0f} ms]")


if __name__ == "__main__":
    main()
