"""Diagnose the occlusion regression: where do hybrid errors come from (gate / open-set / classifier)?"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from scripts.evaluate import corrupt
from src import config as C
from src.data import read_rgb
from src.pipeline import VehicleRecognizer


def main():
    rec = VehicleRecognizer()
    df = pd.read_csv(C.META_DIR / "annotations.csv")
    sub = df[df.split == "test"].sample(1000, random_state=C.SEED).head(400)
    rng = np.random.default_rng(C.SEED)
    cnt = collections.Counter()
    for r in sub.itertuples():
        img = corrupt(read_rgb(r.img_path), "occlusion_30pct", rng, (r.x1, r.y1, r.x2, r.y2))
        res = rec.analyse(img)
        h, d = res["hybrid"], res["deep"]
        hk = h["class_id"] == r.label
        cnt[("status", h["status"])] += 1
        if h["status"] == "open-set":
            cnt[("open-set, closest class correct", h.get("closest_trained_class") == r.class_name)] += 1
        cnt[("hybrid_ok", hk and h["in_training_set"] and h["status"] != "no-car")] += 1
        cnt[("ensemble class right (ignoring routing)", hk)] += 1
        cnt[("deep_ok", d["class_id"] == r.label)] += 1
    for k, v in sorted(cnt.items(), key=str):
        print(k, v)


if __name__ == "__main__":
    main()
