"""Size / latency of the CLIP components; merged into results/optimization.json.

Usage:  python -m scripts.bench_clip
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from src import config as C
from src.clip_classifier import WEIGHTS, ClipFineGrained
from src.data import read_rgb
from src.openvocab import CarVision


def lat(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n


def main():
    im = read_rgb(sorted((C.PROJECT_ROOT / "examples").glob("*.jpg"))[0])
    out = {}
    for dev in ("cuda", "cpu"):
        ft = ClipFineGrained(device=dev); cv = CarVision(device=dev)
        out[f"clip_ft_ms_{dev}"] = lat(lambda: ft.predict_proba([im], tta=True), n=10 if dev == "cpu" else 30)
        out[f"clip_gate_and_naming_ms_{dev}"] = lat(lambda: cv.analyse(im, im), n=10 if dev == "cpu" else 30)
        if dev == "cuda":
            out["clip_ft_params_M"] = round(sum(p.numel() for p in ft.model.parameters()) / 1e6, 1)
            out["clip_zero_shot_visual_params_M"] = round(sum(p.numel() for p in cv.visual.parameters()) / 1e6, 1)
        del ft, cv
    out["clip_ft_size_mb"] = round(WEIGHTS.stat().st_size / 2 ** 20, 1)
    out["note"] = ("CLIP ViT-B/16 fine-tune: FP16 weights; latency includes flip TTA (2 passes). Gate + naming: one "
                   "ViT-B/16 forward for frame and crop, text embeddings precomputed at start-up.")
    p = C.RESULTS_DIR / "optimization.json"
    R = json.loads(p.read_text()); R["clip"] = out; p.write_text(json.dumps(R, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
