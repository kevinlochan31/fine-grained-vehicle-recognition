"""Central configuration: paths and hyper-parameters.

Heavy artefacts (raw dataset, cached crops, checkpoints) live in WORK_DIR, which
defaults to ~/cvh so they are not synced by OneDrive. Override with the
CVH_WORK_DIR environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = Path(os.environ.get("CVH_WORK_DIR", Path.home() / "cvh"))

RAW_DIR = WORK_DIR / "data" / "raw"
TRAIN_IMG_DIR = RAW_DIR / "cars_train" / "cars_train"
TEST_IMG_DIR = RAW_DIR / "cars_test" / "cars_test"
DEVKIT_DIR = RAW_DIR / "car_devkit" / "devkit"

CACHE_DIR = WORK_DIR / "data" / "cache"        # resized crops / full images
YOLO_DIR = WORK_DIR / "data" / "yolo"          # YOLO-format detection dataset
RUNS_DIR = WORK_DIR / "runs"                   # training runs

# Small, deployable artefacts are kept inside the repo.
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
META_DIR = PROJECT_ROOT / "data_meta"          # CSV splits + class names (small)

SEED = 42
VAL_FRACTION = 0.10      # stratified hold-out from the official train split
NUM_CLASSES = 196

# Fine-grained classifier
CLS_BACKBONE = os.environ.get("CVH_BACKBONE", "convnext_tiny.fb_in22k_ft_in1k_384")
CLS_IMG_SIZE = 384
CROP_PAD = 0.08          # context padding added around boxes before classification
CACHE_MAX_SIDE = 512     # cached crops are stored with longest side <= this

# Low-confidence handling (tuned on validation; see scripts/evaluate.py)
CONF_THRESHOLD = 0.50
MAKE_CONF_THRESHOLD = 0.50

# Detection
DET_CONF = 0.25
COCO_VEHICLE_IDS = {2: "car", 5: "bus", 7: "truck"}

for _d in (MODELS_DIR, RESULTS_DIR, META_DIR):
    _d.mkdir(parents=True, exist_ok=True)
