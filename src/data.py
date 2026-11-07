"""Dataset parsing, splits, crop caching and torch Datasets for Stanford Cars."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

from . import config as C

MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")


def split_class_name(name: str) -> tuple[str, str, str]:
    """'Aston Martin V8 Vantage Convertible 2012' -> (make, model, year)."""
    tokens = name.split()
    year = tokens[-1]
    for mw in MULTI_WORD_MAKES:
        if name.startswith(mw + " "):
            make = mw
            model = name[len(mw) + 1: name.rfind(" ")]
            return make, model, year
    return tokens[0], " ".join(tokens[1:-1]), year


def load_class_names() -> list[str]:
    path = C.META_DIR / "class_names.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    meta = sio.loadmat(C.DEVKIT_DIR / "cars_meta.mat")
    names = [str(x[0]) for x in meta["class_names"][0]]
    path.write_text(json.dumps(names, indent=1), encoding="utf8")
    return names


def _parse_annos(mat_path: Path, img_dir: Path, split: str) -> pd.DataFrame:
    annos = sio.loadmat(mat_path)["annotations"][0]
    rows = []
    for a in annos:
        rows.append(dict(
            fname=str(a["fname"][0]),
            x1=int(a["bbox_x1"][0, 0]), y1=int(a["bbox_y1"][0, 0]),
            x2=int(a["bbox_x2"][0, 0]), y2=int(a["bbox_y2"][0, 0]),
            label=int(a["class"][0, 0]) - 1,           # 0-based
            split=split, img_path=str(img_dir / str(a["fname"][0])),
        ))
    return pd.DataFrame(rows)


def build_metadata(force: bool = False) -> pd.DataFrame:
    """Create data_meta/annotations.csv with train/val/test split column."""
    out = C.META_DIR / "annotations.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    names = load_class_names()
    tr = _parse_annos(C.DEVKIT_DIR / "cars_train_annos.mat", C.TRAIN_IMG_DIR, "train")
    te = _parse_annos(C.DEVKIT_DIR / "cars_test_annos_withlabels.mat", C.TEST_IMG_DIR, "test")

    # Stratified validation split carved from the official training set.
    rng = np.random.default_rng(C.SEED)
    val_idx = []
    for _, g in tr.groupby("label"):
        n_val = max(1, int(round(len(g) * C.VAL_FRACTION)))
        val_idx.extend(rng.choice(g.index.values, n_val, replace=False))
    tr.loc[val_idx, "split"] = "val"

    df = pd.concat([tr, te], ignore_index=True)
    widths, heights = [], []
    for p in df.img_path:
        im = cv2.imread(p)
        if im is None:
            raise FileNotFoundError(p)
        heights.append(im.shape[0]); widths.append(im.shape[1])
    df["width"], df["height"] = widths, heights
    # Clip boxes to image bounds (a few annotations exceed by 1 px).
    df["x2"] = np.minimum(df.x2, df.width - 1)
    df["y2"] = np.minimum(df.y2, df.height - 1)
    df["class_name"] = [names[i] for i in df.label]
    parts = [split_class_name(n) for n in df.class_name]
    df["make"] = [p[0] for p in parts]
    df["model"] = [p[1] for p in parts]
    df["year"] = [p[2] for p in parts]
    df["uid"] = df.split.map(lambda s: "test" if s == "test" else "train") + "_" + df.fname.str.replace(".jpg", "", regex=False)
    df.to_csv(out, index=False)
    return df


def class_hierarchy(names: list[str]):
    """Return (make_of_class, makes, makemodel_of_class, makemodels) index arrays."""
    parts = [split_class_name(n) for n in names]
    makes = sorted({p[0] for p in parts})
    mms = sorted({f"{p[0]} {p[1]}" for p in parts})
    make_idx = np.array([makes.index(p[0]) for p in parts])
    mm_idx = np.array([mms.index(f"{p[0]} {p[1]}") for p in parts])
    return make_idx, makes, mm_idx, mms


# ----------------------------------------------------------------------------- crops
def pad_box(box, w, h, pad=C.CROP_PAD):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - pad * bw); y1 = max(0, y1 - pad * bh)
    x2 = min(w - 1, x2 + pad * bw); y2 = min(h - 1, y2 + pad * bh)
    return int(x1), int(y1), int(round(x2)), int(round(y2))


def crop_image(img: np.ndarray, box, pad=C.CROP_PAD) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = pad_box(box, w, h, pad)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return img
    return img[y1:y2 + 1, x1:x2 + 1]


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def cache_path(uid: str, kind: str) -> Path:
    return C.CACHE_DIR / kind / f"{uid}.jpg"


def build_cache(df: pd.DataFrame, kinds=("crop", "full")) -> None:
    """Cache GT-box crops and downsized full images (speeds up training on Windows)."""
    from tqdm import tqdm
    for k in kinds:
        (C.CACHE_DIR / k).mkdir(parents=True, exist_ok=True)
    for r in tqdm(df.itertuples(), total=len(df), desc="caching"):
        todo = [k for k in kinds if not cache_path(r.uid, k).exists()]
        if not todo:
            continue
        img = cv2.imread(r.img_path)
        for k in todo:
            out = crop_image(img, (r.x1, r.y1, r.x2, r.y2)) if k == "crop" else img
            cv2.imwrite(str(cache_path(r.uid, k)), resize_max(out, C.CACHE_MAX_SIDE),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])


def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------- torch
def make_transforms(train: bool, size: int = C.CLS_IMG_SIZE):
    import torch
    from torchvision.transforms import v2 as T
    norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        return T.Compose([
            T.ToImage(),
            T.RandomResizedCrop((size, size), scale=(0.6, 1.0), ratio=(0.6, 1.66), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.3, 0.05)], p=0.8),
            T.RandomGrayscale(p=0.05),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.1),
            T.RandomApply([T.RandomRotation(12)], p=0.3),
            T.ToDtype(torch.float32, scale=True),
            norm,
            T.RandomErasing(p=0.25, scale=(0.02, 0.2)),   # simulates partial occlusion
        ])
    return T.Compose([
        T.ToImage(),
        T.Resize((size, size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        norm,
    ])


class CarsDataset:
    """Reads cached crops ('crop') or cached full images ('full')."""

    def __init__(self, df: pd.DataFrame, train: bool, kind: str = "crop",
                 size: int = C.CLS_IMG_SIZE, lowres_aug: bool = True):
        self.df = df.reset_index(drop=True)
        self.kind, self.train = kind, train
        self.tf = make_transforms(train, size)
        self.lowres_aug = lowres_aug and train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = read_rgb(cache_path(r.uid, self.kind))
        if self.lowres_aug and np.random.rand() < 0.15:          # resolution robustness
            h, w = img.shape[:2]
            s = np.random.uniform(0.2, 0.5)
            img = cv2.resize(cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s)))), (w, h))
        return self.tf(img), int(r.label)
