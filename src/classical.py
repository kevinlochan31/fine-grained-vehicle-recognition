"""Classical computer-vision components.

* Image quality assessment + adaptive preprocessing (gamma / CLAHE / denoise)
* Edge detection (auto-threshold Canny) and contour analysis
* Region-of-interest extraction without deep learning (edge + contour based)
* Hand-crafted features: HOG (shape), LBP (texture), HSV histograms (colour),
  edge-orientation histogram and geometric/contour descriptors.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

FEAT_SIZE = (192, 128)   # (w, h) canonical size for feature extraction


# ----------------------------------------------------------------- quality + preprocessing
def image_stats(rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    # Immerkaer (1996) fast noise-sigma estimate
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    h, w = gray.shape
    conv = cv2.filter2D(gray.astype(np.float64), -1, k)[1:-1, 1:-1]
    noise = float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * max(1, (w - 2) * (h - 2))))
    return dict(
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        sharpness=float(lap.var()),
        noise=noise,
        resolution=int(min(gray.shape)),
    )


def adaptive_preprocess(rgb: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """Choose enhancement steps from measured image characteristics.

    Returns the enhanced RGB image and a list of the steps that were applied
    (shown in the app as intermediate results).
    """
    stats = stats or image_stats(rgb)
    out, steps = rgb.copy(), []
    if stats["resolution"] < 160:                                      # tiny images
        s = 224 / stats["resolution"]
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        steps.append(f"bicubic upscale x{s:.1f}")
    if stats["noise"] > 12:                                            # noisy -> edge-preserving
        out = cv2.bilateralFilter(out, 7, 50, 7)
        steps.append("bilateral denoise")
    if stats["brightness"] < 70:                                       # under-exposed
        gamma = float(np.clip(np.log(0.45) / np.log(max(stats["brightness"], 5) / 255.0), 0.3, 1.0))
        lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
        steps.append(f"gamma {gamma:.2f}")
    elif stats["brightness"] > 200 and stats["contrast"] < 50:         # washed-out / over-exposed
        lut = (np.linspace(0, 1, 256) ** 1.6 * 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
        steps.append("gamma 1.60")
    if stats["contrast"] < 40 or stats["brightness"] < 70:             # low contrast -> CLAHE on L
        clip = float(np.clip(3.0 - stats["contrast"] / 30, 1.5, 3.0))
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        steps.append(f"CLAHE clip={clip:.1f}")
    return out, steps


def gray_world_white_balance(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    f *= f.mean() / (f.reshape(-1, 3).mean(0) + 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- edges / contours / ROI
def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = float(np.median(gray))
    lo, hi = int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v))
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), lo, max(hi, lo + 30))


def classical_roi(rgb: np.ndarray) -> tuple[tuple[int, int, int, int], dict]:
    """Detector-free vehicle ROI: edges -> morphology -> largest contour box.

    Used as (1) a fallback when YOLO finds nothing and (2) the 'classical only'
    localisation baseline.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = auto_canny(gray)
    k = max(3, int(min(h, w) * 0.02)) | 1
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=2)
    closed = cv2.dilate(closed, None, iterations=1)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0, 0, w - 1, h - 1), dict(edges=edges, mask=closed)
    # Score contours by area, penalising those that hug the whole frame border.
    best, best_s = None, -1
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 0.02 * w * h:
            continue
        frame_hug = (bw > 0.98 * w) + (bh > 0.98 * h)
        s = area * (0.6 if frame_hug == 2 else 1.0)
        if s > best_s:
            best, best_s = (x, y, x + bw - 1, y + bh - 1), s
    if best is None:
        best = (0, 0, w - 1, h - 1)
    return best, dict(edges=edges, mask=closed)


def contour_geometry(gray_crop: np.ndarray) -> np.ndarray:
    """Geometric descriptors of the dominant contour in a crop."""
    edges = auto_canny(gray_crop)
    h, w = gray_crop.shape
    cnts, _ = cv2.findContours(cv2.dilate(edges, None), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_density = edges.mean() / 255.0
    if not cnts:
        return np.r_[w / h, edge_density, 0, 0, 0, np.zeros(7)].astype(np.float32)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    x, y, bw, bh = cv2.boundingRect(c)
    hull = cv2.contourArea(cv2.convexHull(c)) + 1e-6
    extent = area / (bw * bh + 1e-6)
    solidity = area / hull
    perim = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (perim ** 2 + 1e-6)
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return np.r_[w / h, edge_density, extent, solidity, circularity, hu].astype(np.float32)


# ----------------------------------------------------------------- feature extraction
def hog_features(gray: np.ndarray) -> np.ndarray:
    # 9 orientations, 8x8 cells, 2x2 blocks (Dalal-Triggs layout) on a 192x128 crop.
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True)[::2]   # stride-2 subsample keeps dim manageable


def lbp_features(gray: np.ndarray, grid=(2, 3), P=8, R=1) -> np.ndarray:
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = P + 2
    h, w = gray.shape
    feats = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            cell = lbp[i * h // grid[0]:(i + 1) * h // grid[0], j * w // grid[1]:(j + 1) * w // grid[1]]
            hist, _ = np.histogram(cell, bins=n_bins, range=(0, n_bins))
            feats.append(hist / (hist.sum() + 1e-6))
    return np.concatenate(feats)


def color_features(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist /= hist.sum() + 1e-6
    moments = np.r_[hsv.reshape(-1, 3).mean(0), hsv.reshape(-1, 3).std(0)] / 255.0
    return np.r_[hist, moments]


def edge_orientation_hist(gray: np.ndarray, bins: int = 12) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(ang % 180, bins=bins, range=(0, 180), weights=mag)
    return hist / (hist.sum() + 1e-6)


def extract_features(rgb_crop: np.ndarray, groups=("hog", "lbp", "color", "edge", "geom")) -> np.ndarray:
    """Full classical descriptor for a vehicle crop (RGB uint8)."""
    img = cv2.resize(rgb_crop, FEAT_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)                       # photometric normalisation
    parts = {
        "hog": lambda: hog_features(gray),
        "lbp": lambda: lbp_features(gray),
        "color": lambda: color_features(img),
        "edge": lambda: edge_orientation_hist(gray),
        "geom": lambda: contour_geometry(gray),
    }
    return np.concatenate([parts[g]().astype(np.float32) for g in groups])


def feature_group_slices() -> dict[str, slice]:
    """Index ranges for each feature group (for ablation)."""
    dummy = np.zeros((FEAT_SIZE[1], FEAT_SIZE[0], 3), np.uint8) + 128
    out, start = {}, 0
    for g in ("hog", "lbp", "color", "edge", "geom"):
        n = len(extract_features(dummy, (g,)))
        out[g] = slice(start, start + n); start += n
    return out


# ----------------------------------------------------------------- visualisation helpers
def visual_steps(rgb: np.ndarray, box=None) -> dict[str, np.ndarray]:
    """Intermediate images for the UI."""
    stats = image_stats(rgb)
    enhanced, steps = adaptive_preprocess(rgb, stats)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    edges = auto_canny(gray)
    roi_box, info = classical_roi(enhanced)
    contour_vis = enhanced.copy()
    cnts, _ = cv2.findContours(info["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_vis, cnts, -1, (0, 255, 0), 2)
    cv2.rectangle(contour_vis, roi_box[:2], roi_box[2:], (255, 0, 0), 3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    lbp = (lbp / lbp.max() * 255).astype(np.uint8)
    return {
        "enhanced": enhanced, "steps": steps, "stats": stats,
        "edges": edges, "gradient": mag, "lbp": lbp,
        "contours": contour_vis, "classical_box": roi_box,
    }
