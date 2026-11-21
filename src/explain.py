"""'Show the working' for one prediction: what the image looked like at each stage and how the
final number was calculated.

    * difference image  - |enhanced - original| as a heat map (what the classical enhancement changed)
    * HOG picture       - the gradient-orientation histograms the classical classifier reads
    * Grad-CAM          - where the ConvNeXt classifier looked for the predicted class
    * calculation       - every number from image statistics to the final decision, with the formulas

Read-only: nothing here changes a prediction; it only explains the hybrid result produced by
VehicleRecognizer.run().
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.feature import hog

from .classical import FEAT_SIZE, adaptive_preprocess, image_stats
from .classifier import preprocess_batch


# ----------------------------------------------------------------------------- images
def _overlay(rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h = cv2.applyColorMap((np.clip(heat01, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) * (1 - alpha) + h.astype(np.float32) * alpha).astype(np.uint8)


def difference_image(crop: np.ndarray) -> dict:
    """What the adaptive enhancement changed, pixel by pixel."""
    enh, steps = adaptive_preprocess(crop)
    if enh.shape[:2] != crop.shape[:2]:                       # the upscale step changes the size
        enh_cmp = cv2.resize(enh, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        enh_cmp = enh
    d = np.abs(enh_cmp.astype(np.int16) - crop.astype(np.int16)).mean(2)       # 0..255 per pixel
    mean_change = float(d.mean())
    vis_max = max(float(np.percentile(d, 99)), 1.0)
    heat = cv2.applyColorMap((np.clip(d / vis_max, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return dict(before=crop, after=enh_cmp, diff=heat, steps=steps, mean_change=mean_change,
                max_change=float(d.max()), changed_pct=float((d > 10).mean() * 100))


def hog_picture(crop: np.ndarray) -> dict:
    """Same canonical size + histogram equalisation as the classical features (src/classical.py)."""
    img = cv2.resize(crop, FEAT_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    _, vis = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                 block_norm="L2-Hys", visualize=True)
    vis = rescale_intensity(vis, in_range=(0, max(float(vis.max()), 1e-6)), out_range=(0, 255)).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    vis = cv2.resize(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), (FEAT_SIZE[0] * 2, FEAT_SIZE[1] * 2),
                     interpolation=cv2.INTER_NEAREST)
    cells = (FEAT_SIZE[0] // 8) * (FEAT_SIZE[1] // 8)
    return dict(image=vis, gray=gray, cells=cells, size=FEAT_SIZE)


def grad_cam(classifier, crop: np.ndarray, class_id: int) -> np.ndarray | None:
    """Grad-CAM (Selvaraju et al., 2017) on the last ConvNeXt stage for `class_id`.
    Returns an RGB overlay at the crop's size, or None for the ONNX backend."""
    if getattr(classifier, "backend", "torch") != "torch":
        return None
    torch = classifier.torch
    model = classifier.model
    target = model.stages[-1]
    store = {}

    def fwd(_, __, out):
        out.retain_grad()
        store["a"] = out

    h = target.register_forward_hook(fwd)
    try:
        x = torch.from_numpy(preprocess_batch([crop], classifier.size)).to(classifier.device)
        if classifier.half:
            x = x.half()
        with torch.enable_grad():
            x.requires_grad_(True)
            z = model(x)
            model.zero_grad(set_to_none=True)
            z[0, class_id].backward()
        a, g = store["a"].float()[0], store["a"].grad.float()[0]          # C x h x w
        w = g.mean(dim=(1, 2))                                           # alpha_k = mean gradient
        cam = torch.relu((w[:, None, None] * a).sum(0)).detach().cpu().numpy()
    finally:
        h.remove()
        model.zero_grad(set_to_none=True)
    if cam.max() <= 0:
        cam = np.zeros_like(cam)
    else:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(cam, (0, 0), max(1.0, min(crop.shape[:2]) / 60))
    cam = cam / max(float(cam.max()), 1e-6)
    return _overlay(crop, cam)


# ----------------------------------------------------------------------------- numbers
def _top(names, p, k=3):
    return [{"label": names[i], "p": float(p[i])} for i in np.argsort(-p)[:k]]


def calculation(rec, res: dict) -> list[dict]:
    """Step-by-step numbers for the hybrid result `res` (output of VehicleRecognizer.run)."""
    cfg, names = rec.cfg, rec.names
    st = image_stats(res["crop"])            # the enhancement rules run on the detected crop, not the full photo
    b, c, n, r = st["brightness"], st["contrast"], st["noise"], st["resolution"]
    steps = []

    gamma = float(np.clip(np.log(0.45) / np.log(max(b, 5) / 255.0), 0.3, 1.0))
    rules = [
        dict(rule="Resolution < 160 px -> bicubic upscale", value=f"{r} px", fired=r < 160),
        dict(rule="Noise > 12 -> bilateral denoise", value=f"{n:.1f}", fired=n > 12),
        dict(rule="Brightness < 70 -> gamma = log(0.45) / log(b/255)", value=f"{b:.1f}"
             + (f" -> gamma {gamma:.2f}" if b < 70 else ""), fired=b < 70),
        dict(rule="Brightness > 200 and contrast < 50 -> gamma 1.6", value=f"{b:.1f} / {c:.1f}",
             fired=b > 200 and c < 50),
        dict(rule="Contrast < 40 or brightness < 70 -> CLAHE, clip = 3 - c/30", value=f"{c:.1f}",
             fired=c < 40 or b < 70),
    ]
    steps.append(dict(id="quality", title="1. Measure the car crop",
                      text="Measured on the detected car crop. Brightness = mean grey level (0-255), contrast = its "
                           "standard deviation, noise = Immerkaer "
                           "estimate, resolution = shorter side. Each rule fires only if its condition is true.",
                      rules=rules))

    box = res["box"]
    steps.append(dict(id="detect", title="2. Find the car",
                      text=f"YOLO11s box ({box[0]}, {box[1]}) to ({box[2]}, {box[3]}), confidence "
                           f"{res['det_conf']:.3f} ({res['det_source']}). The crop adds 8% margin on each side.",
                      values=[dict(k="Detector confidence", v=round(float(res["det_conf"]), 3))]))

    g = float(res.get("car_probability", 1.0))
    gs, gt, gd = cfg.get("gate_strong", 0.9), cfg.get("gate_threshold", 0.5), cfg.get("gate_det_conf", 0.3)
    steps.append(dict(id="gate", title="3. Is it a car? (CLIP gate)",
                      text=f"P(car) = softmax over 21 car prompts and 47 non-car prompts (scale 100), summed over "
                           f"the car prompts. Accept if P(car) >= {gs} or (P(car) >= {gt} and detector >= {gd}).",
                      values=[dict(k="P(car)", v=round(g, 4)),
                              dict(k="Accepted because", v=f"P(car) >= {gs}" if g >= gs
                                   else f"P(car) >= {gt} and detector agrees")]))

    m = res.get("members") or {}
    fp = res.get("final_probs")
    if m and fp is not None:
        T1, T2, w = cfg.get("temperature", 1.0), cfg.get("clip_temperature", 1.0), m.get("ensemble_weight", 0.5)
        k = int(np.argmax(fp))
        rows = []
        for i in np.argsort(-fp)[:3]:
            a, cc = float(m["convnext"][i]), float(m["clip"][i]) if "clip" in m else 0.0
            rows.append(dict(label=names[i], convnext_raw=float(m["convnext_raw"][i]), convnext=a,
                             clip_raw=float(m["clip_raw"][i]) if "clip_raw" in m else None, clip=cc,
                             final=float(fp[i])))
        steps.append(dict(id="members", title="4. Two neural networks vote",
                          text=f"Each network runs on the crop and its mirror image (flip TTA) and the two softmax "
                               f"outputs are averaged. Calibration sharpens or softens them: "
                               f"p_i = p_i^(1/T) / sum_j p_j^(1/T), with T = {T1} for ConvNeXt and T = {T2} for CLIP.",
                          table=rows))
        a, cc, f = float(m["convnext"][k]), float(m.get("clip", m["convnext"])[k]), float(fp[k])
        steps.append(dict(id="ensemble", title="5. Combine the votes",
                          text=f"Final = {w} x ConvNeXt + {round(1 - w, 2)} x CLIP (weights chosen on the validation set).",
                          formula=f"{w} x {a:.4f} + {round(1 - w, 2)} x {cc:.4f} = {f:.4f}",
                          label=names[k]))

        t, mt = cfg["conf_threshold"], cfg["make_threshold"]
        pm = rec._group(fp, rec.make_idx, len(rec.makes))
        pmm = rec._group(fp, rec.mm_idx, len(rec.mms))
        lv = [dict(level="Exact class (make + model + year)", name=names[k], p=f, threshold=t, ok=f >= t),
              dict(level="Make + model (sum over years)", name=rec.mms[int(pmm.argmax())], p=float(pmm.max()),
                   threshold=t, ok=float(pmm.max()) >= t),
              dict(level="Make only (sum over all its models)", name=rec.makes[int(pm.argmax())],
                   p=float(pm.max()), threshold=mt, ok=float(pm.max()) >= mt)]
        steps.append(dict(id="hierarchy", title="6. How sure is it? (hierarchical back-off)",
                          text="Probabilities of classes that share a make (or make + model) are added up. The answer "
                               "is the most specific level whose probability passes its threshold.",
                          levels=lv))

    om = res.get("outside_mass")
    steps.append(dict(id="decision", title="7. Final answer",
                      text=("Open-set check: CLIP compares the crop with ~950 car names; "
                            f"{om * 100:.1f}% of its probability fell on cars outside the 196 training classes. "
                            if om is not None else "") + f"Result: {res['display']} ({res['status']}).",
                      values=[dict(k="Answer", v=res["display"]), dict(k="Status", v=res["status"]),
                              dict(k="Confidence", v=round(float(res["confidence"]), 4))]))
    return steps


def explain(rec, res: dict) -> dict:
    """All 'show the working' material for one hybrid result."""
    crop = res["crop"]
    d = difference_image(crop)
    hp = hog_picture(crop)
    fp = res.get("final_probs")
    cam = grad_cam(rec.classifier, crop, int(np.argmax(fp))) if fp is not None else None
    return dict(diff=d, hog=hp, cam=cam, calc=calculation(rec, res),
                cam_label=rec.names[int(np.argmax(fp))] if fp is not None else None)
