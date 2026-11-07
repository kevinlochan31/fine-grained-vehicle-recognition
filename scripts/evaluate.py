"""Full comparative evaluation on the unseen Stanford Cars test split (8,041 images).

Produces results/eval_results.json + plots + failure-case gallery and tunes the
hybrid fusion weight / confidence thresholds on the validation split
(written to models/hybrid_config.json).

Usage:  python -m scripts.evaluate [--robust_n 1000]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.classical import adaptive_preprocess, image_stats
from src.classifier import FineGrainedClassifier
from src.data import cache_path, class_hierarchy, load_class_names, read_rgb
from src.metrics import class_aware_map, expected_calibration_error, topk_acc

PROBS = C.WORK_DIR / "features" / "deep_probs.npz"
FIG = C.RESULTS_DIR / "figures"


def load_crops(uids, kind, enhance=False):
    def f(u):
        im = read_rgb(cache_path(u, kind))
        return adaptive_preprocess(im)[0] if enhance else im
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(f, uids))


def deep_probs(clf, uids, kind, enhance=False, tta=False, chunk=512):
    out = []
    for i in tqdm(range(0, len(uids), chunk), desc=f"{kind} enh={enhance} tta={tta}", leave=False):
        out.append(clf.predict_proba(load_crops(uids[i:i + chunk], kind, enhance), tta=tta, batch=64))
    return np.concatenate(out).astype(np.float32)


def calibrate(P, T):
    """Temperature scaling applied to probabilities: softmax(log p / T)."""
    z = np.log(np.clip(P, 1e-12, 1)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def summary(P, y, make_idx, mm_idx):
    pred = P.argmax(1)
    return dict(top1=topk_acc(P, y, 1), top5=topk_acc(P, y, 5),
                make_acc=float((make_idx[pred] == make_idx[y]).mean()),
                make_model_acc=float((mm_idx[pred] == mm_idx[y]).mean()),
                mean_conf=float(P.max(1).mean()),
                ece=expected_calibration_error(P.max(1), pred == y))


# ------------------------------------------------------------------------- corruptions
def corrupt(img, kind, rng, box):
    h, w = img.shape[:2]
    if kind == "original":
        return img
    if kind == "dark":
        return (np.power(img / 255.0, 2.2) * 0.6 * 255).astype(np.uint8)
    if kind == "very_dark_noisy":
        d = np.power(img / 255.0, 2.5) * 0.35 * 255
        return np.clip(d + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    if kind == "overexposed":
        return np.clip(img.astype(np.float32) * 1.8 + 40, 0, 255).astype(np.uint8)
    if kind == "low_res_x4":
        return cv2.resize(cv2.resize(img, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA), (w, h))
    if kind == "low_res_x8":
        return cv2.resize(cv2.resize(img, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA), (w, h))
    if kind == "occlusion_30pct":
        x1, y1, x2, y2 = box; bw, bh = x2 - x1, y2 - y1
        ow, oh = int(bw * np.sqrt(0.3)), int(bh * np.sqrt(0.3))
        ox = rng.integers(x1, max(x1 + 1, x2 - ow)); oy = rng.integers(y1, max(y1 + 1, y2 - oh))
        o = img.copy(); o[oy:oy + oh, ox:ox + ow] = rng.integers(0, 255, 3); return o
    if kind == "gaussian_noise":
        return np.clip(img + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)
    if kind == "motion_blur":
        k = np.zeros((15, 15)); k[7] = 1 / 15
        return cv2.filter2D(img, -1, k)
    if kind == "rotation_15deg":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if kind == "perspective_view":
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * 0.12, h * 0.08], [w, 0], [w, h], [w * 0.12, h * 0.92]])
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REFLECT)
    raise ValueError(kind)


CORRUPTIONS = ["original", "dark", "very_dark_noisy", "overexposed", "low_res_x4", "low_res_x8",
               "occlusion_30pct", "gaussian_noise", "motion_blur", "rotation_15deg", "perspective_view"]


def robustness(df_sub, n):
    from src.detector import iou
    from src.pipeline import VehicleRecognizer
    rec = VehicleRecognizer()
    rows = []
    for cond in CORRUPTIONS:
        rng = np.random.default_rng(C.SEED)
        acc = {m: [] for m in ("hybrid", "deep", "classical")}; ious = {m: [] for m in acc}   # hybrid first: fills ctx
        nocar = {m: 0 for m in acc}
        for r in tqdm(df_sub.itertuples(), total=len(df_sub), desc=cond, leave=False):
            img = corrupt(read_rgb(r.img_path), cond, rng, (r.x1, r.y1, r.x2, r.y2))
            ctx = {}
            for m in acc:
                res = rec.run(img, m, ctx)
                # strict: "no car" and "not in training set" answers count as errors
                acc[m].append(res["class_id"] == r.label and res.get("in_training_set", True) and res["status"] != "no-car")
                nocar[m] += res["status"] == "no-car"
                ious[m].append(iou(res["box"], (r.x1, r.y1, r.x2, r.y2)) if res["box"] is not None else 0.0)
        row = dict(condition=cond, n=len(df_sub))
        for m in acc:
            row[f"{m}_top1"] = float(np.mean(acc[m])); row[f"{m}_iou"] = float(np.mean(ious[m]))
        row["gate_rejected"] = nocar["hybrid"] / len(df_sub)
        rows.append(row); print(json.dumps(row), flush=True)
    return rows


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust_n", type=int, default=1000)
    ap.add_argument("--skip_robust", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    names = load_class_names()
    make_idx, makes, mm_idx, mms = class_hierarchy(names)
    df = pd.read_csv(C.META_DIR / "annotations.csv")
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    det = pd.read_csv(C.RESULTS_DIR / "detections_test.csv")
    te = te.merge(det, on="uid", how="left")
    yv, yt = va.label.values, te.label.values
    cl = np.load(C.WORK_DIR / "features" / "classical_probs.npz")

    # ---------------------------------------------------------------- deep probabilities
    D = dict(np.load(PROBS)) if PROBS.exists() else {}
    vu, tu = va.uid.tolist(), te.uid.tolist()
    jobs = {  # key: (uids, cache kind, enhance, tta)
        "val_plain": (vu, "crop", False, False), "val_tta": (vu, "crop", False, True),
        "val_enh_tta": (vu, "crop", True, True),
        "test_det_plain": (tu, "detcrop", False, False), "test_det_tta": (tu, "detcrop", False, True),
        "test_det_enh": (tu, "detcrop", True, False), "test_det_enh_tta": (tu, "detcrop", True, True),
        "test_gt_plain": (tu, "crop", False, False), "test_full_plain": (tu, "full", False, False),
    }
    missing = [k for k in jobs if k not in D]
    if missing:
        clf = FineGrainedClassifier(device="cuda")
        for k in missing:
            u, kind, enh, tta = jobs[k]
            D[k] = deep_probs(clf, u, kind, enhance=enh, tta=tta)
    # robust CLIP ViT-B/16 fine-tune (second ensemble member)
    from src.clip_classifier import WEIGHTS as CLIP_W
    has_clip = CLIP_W.exists()
    if has_clip:
        cj = {f"clip_{k}": v for k, v in jobs.items() if k in ("val_tta", "val_enh_tta", "test_det_tta", "test_det_enh_tta", "test_det_plain")}
        cmiss = [k for k in cj if k not in D]
        if cmiss:
            from src.clip_classifier import ClipFineGrained
            cclf = ClipFineGrained(device="cuda")
            for k in cmiss:
                u, kind, enh, tta = cj[k]
                D[k] = deep_probs(cclf, u, kind, enhance=enh, tta=tta)
            missing += cmiss
    nocrop = C.RUNS_DIR / "cls_nocrop" / "best.pt"
    if nocrop.exists() and "test_full_nocrop_model" not in D:
        D["test_full_nocrop_model"] = deep_probs(FineGrainedClassifier(nocrop, device="cuda"), tu, "full")
        missing.append("nocrop")
    if missing:
        np.savez_compressed(PROBS, **D)

    # Hybrid view-averaging: mean of CNN(original) and CNN(enhanced). For crops where the
    # adaptive enhancement does not trigger, the two views are identical -> no change.
    def views(a, b):
        return (D[a] + D[b]) / 2

    # ---------------------------------------------------------------- tune hybrid on validation
    # Temperature scaling (calibration): label smoothing makes the CNN under-confident.
    Pv_raw = views("val_tta", "val_enh_tta")
    nll = {float(T): float(-np.log(calibrate(Pv_raw, T)[np.arange(len(yv)), yv] + 1e-12).mean())
           for T in np.round(np.arange(0.3, 1.51, 0.05), 2)}
    temperature = min(nll, key=nll.get)
    print("temperature", temperature, "val NLL", nll[temperature], "(T=1:", nll[1.0], ")")
    _raw = views("test_det_tta", "test_det_enh_tta")
    ece_before = expected_calibration_error(_raw.max(1), _raw.argmax(1) == yt)
    for k in list(D):
        if k.startswith(("val_", "test_det", "test_gt", "test_full")):
            D[k] = calibrate(D[k], temperature)
    Pv_deep = views("val_tta", "val_enh_tta")
    clip_T, ens_w = 1.0, 1.0
    if has_clip:
        Pc_raw = views("clip_val_tta", "clip_val_enh_tta")
        nll_c = {float(T): float(-np.log(calibrate(Pc_raw, T)[np.arange(len(yv)), yv] + 1e-12).mean())
                 for T in np.round(np.arange(0.3, 2.01, 0.05), 2)}
        clip_T = min(nll_c, key=nll_c.get)
        for k in list(D):
            if k.startswith("clip_"):
                D[k] = calibrate(D[k], clip_T)
        Pc_v = views("clip_val_tta", "clip_val_enh_tta")
        # ensemble weight: best val top-1, ties broken by lower NLL
        ens_search = {}
        for w_ in np.round(np.arange(0, 1.01, 0.1), 2):
            P_ = w_ * Pv_deep + (1 - w_) * Pc_v
            ens_search[float(w_)] = (topk_acc(P_, yv, 1), -float(-np.log(P_[np.arange(len(yv)), yv] + 1e-12).mean()))
        ens_w = max(ens_search, key=lambda k: ens_search[k])
        print("clip temperature", clip_T, "ensemble weight (ConvNeXt)", ens_w, ens_search[ens_w])
        Pv_deep = ens_w * Pv_deep + (1 - ens_w) * Pc_v

    def hyb(a, b, ca=None, cb=None):
        """ConvNeXt views (a,b) ensembled with the CLIP fine-tune views (ca,cb)."""
        p = views(a, b)
        if has_clip:
            p = ens_w * p + (1 - ens_w) * views(ca or "clip_" + a, cb or "clip_" + b)
        return p
    grid = np.round(np.arange(0, 0.41, 0.025), 3)
    val_fuse = {float(w): topk_acc((1 - w) * Pv_deep + w * cl["val"], yv, 1) for w in grid}
    best_w = max(val_fuse, key=lambda w: (round(val_fuse[w], 4), -w))
    Pv = (1 - best_w) * Pv_deep + best_w * cl["val"]
    # confidence threshold: lowest t whose accepted-set precision on val >= PRECISION_TARGET
    # (98.5 %: with the ensemble already at ~97 % val accuracy a 97 % target would accept everything)
    PRECISION_TARGET = 0.985
    conf_t = 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        acc_m = Pv.max(1) >= t
        if acc_m.any() and (Pv.argmax(1)[acc_m] == yv[acc_m]).mean() >= PRECISION_TARGET:
            conf_t = float(round(t, 2)); break
    old_cfg = json.loads((C.MODELS_DIR / "hybrid_config.json").read_text()) if (C.MODELS_DIR / "hybrid_config.json").exists() else {}
    hybrid_cfg = {**old_cfg, **dict(fusion_weight=best_w, conf_threshold=conf_t, make_threshold=conf_t, tta=True,
                                    enhance=True, temperature=temperature, clip_temperature=clip_T, ensemble_weight=ens_w)}
    (C.MODELS_DIR / "hybrid_config.json").write_text(json.dumps(hybrid_cfg, indent=1))
    print("hybrid config", hybrid_cfg)

    w = best_w
    P_h = (1 - w) * hyb("test_det_tta", "test_det_enh_tta") + w * cl["test_det"]
    P_d = D["test_det_plain"]; P_c = cl["test_det"]
    enh_trig = np.array([bool(adaptive_preprocess(read_rgb(cache_path(u, "detcrop")))[1]) for u in tu])
    R = dict(hybrid_config=hybrid_cfg, val_fusion_search=val_fuse,
             calibration=dict(temperature=temperature, val_nll_T1=nll[1.0], val_nll_best=nll[temperature],
                              test_ece_hybrid_before=ece_before,
                              test_ece_hybrid_after=expected_calibration_error(P_h.max(1), P_h.argmax(1) == yt)))

    # ---------------------------------------------------------------- 1. approach comparison
    R["comparison"] = {
        "classical features (detected crop -> HOG+LBP+colour+edge+geometry -> PCA+LogReg)": summary(P_c, yt, make_idx, mm_idx),
        "deep learning (YOLO crop -> ConvNeXt)": summary(P_d, yt, make_idx, mm_idx),
        ("hybrid (YOLO + adaptive enhancement views + flip TTA + ConvNeXt/CLIP ensemble)" if has_clip else
         "hybrid (YOLO + adaptive classical enhancement views + flip TTA + fusion)"): summary(P_h, yt, make_idx, mm_idx),
    }
    # ---------------------------------------------------------------- 2. ablation
    E = (lambda a, b: ens_w * D[a] + (1 - ens_w) * D[b]) if has_clip else (lambda a, b: D[a])
    abl = {
        "full hybrid": P_h,
        "- classical feature fusion": hyb("test_det_tta", "test_det_enh_tta"),
        "- flip TTA": (1 - w) * (ens_w * views("test_det_plain", "test_det_enh") + (1 - ens_w) * D["clip_test_det_plain"] if has_clip
                                 else views("test_det_plain", "test_det_enh")) + w * cl["test_det"],
        "- adaptive enhancement view": (1 - w) * E("test_det_tta", "clip_test_det_tta") + w * cl["test_det"],
        "enhanced crop replaces original (instead of averaging)": (1 - w) * E("test_det_enh_tta", "clip_test_det_enh_tta") + w * cl["test_det"],
        "- fusion - TTA - enhancement (= deep only)": P_d,
        "- detection crop (whole image -> crop-trained CNN)": D["test_full_plain"],
        "oracle: ground-truth box crop (upper bound for localisation)": D["test_gt_plain"],
    }
    if "test_full_nocrop_model" in D:
        abl["- detection crop (CNN trained & tested on whole images)"] = D["test_full_nocrop_model"]
    if has_clip:
        abl["- CLIP fine-tune (ConvNeXt-only hybrid)"] = (1 - w) * views("test_det_tta", "test_det_enh_tta") + w * cl["test_det"]
        abl["CLIP fine-tune alone (with TTA + enhancement views)"] = views("clip_test_det_tta", "clip_test_det_enh_tta")
        abl["CLIP fine-tune alone, single pass"] = D["clip_test_det_plain"]
    R["ablation"] = {k: dict(top1=topk_acc(v, yt, 1), top5=topk_acc(v, yt, 5)) for k, v in abl.items()}
    m = enh_trig
    R["enhancement_subset"] = dict(
        n_triggered=int(m.sum()), fraction=float(m.mean()),
        deep_top1=topk_acc(P_d[m], yt[m], 1) if m.any() else None,
        tta_only_top1=topk_acc(D["test_det_tta"][m], yt[m], 1) if m.any() else None,
        enhanced_replaces_top1=topk_acc(D["test_det_enh_tta"][m], yt[m], 1) if m.any() else None,
        hybrid_top1=topk_acc(P_h[m], yt[m], 1) if m.any() else None)

    # ---------------------------------------------------------------- 3. localisation / mAP
    R["detection"] = json.loads((C.RESULTS_DIR / "detection.json").read_text())
    ious = te.iou.values; scores = te.det_conf.values
    R["class_aware_mAP50"] = {
        "hybrid": class_aware_map(P_h.argmax(1), scores * P_h.max(1), ious, yt, C.NUM_CLASSES, 0.5),
        "deep": class_aware_map(P_d.argmax(1), scores * P_d.max(1), ious, yt, C.NUM_CLASSES, 0.5),
        "classical": class_aware_map(P_c.argmax(1), scores * P_c.max(1), ious, yt, C.NUM_CLASSES, 0.5),
    }

    # ---------------------------------------------------------------- 4. confidence thresholds
    conf, correct = P_h.max(1), P_h.argmax(1) == yt
    th = []
    for t in np.round(np.arange(0, 1.0, 0.05), 2):
        m = conf >= t
        th.append(dict(threshold=float(t), coverage=float(m.mean()),
                       accuracy_on_accepted=float(correct[m].mean()) if m.any() else None,
                       rejected_error_fraction=float((~correct[~m]).mean()) if (~m).any() else None))
    R["threshold_analysis"] = th
    # hierarchical back-off outcome at the chosen threshold
    pm = np.stack([np.bincount(make_idx, weights=p, minlength=len(makes)) for p in P_h])
    pmm = np.stack([np.bincount(mm_idx, weights=p, minlength=len(mms)) for p in P_h])
    status = np.where(conf >= conf_t, "confident",
              np.where(pmm.max(1) >= conf_t, "year-uncertain",
              np.where(pm.max(1) >= conf_t, "model-uncertain", "low-confidence")))
    lvl_ok = np.where(status == "confident", correct,
             np.where(status == "year-uncertain", pmm.argmax(1) == mm_idx[yt],
             np.where(status == "model-uncertain", pm.argmax(1) == make_idx[yt], np.nan)))
    R["low_confidence_handling"] = {
        s: dict(fraction=float((status == s).mean()),
                accuracy_at_reported_level=(float(np.nanmean(lvl_ok[status == s])) if s != "low-confidence" and (status == s).any() else None),
                fine_grained_accuracy=float(correct[status == s].mean()) if (status == s).any() else None)
        for s in ("confident", "year-uncertain", "model-uncertain", "low-confidence")}

    # ---------------------------------------------------------------- 5. natural variation
    stats = [image_stats(read_rgb(cache_path(u, "detcrop"))) for u in tqdm(te.uid, desc="stats", leave=False)]
    te["brightness"] = [s["brightness"] for s in stats]
    te["box_px"] = np.sqrt((te.x2 - te.x1) * (te.y2 - te.y1))
    te["ok_h"], te["ok_d"], te["ok_c"] = correct, P_d.argmax(1) == yt, P_c.argmax(1) == yt
    nat = {}
    for col, label in (("brightness", "crop brightness"), ("box_px", "vehicle size (sqrt box area, px)")):
        q = pd.qcut(te[col], 5)
        g = te.groupby(q, observed=True)[["ok_c", "ok_d", "ok_h"]].mean()
        nat[label] = [dict(bin=str(i), classical=float(r.ok_c), deep=float(r.ok_d), hybrid=float(r.ok_h))
                      for i, r in g.iterrows()]
    R["natural_variation"] = nat

    # ---------------------------------------------------------------- 6. confusions + failures
    pred = P_h.argmax(1)
    wrong = np.where(~correct)[0]
    same_mm = (mm_idx[pred[wrong]] == mm_idx[yt[wrong]]).mean()
    same_make = (make_idx[pred[wrong]] == make_idx[yt[wrong]]).mean()
    pairs = pd.Series([(names[yt[i]], names[pred[i]]) for i in wrong]).value_counts().head(15)
    R["error_breakdown"] = dict(n_errors=int(len(wrong)), same_make_model_diff_year=float(same_mm),
                                same_make_diff_model=float(same_make - same_mm), different_make=float(1 - same_make),
                                top_confusions=[dict(true=a, pred=b, count=int(c)) for (a, b), c in pairs.items()])
    per_class = pd.Series(correct).groupby(yt).mean()
    R["worst_classes"] = [dict(cls=names[i], acc=float(a)) for i, a in per_class.nsmallest(10).items()]

    fails = {
        "1_wrong_year_same_model": [i for i in wrong if mm_idx[pred[i]] == mm_idx[yt[i]]],
        "2_confident_but_wrong": sorted(wrong, key=lambda i: -conf[i]),
        "3_localisation_failure": sorted(np.where(te.iou.values < 0.5)[0], key=lambda i: te.iou.values[i]),
        "4_dark_or_low_contrast": sorted(wrong, key=lambda i: te.brightness.values[i]),
        "5_small_low_resolution": sorted(wrong, key=lambda i: te.box_px.values[i]),
    }
    fdir = C.RESULTS_DIR / "failures"; fdir.mkdir(exist_ok=True)
    R["failure_cases"] = {}
    for k, idxs in fails.items():
        sel = list(idxs[:4])
        R["failure_cases"][k] = [dict(uid=te.uid[i], true=names[yt[i]], pred=names[pred[i]], conf=float(conf[i]),
                                      iou=float(te.iou[i]), brightness=float(te.brightness[i]),
                                      box_px=float(te.box_px[i])) for i in sel]
        fig, axs = plt.subplots(1, len(sel), figsize=(4.2 * len(sel), 3.6))
        for ax, i in zip(np.atleast_1d(axs), sel):
            r = te.iloc[i]; im = read_rgb(r.img_path)
            cv2.rectangle(im, (r.x1, r.y1), (r.x2, r.y2), (0, 255, 0), 3)
            cv2.rectangle(im, (int(r.px1), int(r.py1)), (int(r.px2), int(r.py2)), (255, 0, 0), 3)
            ax.imshow(im); ax.axis("off")
            ax.set_title(f"GT: {names[yt[i]]}\nPred: {names[pred[i]]} ({conf[i]:.2f})\nIoU {r.iou:.2f}", fontsize=8)
        fig.suptitle(k.replace("_", " ") + "   (green = GT box, red = predicted box)", fontsize=10)
        fig.tight_layout(); fig.savefig(fdir / f"{k}.jpg", dpi=110); plt.close(fig)

    # ---------------------------------------------------------------- 7. robustness (end-to-end)
    if not args.skip_robust:
        sub = df[df.split == "test"].sample(args.robust_n, random_state=C.SEED)
        R["robustness"] = robustness(sub, args.robust_n)
    elif (C.RESULTS_DIR / "eval_results.json").exists():
        old = json.loads((C.RESULTS_DIR / "eval_results.json").read_text())
        if "robustness" in old:
            R["robustness"] = old["robustness"]

    (C.RESULTS_DIR / "eval_results.json").write_text(json.dumps(R, indent=1))
    make_plots(R)
    print(json.dumps({k: R[k] for k in ("comparison", "ablation", "class_aware_mAP50")}, indent=1))


def make_plots(R):
    hist = C.RUNS_DIR / "cls_main" / "history.json"
    if hist.exists():
        h = pd.DataFrame(json.loads(hist.read_text()))
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(h.epoch, h.loss); ax[0].set(title="train loss", xlabel="epoch")
        ax[1].plot(h.epoch, h.val_top1, label="val top-1"); ax[1].plot(h.epoch, h.val_top5, label="val top-5")
        ax[1].set(title="validation accuracy", xlabel="epoch"); ax[1].legend(); fig.tight_layout()
        fig.savefig(FIG / "training_curves.png", dpi=110); plt.close(fig)
    t = pd.DataFrame(R["threshold_analysis"])
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(t.threshold, t.coverage, label="coverage"); ax.plot(t.threshold, t.accuracy_on_accepted, label="accuracy on accepted")
    ax.axvline(R["hybrid_config"]["conf_threshold"], ls="--", c="gray", label="chosen threshold")
    ax.set(xlabel="confidence threshold", title="Confidence threshold analysis (hybrid, test)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "threshold_analysis.png", dpi=110); plt.close(fig)
    if "robustness" in R:
        r = pd.DataFrame(R["robustness"]).set_index("condition")
        ax = r[["classical_top1", "deep_top1", "hybrid_top1"]].plot.bar(figsize=(10, 4), rot=30)
        ax.set(title="End-to-end top-1 under corruptions", ylabel="top-1"); plt.tight_layout()
        plt.savefig(FIG / "robustness.png", dpi=110); plt.close()
    comp = R["ablation"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(comp.keys())[::-1], [v["top1"] for v in list(comp.values())[::-1]])
    ax.set(xlabel="test top-1", title="Ablation"); fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
