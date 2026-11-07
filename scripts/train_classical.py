"""Classical-feature baseline: HOG + LBP + colour + edge-orientation + geometry -> PCA -> LogReg.

Also runs the feature-group ablation and stores class probabilities used for
hybrid late fusion.

Usage:  python -m scripts.train_classical          (run after scripts.run_detection)
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.classical import extract_features, feature_group_slices
from src.data import cache_path, read_rgb


def _feat(path):
    return extract_features(read_rgb(path))


def features_for(df: pd.DataFrame, kind: str) -> np.ndarray:
    out = C.WORK_DIR / "features" / f"{kind}_{df.split.iloc[0]}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return np.load(out)
    t = time.time()
    X = np.stack(Parallel(n_jobs=int(os.environ.get("CVH_JOBS", -1)), batch_size=64)(delayed(_feat)(cache_path(u, kind)) for u in df.uid))
    np.save(out, X)
    print(f"features {kind}/{df.split.iloc[0]}: {X.shape} in {time.time() - t:.0f}s", flush=True)
    return X


def make_model(n_comp=256):
    # PCA dim / C chosen on the validation split (256 / 0.05 -> 15.4% val top-1;
    # tried 128-512 comps, C 0.01-0.05, and an RBF Nystroem kernel which overfit).
    return make_pipeline(StandardScaler(), PCA(n_comp, whiten=True, random_state=C.SEED),
                         LogisticRegression(C=0.05, max_iter=300))


def topk(P, y, k):
    return float((np.argsort(-P, 1)[:, :k] == y[:, None]).any(1).mean())


def main():
    df = pd.read_csv(C.META_DIR / "annotations.csv")
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    Xtr, Xva = features_for(tr, "crop"), features_for(va, "crop")
    Xte_gt = features_for(te, "crop")
    Xte_det = features_for(te, "detcrop")
    ytr, yva, yte = tr.label.values, va.label.values, te.label.values

    res = {}
    t = time.time()
    model = make_model().fit(Xtr, ytr)
    res["fit_sec"] = time.time() - t
    Pva = model.predict_proba(Xva)
    t = time.time(); Pte_det = model.predict_proba(Xte_det); dt = time.time() - t
    Pte_gt = model.predict_proba(Xte_gt)
    res.update(val_top1=topk(Pva, yva, 1), val_top5=topk(Pva, yva, 5),
               test_gt_top1=topk(Pte_gt, yte, 1), test_gt_top5=topk(Pte_gt, yte, 5),
               test_det_top1=topk(Pte_det, yte, 1), test_det_top5=topk(Pte_det, yte, 5),
               clf_ms_per_img=1000 * dt / len(yte))
    print(json.dumps(res, indent=1), flush=True)
    joblib.dump(model, C.MODELS_DIR / "classical_logreg.joblib", compress=3)
    np.savez_compressed(C.WORK_DIR / "features" / "classical_probs.npz",
                        val=Pva.astype(np.float32), test_det=Pte_det.astype(np.float32),
                        test_gt=Pte_gt.astype(np.float32))

    # ---- feature-group ablation (leave-one-group-out + single groups), on validation
    sl = feature_group_slices()
    abl = {}
    for g, s in sl.items():
        keep = np.ones(Xtr.shape[1], bool); keep[s] = False
        m = make_model(min(256, keep.sum() - 1)).fit(Xtr[:, keep], ytr)
        abl[f"without_{g}"] = topk(m.predict_proba(Xva[:, keep]), yva, 1)
        only = ~keep
        m = make_model(min(256, only.sum() - 1)).fit(Xtr[:, only], ytr)
        abl[f"only_{g}"] = topk(m.predict_proba(Xva[:, only]), yva, 1)
        print(g, abl[f"without_{g}"], abl[f"only_{g}"], flush=True)
    res["feature_ablation_val_top1"] = abl
    res["feature_dims"] = {g: s.stop - s.start for g, s in sl.items()}
    (C.RESULTS_DIR / "classical.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
