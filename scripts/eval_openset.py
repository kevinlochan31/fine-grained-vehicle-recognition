"""Evaluate the CLIP vehicle gate and open-set (unseen model) handling; tune its thresholds on val.

* Gate acceptance on all 8,041 Stanford test images (every one contains a car) -> false "no car" rate.
* Gate rejection on a small probe of non-car photos (scikit-image samples + user-reported tulips).
* Zero-shot naming accuracy over the ~950-entry catalogue (make / family / exact class), i.e. how well the
  open-vocabulary branch names cars without any training.
* Rate at which known cars are wrongly sent to the open-set branch, at the tuned threshold.

Usage:  python -m scripts.eval_openset        (after scripts.evaluate)
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config as C
from src.data import cache_path, load_class_names, read_rgb, split_class_name
from src.openvocab import CarVision, same_family

NPZ = C.WORK_DIR / "features" / "clip_zeroshot.npz"


def feats(cv, uids, kind):
    out = []
    with ThreadPoolExecutor(8) as ex:
        for i in tqdm(range(0, len(uids), 256), desc=kind, leave=False):
            ims = list(ex.map(lambda u: read_rgb(cache_path(u, kind)), uids[i:i + 256]))
            out.append(cv.embed(ims).cpu().numpy())
    return np.concatenate(out)


def main():
    cv = CarVision(device="cuda")
    names = load_class_names()
    df = pd.read_csv(C.META_DIR / "annotations.csv")
    va, te = df[df.split == "val"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    F = dict(np.load(NPZ)) if NPZ.exists() else {}
    need = dict(val_crop=(va, "crop"), val_full=(va, "full"), test_det=(te, "detcrop"), test_full=(te, "full"))
    if any(k not in F for k in need):
        for k, (d, kind) in need.items():
            if k not in F:
                F[k] = feats(cv, d.uid.tolist(), kind)
        np.savez_compressed(NPZ, **F)
    T = lambda a: torch.from_numpy(a).to(cv.device)
    # Routing uses the crop; naming an unfamiliar car uses the mean embedding of frame + crop (as in
    # CarVision.analyse). Routing on the mean re-routed more known cars and cost robustness.
    def mean_emb(a, b):
        m = a + b; return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)
    F["val_name"], F["test_name"] = mean_emb(F["val_crop"], F["val_full"]), mean_emb(F["test_det"], F["test_full"])

    R = {}
    # ---- gate on cars (full image and detected crop; the app uses max of both)
    pf, pc = cv.car_probability(T(F["test_full"])), cv.car_probability(T(F["test_det"]))
    g = np.maximum(pf, pc)
    R["gate_on_stanford_test"] = {f"accept@{t}": float((g >= t).mean()) for t in (0.3, 0.5, 0.7)}
    R["gate_on_stanford_test"]["n"] = int(len(g))
    # ---- gate on non-car probes
    from skimage import data as skd
    probes = {}
    for n in ("astronaut", "coffee", "chelsea", "rocket", "hubble_deep_field", "immunohistochemistry", "camera", "page", "brick", "grass", "gravel", "retina", "cell", "horse", "clock"):
        try:
            im = getattr(skd, n)()
            im = np.stack([im] * 3, -1) if im.ndim == 2 else im[..., :3]
            if im.dtype != np.uint8:
                im = (im.astype(float) / im.max() * 255).astype(np.uint8)
            probes[n] = im
        except Exception:
            pass
    tul = C.WORK_DIR / "userimgs" / "tulips.jpg"
    if tul.exists():
        probes["user tulips"] = read_rgb(tul)
    pp = cv.car_probability(cv.embed(list(probes.values())))
    R["gate_on_non_car_probes"] = {k: float(v) for k, v in zip(probes, pp)}
    R["gate_non_car_rejected@0.5"] = float((pp < 0.5).mean())
    # full two-signal rule as used in the app: accept if p >= 0.9, or p >= 0.5 and the YOLO detector agrees
    from src.detector import VehicleDetector
    yolo = VehicleDetector()
    dconf = [max([d["conf"] for d in yolo.detect(im, conf=0.05)] or [0.0]) for im in probes.values()]
    acc = [(p >= 0.9) or (p >= 0.5 and c >= 0.3) for p, c in zip(pp, dconf)]
    R["gate_non_car_rejected_final"] = float(1 - np.mean(acc))
    R["gate_non_car_probe_detector_conf"] = {k: float(c) for k, c in zip(probes, dconf)}

    # ---- zero-shot naming over the catalogue
    yt = te.label.values
    P = cv.name_probs(T(F["test_det"]))                 # routing input (crop)
    E = cv.entries
    tm = [split_class_name(names[y]) for y in yt]

    def zs_acc(PP):
        top = PP.argmax(1)
        return dict(exact_class=float(np.mean([E[t]["stanford"] == y for t, y in zip(top, yt)])),
                    make=float(np.mean([E[t]["make"] == m for t, (m, _, _) in zip(top, tm)])),
                    model_family=float(np.mean([same_family(E[t]["make"], E[t]["model"], m, mo) for t, (m, mo, _) in zip(top, tm)])))
    R["zero_shot_catalogue_test"] = dict(**zs_acc(cv.name_probs(T(F["test_name"]))), input="frame + crop (naming)",
                                         crop_only=zs_acc(P), catalogue_size=len(E),
                                         extra_models=int((~cv.is_stanford).sum()), makes=len(cv.makes))

    # ---- open-set routing: tune outside-mass threshold so <=2% of known val cars are re-routed
    D = dict(np.load(C.WORK_DIR / "features" / "deep_probs.npz"))
    cfg = json.loads((C.MODELS_DIR / "hybrid_config.json").read_text())

    def cal(p, t):
        z = np.log(np.clip(p, 1e-12, 1)) / t; z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)

    def ens(prefix):
        """(ensemble, ConvNeXt member, CLIP member) calibrated probabilities."""
        p = (cal(D[f"{prefix}_tta"], cfg["temperature"]) + cal(D[f"{prefix}_enh_tta"], cfg["temperature"])) / 2
        q = (cal(D[f"clip_{prefix}_tta"], cfg["clip_temperature"]) + cal(D[f"clip_{prefix}_enh_tta"], cfg["clip_temperature"])) / 2
        w = cfg["ensemble_weight"]
        return w * p + (1 - w) * q, p, q

    entry_of = {e["stanford"]: i for i, e in enumerate(E) if e["stanford"] >= 0}

    def routed(Ptriple, Pzs, thr, known_conf, min_top=0.0, agree=False, hide=None):
        """True = sent to the open-set branch. hide=y simulates an UNSEEN model: class y is removed from both
        classifiers (as if never trained) and its catalogue entry counts as an outside-the-training-set model."""
        Pens, Pa, Pb = Ptriple
        out = []
        for i, pz in enumerate(Pzs):
            pe, pa, pb = Pens[i], Pa[i], Pb[i]
            outside_mask = ~cv.is_stanford
            if hide is not None:
                y = hide[i]
                pe, pa, pb = [v.copy() for v in (pe, pa, pb)]
                for v in (pe, pa, pb):
                    v[y] = 0; v /= v.sum()
                outside_mask = outside_mask.copy(); outside_mask[entry_of[y]] = True
            c = int(pe.argmax()); m, mo, _ = split_class_name(names[c])
            ti = int(pz.argmax()); t = E[ti]
            known = ((not outside_mask[ti]) or same_family(t["make"], t["model"], m, mo) or pe[c] >= known_conf
                     or float(pz[outside_mask].sum()) < thr or float(pz[ti]) < min_top
                     or (agree and int(pa.argmax()) == int(pb.argmax())))
            out.append(not known)
        return np.array(out)

    Pv, Pt = ens("val"), ens("test_det")
    Zv, Zt = cv.name_probs(T(F["val_crop"])), P
    yv = va.label.values
    # Grid: false re-routing of KNOWN cars (val) vs. recall on SIMULATED unseen cars (val, class hidden).
    # Unseen recall excludes cars whose family has a sibling class (e.g. another year), where answering with
    # the sibling is the right behaviour.
    sib = np.array([sum(same_family(*split_class_name(names[y])[:2], *split_class_name(n)[:2]) for n in names) > 1 for y in yv])
    grid = []
    for agree in (False, True):
        for mt in (0.0, 0.2, 0.3, 0.4):
            for kc in (0.5, 0.6, 0.7, 0.8, 0.9):
                for thr in np.round(np.arange(0.3, 0.96, 0.05), 2):
                    fr = float(routed(Pv, Zv, thr, kc, mt, agree).mean())
                    rec = float(routed(Pv, Zv, thr, kc, mt, agree, hide=yv)[~sib].mean())
                    grid.append(dict(agree=agree, min_top=mt, known_conf=kc, outside_threshold=float(thr),
                                     known_rerouted=fr, unseen_recall=rec))
    # Tighter budget (0.5%) than before: occlusion showed that re-routing is the main cost on hard inputs.
    ok = [g for g in grid if g["known_rerouted"] <= 0.005] or grid
    best = max(ok, key=lambda g: (round(g["unseen_recall"], 3), -g["known_rerouted"]))
    kc, thr, mt, agree = best["known_conf"], best["outside_threshold"], best["min_top"], best["agree"]
    rt = routed(Pt, Zt, thr, kc, mt, agree)
    sib_t = np.array([sum(same_family(*split_class_name(names[y])[:2], *split_class_name(n)[:2]) for n in names) > 1 for y in yt])
    rec_t = routed(Pt, Zt, thr, kc, mt, agree, hide=yt)
    Pt = Pt[0]
    correct = Pt.argmax(1) == yt
    # simulated-unseen naming quality: when routed, is CLIP's make right?
    tops = [E[int(z.argmax())] for z in Zt]
    make_ok = np.array([t["make"] == split_class_name(names[y])[0] for t, y in zip(tops, yt)])
    R["open_set_routing"] = dict(
        known_conf=kc, outside_threshold=thr, min_top=mt, require_disagreement=agree,
        selection="max simulated-unseen recall s.t. <=0.5% known cars re-routed (val)",
        grid=grid, test_known_rerouted=float(rt.mean()),
        test_known_rerouted_that_were_correct=float((rt & correct).mean()),
        test_top1_before_routing=float(correct.mean()), test_top1_strict=float((correct & ~rt).mean()),
        test_simulated_unseen_recall=float(rec_t[~sib_t].mean()),
        test_simulated_unseen_make_correct_when_routed=float(make_ok[rec_t & ~sib_t].mean()),
        test_simulated_unseen_n=int((~sib_t).sum()))
    # gate with detector agreement (as in the pipeline): p >= 0.9, or p >= 0.5 and YOLO box conf >= 0.3
    det = pd.read_csv(C.RESULTS_DIR / "detections_test.csv").set_index("uid").loc[te.uid]
    acc2 = (g >= 0.9) | ((g >= 0.5) & (det.det_conf.values >= 0.3))
    R["gate_on_stanford_test"]["accept_two_signal"] = float(acc2.mean())
    cfg.update(outside_threshold=thr, known_conf=kc, open_min_top=mt, open_require_disagreement=agree,
               gate_threshold=0.5, gate_strong=0.9, gate_det_conf=0.3)
    (C.MODELS_DIR / "hybrid_config.json").write_text(json.dumps(cfg, indent=1))
    (C.RESULTS_DIR / "openset.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
