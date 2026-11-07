"""Responsible handling of unfamiliar cars: evaluate and tune the "make first, name only when sure" rule.

Simulation (same idea as eval_openset): each validation/test car's true class is hidden from both
classifiers, as if it had never been trained. Two scenarios:
  A. "catalogued"   - the hidden car is still in CLIP's 954-model catalogue (as an outside model)
  B. "uncatalogued" - it is also removed from the catalogue (e.g. a Bugatti special edition, a Ford Figo)
Each answer is scored as
  correct-name  : names the right make + model family
  correct-make  : "<make>, model not recognised" with the right make
  unknown       : "unknown car" (no claim)
  wrong         : a wrong model name, a wrong make, or a (necessarily wrong) trained class
Cars whose family has a sibling class (another year/body) are excluded, as in eval_openset.

Tuned on validation (maximise correct answers minus wrong answers), reported on test, and compared with
the previous rule (always use CLIP's top catalogue model). Writes results/unknown.json and updates
models/hybrid_config.json (unknown_make_weight, open_name_conf, unknown_make_conf).

Usage:  python -m scripts.eval_unknown       (after scripts.eval_openset)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from src import config as C
from src.data import class_hierarchy, load_class_names, split_class_name
from src.openvocab import CarVision, decide_unfamiliar, same_family


def main():
    cv = CarVision(device="cuda")
    names = load_class_names()
    make_idx, makes, mm_idx, _ = class_hierarchy(names)
    mk_map = np.array([cv.makes.index(m) for m in makes])
    df = pd.read_csv(C.META_DIR / "annotations.csv")
    va, te = df[df.split == "val"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    F = dict(np.load(C.WORK_DIR / "features" / "clip_zeroshot.npz"))
    D = dict(np.load(C.WORK_DIR / "features" / "deep_probs.npz"))
    cfg = json.loads((C.MODELS_DIR / "hybrid_config.json").read_text())
    T = lambda a: torch.from_numpy(a).to(cv.device)
    E = cv.entries
    entry_of = {e["stanford"]: i for i, e in enumerate(E) if e["stanford"] >= 0}

    def cal(p, t):
        z = np.log(np.clip(p, 1e-12, 1)) / t; z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)

    def ens(prefix):
        p = (cal(D[f"{prefix}_tta"], cfg["temperature"]) + cal(D[f"{prefix}_enh_tta"], cfg["temperature"])) / 2
        q = (cal(D[f"clip_{prefix}_tta"], cfg["clip_temperature"]) + cal(D[f"clip_{prefix}_enh_tta"], cfg["clip_temperature"])) / 2
        w = cfg["ensemble_weight"]
        return w * p + (1 - w) * q

    def cases(P, Zr, Zn, y, hide, uncat):
        """Per image: (classifier probs, CLIP naming probs, known?, low_conf?, top class, CLIP routing probs)
        exactly as the pipeline sees them (routing on the crop, naming on frame + crop)."""
        out = []
        for i in range(len(y)):
            pe, pz, pn, mask = P[i].copy(), Zr[i].copy(), Zn[i].copy(), ~cv.is_stanford
            if hide:
                pe[y[i]] = 0; pe /= pe.sum()
                mask = mask.copy(); mask[entry_of[y[i]]] = True
                if uncat:
                    pz[entry_of[y[i]]] = 0; pz /= pz.sum()
                    pn[entry_of[y[i]]] = 0; pn /= pn.sum()
            c = int(pe.argmax()); m, mo, _ = split_class_name(names[c])
            ti = int(pz.argmax()); t = E[ti]
            known = ((not mask[ti]) or same_family(t["make"], t["model"], m, mo) or pe[c] >= cfg["known_conf"]
                     or float(pz[mask].sum()) < cfg["outside_threshold"] or float(pz[ti]) < cfg.get("open_min_top", 0))
            pm = np.bincount(make_idx, weights=pe, minlength=len(makes))
            pmm = np.bincount(mm_idx, weights=pe, minlength=int(mm_idx.max()) + 1)
            low = pe[c] < cfg["conf_threshold"] and pmm.max() < cfg["conf_threshold"] and pm.max() < cfg["make_threshold"]
            out.append((pe, pn, known, low, c, pz))
        return out

    def score(cs, y, w, nc, mc, rule="new"):
        k = dict(correct_name=0, correct_make=0, unknown=0, wrong=0, classifier_answer=0)
        for (pe, pz, known, low, c, pr), yy in zip(cs, y):
            tm, tmo, _ = split_class_name(names[yy])
            if known and not low:                       # classifier's trained-class answer (wrong: class hidden)
                k["classifier_answer"] += 1
                ok = c == yy
                k["correct_name" if ok else "wrong"] += 1
                continue
            if rule == "old":                           # previous behaviour: CLIP's top catalogue model, always
                if known:                               # old app said "Uncertain - see top-5" here
                    k["unknown"] += 1
                    continue
                t = E[int(pr.argmax())]                 # old app named from the crop
                k["correct_name" if same_family(t["make"], t["model"], tm, tmo) else "wrong"] += 1
                continue
            pmc = np.zeros(len(cv.makes)); pmc[mk_map] = np.bincount(make_idx, weights=pe, minlength=len(makes))
            make, mp, named, _ = decide_unfamiliar(cv.makes, pmc, E, cv.entry_make, pz, w, nc)
            if not known and named is not None:
                t = E[named]
                k["correct_name" if same_family(t["make"], t["model"], tm, tmo) else "wrong"] += 1
            elif mp >= mc:
                k["correct_make" if make == tm else "wrong"] += 1
            else:
                k["unknown"] += 1
        n = len(y)
        return {a: b / n for a, b in k.items()}

    Pv, Pt = ens("val"), ens("test_det")
    def mean_emb(a, b):
        m = a + b; return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)
    Rv, Rt = cv.name_probs(T(F["val_crop"])), cv.name_probs(T(F["test_det"]))          # routing: crop
    Zv = cv.name_probs(T(mean_emb(F["val_crop"], F["val_full"])))                       # naming: frame + crop
    Zt = cv.name_probs(T(mean_emb(F["test_det"], F["test_full"])))
    yv, yt = va.label.values, te.label.values
    def nosib(y):
        return np.array([sum(same_family(*split_class_name(names[c])[:2], *split_class_name(n)[:2]) for n in names) == 1 for c in y])
    sv, st = nosib(yv), nosib(yt)
    V = {u: cases(Pv[sv], Rv[sv], Zv[sv], yv[sv], True, u) for u in (False, True)}
    Tt = {u: cases(Pt[st], Rt[st], Zt[st], yt[st], True, u) for u in (False, True)}

    grid = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        for nc in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
            for mc in (0.3, 0.4, 0.5, 0.6, 0.7):
                s = [score(V[u], yv[sv], w, nc, mc) for u in (False, True)]
                util = np.mean([x["correct_name"] + x["correct_make"] - x["wrong"] for x in s])
                grid.append(dict(w=w, name_conf=nc, make_conf=mc, val_score=float(util),
                                 val_catalogued=s[0], val_uncatalogued=s[1]))
    best = max(grid, key=lambda g: g["val_score"])
    w, nc, mc = best["w"], best["name_conf"], best["make_conf"]
    R = dict(selection="max (correct name + correct make - wrong) on simulated unseen val cars, mean of both scenarios",
             unknown_make_weight=w, open_name_conf=nc, unknown_make_conf=mc,
             n_test=int(st.sum()), grid=grid)
    for u, lab in ((False, "catalogued"), (True, "uncatalogued")):
        R[f"test_{lab}_new"] = score(Tt[u], yt[st], w, nc, mc)
        R[f"test_{lab}_old"] = score(Tt[u], yt[st], w, nc, mc, rule="old")
    # Known cars (nothing hidden): how often does the new rule change what the user sees, and is it right?
    K = cases(Pt, Rt, Zt, yt, False, False)
    shown = dict(trained_class=0, open_set_named=0, make_only=0, unknown=0)
    make_ok = []
    for (pe, pz, known, low, c, _), yy in zip(K, yt):
        if known and not low:
            shown["trained_class"] += 1; continue
        pmc = np.zeros(len(cv.makes)); pmc[mk_map] = np.bincount(make_idx, weights=pe, minlength=len(makes))
        make, mp, named, _ = decide_unfamiliar(cv.makes, pmc, E, cv.entry_make, pz, w, nc)
        if not known and named is not None:
            shown["open_set_named"] += 1
        elif mp >= mc:
            shown["make_only"] += 1; make_ok.append(make == split_class_name(names[yy])[0])
        else:
            shown["unknown"] += 1
    R["test_known_cars_shown_as"] = {a: b / len(yt) for a, b in shown.items()}
    R["test_known_cars_make_only_accuracy"] = float(np.mean(make_ok)) if make_ok else None
    cfg.update(unknown_make_weight=w, open_name_conf=nc, unknown_make_conf=mc)
    (C.MODELS_DIR / "hybrid_config.json").write_text(json.dumps(cfg, indent=1))
    (C.RESULTS_DIR / "unknown.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({k: v for k, v in R.items() if k != "grid"}, indent=1))


if __name__ == "__main__":
    main()
