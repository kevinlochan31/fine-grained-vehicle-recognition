"""World-car layer, step 3: compare matching methods and backbones, tune thresholds, build the live artefacts.

compare : on the held-out Commons test photos (disjoint from the reference photos, mostly by photographer),
          for CLIP ViT-B/16 (live), CLIP ViT-L/14 and SigLIP ViT-B/16:
            zero-shot text over the 1,992 catalogue models / 2,985 generations (no reference photos),
            1-NN over reference photos, class prototypes, and linear probes (world refs only, or world refs +
            the 196 Stanford classes from the Stanford *train* split),
          make / model / generation accuracy, false "world" answers on Stanford val, image-tower latency.
build   : for the live backbone (ViT-B/16) train the chosen probe, tune the thresholds
            world_route_conf : probe confidence needed to override "this is a trained class" (Stanford val budget)
            world_name_conf  : confidence needed to name make + model + generation (reference-photo CV + leave-model-out)
            world_zs_conf    : zero-shot text confidence needed to name a model that has no reference photos
          and write models/world_gallery.npz, models/world_text_b16.npz, data_meta/world_refs.csv,
          web/world_refs/*.jpg (shipped reference thumbnails), docs/world_photo_credits.md,
          world_* keys in models/hybrid_config.json and results/world_cars.json.
e2e     : run the full app pipeline (VehicleRecognizer, hybrid mode) on every held-out test photo and on a
          random sample of Stanford test images (before/after comparison of what the user would see).

Usage:  python -m scripts.eval_world compare|build|e2e
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch

from src import config as C
from src.data import load_class_names, split_class_name
from src.openvocab import NAME_TEMPLATES, same_family

W = C.WORK_DIR / "world_cars"
OUT = C.RESULTS_DIR / "world_cars.json"
NAMES = load_class_names()
BACKBONES = {"b16": ("ViT-B-16", "laion2b_s34b_b88k"), "l14": ("ViT-L-14", "laion2b_s32b_b82k"),
             "siglip": ("ViT-B-16-SigLIP", "webli")}


def norm(a):
    return (a / np.linalg.norm(a, axis=-1, keepdims=True)).astype(np.float32)


def mk(m: str) -> str:
    m = m.lower()
    return "suzuki" if m in ("maruti suzuki", "maruti") else m      # Suzuki is sold as Maruti Suzuki in India


def catalogue():
    return json.loads((C.META_DIR / "world_catalogue.json").read_text(encoding="utf8"))["entries"]


def load_world():
    df = pd.read_csv(W / "selected.csv")
    df = df[df.set.notna()].reset_index(drop=True)
    keys = list(dict.fromkeys(df.key))
    cls = []
    for k in keys:
        r = df[df.key == k].iloc[0]
        ys, ye = (r.years.split("–") + [""])[:2] if isinstance(r.years, str) else ("", "")
        cls.append(dict(key=k, make=r.make, model=r.model, gen=r.gen if isinstance(r.gen, str) else None,
                        years=r.years if isinstance(r.years, str) else "", y0=int(ys) if ys.isdigit() else None,
                        y1=int(ye) if ye.isdigit() else 2026, stanford=-1))
    df["cls"] = df.key.map({k: i for i, k in enumerate(keys)})
    return df, cls


def stanford_classes():
    out = []
    for i, n in enumerate(NAMES):
        m, mo, y = split_class_name(n)
        out.append(dict(key=f"stanford_{i}", make=m, model=mo, gen=None, years=y, y0=int(y), y1=int(y), stanford=i))
    return out


def judge(c, t):
    """(make_ok, model_ok, gen_ok) of predicted class/entry c for a world test photo of class t."""
    make_ok = mk(c["make"]) == mk(t["make"])
    if c.get("stanford", -1) >= 0:          # a Stanford class: same family, and its year inside the generation
        model_ok = make_ok and same_family(mk(c["make"]), c["model"], mk(t["make"]), t["model"])
        gen_ok = model_ok and t["y0"] is not None and t["y0"] <= c["y0"] <= t["y1"]
    else:
        model_ok = make_ok and c["model"].lower() == t["model"].lower()
        gen_ok = model_ok and (c.get("y0") == t["y0"])
    return make_ok, model_ok, gen_ok


def acc(pred_classes, truths):
    j = np.array([judge(c, t) for c, t in zip(pred_classes, truths)])
    return dict(make=float(j[:, 0].mean()), model=float(j[:, 1].mean()), generation=float(j[:, 2].mean()), n=len(j))


def fit_probe(X, y, n_cls, wd=1e-4, epochs=400, balanced=True, init_scale=10.0):
    """Multinomial logistic regression on L2-normalised embeddings (full batch, Adam, on the GPU)."""
    X = torch.from_numpy(X).cuda(); y = torch.from_numpy(np.asarray(y)).long().cuda()
    cnt = torch.bincount(y, minlength=n_cls).float().clamp(min=1)
    wts = (cnt.sum() / (n_cls * cnt)) if balanced else torch.ones(n_cls, device="cuda")
    lin = torch.nn.Linear(X.shape[1], n_cls).cuda()
    with torch.no_grad():                      # start from scaled class means (prototype init)
        P = torch.zeros(n_cls, X.shape[1], device="cuda").index_add_(0, y, X) / cnt[:, None]
        lin.weight.copy_(init_scale * torch.nn.functional.normalize(P, dim=1)); lin.bias.zero_()
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(X), y, weight=wts)
        loss.backward(); opt.step()
    return lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy()


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def text_embeddings(name, labels_model, labels_gen):
    import open_clip
    arch, pre = BACKBONES[name]
    m, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pre)
    m = m.eval().cuda(); tok = open_clip.get_tokenizer(arch)
    scale = float(m.logit_scale.exp())

    @torch.no_grad()
    def enc(texts):
        out = []
        for i in range(0, len(texts), 512):
            e = m.encode_text(tok(texts[i:i + 512]).cuda()).float(); out.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(out).cpu().numpy()

    def enc_names(names, templates):
        e = enc([t.format(n) for n in names for t in templates]).reshape(len(names), len(templates), -1).mean(1)
        return norm(e)

    tm = enc_names(labels_model, NAME_TEMPLATES)
    tg = enc_names(labels_gen, ["a photo of a {}, a type of car.", "a photo of the {}."])
    del m; torch.cuda.empty_cache()
    return tm, tg, scale


def latency(name, n=30):
    import open_clip
    arch, pre = BACKBONES[name]
    m, _, prep = open_clip.create_model_and_transforms(arch, pretrained=pre)
    v = m.visual.eval()
    size = prep.transforms[0].size if hasattr(prep.transforms[0], "size") else 224
    size = size if isinstance(size, int) else size[0]
    x = torch.randn(1, 3, size, size)
    torch.set_num_threads(2)                    # the Modal container has 2 CPU cores
    with torch.no_grad():
        for _ in range(3):
            v(x)
        t = time.perf_counter()
        for _ in range(n // 3):
            v(x)
        cpu = 1000 * (time.perf_counter() - t) / (n // 3)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    vg = v.cuda().half(); xg = x.cuda().half()
    with torch.no_grad():
        for _ in range(5):
            vg(xg)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n):
            vg(xg)
        torch.cuda.synchronize(); gpu = 1000 * (time.perf_counter() - t) / n
    params = sum(p.numel() for p in v.parameters()) / 1e6
    del m, v, vg; torch.cuda.empty_cache()
    return dict(cpu_ms_2threads=round(cpu, 1), gpu_fp16_ms=round(gpu, 2), image_tower_params_m=round(params, 1), input=size)


def compare():
    df, wcls = load_world()
    scls = stanford_classes()
    E = catalogue()
    models = list(dict.fromkeys((e["make"], e["model"]) for e in E))
    model_entries = [dict(make=a, model=b, y0=None, y1=None) for a, b in models]
    gen_entries = [e for e in E if e["years"][0]]
    for e in gen_entries:
        e["y0"], e["y1"] = e["years"][0], e["years"][1] or 2026
    gen_labels = [f"{e['years'][0]} {e['make']} {e['model']}" for e in gen_entries]
    ref, test = (df.set == "ref").values, (df.set == "test").values
    truth = [wcls[c] for c in df.cls[test]]
    R = dict(n_models=len(wcls), n_ref=int(ref.sum()), n_test=int(test.sum()),
             author_disjoint_models=float(df.groupby("key").author_disjoint.first().mean()), backbones={})
    for name in BACKBONES:
        F = dict(np.load(W / f"feats_{name}.npz", allow_pickle=True))
        assert list(F["world_file"]) == list(df.file), "feature file out of date: re-run build_world_gallery embed"
        Xn = norm(F["world_crop"] + F["world_full"]); Xc = F["world_crop"]
        Sn = norm(F["stanford_crop"] + F["stanford_full"]); sl = F["stanford_label"]; ss = F["stanford_split"]
        tm, tg, scale = text_embeddings(name, [f"{a} {b}" for a, b in models], gen_labels)
        res = {}
        for inp, X in (("frame+crop", Xn), ("crop", Xc)):
            Xt, Xr, yr = X[test], X[ref], df.cls[ref].values
            r = {}
            r["zero_shot_models"] = acc([model_entries[i] for i in (Xt @ tm.T).argmax(1)], truth)
            r["zero_shot_generations"] = acc([gen_entries[i] for i in (Xt @ tg.T).argmax(1)], truth)
            r["nn1_refs"] = acc([wcls[yr[i]] for i in (Xt @ Xr.T).argmax(1)], truth)
            P = norm(np.stack([Xr[yr == k].mean(0) for k in range(len(wcls))]))
            r["prototypes"] = acc([wcls[i] for i in (Xt @ P.T).argmax(1)], truth)
            Wt, b = fit_probe(Xr, yr, len(wcls))
            r["probe_world"] = acc([wcls[i] for i in (Xt @ Wt.T + b).argmax(1)], truth)
            Xs_tr = (Sn if inp == "frame+crop" else F["stanford_crop"])[ss == "train"]
            Xall = np.concatenate([Xr, Xs_tr]); yall = np.concatenate([yr, len(wcls) + sl[ss == "train"]])
            Wt, b = fit_probe(Xall, yall, len(wcls) + 196)
            allc = wcls + scls
            r["probe_world+stanford"] = acc([allc[i] for i in (Xt @ Wt.T + b).argmax(1)], truth)
            # how often a Stanford val car gets a "world" answer from this probe (any confidence / p >= 0.5 / 0.8)
            Xv = (Sn if inp == "frame+crop" else F["stanford_crop"])[ss == "val"]
            pv = softmax(Xv @ Wt.T + b)
            kv = pv.argmax(1)
            r["probe_world+stanford"]["stanford_val_top1"] = float((kv - len(wcls) == sl[ss == "val"]).mean())
            for t in (0.0, 0.5, 0.8):
                r["probe_world+stanford"][f"stanford_val_world_answer@{t}"] = float(((kv < len(wcls)) & (pv.max(1) >= t)).mean())
            res[inp] = r
            print(name, inp, json.dumps({k: {a: round(v, 3) for a, v in d.items() if a in ('make', 'model', 'generation')} for k, d in r.items()}), flush=True)
        res["latency"] = latency(name)
        print(name, res["latency"], flush=True)
        R["backbones"][name] = res
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    old["comparison"] = R
    OUT.write_text(json.dumps(old, indent=1))


# ---------------------------------------------------------------------------------------------------- build
def stanford_test_feats(name):
    """Stanford test embeddings (detected crop, full frame) for a backbone; B/16 ones are cached by eval_openset."""
    if name == "b16":
        Z = dict(np.load(C.WORK_DIR / "features" / "clip_zeroshot.npz"))
        return Z["test_det"], Z["test_full"]
    p = W / f"stanford_test_{name}.npz"
    if not p.exists():
        import open_clip
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        from src.data import cache_path, read_rgb
        m, _, prep = open_clip.create_model_and_transforms(*BACKBONES[name]); m = m.eval().cuda().half()
        te = pd.read_csv(C.META_DIR / "annotations.csv"); te = te[te.split == "test"]
        out = {}
        for kind in ("detcrop", "full"):
            fs = []
            with ThreadPoolExecutor(8) as ex:
                for i in range(0, len(te), 256):
                    ims = list(ex.map(lambda u: read_rgb(cache_path(u, kind)), te.uid[i:i + 256]))
                    with torch.no_grad():
                        f = m.encode_image(torch.stack([prep(Image.fromarray(x)) for x in ims]).cuda().half()).float()
                    fs.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
            out[kind] = np.concatenate(fs)
        np.savez_compressed(p, **out); del m; torch.cuda.empty_cache()
    z = np.load(p)
    return z["detcrop"], z["full"]


def build(name="b16"):
    from src.openvocab import CarVision
    df, wcls = load_world()
    scls = stanford_classes()
    E = catalogue()
    F = dict(np.load(W / f"feats_{name}.npz", allow_pickle=True))
    assert list(F["world_file"]) == list(df.file)
    Xn = norm(F["world_crop"] + F["world_full"])
    Sn = norm(F["stanford_crop"] + F["stanford_full"]); sl = F["stanford_label"]; ss = F["stanford_split"]
    ref, test = (df.set == "ref").values, (df.set == "test").values
    yr = df.cls[ref].values
    K = len(wcls)
    allc = wcls + scls
    Xs_tr, ys_tr = Sn[ss == "train"], K + sl[ss == "train"]
    Xs_va, ys_va = Sn[ss == "val"], sl[ss == "val"]

    def train(Xr, yr_):
        return fit_probe(np.concatenate([Xr, Xs_tr]), np.concatenate([yr_, ys_tr]), K + 196)

    # ---- world_name_conf: CV on the reference photos (hold out one reference per model per fold) plus
    # leave-models-out negatives (their photos must not be named as some other world model).
    Xr_all = Xn[ref]
    # Cross-validated LOGITS: positives (one reference per model held out per fold) and leave-models-out
    # negatives (a model removed entirely; its photos must not be named as some other world model).
    pos_l, pos_t, neg_l, neg_t = [], [], [], []
    rank = df[ref].groupby("cls").cumcount().values
    for f in range(5):
        tr = rank % 5 != f
        Wt, b = train(Xr_all[tr], yr[tr])
        pos_l.append(Xr_all[~tr] @ Wt.T + b); pos_t += list(yr[~tr])
    rng = np.random.default_rng(0)
    for g in np.array_split(rng.permutation(K), 5):
        keep = ~np.isin(yr, g)
        Wt, b = train(Xr_all[keep], yr[keep])
        neg_l.append(Xr_all[~keep] @ Wt.T + b); neg_t += list(yr[~keep])
    pos_l, neg_l = np.concatenate(pos_l), np.concatenate(neg_l)

    # ---- final probe on all reference photos, then temperature scaling (the probe is under-confident):
    # T minimises the NLL on CV positives (world labels) + Stanford val (Stanford labels). Never on test.
    Wt, b = train(Xr_all, yr)
    lv = Xs_va @ Wt.T + b
    def nll(T):
        a = np.log(softmax(pos_l / T)[np.arange(len(pos_t)), pos_t] + 1e-12).mean()
        c = np.log(softmax(lv / T)[np.arange(len(ys_va)), K + ys_va] + 1e-12).mean()
        return -(a + c) / 2
    temps = np.exp(np.linspace(np.log(0.02), np.log(2.0), 80))
    T = float(min(temps, key=nll))
    Wt, b = Wt / T, b / T
    pos = [(float(pi.max()), int(pi.argmax()) < K, judge(allc[int(pi.argmax())], wcls[t])) for pi, t in zip(softmax(pos_l / T), pos_t)]
    neg = [(float(pi.max()), int(pi.argmax()) < K, judge(allc[int(pi.argmax())], wcls[t])) for pi, t in zip(softmax(neg_l / T), neg_t)]

    def utility(tau):
        """name a world model only if p >= tau: +1 right generation, -1 wrong model (make-only right counts 0)."""
        s = []
        for L in (pos, neg):
            u = [(1 if j[2] else 0 if j[1] else -1) for p, w, j in L if w and p >= tau]
            s.append(sum(u) / len(L))
        return float(np.mean(s))
    grid_name = {round(t, 2): utility(t) for t in np.arange(0.2, 0.96, 0.05)}
    name_conf = max(grid_name, key=lambda t: (round(grid_name[t], 3), t))

    # ---- middle tiers: model (summed over its generations) and make (summed over all its classes).
    # Threshold = lowest value whose answers are >= 95% correct on the CV positives + leave-models-out negatives.
    from src.worldcars import class_groups
    mg, mkk = class_groups(allc)
    is_world_group = {g for g, c in zip(mg, allc) if c.get("stanford", -1) < 0}
    ug, umk = sorted(set(mg)), sorted(set(mkk))
    Gm = np.zeros((len(allc), len(ug))); Gm[np.arange(len(allc)), [ug.index(g) for g in mg]] = 1
    Gk = np.zeros((len(allc), len(umk))); Gk[np.arange(len(allc)), [umk.index(g) for g in mkk]] = 1
    cv_p = np.concatenate([softmax(pos_l / T), softmax(neg_l / T)]); cv_t = list(pos_t) + list(neg_t)
    pm_, pk_ = cv_p @ Gm, cv_p @ Gk
    truth_g = [mg[t] for t in cv_t]; truth_k = [mkk[t] for t in cv_t]

    def tier(P, labels, truth, allowed=None, prec=0.95):
        top = P.argmax(1); p = P.max(1)
        ok = np.array([labels[i] == t for i, t in zip(top, truth)])
        valid = np.array([allowed is None or labels[i] in allowed for i in top])
        grid = {}
        for t in np.arange(0.3, 0.96, 0.05):
            a = valid & (p >= t)
            grid[round(t, 2)] = dict(coverage=float(a.mean()), precision=float(ok[a].mean()) if a.any() else None)
        good = [t for t, v in grid.items() if v["precision"] is not None and v["precision"] >= prec]
        return (min(good) if good else 0.99), grid
    model_conf, grid_model = tier(pm_, ug, truth_g, is_world_group)
    make_conf, grid_make = tier(pk_, umk, truth_k)

    # ---- world_route_conf: Stanford val cars answered as a world model must stay <= 0.1%
    pv = softmax(Xs_va @ Wt.T + b)
    kv = pv.argmax(1)
    grid_route = {round(t, 2): float(((kv < K) & (pv.max(1) >= t)).mean()) for t in np.arange(0.3, 0.99, 0.02)}
    ok = [t for t, v in grid_route.items() if v <= 0.001]
    route_conf = min(ok) if ok else 0.98
    route_conf = max(route_conf, name_conf)

    # ---- zero-shot text naming for models without reference photos (leave-model-out on the reference photos)
    models = list(dict.fromkeys((e["make"], e["model"]) for e in E))
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(*BACKBONES[name])
    m = m.eval().cuda(); tok = open_clip.get_tokenizer(BACKBONES[name][0]); scale = float(m.logit_scale.exp())
    with torch.no_grad():
        labels = [f"{a} {c}" for a, c in models]
        e = torch.cat([m.encode_text(tok([t.format(n) for n in labels[i:i + 128] for t in NAME_TEMPLATES]).cuda()).float()
                       for i in range(0, len(labels), 128)])
        e = e / e.norm(dim=-1, keepdim=True)
        tm = norm(e.view(len(labels), len(NAME_TEMPLATES), -1).mean(1).cpu().numpy())
    del m; torch.cuda.empty_cache()
    pz = softmax(scale * Xr_all @ tm.T)
    zs = []
    for pi, t in zip(pz, yr):
        k = int(pi.argmax()); c = dict(make=models[k][0], model=models[k][1], y0=None)
        zs.append((float(pi[k]), judge(c, wcls[t])))
    grid_zs = {}
    for t in np.arange(0.1, 0.96, 0.05):
        u = [(1 if j[1] else -1) for p, j in zs if p >= t]
        grid_zs[round(t, 2)] = sum(u) / len(zs)
    zs_conf = max(grid_zs, key=lambda t: (round(grid_zs[t], 3), t))

    # ---- held-out test with the final probe (forced choice and thresholded)
    Xt = Xn[test]; truth = [wcls[c] for c in df.cls[test]]
    pt = softmax(Xt @ Wt.T + b); kt = pt.argmax(1)
    forced = acc([allc[k] for k in kt], truth)
    named = [(judge(allc[k], t), p[k]) for k, p, t in zip(kt, pt, truth) if k < K and p[k] >= name_conf]
    thr = dict(coverage=len(named) / len(truth),
               precision_generation=float(np.mean([j[2] for j, _ in named])) if named else None,
               precision_model=float(np.mean([j[1] for j, _ in named])) if named else None,
               precision_make=float(np.mean([j[0] for j, _ in named])) if named else None)
    # zero-shot only (as if the model had no reference photos)
    pzt = softmax(scale * Xt @ tm.T)
    zs_forced = acc([dict(make=models[k][0], model=models[k][1], y0=None) for k in pzt.argmax(1)], truth)

    # ---- Stanford test: re-routing of known cars before/after the world override (all 8,041 images)
    cv = CarVision(device="cuda")
    Z = dict(np.load(C.WORK_DIR / "features" / "clip_zeroshot.npz"))
    D = dict(np.load(C.WORK_DIR / "features" / "deep_probs.npz"))
    cfg = json.loads((C.MODELS_DIR / "hybrid_config.json").read_text())
    ann = pd.read_csv(C.META_DIR / "annotations.csv"); te = ann[ann.split == "test"].reset_index(drop=True)

    def cal(p, t):
        z = np.log(np.clip(p, 1e-12, 1)) / t; z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)
    pe = cfg["ensemble_weight"] * (cal(D["test_det_tta"], cfg["temperature"]) + cal(D["test_det_enh_tta"], cfg["temperature"])) / 2 \
        + (1 - cfg["ensemble_weight"]) * (cal(D["clip_test_det_tta"], cfg["clip_temperature"]) + cal(D["clip_test_det_enh_tta"], cfg["clip_temperature"])) / 2
    Pr = cv.name_probs(torch.from_numpy(Z["test_det"]).cuda())
    old_routed = []
    for i in range(len(te)):
        c = int(pe[i].argmax()); mm, mo, _ = split_class_name(NAMES[c]); ti = int(Pr[i].argmax()); t = cv.entries[ti]
        known = (t["stanford"] >= 0 or same_family(t["make"], t["model"], mm, mo) or pe[i, c] >= cfg["known_conf"]
                 or float(Pr[i][~cv.is_stanford].sum()) < cfg["outside_threshold"] or float(Pr[i, ti]) < cfg["open_min_top"])
        old_routed.append(not known)
    old_routed = np.array(old_routed)
    Td, Tf = stanford_test_feats(name)
    Xte = norm(Td + Tf)
    pte = softmax(Xte @ Wt.T + b); kte = pte.argmax(1)
    world_route = (kte < K) & (pte.max(1) >= route_conf)
    correct = pe.argmax(1) == te.label.values
    stanford_test = dict(n=len(te), top1_before_routing=float(correct.mean()),
                         known_rerouted_before=float(old_routed.mean()),
                         known_rerouted_after=float((old_routed | world_route).mean()),
                         extra_rerouted_by_world_layer=float((world_route & ~old_routed).mean()),
                         extra_rerouted_that_were_correct=float((world_route & ~old_routed & correct).mean()),
                         probe_top1_on_stanford_test=float((kte - K == te.label.values).mean()))

    # ---- artefacts
    rows = write_ref_artefacts(df, ref)
    meta = dict(backbone=" ".join(BACKBONES[name]), arch=BACKBONES[name][0], pretrained=BACKBONES[name][1], name=name,
                n_world=K, classes=[{k: c[k] for k in ("key", "make", "model", "gen", "years", "y0", "y1")} for c in wcls],
                stanford_classes=NAMES, method="linear probe over world reference photos + Stanford train (196 classes)")
    _build_tail(name, meta, Xr_all, Xs_tr, yr, ys_tr, T, Wt, b, tm, scale, models, cfg, route_conf, name_conf,
                model_conf, make_conf, zs_conf, K, ref, test, grid_name, grid_model, grid_make, grid_route, grid_zs,
                pos, neg, forced, thr, zs_forced, stanford_test)


def write_ref_artefacts(df, ref):
    """Reference thumbnails (web/world_refs), data_meta/world_refs.csv and docs/world_photo_credits.md."""
    keep_ref = df[ref].reset_index(drop=True)
    thumbs = C.PROJECT_ROOT / "web" / "world_refs"; thumbs.mkdir(parents=True, exist_ok=True)
    for old in thumbs.glob("*.jpg"):
        old.unlink()
    rows = []
    for i, r in keep_ref.iterrows():
        im = cv2.imread(str(W / r.file)); h, w = im.shape[:2]; s = 320 / max(h, w)
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        fname = r.file.split("/")[-1]
        cv2.imwrite(str(thumbs / fname), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
        rows.append(dict(ref=i, cls=int(r.cls), make=r.make, model=r.model, gen=r.gen, years=r.years,
                         image=f"world_refs/{fname}", title=r.title, author=r.author, license=r.license,
                         license_url=r.license_url if isinstance(r.license_url, str) else "", source=r.source))
    pd.DataFrame(rows).to_csv(C.META_DIR / "world_refs.csv", index=False, encoding="utf8")
    credits = ["# Reference photo credits (world-car layer)", "",
               "The reference thumbnails in `web/world_refs/` are downscaled (320 px) copies of photos from "
               "[Wikimedia Commons](https://commons.wikimedia.org/), used under their free licences. "
               "Each is credited below with its author, licence and source page. They are used only by the "
               "world-car layer, never to evaluate the 196-class Stanford Cars models.", "",
               "| Car | Thumbnail | Author | Licence | Source |", "|---|---|---|---|---|"]
    for r in rows:
        au = (r["author"] or "unknown").replace("|", "/")
        credits.append(f"| {r['make']} {r['model']} ({r['years']}) | `{r['image']}` | {au} | {r['license']} | [Commons]({r['source']}) |")
    (C.PROJECT_ROOT / "docs" / "world_photo_credits.md").write_text("\n".join(credits) + "\n", encoding="utf8")
    return rows


def _build_tail(name, meta, Xr_all, Xs_tr, yr, ys_tr, T, Wt, b, tm, scale, models, cfg, route_conf, name_conf,
                model_conf, make_conf, zs_conf, K, ref, test, grid_name, grid_model, grid_make, grid_route, grid_zs,
                pos, neg, forced, thr, zs_forced, stanford_test):
    # training set of the probe (for teaching new cars at run time; stays in WORK_DIR, not in the repo)
    np.savez_compressed(W / f"probe_train_{name}.npz", X=np.concatenate([Xr_all, Xs_tr]).astype(np.float16),
                        y=np.concatenate([yr, ys_tr]).astype(np.int32), T=np.float32(T))
    np.savez_compressed(C.MODELS_DIR / "world_gallery.npz", W=Wt.astype(np.float16), b=b.astype(np.float32),
                        ref_emb=Xr_all.astype(np.float16), ref_cls=yr.astype(np.int16), meta=json.dumps(meta))
    for old in C.MODELS_DIR.glob("world_text_*.npz"):
        old.unlink()
    np.savez_compressed(C.MODELS_DIR / f"world_text_{name}.npz", t=tm.astype(np.float16), scale=np.float32(scale),
                        makes=np.array([a for a, _ in models]), models=np.array([c for _, c in models]))
    cfg.update(world_route_conf=round(float(route_conf), 2), world_name_conf=round(float(name_conf), 2),
               world_model_conf=round(float(model_conf), 2), world_make_conf=round(float(make_conf), 2),
               world_zs_conf=round(float(zs_conf), 2))
    (C.MODELS_DIR / "hybrid_config.json").write_text(json.dumps(cfg, indent=1))
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    R["live"] = dict(backbone=" ".join(BACKBONES[name]), method=meta["method"], n_models=K,
                     n_ref=int(ref.sum()), n_test=int(test.sum()),
                     thresholds=dict(world_route_conf=route_conf, world_name_conf=name_conf, world_model_conf=model_conf,
                                     world_make_conf=make_conf, world_zs_conf=zs_conf),
                     tuning=dict(temperature=T, name_conf_grid=grid_name, model_conf_grid=grid_model, make_conf_grid=grid_make, route_conf_stanford_val_world_rate=grid_route, zs_conf_grid=grid_zs,
                                 cv_positives=len(pos), leave_models_out_negatives=len(neg)),
                     test_forced_choice=forced, test_named_at_threshold=thr, test_zero_shot_only=zs_forced,
                     stanford_test=stanford_test)
    OUT.write_text(json.dumps(R, indent=1))
    print(json.dumps({k: v for k, v in R["live"].items() if k != "tuning"}, indent=1))


# ---------------------------------------------------------------------------------------------------- teach
def teach_sim(n_models=30, shots=(3, 5)):
    """Simulate a user teaching a car the gallery does not know: the model is removed (its row disabled, its
    reference photos and training rows dropped), then taught back from `shots` of its reference photos with
    WorldCars.teach. Measured on its held-out test photos (named with the right generation at the live
    threshold) and on all OTHER test photos (now wrongly named as the taught car)."""
    from src.worldcars import WorldCars
    df, wcls = load_world()
    cfg = json.loads((C.MODELS_DIR / "hybrid_config.json").read_text())
    F = dict(np.load(W / "feats_siglip.npz", allow_pickle=True))
    Xn = norm(F["world_crop"] + F["world_full"])
    ref, test = (df.set == "ref").values, (df.set == "test").values
    rng = np.random.default_rng(0)
    eligible = [k for k in range(len(wcls)) if (df[ref].cls == k).sum() >= max(shots) and (df[test].cls == k).sum() >= 1]
    picks = rng.choice(eligible, size=min(n_models, len(eligible)), replace=False)
    base = WorldCars(device="cuda", _skip_replay=True)
    R = {}
    for n in shots:
        hit = tot = false_named = others = 0
        for m in picks:
            wc = WorldCars.__new__(WorldCars); wc.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else
                                                                     list(v) if isinstance(v, list) else v)
                                                                 for k, v in base.__dict__.items()})
            wc.taught = []
            wc.classes = [dict(c) for c in wc.classes]
            wc.classes[m]["user"] = True                     # treat as a brand-new car: no tuned row to start from
            keep = wc.train_y != m
            wc.train_X, wc.train_y = wc.train_X[keep], wc.train_y[keep]
            keep_r = wc.ref_cls != m
            wc.ref_emb, wc.ref_cls = wc.ref_emb[keep_r], wc.ref_cls[keep_r]
            wc.refs = [r for r, k in zip(wc.refs, keep_r) if k]
            c = wcls[m]
            shots_X = Xn[ref & (df.cls == m).values][:n]
            wc.teach(c["make"], c["model"], c["gen"], c["years"], c["y0"], c["y1"], list(shots_X),
                     [f"sim_{i}.jpg" for i in range(n)], save=False)
            for i in np.where(test)[0]:
                r = wc.match(Xn[i])
                named = r["top_is_world"] and r["p"] >= cfg["world_name_conf"]
                if df.cls[i] == m:
                    tot += 1; hit += named and judge(r["cls"], c)[2]
                else:
                    others += 1; false_named += named and r["cls"].get("key") == wc.classes[m].get("key") and judge(r["cls"], wcls[df.cls[i]])[1] is False
        R[f"{n}_photos"] = dict(models=len(picks), taught_car_test_photos=tot, named_right_generation=hit / tot,
                                other_photos_wrongly_named_as_taught_car=false_named / others)
        print(n, R[f"{n}_photos"], flush=True)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out["teach_simulation"] = R
    OUT.write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------------- e2e
def e2e(test_set="test"):
    from src.pipeline import VehicleRecognizer
    from src.data import read_rgb
    df, wcls = load_world()
    rec = VehicleRecognizer(device="cuda")
    test = df[df.set == test_set].reset_index(drop=True)
    name = "e2e_world_test" if test_set == "test" else f"e2e_{test_set}"     # e.g. e2e_test_fgvd (street photos)
    out, rows = dict(correct_generation=0, correct_model=0, correct_make_only=0, unknown=0, wrong=0, no_car=0), []
    for r in test.itertuples():
        res = rec.run(read_rgb(W / r.file), "hybrid")
        t = wcls[r.cls]
        st = res["status"]
        if st == "no-car":
            k = "no_car"
        elif st in ("unknown", "not-sure"):               # no claim made
            k = "unknown"
        else:
            if res.get("world"):
                wc = res["world"]; c = dict(make=wc["make"], model=wc["model"], y0=wc.get("y0"), stanford=-1)
                if wc.get("gen") is None:
                    c["y0"] = -1
            elif res["in_training_set"]:
                mm, mo, y = split_class_name(res["label"]) if res["class_id"] >= 0 else (res["make"], "", "0")
                c = dict(make=res["make"], model=res["model"], y0=int(y) if str(y).isdigit() else -1,
                         stanford=res["class_id"])
            else:
                c = dict(make=res["make"], model=res["model"] if res["status"] == "open-set" else "", y0=-1, stanford=-1)
            if st in ("unknown-model", "model-uncertain"):
                c["model"] = ""
            j = judge(c, t)
            k = "correct_generation" if j[2] else "correct_model" if j[1] else "correct_make_only" if j[0] and not c["model"] \
                else "wrong"
        out[k] += 1
        rows.append(dict(file=r.file, truth=f"{t['make']} {t['model']} {t['years']}", shown=res["display"], status=st, outcome=k))
    n = len(test)
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    R[name] = dict(n=n, **{k: v / n for k, v in out.items()})
    pd.DataFrame(rows).to_csv(W / f"{name}.csv", index=False)
    # Same photos with the world layer switched off (what the app answered before this feature)
    world, rec.world = rec.world, None
    before = dict(correct_model_or_better=0, correct_make_only=0, unknown_or_no_car=0, wrong=0)
    for r in test.itertuples():
        res = rec.run(read_rgb(W / r.file), "hybrid"); t = wcls[r.cls]
        if res["status"] in ("unknown", "no-car"):
            before["unknown_or_no_car"] += 1; continue
        model = "" if res["status"] in ("unknown-model", "model-uncertain") else res["model"]
        c = dict(make=res["make"] if res["status"] != "model-uncertain" else res["best_make"], model=model,
                 y0=-1, stanford=res["class_id"] if res["in_training_set"] and model else -1)
        j = judge(c, t)
        before["correct_model_or_better" if j[1] else "correct_make_only" if j[0] and not model else "wrong"] += 1
    R[f"{name}_without_world_layer"] = {k: v / n for k, v in before.items()}
    if test_set != "test":
        OUT.write_text(json.dumps(R, indent=1))
        print(json.dumps({k: R[k] for k in (name, f"{name}_without_world_layer")}, indent=1))
        return
    # Stanford test sample: does the world layer change what users see for the 196 trained classes?
    ann = pd.read_csv(C.META_DIR / "annotations.csv"); te = ann[ann.split == "test"].sample(500, random_state=0)
    changed = right_before = right_after = 0
    for r in te.itertuples():
        im = read_rgb(r.img_path)
        rec.world = None; a = rec.run(im, "hybrid")
        rec.world = world; b = rec.run(im, "hybrid")
        changed += a["display"] != b["display"]
        right_before += a["status"] == "confident" and a["class_id"] == r.label
        right_after += b["status"] == "confident" and b["class_id"] == r.label and b.get("world") is None
    R["e2e_stanford_test_sample"] = dict(n=len(te), answers_changed=changed / len(te),
                                         confident_and_correct_without_world=right_before / len(te),
                                         confident_and_correct_with_world=right_after / len(te))
    OUT.write_text(json.dumps(R, indent=1))
    print(json.dumps({k: R[k] for k in ("e2e_world_test", "e2e_world_test_without_world_layer", "e2e_stanford_test_sample")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["compare", "build", "e2e", "teach"])
    ap.add_argument("--backbone", default="b16", choices=list(BACKBONES))
    ap.add_argument("--set", default="test", help="e2e: which held-out set (test = Commons, test_fgvd = street photos)")
    a = ap.parse_args()
    if a.stage == "build":
        build(a.backbone)
    elif a.stage == "e2e":
        e2e(a.set)
    else:
        {"compare": compare, "teach": teach_sim}[a.stage]()
