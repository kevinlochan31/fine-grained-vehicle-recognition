"""World-car layer, Phase II: fine-tune SigLIP ViT-B/16 end to end (instead of a linear probe on frozen features).

The network is trained on the world reference photos (Wikimedia Commons, selected by build_world_gallery.py)
together with the Stanford Cars *train* split, so one head covers K world generations + the 196 Stanford classes.
The frozen held-out test photos (test_frozen.csv + the test split of newly added models) are never trained on.

stages
  crops : YOLO crop of every selected world photo -> WORK_DIR/world_cars/crops/ (the app embeds frame + crop)
  train --run tune  : holds out every 5th reference photo of each model (positives) and 10% of the models
                      entirely (leave-models-out negatives); used ONLY to calibrate the temperature and thresholds
  train --run final : all reference photos; this is the model that ships
  build : temperature + thresholds from the tune run, network + head from the final run ->
          models/world_siglip_ft.pt, models/world_gallery.npz, world_* thresholds, results/world_cars.json

Usage:  python -m scripts.train_world crops | train --run tune | train --run final | build
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image

from src import config as C
from src.data import cache_path, load_class_names

W = C.WORK_DIR / "world_cars"
CROPS = W / "crops"
ARCH, PRETRAINED = "ViT-B-16-SigLIP", "webli"
WEIGHTS = C.MODELS_DIR / "world_siglip_ft.pt"
HP = dict(epochs=14, batch=48, lr_backbone=2e-5, lr_head=1e-3, wd=0.05, smoothing=0.1, warmup=1, layer_decay=0.8)


# ------------------------------------------------------------------------------------------------ data
def world_table():
    from scripts.eval_world import load_world
    df, wcls = load_world()
    df["crop"] = df.file.map(lambda f: str(CROPS / f.split("/")[-1]))
    df["full"] = df.file.map(lambda f: str(W / f))
    return df, wcls


def stanford_table(split):
    ann = pd.read_csv(C.META_DIR / "annotations.csv")
    s = ann[ann.split == split].reset_index(drop=True)
    crop_kind = "detcrop" if split == "test" else "crop"
    s["crop"] = [str(cache_path(u, crop_kind)) for u in s.uid]
    s["full"] = [str(cache_path(u, "full")) for u in s.uid]
    return s


class Pairs(torch.utils.data.Dataset):
    """(full frame, crop) photo pairs. Training: one of the two at random; eval: both."""

    def __init__(self, full, crop, y, tf, train):
        self.full, self.crop, self.y, self.tf, self.train = list(full), list(crop), np.asarray(y), tf, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.train:
            p = self.crop[i] if np.random.rand() < 0.6 else self.full[i]
            return self.tf(Image.open(p).convert("RGB")), int(self.y[i])
        return (self.tf(Image.open(self.full[i]).convert("RGB")), self.tf(Image.open(self.crop[i]).convert("RGB")),
                int(self.y[i]))


def transforms():
    import torchvision.transforms as T
    import open_clip
    _, _, prep = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
    norm = [t for t in prep.transforms if isinstance(t, T.Normalize)][0]
    train = T.Compose([T.RandomResizedCrop(224, scale=(0.45, 1.0), ratio=(0.75, 1.45), interpolation=T.InterpolationMode.BICUBIC),
                       T.RandomHorizontalFlip(), T.RandomApply([T.ColorJitter(0.35, 0.35, 0.3, 0.04)], p=0.8),
                       T.RandomGrayscale(0.05), T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.15),
                       T.ToTensor(), norm, T.RandomErasing(p=0.2, scale=(0.02, 0.12))])
    return train, prep


# ------------------------------------------------------------------------------------------------ model
class CosineHead(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(n))
        self.log_s = nn.Parameter(torch.tensor(math.log(20.0)))

    def forward(self, f):
        return self.log_s.exp() * Fn.normalize(f, dim=-1) @ Fn.normalize(self.w, dim=-1).T + self.b


def load_visual(weights=None):
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
    v = m.visual
    del m
    if weights is not None:
        v.load_state_dict(torch.load(weights, map_location="cpu"))
    return v


@torch.no_grad()
def embed_pairs(vis, ds, bs=128):
    """L2-normalised mean of frame + crop embeddings (exactly what WorldCars.embed computes in the app)."""
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=6, persistent_workers=False)
    out = []
    vis.eval()
    for a, b, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fa, fb = vis(a.cuda(non_blocking=True)).float(), vis(b.cuda(non_blocking=True)).float()
        f = Fn.normalize(Fn.normalize(fa, dim=-1) + Fn.normalize(fb, dim=-1), dim=-1)
        out.append(f.cpu())
    return torch.cat(out).numpy()


def param_groups(vis, head):
    """Layer-wise lr decay: the last transformer blocks move most, the patch embedding least."""
    names = [n for n, _ in vis.named_parameters()]
    blocks = sorted({int(n.split("blocks.")[1].split(".")[0]) for n in names if "blocks." in n})
    depth = len(blocks) + 1
    groups = {}
    for n, p in vis.named_parameters():
        if "blocks." in n:
            lid = int(n.split("blocks.")[1].split(".")[0]) + 1
        elif any(k in n for k in ("patch_embed", "pos_embed", "cls_token", "reg_token")):
            lid = 0
        else:
            lid = depth                                          # final norm, attention pool
        scale = HP["layer_decay"] ** (depth - lid)
        nd = p.ndim == 1 or n.endswith(".bias")
        key = (lid, nd)
        groups.setdefault(key, dict(params=[], lr=HP["lr_backbone"] * scale, weight_decay=0.0 if nd else HP["wd"]))
        groups[key]["params"].append(p)
    out = list(groups.values())
    out.append(dict(params=list(head.parameters()), lr=HP["lr_head"], weight_decay=1e-4))
    for g in out:
        g["base_lr"] = g["lr"]
    return out


# ------------------------------------------------------------------------------------------------ stages
def crops():
    from src.detector import VehicleDetector
    from src.data import crop_image, read_rgb
    df = pd.read_csv(W / "selected.csv")
    df = df[df.set.notna()]
    CROPS.mkdir(exist_ok=True)
    det = VehicleDetector(device="cuda")
    n = 0
    for f in df.file:
        dst = CROPS / f.split("/")[-1]
        if dst.exists():
            continue
        im = read_rgb(W / f)
        d, _ = det.primary(im)
        Image.fromarray(crop_image(im, d["box"])).save(dst, quality=92)
        n += 1
    print(f"crops written: {n} (total {len(df)})")


def splits(df, K, run):
    ref = (df.set == "ref").values
    yr = df.cls.values
    if run == "final":
        return ref, np.zeros(len(df), bool), np.zeros(len(df), bool), []
    rng = np.random.default_rng(0)
    held_models = sorted(rng.choice(K, size=max(1, K // 10), replace=False).tolist())
    rank = df[ref].groupby("cls").cumcount().reindex(df.index).fillna(-1).values
    neg = ref & np.isin(yr, held_models)
    val = ref & ~neg & (rank % 5 == 4)
    train = ref & ~neg & ~val
    return train, val, neg, held_models


def train(run):
    torch.backends.cuda.matmul.allow_tf32 = True
    df, wcls = world_table()
    K = len(wcls)
    st = stanford_table("train")
    tr, val, neg, held = splits(df, K, run)
    tf_train, tf_eval = transforms()
    full = list(df.full[tr]) + list(st.full); crop = list(df.crop[tr]) + list(st.crop)
    y = np.concatenate([df.cls[tr].values, K + st.label.values])
    ds = Pairs(full, crop, y, tf_train, train=True)
    cnt = np.bincount(y, minlength=K + 196).astype(float)
    sampler = torch.utils.data.WeightedRandomSampler(1.0 / np.sqrt(cnt[y]), num_samples=len(y), replacement=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=HP["batch"], sampler=sampler, num_workers=8, drop_last=True,
                                     persistent_workers=True, pin_memory=True)
    vis = load_visual().cuda()
    if hasattr(vis, "set_grad_checkpointing"):
        vis.set_grad_checkpointing(True)
    head = CosineHead(vis(torch.zeros(1, 3, 224, 224, device="cuda")).shape[-1], K + 196).cuda()
    # prototype init of the head from the pretrained features (fast convergence, no random start)
    proto_ds = Pairs(full, crop, y, tf_eval, train=False)
    Fp = torch.from_numpy(embed_pairs(vis, proto_ds)).cuda()
    yt = torch.from_numpy(y).cuda()
    P = torch.zeros(K + 196, Fp.shape[1], device="cuda").index_add_(0, yt, Fp)
    with torch.no_grad():
        head.w.copy_(Fn.normalize(P, dim=-1))
    opt = torch.optim.AdamW(param_groups(vis, head), betas=(0.9, 0.999))
    steps = HP["epochs"] * len(dl); warm = HP["warmup"] * len(dl)
    step, t0 = 0, time.time()
    print(f"[{run}] world classes {K}, train photos world {int(tr.sum())} + stanford {len(st)}, "
          f"val {int(val.sum())}, held-out models {len(held)} ({int(neg.sum())} photos), steps {steps}", flush=True)
    for ep in range(HP["epochs"]):
        vis.train(); head.train()
        tot = cor = 0; loss_sum = 0.0
        for x, yy in dl:
            lr_f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_f
            x, yy = x.cuda(non_blocking=True), yy.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(vis(x).float())
            loss = Fn.cross_entropy(logits.float(), yy, label_smoothing=HP["smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(vis.parameters()) + list(head.parameters()), 1.0)
            opt.step(); step += 1
            loss_sum += float(loss) * len(yy); tot += len(yy); cor += int((logits.argmax(1) == yy).sum())
        msg = f"[{run}] epoch {ep + 1}/{HP['epochs']} loss {loss_sum / tot:.3f} train-acc {cor / tot:.3f} {time.time() - t0:.0f}s"
        if val.any():
            fv = torch.from_numpy(embed_pairs(vis, Pairs(df.full[val], df.crop[val], df.cls[val], tf_eval, False))).cuda()
            with torch.no_grad():
                pv = head(fv).argmax(1).cpu().numpy()
            msg += f"  world-val top1 {float((pv == df.cls[val].values).mean()):.3f}"
        print(msg, flush=True)
    vis.eval()
    # ---- features for everything the build stage needs (frame + crop mean, as the app embeds)
    out = dict(head_w=head.w.detach().cpu().numpy(), head_b=head.b.detach().cpu().numpy(),
               head_s=float(head.log_s.exp()), K=K, held_models=np.array(held, int),
               world_file=df.file.values.astype(str), world_cls=df.cls.values, world_set=df.set.values.astype(str),
               train_mask=tr, val_mask=val, neg_mask=neg)
    out["world_feat"] = embed_pairs(vis, Pairs(df.full, df.crop, df.cls, tf_eval, False))
    for split in ("train", "val", "test"):
        s = stanford_table(split)
        out[f"stanford_{split}_feat"] = embed_pairs(vis, Pairs(s.full, s.crop, s.label, tf_eval, False))
        out[f"stanford_{split}_label"] = s.label.values
    np.savez_compressed(W / f"ft_{run}.npz", **out)
    torch.save({k: v.half() for k, v in vis.state_dict().items()}, W / f"ft_{run}_visual.pt")
    print(f"[{run}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def head_logits(Z, X):
    Wn = Z["head_w"] / np.linalg.norm(Z["head_w"], axis=-1, keepdims=True)
    return Z["head_s"] * X @ Wn.T + Z["head_b"]


def build():
    import shutil
    from scripts.eval_world import acc, judge, stanford_classes, write_ref_artefacts
    from src.worldcars import class_groups
    df, wcls = world_table()
    K = len(wcls); allc = wcls + stanford_classes()
    Zt, Zf = (dict(np.load(W / f"ft_{r}.npz", allow_pickle=True)) for r in ("tune", "final"))   # our own files
    for Z in (Zt, Zf):
        for k in ("train_mask", "val_mask", "neg_mask"):
            Z[k] = Z[k].astype(bool)
        Z["head_s"] = float(Z["head_s"])
    assert list(Zf["world_file"]) == list(df.file) == list(Zt["world_file"]), "features out of date"
    held = set(Zt["held_models"].tolist())
    # ---------------- calibration on the tune run (never on test)
    Lw = head_logits(Zt, Zt["world_feat"])
    Lw_neg = Lw.copy(); Lw_neg[:, list(held)] = -1e9                 # held-out models do not exist for that net
    pos_l, pos_t = Lw[Zt["val_mask"]], df.cls.values[Zt["val_mask"]]
    neg_l, neg_t = Lw_neg[Zt["neg_mask"]], df.cls.values[Zt["neg_mask"]]
    lv, ys_va = head_logits(Zt, Zt["stanford_val_feat"]), Zt["stanford_val_label"]

    def nll(T):
        a = np.log(softmax(pos_l / T)[np.arange(len(pos_t)), pos_t] + 1e-12).mean()
        c = np.log(softmax(lv / T)[np.arange(len(ys_va)), K + ys_va] + 1e-12).mean()
        return -(a + c) / 2
    temps = np.exp(np.linspace(np.log(0.2), np.log(5.0), 120))
    T = float(min(temps, key=nll))
    pos = [(float(p.max()), int(p.argmax()) < K, judge(allc[int(p.argmax())], wcls[t])) for p, t in zip(softmax(pos_l / T), pos_t)]
    neg = [(float(p.max()), int(p.argmax()) < K, judge(allc[int(p.argmax())], wcls[t])) for p, t in zip(softmax(neg_l / T), neg_t)]

    def utility(tau):
        s = []
        for L in (pos, neg):
            u = [(1 if j[2] else 0 if j[1] else -1) for p, w, j in L if w and p >= tau]
            s.append(sum(u) / len(L))
        return float(np.mean(s))
    grid_name = {round(t, 2): utility(t) for t in np.arange(0.3, 0.96, 0.05)}
    name_conf = max(grid_name, key=lambda t: (round(grid_name[t], 3), t))

    mg, mkk = class_groups(allc)
    is_world_group = {g for g, c in zip(mg, allc) if c.get("stanford", -1) < 0}
    ug, umk = sorted(set(mg)), sorted(set(mkk))
    Gm = np.zeros((len(allc), len(ug))); Gm[np.arange(len(allc)), [ug.index(g) for g in mg]] = 1
    Gk = np.zeros((len(allc), len(umk))); Gk[np.arange(len(allc)), [umk.index(g) for g in mkk]] = 1
    cv_p = np.concatenate([softmax(pos_l / T), softmax(neg_l / T)]); cv_t = list(pos_t) + list(neg_t)
    pm_, pk_ = cv_p @ Gm, cv_p @ Gk

    # a wrong make or model costs more than "not sure", so each tier must be right >= 99% of the time on held-out
    # photos (95% let too many wrong makes through once the catalogue grew to 735 generations)
    def tier(P, labels, truth, allowed=None, prec=0.99):
        top = P.argmax(1); p = P.max(1)
        ok = np.array([labels[i] == t for i, t in zip(top, truth)])
        valid = np.array([allowed is None or labels[i] in allowed for i in top])
        grid = {}
        for t in list(np.arange(0.3, 0.96, 0.05)) + [0.97, 0.99]:
            a = valid & (p >= t)
            grid[round(t, 2)] = dict(coverage=float(a.mean()), precision=float(ok[a].mean()) if a.any() else None)
        good = [t for t, v in grid.items() if v["precision"] is not None and v["precision"] >= prec]
        return (min(good) if good else 0.99), grid
    model_conf, grid_model = tier(pm_, ug, [mg[t] for t in cv_t], is_world_group)
    make_conf, grid_make = tier(pk_, umk, [mkk[t] for t in cv_t])
    pv = softmax(lv / T); kv = pv.argmax(1)
    grid_route = {round(t, 2): float(((kv < K) & (pv.max(1) >= t)).mean()) for t in np.arange(0.3, 0.99, 0.02)}
    ok_r = [t for t, v in grid_route.items() if v <= 0.001]
    route_conf = max(min(ok_r) if ok_r else 0.98, name_conf)
    # "Not sure" cross-check: doubt a confident classifier answer when the head puts >= doubt_mass of its belief on
    # world (non-Stanford) classes. Lowest threshold that doubts <= 0.5% of Stanford val cars; its catch rate on
    # leave-models-out negatives (car models the net has never seen, like a new Thar) is reported.
    lv_m = lv.copy(); lv_m[:, list(held)] = -1e9
    mass_sv = softmax(lv_m / T)[:, :K].sum(1)
    mass_neg = softmax(neg_l / T)[:, :K].sum(1)
    grid_doubt = {round(t, 2): dict(stanford_val_doubted=float((mass_sv >= t).mean()),
                                    unseen_models_caught=float((mass_neg >= t).mean()))
                  for t in np.arange(0.2, 0.91, 0.05)}
    ok_d = [t for t, v in grid_doubt.items() if v["stanford_val_doubted"] <= 0.005]
    doubt_mass = min(ok_d) if ok_d else 0.9

    # ---------------- the shipped network: final run
    Wn = Zf["head_w"] / np.linalg.norm(Zf["head_w"], axis=-1, keepdims=True)
    Wg, bg = (Zf["head_s"] * Wn) / T, Zf["head_b"] / T
    X = Zf["world_feat"]
    ref, test = (df.set == "ref").values, (df.set == "test").values
    truth = [wcls[c] for c in df.cls[test]]
    pt = softmax(X[test] @ Wg.T + bg); kt = pt.argmax(1)
    forced = acc([allc[k] for k in kt], truth)
    named = [(judge(allc[k], t), p[k]) for k, p, t in zip(kt, pt, truth) if k < K and p[k] >= name_conf]
    thr = dict(coverage=len(named) / len(truth),
               precision_generation=float(np.mean([j[2] for j, _ in named])) if named else None,
               precision_model=float(np.mean([j[1] for j, _ in named])) if named else None,
               precision_make=float(np.mean([j[0] for j, _ in named])) if named else None)
    frozen = set(pd.read_csv(W / "test_frozen.csv").file)
    old_test = test & df.file.isin(frozen).values
    forced_old = acc([allc[k] for k in softmax(X[old_test] @ Wg.T + bg).argmax(1)], [wcls[c] for c in df.cls[old_test]])
    pte = softmax(Zf["stanford_test_feat"] @ Wg.T + bg); kte = pte.argmax(1)
    stanford_test = dict(n=len(kte), world_answer_rate_at_route_conf=float(((kte < K) & (pte.max(1) >= route_conf)).mean()),
                         head_top1_on_stanford_test=float((kte - K == Zf["stanford_test_label"]).mean()),
                         doubted_at_doubt_mass=float((pte[:, :K].sum(1) >= doubt_mass).mean()))

    # ---------------- artefacts
    shutil.copy(W / "ft_final_visual.pt", WEIGHTS)
    yr = df.cls[ref].values
    meta = dict(backbone=f"{ARCH} {PRETRAINED} (fine-tuned)", arch=ARCH, pretrained=PRETRAINED, name="siglip_ft",
                weights=WEIGHTS.name, zero_shot_backbone="siglip", n_world=K,
                classes=[{k: c[k] for k in ("key", "make", "model", "gen", "years", "y0", "y1")} for c in wcls],
                stanford_classes=load_class_names(),
                method="SigLIP ViT-B/16 fine-tuned end to end on world reference photos + Stanford train (cosine head)")
    np.savez_compressed(C.MODELS_DIR / "world_gallery.npz", W=Wg.astype(np.float16), b=bg.astype(np.float32),
                        ref_emb=X[ref].astype(np.float16), ref_cls=yr.astype(np.int16), meta=json.dumps(meta))
    st_tr = Zf["stanford_train_feat"]
    np.savez_compressed(W / "probe_train_siglip_ft.npz", X=np.concatenate([X[ref], st_tr]).astype(np.float16),
                        y=np.concatenate([yr, K + Zf["stanford_train_label"]]).astype(np.int32), T=np.float32(1.0))
    write_ref_artefacts(df, ref)
    cfg = json.loads((C.MODELS_DIR / "hybrid_config.json").read_text())
    cfg.update(world_route_conf=round(float(route_conf), 2), world_name_conf=round(float(name_conf), 2),
               world_model_conf=round(float(model_conf), 2), world_make_conf=round(float(make_conf), 2),
               world_doubt_mass=round(float(doubt_mass), 2))
    (C.MODELS_DIR / "hybrid_config.json").write_text(json.dumps(cfg, indent=1))
    OUT = C.RESULTS_DIR / "world_cars.json"
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "live" in R and "round1_linear_probe" not in R:
        R["round1_linear_probe"] = R["live"]
    R["live"] = dict(backbone=meta["backbone"], method=meta["method"], hyperparameters=HP, n_models=K,
                     n_ref=int(ref.sum()), n_test=int(test.sum()), n_test_round1_frozen=int(old_test.sum()),
                     thresholds=dict(world_route_conf=route_conf, world_name_conf=name_conf, world_model_conf=model_conf,
                                     world_make_conf=make_conf, world_doubt_mass=doubt_mass),
                     tuning=dict(temperature=T, name_conf_grid=grid_name, doubt_mass_grid=grid_doubt, model_conf_grid=grid_model, make_conf_grid=grid_make,
                                 route_conf_stanford_val_world_rate=grid_route, val_positives=len(pos),
                                 leave_models_out_negatives=len(neg), held_out_models=len(held)),
                     test_forced_choice=forced, test_forced_choice_round1_photos=forced_old,
                     test_named_at_threshold=thr, stanford_test=stanford_test)
    OUT.write_text(json.dumps(R, indent=1))
    print(json.dumps({k: v for k, v in R["live"].items() if k != "tuning"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["crops", "train", "build"])
    ap.add_argument("--run", default="final", choices=["tune", "final"])
    a = ap.parse_args()
    {"crops": crops, "build": build}.get(a.stage, lambda: train(a.run))()
