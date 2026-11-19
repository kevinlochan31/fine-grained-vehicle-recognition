"""World-car layer, step 2: select usable Commons photos, split them, and embed them with each backbone.

select : YOLO11s finds the car (box conf >= 0.5, box >= 15% of the frame), the CLIP gate says "car"
         (p >= 0.9) and a zero-shot exterior check rejects interiors, engine bays, badges, wheels, scale
         models and drawings. Near-duplicates (cosine >= 0.95 within a model) are dropped.
         Per model the photos are split into a reference set (<= 20) and a held-out test set (<= 3).
         The split is by photographer where possible (test authors never appear in the reference set),
         so the test measures recognition of a different car/photo session, not a re-shot of the same car.
embed  : for each backbone (live CLIP ViT-B/16 + the candidates ViT-L/14 and SigLIP ViT-B/16) embed the
         YOLO crop and the full frame of every selected photo, plus Stanford Cars train/val crops+frames
         (needed for a probe that also knows the 196 trained classes and for false-routing estimates).

Everything is written to WORK_DIR/world_cars (outside the project).
Usage:  python -m scripts.build_world_gallery select|embed [--backbones b16,l14,siglip]
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config as C
from src.data import cache_path, crop_image, read_rgb

W = C.WORK_DIR / "world_cars"
MAX_REF = 20          # reference photos per model (all usable photos that are not in the test set)
# Frozen held-out test set: once written, later re-selections (e.g. after downloading more photos) keep exactly
# these test photos, so accuracy before/after stays comparable.
_FZ = W / "test_frozen.csv"
FROZEN = set(pd.read_csv(_FZ).file) if _FZ.exists() else set()
BACKBONES = {"b16": ("ViT-B-16", "laion2b_s34b_b88k"), "l14": ("ViT-L-14", "laion2b_s32b_b82k"),
             "siglip": ("ViT-B-16-SigLIP", "webli")}
EXT = ["a photo of a car's exterior.", "a photo of a whole car taken from outside.", "a photo of a car on the road."]
NOT_EXT = ["a photo of a car's interior, dashboard and seats.", "a photo of a car engine bay.",
           "a close-up photo of a car badge or emblem.", "a close-up photo of a car wheel.",
           "a close-up photo of a car's headlight or taillight.", "a photo of a toy or scale model car.",
           "a drawing of a car.", "a photo of a car's boot or trunk space."]


def select():
    from src.detector import VehicleDetector
    from src.openvocab import CarVision
    import open_clip
    df = pd.read_csv(W / "photos.csv")
    cv, det = CarVision(device="cuda"), VehicleDetector(device="cuda")
    m, _, _ = open_clip.create_model_and_transforms(*BACKBONES["b16"])
    m = m.eval().cuda(); tok = open_clip.get_tokenizer("ViT-B-16")
    with torch.no_grad():
        t = m.encode_text(tok(EXT + NOT_EXT).cuda()).float(); t = t / t.norm(dim=-1, keepdim=True)
    del m
    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="filter"):
        try:
            im = read_rgb(W / r.file)
        except Exception:
            continue
        d, _ = det.primary(im)
        h, w = im.shape[:2]
        x1, y1, x2, y2 = d["box"]
        area = (x2 - x1) * (y2 - y1) / (h * w)
        f = cv.embed([im, crop_image(im, d["box"])])
        pc = float(cv.car_probability(f[:1])[0])
        pe = float((cv.scale * f[:1] @ t.T).softmax(-1)[0, :len(EXT)].sum())
        fm = f.mean(0); fm = (fm / fm.norm()).cpu().numpy()
        ok = d["source"] == "yolo" and d["conf"] >= 0.5 and area >= 0.15 and pc >= 0.9 and pe >= 0.6
        rows.append(dict(file=r.file, det_conf=d["conf"], box_area=area, car_p=pc, exterior_p=pe, keep=ok, emb=fm))
    q = pd.DataFrame(rows)
    df = df.merge(q.drop(columns="emb"), on="file")
    emb = dict(zip(q.file, q.emb))
    rng = random.Random(0)
    split = {}
    for key, g in df[df.keep].groupby("key", sort=False):
        files = list(g.file)
        # near-duplicate removal within the model
        kept = []
        for fl in files:
            if all(float(emb[fl] @ emb[k]) < 0.95 for k in kept):
                kept.append(fl)
        g = g[g.file.isin(kept)]
        frozen = [f for f in g.file if f in FROZEN]
        if frozen:
            # frozen test set (test_frozen.csv): keep exactly the earlier test photos; every other usable photo by
            # a photographer who is not in the test set becomes a reference (up to MAX_REF)
            t_auth = set(g[g.file.isin(frozen)].author)
            # never let a reference be a near-copy of a test photo (cosine >= 0.95)
            clean = g[[all(float(emb[f] @ emb[t]) < 0.95 for t in frozen) or f in frozen for f in g.file]]
            rest = [L for a, L in clean[~clean.file.isin(frozen) & ~clean.author.isin(t_auth)].groupby("author").file.apply(list).items()]
            disjoint = bool(rest)
            if not rest:                                       # single photographer: any non-test photo
                rest = [list(clean[~clean.file.isin(frozen)].file)]
            ref = []
            while len(ref) < MAX_REF and any(rest):
                for L in rest:
                    if L and len(ref) < MAX_REF:
                        ref.append(L.pop(0))
            for fl in ref:
                split[fl] = ("ref", disjoint)
            for fl in frozen:
                split[fl] = ("test", disjoint)
            continue
        if FROZEN or len(g) < 4:
            continue
        authors = list(g.groupby("author").file.apply(list).items())
        rng.shuffle(authors)
        authors.sort(key=lambda a: len(a[1]))                  # small author groups go to test first
        test, ref, disjoint = [], [], True
        n_test = 3 if len(g) >= 7 else 2 if len(g) >= 5 else 1
        ai = 0
        while len(test) < n_test and ai < len(authors) - 1:    # keep at least one author for the references
            test += authors[ai][1][: n_test - len(test)]; ai += 1
        rest = [a[1] for a in authors[ai:]]
        while len(ref) < MAX_REF and any(rest):                      # round-robin over the remaining authors
            for L in rest:
                if L and len(ref) < MAX_REF:
                    ref.append(L.pop(0))
        if len(test) < n_test or len(ref) < 2:                 # one photographer only: split by photo
            disjoint = False
            allf = list(g.file); rng.shuffle(allf)
            test, ref = allf[:n_test], allf[n_test:n_test + MAX_REF]
        for fl in ref:
            split[fl] = ("ref", disjoint)
        for fl in test:
            split[fl] = ("test", disjoint)
    df["set"] = df.file.map(lambda f: split.get(f, (None, None))[0])
    df["author_disjoint"] = df.file.map(lambda f: split.get(f, (None, None))[1])
    df.to_csv(W / "selected.csv", index=False)
    s = df[df.set.notna()]
    print(f"candidates {len(df)}, passed filter {int(df.keep.sum())}, used {len(s)} "
          f"(ref {int((s.set == 'ref').sum())}, test {int((s.set == 'test').sum())}) for {s.key.nunique()} models; "
          f"author-disjoint models {s.groupby('key').author_disjoint.first().mean():.1%}")


def embed(names):
    import open_clip
    from PIL import Image
    df = pd.read_csv(W / "selected.csv")
    df = df[df.set.notna()].reset_index(drop=True)
    from src.detector import VehicleDetector
    det = VehicleDetector(device="cuda")
    boxes = [det.primary(read_rgb(W / f))[0]["box"] for f in tqdm(df.file, desc="boxes")]
    ann = pd.read_csv(C.META_DIR / "annotations.csv")
    st = ann[ann.split.isin(["train", "val"])].reset_index(drop=True)
    for name in names:
        arch, pre = BACKBONES[name]
        if name == "b16":            # the live model: embed exactly as the app does (CarVision.embed)
            from src.openvocab import CarVision
            model = CarVision(device="cuda")
            enc = lambda ims: model.embed(ims).cpu().numpy()
        else:
            model, _, prep = open_clip.create_model_and_transforms(arch, pretrained=pre)
            model = model.eval().cuda().half()

            @torch.no_grad()
            def enc(ims):
                x = torch.stack([prep(Image.fromarray(i)) for i in ims]).cuda().half()
                f = model.encode_image(x).float()
                return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

        out = {}
        ims = [read_rgb(W / f) for f in df.file]
        out["world_full"] = np.concatenate([enc(ims[i:i + 64]) for i in range(0, len(ims), 64)])
        crops = [crop_image(im, b) for im, b in zip(ims, boxes)]
        out["world_crop"] = np.concatenate([enc(crops[i:i + 64]) for i in range(0, len(crops), 64)])
        for kind in ("crop", "full"):
            fs = []
            with ThreadPoolExecutor(8) as ex:
                for i in tqdm(range(0, len(st), 256), desc=f"{name} stanford {kind}", leave=False):
                    fs.append(enc(list(ex.map(lambda u: read_rgb(cache_path(u, kind)), st.uid[i:i + 256]))))
            out[f"stanford_{kind}"] = np.concatenate(fs)
        out["stanford_label"] = st.label.values
        out["stanford_split"] = st.split.values.astype(str)
        out["world_file"] = df.file.values.astype(str)
        np.savez_compressed(W / f"feats_{name}.npz", **out)
        print(name, {k: v.shape for k, v in out.items()})
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "embed"])
    ap.add_argument("--backbones", default="b16,l14,siglip")
    a = ap.parse_args()
    select() if a.stage == "select" else embed(a.backbones.split(","))
