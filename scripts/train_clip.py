"""Fine-tune CLIP ViT-B/16 (LAION-2B) for the 196 classes with strong real-world augmentation.

Why: web-scale CLIP pre-training generalises far better to photos unlike Stanford Cars (phone shots,
night, odd angles, compression). The head is initialised from CLIP text embeddings of the class
names (zero-shot init), which keeps that robustness during fine-tuning. Ensembled with ConvNeXt.

Usage:  python -m scripts.train_clip --epochs 15
"""
from __future__ import annotations

import argparse
import json
import math
import time

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from src import config as C
from src.data import cache_path, load_class_names, read_rgb

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
ARCH, PRETRAINED = "ViT-B-16", "laion2b_s34b_b88k"


class ClipClassifier(nn.Module):
    def __init__(self, visual: nn.Module, head_w: torch.Tensor):
        super().__init__()
        self.visual = visual
        self.head = nn.Linear(head_w.shape[1], head_w.shape[0], bias=True)
        with torch.no_grad():
            self.head.weight.copy_(head_w); self.head.bias.zero_()

    def forward(self, x, features=False):
        f = self.visual(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return (self.head(f), f) if features else self.head(f)


def build(pretrained=True):
    model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED if pretrained else None)
    return model


def text_head(model, names, device):
    tok = open_clip.get_tokenizer(ARCH)
    templates = ["a photo of a {}, a type of car.", "a photo of the {}.", "a {} car."]
    with torch.no_grad():
        ws = []
        for n in names:
            e = model.encode_text(tok([t.format(n) for t in templates]).to(device)).float()
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0); ws.append(e / e.norm())
        return torch.stack(ws) * model.logit_scale.exp().item()


class RobustAug:
    """Extra degradations not in torchvision: gamma darkness, low resolution, sensor noise."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        r = np.random.rand
        if r() < 0.2:                                      # under / over exposure
            g = np.random.uniform(1.4, 3.0) if r() < 0.75 else np.random.uniform(0.4, 0.7)
            img = (np.power(img / 255.0, g) * np.random.uniform(0.5, 1.0) * 255).astype(np.uint8)
        if r() < 0.2:                                      # low resolution
            h, w = img.shape[:2]; s = np.random.uniform(0.12, 0.45)
            img = cv2.resize(cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA), (w, h))
        if r() < 0.15:                                     # sensor noise
            img = np.clip(img + np.random.normal(0, np.random.uniform(5, 25), img.shape), 0, 255).astype(np.uint8)
        return img


def transforms(train: bool, size=224):
    norm = T.Normalize(CLIP_MEAN, CLIP_STD)
    if not train:
        return T.Compose([T.ToImage(), T.Resize((size, size), antialias=True), T.ToDtype(torch.float32, scale=True), norm])
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop((size, size), scale=(0.5, 1.0), ratio=(0.6, 1.66), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomPerspective(0.35, p=1.0)], p=0.3),        # viewpoint
        T.RandomApply([T.RandomRotation(15)], p=0.3),
        T.RandomApply([T.ColorJitter(0.5, 0.5, 0.4, 0.06)], p=0.8),        # lighting / white balance
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(7, sigma=(0.1, 3.0))], p=0.2),      # focus / motion
        T.RandomApply([T.JPEG((15, 90))], p=0.3),                         # compression artefacts
        T.ToDtype(torch.float32, scale=True), norm,
        T.RandomErasing(p=0.3, scale=(0.02, 0.25)),                       # occlusion
    ])


class DS:
    def __init__(self, df, train, kind="crop"):
        self.df, self.kind, self.train = df.reset_index(drop=True), kind, train
        self.tf, self.aug = transforms(train), RobustAug()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = read_rgb(cache_path(r.uid, self.kind))
        if self.train:
            img = self.aug(img)
        return self.tf(img), int(r.label)


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval(); c1 = c5 = n = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in dl:
            z = model(x.to(device)).float(); y = y.to(device)
            t5 = z.topk(5, 1).indices
            c1 += (t5[:, 0] == y).sum().item(); c5 += (t5 == y[:, None]).any(1).sum().item(); n += len(y)
    return c1 / n, c5 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    device = "cuda"
    names = load_class_names()
    clip = build().to(device)
    head = text_head(clip, names, device)
    model = ClipClassifier(clip.visual, head).to(device)
    del clip.transformer                                    # text tower no longer needed
    model.visual.set_grad_checkpointing(True)

    df = pd.read_csv(C.META_DIR / "annotations.csv")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    dl_tr = DataLoader(DS(tr, True), a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(DS(va, False), 96, num_workers=a.workers, pin_memory=True, persistent_workers=True)
    zs1, zs5 = evaluate(model, dl_va, device)
    print(json.dumps({"epoch": 0, "zero_shot_val_top1": zs1, "zero_shot_val_top5": zs5}), flush=True)

    # layer-wise LR decay: earlier ViT blocks move less (protects pre-trained robustness)
    blocks = model.visual.transformer.resblocks
    groups, decay = [], 0.75
    groups.append({"params": list(model.head.parameters()), "lr": a.lr * 20})
    groups.append({"params": [p for n_, p in model.visual.named_parameters() if n_.startswith(("ln_post", "proj"))] + ([model.visual.proj] if isinstance(model.visual.proj, nn.Parameter) else []), "lr": a.lr})
    for i, b in enumerate(blocks):
        groups.append({"params": list(b.parameters()), "lr": a.lr * decay ** (len(blocks) - 1 - i)})
    emb = [p for n_, p in model.visual.named_parameters() if n_.startswith(("conv1", "class_embedding", "positional_embedding", "ln_pre"))]
    groups.append({"params": emb, "lr": a.lr * decay ** len(blocks)})
    seen, uniq = set(), []
    for g in groups:                                        # de-duplicate parameters across groups
        ps = []
        for p in g["params"]:
            if id(p) not in seen:
                seen.add(id(p)); ps.append(p)
        if ps: uniq.append({**g, "params": ps, "base_lr": g["lr"]})
    opt = torch.optim.AdamW(uniq, weight_decay=0.05)
    steps, warm = a.epochs * len(dl_tr), len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    out = C.RUNS_DIR / "clip_ft"; out.mkdir(parents=True, exist_ok=True)
    best, hist = 0, [{"epoch": 0, "val_top1": zs1, "val_top5": zs5, "zero_shot": True}]
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()
        v1, v5 = evaluate(model, dl_va, device)
        rec = dict(epoch=ep + 1, loss=tot / len(dl_tr), val_top1=v1, val_top5=v5, sec=time.time() - t0)
        hist.append(rec); print(json.dumps(rec), flush=True)
        if v1 >= best:
            best = v1
            torch.save({"arch": ARCH, "pretrained": PRETRAINED, "state_dict": {k: v.half() for k, v in model.state_dict().items()},
                        "val_top1": v1}, out / "best_fp16.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print("best", best)


if __name__ == "__main__":
    main()
