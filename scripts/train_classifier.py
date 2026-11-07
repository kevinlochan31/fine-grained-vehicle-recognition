"""Fine-tune a pretrained CNN (transfer learning) for 196-way fine-grained classification.

Usage:
    python -m scripts.train_classifier                    # main model on detected/GT crops
    python -m scripts.train_classifier --kind full --epochs 12 --tag nocrop   # ablation
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config as C
from src.data import CarsDataset


def evaluate(model, loader, device):
    model.eval()
    correct1 = correct5 = n = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device)
            logits = model(x).float()
            top5 = logits.topk(5, 1).indices
            correct1 += (top5[:, 0] == y).sum().item()
            correct5 += (top5 == y[:, None]).any(1).sum().item()
            n += len(y)
    return correct1 / n, correct5 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=C.CLS_BACKBONE)
    ap.add_argument("--kind", default="crop", choices=["crop", "full"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--size", type=int, default=C.CLS_IMG_SIZE)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop early")
    args = ap.parse_args()

    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    df = pd.read_csv(C.META_DIR / "annotations.csv")
    tr, va = df[df.split == "train"], df[df.split == "val"]

    dl_tr = DataLoader(CarsDataset(tr, True, args.kind, args.size), args.bs, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(CarsDataset(va, False, args.kind, args.size), 64, shuffle=False,
                       num_workers=args.workers, pin_memory=True, persistent_workers=True)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=C.NUM_CLASSES,
                              drop_path_rate=0.2).to(device).to(memory_format=torch.channels_last)
    # Discriminative LR: new head learns 10x faster than the pretrained backbone.
    head = [p for n, p in model.named_parameters() if n.startswith("head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}],
                            weight_decay=0.05)
    steps_per_epoch = len(dl_tr)
    total, warm = args.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    out_dir = C.RUNS_DIR / f"cls_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best, history, step = 0.0, [], 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot_loss = 0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); step += 1
            tot_loss += loss.item()
            if args.max_steps and step >= args.max_steps:
                print(f"debug: {step} steps in {time.time() - t0:.1f}s"); return
        top1, top5 = evaluate(model, dl_va, device)
        rec = dict(epoch=ep + 1, loss=tot_loss / steps_per_epoch, val_top1=top1, val_top5=top5,
                   sec=time.time() - t0)
        history.append(rec); print(json.dumps(rec), flush=True)
        if top1 >= best:
            best = top1
            torch.save({"backbone": args.backbone, "size": args.size, "kind": args.kind,
                        "state_dict": model.state_dict(), "val_top1": top1}, out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))

    if args.tag == "main":   # export compact fp16 weights into the repo for deployment
        ck = torch.load(out_dir / "best.pt", map_location="cpu")
        ck["state_dict"] = {k: v.half() for k, v in ck["state_dict"].items()}
        torch.save(ck, C.MODELS_DIR / "classifier_fp16.pt")
    print("best val top1", best)


if __name__ == "__main__":
    main()
