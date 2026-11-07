"""Model optimisation: reduced precision (FP16), INT8 static quantisation (ONNX Runtime)
and global magnitude pruning. Measures size, latency and accuracy of each variant.

Usage:  python -m scripts.optimize
"""
from __future__ import annotations

import gzip
import io
import json
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import timm
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src import config as C
from src.classifier import preprocess_batch, softmax
from src.data import cache_path, read_rgb
from src.metrics import topk_acc

ONNX_FP32 = C.WORK_DIR / "classifier_fp32.onnx"
ONNX_PRE = C.WORK_DIR / "classifier_fp32_pre.onnx"
ONNX_INT8 = C.MODELS_DIR / "classifier_int8.onnx"


def load_model():
    ck = torch.load(C.RUNS_DIR / "cls_main" / "best.pt", map_location="cpu", weights_only=False)
    m = timm.create_model(ck["backbone"], pretrained=False, num_classes=C.NUM_CLASSES)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["size"]


def mb(nbytes):
    return round(nbytes / 2 ** 20, 2)


def state_bytes(sd, compress=False):
    buf = io.BytesIO(); torch.save(sd, buf); b = buf.getvalue()
    return len(gzip.compress(b, 6)) if compress else len(b)


def torch_eval(model, X, y, device, half=False):
    model = model.to(device); model = model.half() if half else model.float()
    P = []
    with torch.inference_mode():
        for i in range(0, len(X), 64):
            t = torch.from_numpy(X[i:i + 64]).to(device)
            P.append(softmax(model(t.half() if half else t).float().cpu().numpy()))
    P = np.concatenate(P)
    return topk_acc(P, y, 1), topk_acc(P, y, 5)


def onnx_eval(path, X, y):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    P = np.concatenate([softmax(s.run(None, {n: X[i:i + 16]})[0]) for i in range(0, len(X), 16)])
    return topk_acc(P, y, 1), topk_acc(P, y, 5)


def latency(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return 1000 * (time.perf_counter() - t) / n


class Calib(CalibrationDataReader):
    def __init__(self, X, name):
        self.it = iter([{name: X[i:i + 1]} for i in range(len(X))])

    def get_next(self):
        return next(self.it, None)


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, size = load_model()
    df = pd.read_csv(C.META_DIR / "annotations.csv")
    va = df[df.split == "val"]
    te = df[df.split == "test"].sample(1500, random_state=C.SEED)
    load = lambda d, kind: preprocess_batch([read_rgb(cache_path(u, kind)) for u in d.uid], size)
    Xv, yv = load(va, "crop"), va.label.values
    Xt, yt = load(te, "detcrop"), te.label.values
    calib = load(df[df.split == "train"].sample(200, random_state=C.SEED), "crop")
    x1 = torch.from_numpy(Xv[:1])
    R = {"eval_sets": {"val": int(len(yv)), "test_subset": int(len(yt))}, "variants": {}}

    def add(name, size_mb, acc_v, acc_t, lat_gpu=None, lat_cpu=None, note=""):
        R["variants"][name] = dict(size_mb=size_mb, val_top1=acc_v[0], val_top5=acc_v[1],
                                   test_top1=acc_t[0], test_top5=acc_t[1],
                                   latency_gpu_ms=lat_gpu, latency_cpu_ms=lat_cpu, note=note)
        print(name, R["variants"][name], flush=True)

    sd = model.state_dict()
    n_params = sum(p.numel() for p in model.parameters())
    R["classifier_params_M"] = round(n_params / 1e6, 2)

    # ---- FP32 baseline
    lat_cpu = latency(lambda: model.float().cpu()(x1), n=10)
    mg = model.cuda()
    lat_gpu = latency(lambda: (mg(x1.cuda()), torch.cuda.synchronize()))
    add("fp32 (PyTorch)", mb(state_bytes(sd)), torch_eval(model, Xv, yv, "cuda"), torch_eval(model, Xt, yt, "cuda"),
        lat_gpu, lat_cpu)

    # ---- FP16 reduced precision
    mh = model.cuda().half()
    lat_gpu16 = latency(lambda: (mh(x1.cuda().half()), torch.cuda.synchronize()))
    add("fp16 (PyTorch, GPU)", mb(state_bytes({k: v.half() for k, v in sd.items()})),
        torch_eval(model, Xv, yv, "cuda", half=True), torch_eval(model, Xt, yt, "cuda", half=True), lat_gpu16, None,
        "reduced precision; deployed weights file")
    model = model.float().cpu()

    # ---- ONNX FP32 + INT8 static quantisation (QDQ, per-channel)
    torch.onnx.export(model, x1, str(ONNX_FP32), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "b"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)
    s32 = ort.InferenceSession(str(ONNX_FP32), providers=["CPUExecutionProvider"])
    add("fp32 (ONNX Runtime, CPU)", mb(ONNX_FP32.stat().st_size), onnx_eval(ONNX_FP32, Xv, yv),
        onnx_eval(ONNX_FP32, Xt, yt), None, latency(lambda: s32.run(None, {"input": Xv[:1]}), n=20))
    quant_pre_process(str(ONNX_FP32), str(ONNX_PRE))
    quantize_static(str(ONNX_PRE), str(ONNX_INT8), Calib(calib, "input"), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    s8 = ort.InferenceSession(str(ONNX_INT8), providers=["CPUExecutionProvider"])
    add("int8 static quant (ONNX Runtime, CPU)", mb(ONNX_INT8.stat().st_size), onnx_eval(ONNX_INT8, Xv, yv),
        onnx_eval(ONNX_INT8, Xt, yt), None, latency(lambda: s8.run(None, {"input": Xv[:1]}), n=20),
        "200 training crops for calibration")

    # ---- Global L1 unstructured pruning
    import torch.nn.utils.prune as prune
    for amount in (0.3, 0.5):
        m, _ = load_model()
        params = [(mod, "weight") for mod in m.modules() if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear))]
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
        for mod, n in params:
            prune.remove(mod, n)
        psd = {k: v.half() for k, v in m.state_dict().items()}
        add(f"pruned {int(amount * 100)}% (global L1) + fp16", mb(state_bytes(psd, compress=True)),
            torch_eval(m, Xv, yv, "cuda"), torch_eval(m, Xt, yt, "cuda"), None, None,
            "size = gzip-compressed sparse weights; no fine-tuning after pruning")

    # ---- detector
    from ultralytics import YOLO
    yolo = YOLO(str(C.MODELS_DIR / "yolo_cars.pt"))
    img = read_rgb(df.img_path.iloc[0])[..., ::-1]
    R["detector"] = dict(weights="yolo_cars.pt (YOLO11s fine-tuned)",
                         size_mb=mb((C.MODELS_DIR / "yolo_cars.pt").stat().st_size),
                         params_M=round(sum(p.numel() for p in yolo.model.parameters()) / 1e6, 2),
                         latency_gpu_ms=latency(lambda: yolo.predict(img, verbose=False, device=0)),
                         latency_cpu_ms=latency(lambda: yolo.predict(img, verbose=False, device="cpu"), n=10))
    print(R["detector"])
    (C.RESULTS_DIR / "optimization.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
