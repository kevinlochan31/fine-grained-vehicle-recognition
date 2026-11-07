"""Fine-grained classifier wrapper (PyTorch or ONNX Runtime backend)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config as C

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess_batch(crops: list[np.ndarray], size: int) -> np.ndarray:
    """RGB uint8 crops -> normalised NCHW float32 (same as eval transform)."""
    out = np.empty((len(crops), 3, size, size), np.float32)
    for i, c in enumerate(crops):
        x = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA if min(c.shape[:2]) > size else cv2.INTER_LINEAR)
        out[i] = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class FineGrainedClassifier:
    def __init__(self, weights: str | Path | None = None, device: str | None = None,
                 backend: str = "torch", onnx_path: str | Path | None = None):
        self.backend = backend
        if backend == "onnx":
            import onnxruntime as ort
            onnx_path = onnx_path or C.MODELS_DIR / "classifier_int8.onnx"
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            self.size = self.sess.get_inputs()[0].shape[-1]
            if not isinstance(self.size, int):
                self.size = C.CLS_IMG_SIZE
            return
        import timm
        import torch
        weights = weights or C.MODELS_DIR / "classifier_fp16.pt"
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.size = ck.get("size", C.CLS_IMG_SIZE)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = timm.create_model(ck["backbone"], pretrained=False, num_classes=C.NUM_CLASSES)
        m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
        self.model = m.eval().to(self.device)
        self.half = self.device == "cuda"
        if self.half:
            self.model = self.model.half()
        self.torch = torch

    def logits(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            return self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0]
        t = self.torch.from_numpy(x).to(self.device)
        if self.half:
            t = t.half()
        with self.torch.inference_mode():
            return self.model(t).float().cpu().numpy()

    def predict_proba(self, crops: list[np.ndarray], tta: bool = False, batch: int = 32) -> np.ndarray:
        probs = []
        for i in range(0, len(crops), batch):
            x = preprocess_batch(crops[i:i + batch], self.size)
            z = self.logits(x)
            if tta:                                   # horizontal-flip test-time augmentation
                z = (z + self.logits(np.ascontiguousarray(x[..., ::-1]))) / 2
            probs.append(softmax(z))
        return np.concatenate(probs)
