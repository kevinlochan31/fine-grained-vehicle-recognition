"""Web app - Intelligent Fine-Grained Vehicle Recognition (Problem Statement P5).

FastAPI backend + custom animated frontend (web/, Motion for animations).
Run locally:   python app.py          -> http://127.0.0.1:7860
(The previous Gradio UI is kept as app_gradio.py.)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import config as C
from src.classical import visual_steps
from src.pipeline import InputError, VehicleRecognizer, validate_image

WEB = C.PROJECT_ROOT / "web"
MAX_BYTES = 15 * 2 ** 20
BACKEND = os.environ.get("CVH_BACKEND", "torch")      # 'torch' or 'onnx' (INT8, CPU)

REC = VehicleRecognizer(backend=BACKEND)
MODES = ("hybrid", "deep", "classical") if REC.classical is not None else ("hybrid", "deep")
# All inference on ONE dedicated thread: the first CUDA call in each new thread costs ~1 s.
EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _warmup():
    w = np.full((240, 320, 3), 128, np.uint8); w[60:180, 40:280] = 30
    REC.analyse(w)
    ex = sorted((C.PROJECT_ROOT / "examples").glob("*.jpg"))
    if ex:                                     # a real car warms the classifier branches too
        REC.analyse(cv2.cvtColor(cv2.imread(str(ex[0])), cv2.COLOR_BGR2RGB))


EXECUTOR.submit(_warmup).result()


def _b64(img: np.ndarray, max_side: int = 480, gray: bool = False) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if not gray and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _pred(r: dict) -> dict:
    keys = ("label", "make", "model", "year", "confidence", "status", "display", "best_make",
            "make_confidence", "best_makemodel", "makemodel_confidence", "det_source", "det_conf",
            "enhancement_steps", "timings")
    out = {k: r[k] for k in keys}
    out["box"] = [int(v) for v in r["box"]] if r["box"] is not None else None
    out["top5"] = [{"label": n, "p": float(p)} for n, p in r["top5"]]
    out["in_training_set"] = bool(r.get("in_training_set", True))
    out["car_probability"] = float(r.get("car_probability", 1.0))
    out["closest_trained_class"] = r.get("closest_trained_class")
    out["outside_mass"] = r.get("outside_mass")
    out["possibly_outside"] = bool(r.get("possibly_outside", False))
    return out


app = FastAPI(title="P5 Vehicle Recognition", docs_url="/api/docs", redoc_url=None)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(422, "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 15 MB. Please upload a smaller photo.")
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(422, "This file is not a readable image. Please upload a JPG, PNG or WebP photo.")
    try:
        payload = await asyncio.wrap_future(EXECUTOR.submit(_process, arr))
    except InputError as e:
        raise HTTPException(422, str(e))
    except Exception as e:                                   # never leak a stack trace
        traceback.print_exc()
        raise HTTPException(500, f"Processing failed: {type(e).__name__}")
    return JSONResponse(payload)


def _process(arr: np.ndarray) -> dict:
    rgb = validate_image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    results = REC.analyse(rgb)
    h = results["hybrid"]
    if h["status"] == "no-car":
        return {"width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "image": _b64(rgb, 1280),
                "quality": {k: round(float(v), 1) for k, v in h["stats"].items()},
                "results": {m: _pred(r) for m, r in results.items()}, "steps": [],
                "thresholds": {"confidence": REC.cfg["conf_threshold"], "gate": REC.cfg.get("gate_threshold", 0.5)},
                "no_car": True}
    vs = visual_steps(h["crop"])
    steps = [
        ("Detected region", "YOLO11 crop with 8% context", h["crop"], False),
        ("Adaptive enhancement", ", ".join(vs["steps"]) or "not needed for this image", vs["enhanced"], False),
        ("Canny edges", "auto threshold from median intensity", vs["edges"], True),
        ("Gradient magnitude", "Sobel x / y", vs["gradient"], True),
        ("LBP texture", "uniform local binary patterns", vs["lbp"], True),
        ("Contours + classical ROI", "morphology, largest contour (red box)", vs["contours"], False),
    ]
    return {
        "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
        "image": _b64(rgb, 1280),
        "quality": {k: round(float(v), 1) for k, v in h["stats"].items()},
        "results": {m: _pred(r) for m, r in results.items()},
        "steps": [{"title": t, "caption": c, "img": _b64(i, 420, g)} for t, c, i, g in steps],
        "thresholds": {"confidence": REC.cfg["conf_threshold"], "gate": REC.cfg.get("gate_threshold", 0.5)},
        "no_car": False,
    }


@app.get("/api/performance")
def performance():
    p = C.RESULTS_DIR / "eval_results.json"
    if not p.exists():
        raise HTTPException(404, "Evaluation results not found - run python -m scripts.evaluate")
    R = json.loads(p.read_text())
    opt = C.RESULTS_DIR / "optimization.json"
    return {
        "comparison": R["comparison"], "ablation": R["ablation"], "detection": R["detection"],
        "class_aware_mAP50": R["class_aware_mAP50"], "robustness": R.get("robustness", []),
        "threshold_analysis": R["threshold_analysis"], "low_confidence": R["low_confidence_handling"],
        "calibration": R.get("calibration"), "errors": R["error_breakdown"],
        "optimization": json.loads(opt.read_text()) if opt.exists() else None,
        "hybrid_config": json.loads((C.MODELS_DIR / "hybrid_config.json").read_text()),
        "openset": json.loads((C.RESULTS_DIR / "openset.json").read_text()) if (C.RESULTS_DIR / "openset.json").exists() else None,
        "unknown": {k: v for k, v in json.loads((C.RESULTS_DIR / "unknown.json").read_text()).items() if k != "grid"}
        if (C.RESULTS_DIR / "unknown.json").exists() else None,
    }


@app.get("/api/examples")
def examples():
    return [f"/examples/{p.name}" for p in sorted((C.PROJECT_ROOT / "examples").glob("*.jpg"))]


@app.get("/api/classes")
def classes():
    return REC.names


@app.get("/api/health")
def health():
    return {"status": "ok", "modes": MODES, "backend": BACKEND}


app.mount("/examples", StaticFiles(directory=C.PROJECT_ROOT / "examples"), name="examples")
app.mount("/docs-assets", StaticFiles(directory=C.PROJECT_ROOT / "docs"), name="docs")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), log_level="info")
