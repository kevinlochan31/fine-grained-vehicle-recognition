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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import config as C
from src.classical import visual_steps
from src.explain import explain as explain_result
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
        r = REC.analyse(cv2.cvtColor(cv2.imread(str(ex[0])), cv2.COLOR_BGR2RGB))["hybrid"]
        if r["status"] != "no-car":
            try:
                explain_result(REC, r)         # first Grad-CAM backward pass is slow; do it at start-up
            except Exception:
                traceback.print_exc()


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
    w = r.get("world")                       # world-car layer answer (extra layer, outside the 196 classes)
    wk = r.get("world_make")
    out["world_make"] = None if wk is None else dict(make=wk["make"], p=float(wk["p"]), candidates=list(wk["candidates"]))
    out["not_sure"] = r.get("not_sure")
    out["best_guess"] = r.get("best_guess")
    out["world"] = None if w is None else {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in w.items()}
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


def _explain(h: dict) -> dict:
    """'Show the working': difference image, HOG picture, Grad-CAM and the step-by-step numbers."""
    E = explain_result(REC, h)
    d = E["diff"]
    return {
        "before": _b64(d["before"], 640), "after": _b64(d["after"], 640), "diff": _b64(d["diff"], 640),
        "enh_steps": d["steps"], "mean_change": round(d["mean_change"], 2),
        "max_change": round(d["max_change"], 1), "changed_pct": round(d["changed_pct"], 1),
        "hog": _b64(E["hog"]["image"], 480), "hog_gray": _b64(E["hog"]["gray"], 480, gray=True),
        "hog_cells": E["hog"]["cells"],
        "cam": _b64(E["cam"], 640) if E["cam"] is not None else None, "cam_label": E["cam_label"],
        "calc": E["calc"],
    }


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
    learnt = _self_learn(rgb, h)
    vs = visual_steps(h["crop"])
    steps = [
        ("Detected region", "YOLO11 crop with 8% context", h["crop"], False),
        ("Adaptive enhancement", ", ".join(vs["steps"]) or "not needed for this image", vs["enhanced"], False),
        ("Canny edges", "auto threshold from median intensity", vs["edges"], True),
        ("Gradient magnitude", "Sobel x / y", vs["gradient"], True),
        ("LBP texture", "uniform local binary patterns", vs["lbp"], True),
        ("Contours + classical ROI", "morphology, largest contour (red box)", vs["contours"], False),
    ]
    try:
        explain = _explain(h)
    except Exception:                                        # the explanation must never break a scan
        traceback.print_exc()
        explain = None
    return {
        "explain": explain,
        "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
        "image": _b64(rgb, 1280),
        "quality": {k: round(float(v), 1) for k, v in h["stats"].items()},
        "results": {m: _pred(r) for m, r in results.items()},
        "steps": [{"title": t, "caption": c, "img": _b64(i, 420, g)} for t, c, i, g in steps],
        "thresholds": {"confidence": REC.cfg["conf_threshold"], "gate": REC.cfg.get("gate_threshold", 0.5)},
        "no_car": False,
        "self_learnt": learnt,
    }


def _self_learn(rgb: np.ndarray, h: dict) -> bool:
    """Self-learning: when the world-car layer names a generation from reference photos with very high
    confidence, keep this photo as one more reference of that car (thumbnail + embedding outside the repo),
    so it adapts to the kind of photos people actually upload. Set CVH_SELF_LEARN=0 to switch it off."""
    W, w = REC.world, h.get("world")
    if (os.environ.get("CVH_SELF_LEARN", "1") == "0" or W is None or not W.can_teach or h["status"] != "world"
            or not w or w.get("level") != "generation" or w.get("source") != "reference photos"
            or float(w.get("p", 0)) < REC.cfg.get("world_self_learn_conf", 0.97) or h.get("crop") is None):
        return False
    import uuid
    from src.worldcars import USER_DIR
    feat = W.embed(rgb, h["crop"] if h.get("det_source", "").startswith("yolo") else None)
    name = f"auto_{uuid.uuid4().hex[:12]}.jpg"
    ok = W.auto_learn(w["make"], w["model"], w.get("gen"), w.get("years"), w.get("y0"), None, feat, name)
    if ok:
        (USER_DIR / "images").mkdir(parents=True, exist_ok=True)
        hh, ww = rgb.shape[:2]; s = min(1.0, 320 / max(hh, ww))
        th = cv2.resize(rgb, (int(ww * s), int(hh * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(USER_DIR / "images" / name), cv2.cvtColor(th, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return ok


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


# ---------------------------------------------------------------------------- teach AutoLens your own car photos
_CATALOGUE = None


@app.get("/api/catalogue")
def catalogue():
    """World catalogue for the teach form: {make: {model: [[gen, years, y0, y1], ...]}}."""
    global _CATALOGUE
    if _CATALOGUE is None:
        from src.worldcars import load_world_catalogue
        out = {}
        for e in load_world_catalogue():
            out.setdefault(e["make"], {}).setdefault(e["model"], []).append([e["gen"], e["years_label"], e["years"][0], e["years"][1]])
        _CATALOGUE = out
    return {"can_teach": bool(REC.world is not None and REC.world.can_teach), "makes": _CATALOGUE}


@app.post("/api/teach")
async def teach(make: str = Form(...), model: str = Form(...), y0: str = Form(""), files: list[UploadFile] = File(...)):
    W = REC.world
    if W is None or not W.can_teach:
        raise HTTPException(503, "Teaching is not available (world-car layer or its training set is missing).")
    make, model = make.strip(), model.strip()
    if not make or not model:
        raise HTTPException(422, "Please give the make and the model.")
    if not 1 <= len(files) <= 20:
        raise HTTPException(422, "Add between 1 and 20 photos.")
    datas = [await f.read() for f in files]
    if any(len(d) > MAX_BYTES or not d for d in datas):
        raise HTTPException(413, "Each photo must be a non-empty image up to 15 MB.")
    try:
        return await asyncio.wrap_future(EXECUTOR.submit(_teach, make, model, y0, datas))
    except InputError as e:
        raise HTTPException(422, str(e))


def _teach(make, model, y0, datas):
    import uuid
    from src.data import crop_image
    from src.worldcars import USER_DIR, load_world_catalogue
    y0 = int(y0) if str(y0).strip().isdigit() else None
    e = next((e for e in load_world_catalogue() if e["make"] == make and e["model"] == model and e["years"][0] == y0), None)
    gen, years, y1 = (e["gen"], e["years_label"], e["years"][1]) if e else (None, "", None)
    (USER_DIR / "images").mkdir(parents=True, exist_ok=True)
    feats, names, skipped = [], [], 0
    for d in datas:
        arr = cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            skipped += 1; continue
        rgb = validate_image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        det, _ = REC.detector.primary(rgb)
        if not det["source"].startswith("yolo"):          # only photos where the car detector finds a car
            skipped += 1; continue
        feats.append(REC.world.embed(rgb, crop_image(rgb, det["box"])))
        name = f"{uuid.uuid4().hex[:12]}.jpg"
        h, w = rgb.shape[:2]; s = min(1.0, 320 / max(h, w))
        th = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(USER_DIR / "images" / name), cv2.cvtColor(th, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        names.append(name)
    if not feats:
        raise InputError("No car was found in these photos. Use clear photos where the whole car is visible.")
    out = REC.world.teach(make, model, gen, years, y0, y1, feats, names)
    out["skipped"] = skipped
    return out


@app.get("/api/taught")
def taught():
    W = REC.world
    if W is None:
        return []
    return [dict(index=i, make=e["make"], model=e["model"], gen=e["gen"], years=e["years"], photos=len(e["images"]))
            for i, e in enumerate(W.taught)]


@app.delete("/api/taught/{index}")
async def forget(index: int):
    W = REC.world
    if W is None or not 0 <= index < len(W.taught):
        raise HTTPException(404, "No such taught car.")
    await asyncio.wrap_future(EXECUTOR.submit(W.forget, index))
    return taught()


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
if (WEB / "world_refs").exists():          # reference photos of the world-car layer (Wikimedia Commons, credited)
    app.mount("/world_refs", StaticFiles(directory=WEB / "world_refs"), name="world_refs")
from src.worldcars import USER_DIR as _USER_DIR     # photos you taught AutoLens (kept outside the repository)
(_USER_DIR / "images").mkdir(parents=True, exist_ok=True)
app.mount("/user_refs", StaticFiles(directory=_USER_DIR / "images"), name="user_refs")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def no_stale_ui(request, call_next):
    # the page, script and stylesheet change between versions: make the browser revalidate them every time
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), log_level="info")
