"""Gradio web app - Intelligent Fine-Grained Vehicle Recognition (Problem Statement P5).

Run locally:   python app.py          -> http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import numpy as np
import pandas as pd

from src import config as C
from src.classical import visual_steps
from src.pipeline import InputError, VehicleRecognizer, draw_result, validate_image

BACKEND = os.environ.get("CVH_BACKEND", "torch")      # 'torch' or 'onnx' (INT8, CPU)
REC = VehicleRecognizer(backend=BACKEND)
MODES = ("hybrid", "deep", "classical") if REC.classical is not None else ("hybrid", "deep")
# All inference runs on ONE dedicated thread: the first CUDA call in every new thread costs ~1 s
# of per-thread setup, and Gradio would otherwise use a different worker thread per request.
EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _warmup():
    w = np.full((240, 320, 3), 128, np.uint8); w[60:180, 40:280] = 30
    for m in MODES:
        REC.run(w, m)


EXECUTOR.submit(_warmup).result()

STATUS_TEXT = {
    "confident": "Confident fine-grained prediction",
    "year-uncertain": "Make & model confident, manufacturing year uncertain",
    "model-uncertain": "Only the manufacturer is reliable",
    "low-confidence": "Low confidence - showing candidates only",
}


def _card(res: dict) -> str:
    return f"""
### {res['display']}
| | |
|---|---|
| **Make** | {res['make']} |
| **Model** | {res['model']} |
| **Year** | {res['year']} |
| **Confidence** | {res['confidence'] * 100:.1f}% |
| **Make-level confidence** | {res['best_make']} - {res['make_confidence'] * 100:.1f}% |
| **Status** | {STATUS_TEXT.get(res['status'], res['status'])} |
| **Box source** | {res['det_source']} (det. conf {res['det_conf']:.2f}) |
| **Enhancement applied** | {', '.join(res['enhancement_steps']) or 'none needed'} |
| **Latency** | detection {res['timings']['detection_ms']:.0f} ms - classification {res['timings']['classification_ms']:.0f} ms - total {res['timings']['total_ms']:.0f} ms |
"""


def analyse(image):
    if image is None:
        raise gr.Error("Please upload an image first.")
    try:
        rgb = validate_image(image)
        results = EXECUTOR.submit(lambda: {m: REC.run(rgb, m) for m in MODES}).result()
    except InputError as e:
        raise gr.Error(str(e))
    except Exception as e:                      # never show a stack trace to the user
        traceback.print_exc()
        raise gr.Error(f"Processing failed: {e}")
    h = results["hybrid"]
    vs = visual_steps(h["crop"])
    gallery = [
        (h["crop"], "1. Detected ROI (YOLO crop)"),
        (vs["enhanced"], "2. Adaptive enhancement: " + (", ".join(vs["steps"]) or "none")),
        (vs["edges"], "3. Canny edges (auto threshold)"),
        (vs["gradient"], "4. Sobel gradient magnitude"),
        (vs["lbp"], "5. LBP texture map"),
        (vs["contours"], "6. Contours + classical ROI (blue)"),
    ]
    comp = pd.DataFrame([
        dict(Approach=m, Prediction=r["label"], Confidence=f"{r['confidence'] * 100:.1f}%",
             **{"Latency (ms)": round(r["timings"]["total_ms"])},
             **{"Box source": r["det_source"]})
        for m, r in results.items()])
    s = h["stats"]
    quality = (f"brightness {s['brightness']:.0f}/255 - contrast {s['contrast']:.0f} - "
               f"noise {s['noise']:.1f} - sharpness {s['sharpness']:.0f} - min side {s['resolution']} px")
    top5 = {n: p for n, p in h["top5"]}
    return draw_result(h), _card(h), top5, gallery, comp, quality


def performance_md() -> str:
    p = C.RESULTS_DIR / "eval_results.json"
    if not p.exists():
        return "Evaluation results not found - run `python -m scripts.evaluate`."
    R = json.loads(p.read_text())
    lines = ["### Test-set comparison (8,041 unseen images)", "",
             "| Approach | Top-1 | Top-5 | Make acc. |", "|---|---|---|---|"]
    for k, v in R["comparison"].items():
        lines.append(f"| {k} | {v['top1'] * 100:.2f}% | {v['top5'] * 100:.2f}% | {v['make_acc'] * 100:.2f}% |")
    lines += ["", "### Ablation", "", "| Variant | Top-1 |", "|---|---|"]
    lines += [f"| {k} | {v['top1'] * 100:.2f}% |" for k, v in R["ablation"].items()]
    lines += ["", "### Detection", "", "| Detector | mean IoU | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|"]
    lines += [f"| {d['detector']} | {d['mean_iou']:.3f} | {d['mAP50']:.3f} | {d['mAP50_95']:.3f} |" for d in R["detection"]]
    return "\n".join(lines)


def figures():
    f = C.RESULTS_DIR / "figures"
    return [str(p) for p in sorted(f.glob("*.png"))] if f.exists() else []


EXAMPLES = sorted(str(p) for p in (C.PROJECT_ROOT / "examples").glob("*.jpg"))

GR6 = int(gr.__version__.split(".")[0]) >= 6
THEME = gr.themes.Soft()

with gr.Blocks(title="Fine-Grained Vehicle Recognition", **({} if GR6 else {"theme": THEME})) as demo:
    gr.Markdown("# Fine-Grained Vehicle Recognition\n"
                "Upload a car photo. The system **detects** the vehicle (YOLO11), analyses image quality and "
                "applies **adaptive classical enhancement** where needed, and identifies **make / model / year** "
                "among 196 Stanford Cars classes with a fine-tuned ConvNeXt. The table below compares it with a "
                "purely classical pipeline (edge/contour ROI + HOG/LBP/colour/geometry features).")
    with gr.Tab("Recognise"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="numpy", label="Vehicle image", height=360)
                btn = gr.Button("Analyse", variant="primary")
                if EXAMPLES:
                    gr.Examples(EXAMPLES, inputs=inp, label="Unseen test images")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Detection + prediction", height=360)
                card = gr.Markdown()
        with gr.Row():
            top5 = gr.Label(num_top_classes=5, label="Top-5 (hybrid)")
            comp = gr.Dataframe(label="Classical vs Deep vs Hybrid on this image", wrap=True)
        quality = gr.Textbox(label="Image-quality analysis", interactive=False)
        gallery = gr.Gallery(label="Intermediate image-processing results", columns=6, height=260)
        btn.click(analyse, inp, [out_img, card, top5, gallery, comp, quality], show_progress="full")
        inp.upload(analyse, inp, [out_img, card, top5, gallery, comp, quality], show_progress="full")
    with gr.Tab("Model performance"):
        gr.Markdown(performance_md())
        gr.Gallery(figures(), label="Evaluation figures", columns=2, height=600)
    gr.Markdown("<small>Trained only on Stanford Cars (196 classes, mostly US-market cars 1991-2012). "
                "Vehicles outside these classes will be mapped to the nearest known class - check the "
                "confidence and status. No number-plate recognition or tracking is performed.</small>")

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                                   **({"theme": THEME} if GR6 else {}))
