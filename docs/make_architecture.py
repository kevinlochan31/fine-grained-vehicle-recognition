"""Render docs/architecture.png (system architecture diagram) with matplotlib.

Every box names the module that implements it, so the diagram can be checked against the code.
Usage:  python docs/make_architecture.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).with_name("architecture.png")
IO, CL, DL, HY = "#e8eef9", "#fdf1dc", "#dff3e4", "#f3e3f7"
X = [0.2, 4.1, 8.1, 12.2, 16.3]            # column x positions
W = [3.3, 3.4, 3.5, 3.5, 3.5]              # column widths
H = 1.05

BOX = {  # key: (col, y, colour, title, detail, module)
    "ui":    (0, 8.6, IO, "Web UI (browser)", "upload / paste / examples", "web/ + Motion"),
    "api":   (0, 7.2, IO, "FastAPI  POST /api/analyze", "15 MB limit, decode check", "app.py"),
    "val":   (0, 5.8, IO, "Input validation", "grey/RGBA, tiny, blank, huge", "pipeline.validate_image"),
    "qa":    (0, 4.4, CL, "Image-quality analysis", "brightness, contrast, noise σ, sharpness", "classical.image_stats"),
    "yolo":  (1, 8.6, DL, "YOLO11s detector", "fine-tuned, 1 class 'car'", "detector.py"),
    "retry": (1, 7.2, CL, "Enhance whole image", "+ retry YOLO (only if no box)", "classical.adaptive_preprocess"),
    "gate":  (1, 5.8, HY, "CLIP vehicle gate", "p >= 0.9, or p >= 0.5 + YOLO agrees", "openvocab.CarVision"),
    "nocar": (1, 4.4, IO, "\"No car detected\"", "returned for any non-car input", "pipeline._no_car"),
    "views": (2, 7.2, CL, "Two views of the crop", "original + enhanced (if needed)", "steps picked from quality stats"),
    "crop":  (2, 5.8, IO, "ROI crop", "YOLO box +8% (or full frame)", "data.crop_image"),
    "feat":  (2, 4.4, CL, "Classical features", "HOG, LBP, HSV hist, edges, geometry", "classical.extract_features"),
    "cnn":   (3, 7.2, DL, "Ensemble classifier", "0.3 ConvNeXt-T + 0.7 CLIP ViT-B/16 FT", "classifier.py + clip_classifier.py"),
    "temp":  (3, 5.8, HY, "Temperature scaling", "T = 0.65 / 0.5, flip TTA, view mean", "pipeline._calibrate"),
    "lr":    (3, 4.4, CL, "PCA + Logistic Regression", "196 classes", "models/classical_logreg"),
    "fuse":  (4, 7.2, HY, "Late fusion", "(1-w) p_ens + w p_classical", "w = 0 chosen on val"),
    "conf":  (4, 5.8, HY, "Low-confidence back-off", "t = 0.67: class -> make+model -> make", "pipeline.describe"),
    "open":  (4, 4.4, HY, "Open-set check (CLIP zero-shot)", "make first: name / make only / unknown", "pipeline._open_set"),
    "out":   (4, 3.0, IO, "JSON response -> UI", "box, make, model, year, conf, top-5, steps", "app.py"),
}

fig, ax = plt.subplots(figsize=(20.5, 10))
ax.set_xlim(0, 20.0); ax.set_ylim(1.5, 10.4); ax.axis("off")
geom = {}
for k, (c, y, col, t, d, m) in BOX.items():
    x, w = X[c], W[c]
    ax.add_patch(FancyBboxPatch((x, y), w, H, boxstyle="round,pad=0.05", fc=col, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + H - 0.27, t, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x + w / 2, y + H / 2 - 0.05, d, ha="center", va="center", fontsize=8.6)
    ax.text(x + w / 2, y + 0.17, m, ha="center", va="center", fontsize=7.6, family="monospace", color="#555")
    geom[k] = (x, y, w)

L = lambda k: (geom[k][0], geom[k][1] + H / 2)
R = lambda k: (geom[k][0] + geom[k][2], geom[k][1] + H / 2)
T = lambda k: (geom[k][0] + geom[k][2] / 2, geom[k][1] + H)
B = lambda k: (geom[k][0] + geom[k][2] / 2, geom[k][1])


def arrow(p, q, label=None, colour="#333", dashed=False, dx=0.08, dy=0.08, rot=0):
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="->", lw=1.4, color=colour,
                                          linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2))
    if label:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label, fontsize=8.2, color=colour,
                rotation=rot, va="center")


arrow(B("ui"), T("api")); arrow(B("api"), T("val")); arrow(B("val"), T("qa"))
arrow(R("qa"), L("yolo"))
arrow(B("yolo"), T("retry"), "no box", "#a33", dx=0.12, dy=0)
arrow(B("retry"), T("gate"))
# "box found" path runs down the gutter right of column 1 into the gate
gx = X[1] + W[1] + 0.28
ax.plot([R("yolo")[0], gx, gx], [R("yolo")[1], R("yolo")[1], 6.6], color="#2a7", lw=1.4)
arrow((gx, 6.6), (X[1] + W[1], 6.6), None, "#2a7")
ax.text(gx + 0.06, 8.6, "box\nfound", fontsize=8, color="#2a7", va="center")
arrow((X[1] + W[1], 6.05), (X[2], 6.05), "car", "#2a7", dx=-0.1, dy=-0.2)
arrow(B("gate"), T("nocar"), "not a car", "#a33", dx=0.12, dy=0)
arrow(T("crop"), B("views")); arrow(B("crop"), T("feat"))
arrow(R("views"), L("cnn")); arrow(B("cnn"), T("temp")); arrow(R("feat"), L("lr"))
arrow(R("temp"), (X[4], 7.2 + 0.3))
arrow(R("lr"), (X[4], 7.2 + 0.1), None, "#777", dashed=True)
ax.text(15.6, 5.62, "weight w = 0 (no gain on val)", fontsize=8, color="#666", ha="right", va="center")
arrow(B("fuse"), T("conf")); arrow(B("conf"), T("open")); arrow(B("open"), T("out"))

ax.text(0.2, 10.05, "P5 - Fine-Grained Vehicle Recognition: system architecture (hybrid request path)",
        fontsize=15, weight="bold")
ax.text(8.1, 3.55, "For the on-screen comparison the same request also runs:\n"
        "  Deep only       YOLO crop -> ConvNeXt, single pass\n"
        "  Classical only  edge/contour ROI -> features -> PCA+LogReg\n"
        "(both behind the same vehicle gate)",
        fontsize=8.8, va="top", family="monospace", bbox=dict(boxstyle="round", fc="#ffffff", ec="#999"))
ax.text(0.2, 3.85, "Offline training (scripts/, python -m scripts.run_all):\n"
        "prepare_data -> train_classifier (ConvNeXt-T)\n-> train_clip (CLIP ViT-B/16) -> train_detector\n"
        "-> run_detection -> train_classical -> evaluate\n-> eval_openset -> optimize -> write_report",
        fontsize=8.8, va="top", family="monospace", bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#999"))
for x, label, c in ((0.2, "Classical CV", CL), (2.6, "Deep learning", DL),
                    (5.2, "Hybrid integration", HY), (8.2, "I/O & serving", IO)):
    ax.add_patch(FancyBboxPatch((x, 1.75), 0.5, 0.35, boxstyle="round,pad=0.02", fc=c, ec="#333"))
    ax.text(x + 0.65, 1.92, label, va="center", fontsize=9.5)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("saved", OUT)
