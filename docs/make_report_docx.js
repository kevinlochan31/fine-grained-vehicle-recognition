const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  HeadingLevel, ImageRun, PageBreak, LevelFormat, BorderStyle, TableOfContents, Footer, Header, PageNumber,
  TabStopType,
} = require("docx");

const ROOT = "C:\\Users\\Kevin\\OneDrive\\Desktop\\cv hackathon";
const OUT = path.join(ROOT, "docs", "P5_Technical_Report.docx");
const FONT = "Calibri";
const W = 9026; // A4 text width in DXA with 1" margins (11906 - 2*1440)
const ACCENT = "1F4E79", HEAD_FILL = "D9E2F3", ALT_FILL = "F2F6FC", HL_FILL = "E2EFDA";

// ------------------------------------------------------------------ helpers
const t = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 22, bold: o.bold, italics: o.italics, color: o.color });
function rich(s, o = {}) { // **bold** markup -> runs
  return s.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((p) => p.startsWith("**") ? t(p.slice(2, -2), { ...o, bold: true }) : t(p, o));
}
const P = (s, o = {}) => new Paragraph({ children: rich(s, o), spacing: { after: o.after ?? 120, line: 276 }, alignment: o.align });
const H1 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: s })] });
const H2 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: s })] });
const H3 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: s })] });
const B = (s, lvl = 0) => new Paragraph({ numbering: { reference: "bullets", level: lvl }, children: rich(s), spacing: { after: 60, line: 264 } });
const N = (s, ref = "numbers") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: rich(s), spacing: { after: 60, line: 264 } });
const code = (lines) => lines.map((l, i) => new Paragraph({
  children: [new TextRun({ text: l, font: "Consolas", size: 19, color: l.startsWith("#") ? "6A737D" : "1F2328" })],
  shading: { type: ShadingType.CLEAR, fill: "F3F4F6" }, spacing: { after: 0, before: i === 0 ? 60 : 0 },
  indent: { left: 200, right: 200 },
}));
const caption = (s) => new Paragraph({ children: [t(s, { italics: true, size: 19, color: "555555" })], alignment: AlignmentType.CENTER, spacing: { after: 200 } });
const brk = () => new Paragraph({ children: [new PageBreak()] });

function img(file, widthPx, ratio) { // ratio = h / w
  const data = fs.readFileSync(path.join(ROOT, file));
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: file.toLowerCase().endsWith(".png") ? "png" : "jpg", data,
      transformation: { width: widthPx, height: Math.round(widthPx * ratio) },
      altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } })],
  });
}

// table(header, rows, colFractions, {highlight: rowIndex[]})
function table(header, rows, fr, o = {}) {
  const cols = fr.map((f) => Math.round(f * W));
  cols[cols.length - 1] = W - cols.slice(0, -1).reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "B4C6E7" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, r, isHead) => new TableCell({
    width: { size: cols[i], type: WidthType.DXA }, borders,
    shading: { type: ShadingType.CLEAR, fill: isHead ? HEAD_FILL : (o.highlight || []).includes(r) ? HL_FILL : r % 2 ? ALT_FILL : "FFFFFF" },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: i > 0 && !o.leftAll ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: rich(String(text), { size: 19, bold: isHead || (o.highlight || []).includes(r) ? true : undefined }),
    })],
  });
  return [new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: cols,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, -1, true)) }),
      ...rows.map((row, r) => new TableRow({ children: row.map((c, i) => cell(c, i, r, false)) }))],
  }), new Paragraph({ children: [], spacing: { after: 120 } })];
}

// ------------------------------------------------------------------ content
const cover = [
  new Paragraph({ children: [], spacing: { before: 1800 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [t("Computer Vision Hackathon 2026", { size: 26, color: "555555" })], spacing: { after: 120 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [t("Hackathon Technical Solution Report", { size: 30, color: ACCENT, bold: true })], spacing: { after: 480 } }),
  new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Intelligent Fine-Grained Vehicle Recognition and Analysis System" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [t("AutoLens: detect the car, name its make, model and year, and know when it does not know", { size: 24, italics: true, color: "444444" })], spacing: { before: 120, after: 720 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [t("Problem Statement 5 (P5)  ·  Stanford Cars dataset", { size: 24 })], spacing: { after: 120 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [t("Kevin Isaac · Kokila C G · Dhruthishree V", { size: 24 })], spacing: { after: 1200 } }),
  ...table(["Headline result (official test split, 8,041 images)", "Value"], [
    ["Fine-grained top-1 / top-5 accuracy (hybrid, 196 classes)", "95.90% / 99.75%"],
    ["Detection mAP@0.5 / mAP@0.5:0.95 / mean IoU", "0.989 / 0.931 / 0.954"],
    ["Robustness gain over deep-only, 8× low resolution", "75.8% → 82.7%"],
    ["Non-car photos rejected / real cars accepted", "16 of 16 / 99.89%"],
    ["Deployed classifier size (FP16) / INT8 size", "53.4 MB / 27.3 MB"],
  ], [0.68, 0.32]),
  brk(),
];

const info = [
  H1("Submission Details"),
  ...table(["Field", "Details"], [
    ["**Problem ID**", "P5"],
    ["**Problem Title**", "Intelligent Fine-Grained Vehicle Recognition and Analysis System"],
    ["**Team Name**", "Not registered (Team ID: T80)"],
    ["**Team Members**", "Kevin Isaac (1RUA24CSE0200), Kokila C G (1RUA24CSE0206), Dhruthishree V (1RUA24CSE0130)"],
    ["**GitHub Repository**", "https://github.com/kevinlochan31/fine-grained-vehicle-recognition (commit evaluated: 7034f4d)"],
    ["**Live Demo / Deployment**", "https://kevinisaacbtech24--autolens.us-east.modal.run — web app \"AutoLens\" on Modal (runs locally with python app.py)"],
    ["**Evaluation file**", "eval_P5_<TeamID>.md (first line: comma-separated USNs)"],
  ], [0.3, 0.7], { leftAll: true }),
  new Paragraph({ children: [t("Contents", { size: 26, bold: true, color: "2E74B5" })], spacing: { before: 240, after: 120 }, keepNext: true }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
];

const s1 = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun({ text: "1. Problem Understanding & Solution Strategy" })] }),
  H2("1.1 Core Technical Challenge"),
  P("The system must recognise a car's **make, model and manufacturing year** among the 196 classes of the Stanford Cars dataset and localise it with a bounding box, showing a confidence score and the intermediate image-processing steps."),
  P("**The primary technical barrier is fine-grained similarity.** Many classes are the same body in different trims or years, and each class has only 24–68 training images (median 42). Examples:"),
  B("Audi S5 Coupe vs Audi A5 Coupe (2012)"),
  B("Chevrolet Express Van vs Express Cargo Van (2007)"),
  B("Dodge Caliber Wagon 2007 vs 2012"),
  P("**Failure modes targeted:**", { after: 60 }),
  B("darkness, over-exposure and sensor noise;"),
  B("motion blur and low resolution (small or distant cars);"),
  B("rotation and different viewing angles;"),
  B("partial occlusion;"),
  B("**bad input**: photos without a car, and cars outside the 196 training classes (the dataset ends at 2012 and is mostly US-market)."),
  P("**Constraints:**", { after: 60 }),
  B("the public URL must respond within about 10 s on phone and laptop;"),
  B("the system runs on an RTX 5060 Laptop GPU (8 GB) or a CPU-only free host;"),
  B("model files must stay small enough to deploy."),
  P("**Dataset.** Stanford Cars: 16,185 images, 196 classes, 49 makes, model years 1991–2012."),
  ...table(["Split", "Images", "Use"], [
    ["Train", "7,356", "Training"],
    ["Validation", "788", "Stratified 10% of the official training set; model selection and all threshold tuning"],
    ["Test", "8,041", "Official test split; final numbers only (never used for training or tuning)"],
  ], [0.18, 0.14, 0.68]),
  H2("1.2 Proposed Methodology"),
  P("**Pipeline.** A fine-tuned **YOLO11s** detector localises the car, and a **CLIP vehicle gate** rejects images without a car. The detected region is classified by a calibrated **ensemble of ConvNeXt-Tiny and a robustly fine-tuned CLIP ViT-B/16**."),
  P("**Adaptive mechanism.**"),
  B("Classical enhancement (gamma, CLAHE, bilateral denoising, upscaling) is triggered only when measured image statistics call for it. It is added as a **second view** next to the original rather than replacing it."),
  B("Confidences are temperature-calibrated, and the answer **backs off hierarchically**: full class → make + model → make → unknown."),
  B("Cars outside the 196 classes are detected and answered **make first**: a model is named only when the evidence is strong."),
  P("**Why not static baselines.** A fixed enhancement hurt clean images in our tests. A closed-set classifier always forces one of 196 labels, even onto flowers or a 2024 car."),
];

const s2 = [
  H1("2. Technical Architecture & Implementation"),
  P("Raw code is not reproduced here; each component names its repository module. The architecture diagram is generated from docs/make_architecture.py, and the full design rationale is in docs/design_document.md."),
  img("docs/architecture.png", 620, 1027 / 2091),
  caption("Figure 1. System architecture (hybrid request path). Colours: classical CV, deep learning, hybrid integration, I/O."),
  H2("2.1 Classical Algorithmic Baseline"),
  P("**Algorithms utilised** (src/classical.py):", { after: 60 }),
  ...table(["Requirement (brief)", "Implementation"], [
    ["Pre-processing & normalisation", "Mean/std normalisation; histogram equalisation; adaptive gamma, CLAHE on the L channel, bilateral edge-preserving denoising, bicubic upscaling"],
    ["Edge detection & contour analysis", "Automatic Canny (median-based thresholds), morphological closing, external contours, Hu moments"],
    ["Colour & texture features", "HSV 8×4×4 joint histogram + colour moments; uniform LBP (P = 8, R = 1) on a 2×3 grid; Sobel edge-orientation histogram"],
    ["Shape", "HOG (9 orientations, 8×8 cells, 2×2 blocks) on a 192×128 canonical crop"],
    ["Region-of-interest extraction", "Edge/contour ROI (classical detector and fallback) + ROI crop with 8% context"],
    ["Geometric features", "Aspect ratio, edge density, contour extent, solidity, circularity, log-Hu moments"],
    ["Classifier", "PCA + multinomial logistic regression (models/classical_logreg.joblib)"],
  ], [0.3, 0.7], { leftAll: true }),
  P("**Adaptive selection logic.** Each image is first measured: mean brightness, RMS contrast, Immerkaer noise σ, Laplacian sharpness and minimum side length. Parameters then follow those numbers:"),
  B("gamma correction only if mean brightness < 70;"),
  B("CLAHE clip limit increases as contrast drops;"),
  B("bilateral filtering only if the estimated noise σ > 12;"),
  B("upscaling only for small crops;"),
  B("Canny thresholds derived from the median intensity."),
  P("On normal photos the enhancement is usually a no-op, so clean inputs are not harmed."),
  H2("2.2 Deep Learning Pipeline"),
  ...table(["Component", "Model & rationale", "Training"], [
    ["Detector (src/detector.py)", "YOLO11s, COCO-pretrained, fine-tuned as one class \"car\". Real-time; fine-tuning teaches the dataset's \"main car, tight box\" convention", "15 epochs, 640 px, cosine LR, mosaic off in the last 3 epochs"],
    ["Classifier 1 (src/classifier.py)", "ConvNeXt-Tiny, ImageNet-22k pretrained. High fine-grained accuracy at 28 M parameters, small enough for CPU hosting", "384 px, 30 epochs, AdamW (backbone LR 2e-4, head 10×), weight decay 0.05, drop-path 0.2"],
    ["Classifier 2 (src/clip_classifier.py)", "CLIP ViT-B/16, LAION-2B pretrained. Web-scale pre-training generalises to real-world photos; head initialised from text embeddings of the class names", "15 epochs, layer-wise LR decay 0.75, harsh augmentation"],
    ["Vehicle gate + open-set naming (src/openvocab.py)", "Zero-shot CLIP ViT-B/16 with 21 car prompts, about 45 non-car prompts and a 954-model catalogue (83 makes)", "No training"],
  ], [0.22, 0.45, 0.33], { leftAll: true }),
  P("**Loss functions & training.**"),
  B("Cross-entropy with **label smoothing 0.1**."),
  B("AdamW, 1-epoch warm-up + cosine schedule, bf16 mixed precision."),
  B("The best checkpoint is chosen on the validation split."),
  P("**Augmentation.**"),
  B("**ConvNeXt:** random resized crop (scale 0.6–1), flip, colour jitter, greyscale, blur, rotation ±12°, random erasing (occlusion), down-scaling to 20–50%."),
  B("**CLIP (harsher):** random gamma darkening and over-exposure, down-scaling to 12–45%, sensor noise, JPEG quality 15–90, perspective warp, rotation, strong colour jitter, blur, random erasing."),
  P("**How overfitting was controlled.**"),
  B("transfer learning from large pre-trained models;"),
  B("layer-wise learning-rate decay (CLIP);"),
  B("label smoothing, weight decay and stochastic depth;"),
  B("heavy augmentation;"),
  B("model selection on a held-out validation split;"),
  B("an ensemble of a CNN and a transformer, which make different mistakes."),
  P("Best validation top-1: ConvNeXt 95.7%, CLIP fine-tune 96.7%. Zero-shot CLIP with the text-initialised head already gave 87.2% before training."),
  H2("2.3 Hybrid Integration & Adaptive Mechanism"),
  P("**Fusion strategy** (src/pipeline.py):", { after: 60 }),
  N("**Detection-driven ROI.** The classifiers see the detected region (+8% context), not the whole image. If YOLO finds no box, the whole image is classically enhanced and YOLO is retried."),
  N("**Classical enhancement as a second view.** When the quality analysis asks for it, the enhanced crop is classified as well as the original and the predictions are averaged, each with horizontal-flip test-time augmentation."),
  N("**Calibrated ensemble.** Each network is temperature-scaled (T = 0.65 for ConvNeXt, 0.5 for CLIP) and they are averaged 0.3 / 0.7. Weights and temperatures were fitted on validation."),
  N("**Late fusion with the classical classifier.** Tested over w = 0–0.2; the validation optimum is w = 0, so it is reported honestly rather than forced. The classical branch contributes through enhancement, the ROI fallback and the interpretable intermediate outputs."),
  P("**Decision logic:**", { after: 60 }),
  B("**Vehicle gate.** Accept if P(car) ≥ 0.9, or if P(car) ≥ 0.5 and YOLO agrees (box confidence ≥ 0.3). Otherwise the answer is \"No car detected\"."),
  B("**Hierarchical back-off.** The threshold t = 0.67 is the lowest giving ≥ 98.5% precision on validation. Below t, the system reports \"make + model (year uncertain)\" or \"make only\". The UI never presents a model it did not commit to."),
  B("**Open-set check.** CLIP on the crop scores the 954-model catalogue. A car is flagged as outside the 196 classes when CLIP's best match is an outside model of a different family, CLIP is ≥ 20% sure, at least 30% of its probability lies outside the training set, and the ensemble is below 0.7. Tuned on validation by hiding each image's true class, with at most 0.5% of known cars re-routed."),
  B("**Responsible answer for unfamiliar cars (make first).** The make probability is fused 50/50 from the classifier and from CLIP; for naming, CLIP uses the mean of the whole-frame and crop embeddings."),
  B("A catalogue model is named only if CLIP is ≥ 80% sure and it matches that make.", 1),
  B("Otherwise the answer is \"<make>, model not recognised\" (make ≥ 40%), or \"Unknown car (outside the supported classes)\".", 1),
  B("The best unconfirmed guess and the closest trained class stay visible as candidates.", 1),
  B("The thresholds maximise correct minus wrong answers on simulated unseen validation cars.", 1),
];

const s3 = [
  H1("3. Experimental Benchmarks & Evaluation"),
  P("All numbers are on the official Stanford Cars **test split of 8,041 images**, in the order required by the brief. Every value is reproduced by python -m scripts.run_all; raw values are in results/*.json and in eval_P5_<TeamID>.md."),
  H2("3.1 Quantitative Benchmarks"),
  H3("Classification performance (Top-1 / Top-5)"),
  ...table(["Approach", "Top-1 ↑", "Top-5 ↑", "Make acc.", "Make+Model acc.", "ECE ↓"], [
    ["Classical baseline (detected crop → HOG+LBP+colour+edge+geometry → PCA+LogReg)", "14.53%", "32.94%", "24.01%", "14.91%", "0.035"],
    ["Deep learning only (YOLO11s crop → ConvNeXt-T)", "94.52%", "99.19%", "98.41%", "95.10%", "0.017"],
    ["**Proposed hybrid system**", "**95.90%**", "**99.75%**", "**99.10%**", "**96.38%**", "**0.011**"],
  ], [0.4, 0.11, 0.11, 0.12, 0.14, 0.12], { highlight: [2] }),
  P("Chance level is 0.51% (1 in 196). With unseen-model routing switched on, a known car wrongly flagged as unseen counts as an error, and the strict hybrid top-1 is 95.45%."),
  H3("Object localisation (IoU) and detection (mAP)"),
  ...table(["Detector", "Mean IoU", "IoU ≥ 0.5", "IoU ≥ 0.75", "mAP@0.5", "mAP@0.5:0.95"], [
    ["**YOLO11s fine-tuned (ours)**", "**0.954**", "**99.90%**", "**99.15%**", "**0.989**", "**0.931**"],
    ["YOLO11n COCO-pretrained", "0.944", "99.58%", "98.61%", "0.854", "0.784"],
    ["Classical edge/contour ROI", "0.654", "79.52%", "31.63%", "0.641", "0.225"],
  ], [0.32, 0.12, 0.13, 0.13, 0.13, 0.17], { highlight: [0] }),
  P("Class-aware mAP@0.5 counts a detection only if IoU ≥ 0.5 **and** the make/model/year is correct: hybrid 0.944, deep 0.927, classical 0.072."),
  H3("Model efficiency (size and inference time)"),
  ...table(["Component", "Parameters", "Size", "GPU latency", "CPU latency"], [
    ["YOLO11s detector", "9.4 M", "18.3 MB", "8.0 ms", "38.5 ms"],
    ["ConvNeXt-T classifier, FP16 (deployed)", "28.0 M", "53.4 MB", "5.5 ms", "—"],
    ["CLIP ViT-B/16 fine-tune, FP16, with flip TTA", "86.3 M", "164.6 MB", "8.9 ms", "135 ms"],
    ["CLIP gate + open-vocabulary naming", "86.2 M", "—", "6.6 ms", "122 ms"],
    ["Classical features + classifier", "—", "—", "~15 ms features + 0.02 ms classifier", ""],
  ], [0.4, 0.13, 0.13, 0.17, 0.17]),
  P("Latency is batch 1, the mean of 20–30 runs after warm-up, on an RTX 5060 Laptop GPU. A full web request (all three modes, intermediate images and JSON) takes about 0.15–0.2 s on the GPU in the app."),
  P("**Optimisation (at least one required; three applied)** — ConvNeXt-T classifier:", { after: 60 }),
  ...table(["Variant", "Size", "Val top-1", "Test top-1 (1,500 subset)", "Latency"], [
    ["FP32 (PyTorch)", "106.8 MB", "95.69%", "93.87%", "8.2 ms GPU / 74.2 ms CPU"],
    ["**FP16 reduced precision (deployed)**", "**53.4 MB**", "95.69%", "93.87%", "**5.5 ms GPU**"],
    ["**INT8 static quantisation (ONNX Runtime)**", "**27.3 MB**", "94.92%", "93.73%", "40.6 ms CPU"],
    ["Global L1 pruning 30% (no fine-tune)", "41.0 MB*", "90.86%", "89.67%", "—"],
    ["Global L1 pruning 50% (no fine-tune)", "32.9 MB*", "16.88%", "16.00%", "—"],
  ], [0.34, 0.12, 0.12, 0.18, 0.24], { highlight: [1, 2] }),
  P("* gzip-compressed sparse weights. FP16 halves the size with no accuracy loss; INT8 is 3.9× smaller for −0.14 points; pruning without re-training is not worthwhile."),
  H3("Robustness (performance under varying conditions)"),
  P("End-to-end top-1 (detector + classifier) on 1,000 random test images per condition. Scoring is strict: \"no car\" and \"unseen\" answers count as errors."),
  ...table(["Condition", "Classical", "Deep only", "Hybrid", "Hybrid mean IoU"], [
    ["Original", "5.0%", "94.7%", "**94.8%**", "0.947"],
    ["Dark", "4.4%", "94.1%", "**95.0%**", "0.941"],
    ["Very dark + noisy", "2.6%", "89.7%", "**93.0%**", "0.923"],
    ["Overexposed", "4.3%", "93.4%", "**93.8%**", "0.941"],
    ["Low resolution 4×", "2.3%", "90.7%", "**93.5%**", "0.936"],
    ["Low resolution 8×", "2.5%", "75.8%", "**82.7%**", "0.897"],
    ["30% occlusion", "1.8%", "**88.7%**", "87.5%", "0.927"],
    ["Gaussian noise", "3.7%", "93.2%", "**94.5%**", "0.927"],
    ["Motion blur", "2.5%", "89.4%", "**90.5%**", "0.920"],
    ["Rotation 15°", "1.7%", "92.4%", "**94.8%**", "0.816"],
    ["Perspective (viewpoint)", "3.7%", "94.2%", "**95.2%**", "0.796"],
  ], [0.32, 0.15, 0.15, 0.15, 0.23]),
  P("The classical column is fully classical end-to-end (classical ROI + hand-crafted features). The lower IoU under rotation and perspective reflects the unwarped ground-truth box, not a detector failure. On natural variation across the full test set (quintiles of brightness and of car size), the hybrid stays between 95.4% and 96.4% in every bin."),
  H3("Prediction confidence and threshold analysis"),
  ...table(["Threshold", "Coverage", "Accuracy on accepted"], [
    ["0.0", "100.00%", "95.90%"], ["0.5", "98.74%", "96.50%"], ["0.6", "97.21%", "97.21%"],
    ["**0.67 (chosen on validation)**", "—", "≥ 98.5% precision on validation"], ["0.7", "95.03%", "97.80%"],
    ["0.8", "91.85%", "98.39%"], ["0.9", "87.76%", "98.75%"],
  ], [0.4, 0.25, 0.35], { highlight: [3] }),
  ...table(["Reported level (t = 0.67)", "Share of test images", "Accuracy at that level"], [
    ["Confident (make + model + year)", "95.73%", "97.62%"],
    ["Year uncertain (make + model)", "0.52%", "100.00%"],
    ["Model uncertain (make only)", "2.67%", "98.60%"],
    ["Low confidence (no claim / unknown)", "1.07%", "—"],
  ], [0.44, 0.26, 0.3]),
  H3("Bad input and unfamiliar cars"),
  ...table(["Check", "Result"], [
    ["Real test cars accepted by the vehicle gate", "99.89%"],
    ["Non-car probe photos rejected (people, food, animals, textures, clock, flowers)", "16 / 16"],
    ["Zero-shot naming over 954 models (frame + crop), no training", "Make 93.3%, model family 85.0%"],
    ["Simulated unseen models flagged as \"not in the training set\" (n = 5,266)", "79.97%"],
    ["Known cars wrongly flagged as unseen", "0.80%"],
  ], [0.66, 0.34], { leftAll: true }),
  P("**Responsible answers on simulated unseen cars** (true class hidden from both classifiers). The make-first rule is compared with the old rule, which always used CLIP's top model:", { after: 60 }),
  ...table(["Scenario", "Correct model", "Correct make only", "No claim", "Wrong answer ↓"], [
    ["Unseen car in the catalogue — old rule", "69.0%", "—", "5.2%", "25.9%"],
    ["Unseen car in the catalogue — **make-first rule**", "55.6%", "18.8%", "8.1%", "**17.4%**"],
    ["Unseen car not in the catalogue — old rule", "5.5%", "—", "28.1%", "66.4%"],
    ["Unseen car not in the catalogue — **make-first rule**", "2.1%", "38.3%", "25.0%", "**34.6%**"],
  ], [0.4, 0.14, 0.16, 0.13, 0.17], { highlight: [1, 3] }),
  P("**Analysis.**"),
  B("**Classical vs deep.** Hand-crafted features cannot separate 196 near-identical classes (14.5%, still about 28× chance). Transfer-learned deep models reach 94.5%."),
  B("**Hybrid gain.** +1.4 points on clean images and far more on hard ones (+6.9 at 8× low resolution, +3.3 when very dark and noisy), mostly from the robust CLIP member."),
  B("**Speed.** About 10× the latency of deep-only, still two orders of magnitude below the 10 s budget."),
  B("**Weak spot.** Occlusion is the only condition without a gain; the −1.2 points are within noise (±1 point at n = 1,000)."),
  H2("3.2 Visual Outputs & Intermediate Representations"),
  P("For every upload the web application shows:", { after: 60 }),
  B("the detected bounding box with label and confidence;"),
  B("an image-quality telemetry panel (brightness, contrast, noise, sharpness, resolution);"),
  B("the top-5 candidates;"),
  B("a live classical vs deep vs hybrid comparison;"),
  B("six intermediate representations: detected region, adaptive enhancement, Canny edges, Sobel gradient magnitude, LBP texture, and contours + classical ROI (src/classical.py)."),
  P("The figures below are from the evaluation."),
  img("results/figures/robustness.png", 600, 440 / 1100),
  caption("Figure 2. Robustness: end-to-end top-1 per condition (results/figures/robustness.png)."),
  img("results/figures/threshold_analysis.png", 400, 418 / 605),
  caption("Figure 3. Confidence threshold analysis: coverage vs accuracy on accepted predictions."),
  img("results/figures/training_curves.png", 600, 385 / 1100),
  caption("Figure 4. Training curves."),
  img("results/failures/4_dark_or_low_contrast.jpg", 600, 396 / 1848),
  caption("Figure 5. Hard inputs: dark / low-contrast test images (green = ground-truth box, red = our box)."),
  img("results/failures/5_small_low_resolution.jpg", 600, 396 / 1848),
  caption("Figure 6. Hard inputs: very small / low-resolution vehicles."),
  H2("3.3 Ablation Study"),
  ...table(["Variant (test)", "Top-1", "Top-5", "Δ top-1 vs full"], [
    ["**Full hybrid**", "**95.90%**", "**99.75%**", "—"],
    ["− flip TTA", "95.73%", "99.73%", "−0.17"],
    ["− adaptive enhancement view", "95.86%", "99.75%", "−0.04"],
    ["Enhanced crop replaces original (fixed instead of adaptive view)", "95.90%", "99.71%", "0.00"],
    ["− classical feature fusion (validation weight = 0)", "95.90%", "99.75%", "0.00"],
    ["− robust CLIP fine-tune (ConvNeXt-only hybrid)", "94.62%", "99.17%", "−1.28"],
    ["CLIP fine-tune alone (TTA + enhancement views)", "95.68%", "99.76%", "−0.22"],
    ["Deep only (no TTA, no enhancement, no ensemble)", "94.52%", "99.19%", "−1.38"],
    ["− detection crop (whole image → crop-trained CNN)", "90.32%", "97.44%", "−5.58"],
    ["− detection crop (CNN trained and tested on whole images)", "92.85%", "99.23%", "−3.05"],
    ["Oracle: ground-truth box crop (deep)", "94.64%", "99.22%", "upper bound for localisation"],
  ], [0.52, 0.13, 0.13, 0.22], { highlight: [0] }),
  ...table(["Classical feature-group ablation (validation)", "Top-1"], [
    ["HOG only", "11.42%"], ["Without HOG", "10.28%"], ["Without LBP", "13.96%"], ["Without colour", "12.94%"],
    ["Without edges / without geometry", "14.59%"], ["LBP only / colour only / edges only / geometry only", "4.06% / 4.82% / 2.54% / 1.27%"],
  ], [0.7, 0.3]),
  img("results/figures/ablation.png", 520, 440 / 880),
  caption("Figure 7. Ablation of the hybrid pipeline."),
  P("**Findings.**"),
  B("**The detection crop is the largest single factor** (+5.6 points). Our detector crop is as good as the oracle ground-truth box."),
  B("**The robust CLIP member is the largest gain inside the hybrid** (+1.3 points)."),
  B("**Fixed vs adaptive enhancement.** Replacing the original crop with the enhanced one gave no gain, and an earlier version hurt dark images. Adding it as a view helps under hard conditions."),
  B("**HOG is the most useful hand-crafted feature.** Colour alone is weak (4.8%) because each model comes in many colours."),
];

const s4 = [
  H1("4. Failure Modes & Limitations"),
  P("**Error breakdown.** 330 hybrid errors on the test set:"),
  B("66.4%: right make, wrong model;"),
  B("11.8%: right make + model, wrong year;"),
  B("21.8%: a different make (mostly badge-engineered twins)."),
  ...table(["Most frequent confusion (true → predicted)", "Count"], [
    ["Audi TTS Coupe 2012 → Audi TT Hatchback 2011", "17"],
    ["Dodge Caliber Wagon 2012 ↔ 2007", "13 + 9"],
    ["Chevrolet Express Van 2007 ↔ Express Cargo Van 2007", "13 + 10"],
    ["Audi S5 Coupe 2012 → Audi A5 Coupe 2012", "12"],
    ["Chevrolet Silverado 1500 Regular Cab → 2500HD Regular Cab (2012)", "9"],
    ["Bentley Continental GT Coupe 2012 → 2007", "9"],
  ], [0.82, 0.18]),
  ...table(["#", "Observed failure case", "Root cause", "Mitigation"], [
    ["1", "Year / generation confusion (Bentley Continental GT, Dodge Caliber 2012 vs 2007)", "Generations share the same silhouette; the differences (lamps, bumpers) are a few pixels and vanish in side or rear views", "\"Year uncertain\" back-off (implemented); part-based attention on lamps and grille (future)"],
    ["2", "\"Confident but wrong\": GMC Yukon vs Terrain, Jeep Compass vs Grand Cherokee, Rolls-Royce Phantom vs Ghost", "Dataset label noise: the prediction is right and the test label is wrong", "Reported, not \"fixed\"; it caps achievable accuracy"],
    ["3", "Multi-car / collage images (IoU 0.07–0.23)", "Only one car per image is annotated; which one is arbitrary", "Box ranked by confidence × size; multi-car classification (future)"],
    ["4", "Dark body-style variants (wagon vs sedan, coupe vs convertible)", "The distinguishing part is in shadow or not visible from the front", "Adaptive enhancement view (+3.3 points very dark); medium confidence → make-level back-off"],
    ["5", "Very small / low-resolution cars (Audi V8 vs Audi 100, 1994)", "Same body; only the grille differs at 80–95 px", "Low-resolution augmentation in both networks (+6.9 points at 8×)"],
    ["6", "User test: tulips labelled \"Lamborghini Gallardo\" (first version)", "The single-class detector fires on anything (0.53), and a closed-set softmax must pick one of 196 classes", "**Fixed**: CLIP vehicle gate (P(car) = 0.00003) → \"No car detected\""],
    ["7", "User test: 2024 Aston Martin labelled \"Lamborghini\" / \"Hyundai Veloster\"", "The model is not among the 196 classes (the newest Aston Martin in the dataset is from 2012)", "**Fixed**: open-set check → \"Aston Martin, model not recognised\", best guess Vantage"],
    ["8", "User test: Bugatti special edition named \"Ferrari F8 Tributo\" (46%)", "Not in the classes or the catalogue; CLIP's uncertain top model overrode the classifier's correct make (Bugatti, 67%)", "**Fixed**: make-first rule → \"Bugatti, model not recognised\", best guess Bugatti Divo"],
    ["9", "User test: Indian-market Ford Figo shown as \"Ford Fiesta Sedan 2012\" (21%)", "Not in the dataset or the catalogue; probability split between Ford and Hyundai hatchbacks", "**Fixed**: \"Unknown car (outside the supported classes?)\"; adding regional models to the catalogue (future)"],
  ], [0.05, 0.3, 0.33, 0.32], { leftAll: true }),
  img("results/failures/1_wrong_year_same_model.jpg", 600, 396 / 1848),
  caption("Figure 8. Failure case 1: right model, wrong year."),
  img("results/failures/2_confident_but_wrong.jpg", 600, 396 / 1848),
  caption("Figure 9. Failure case 2: \"confident but wrong\", mostly label noise in the test set."),
  img("results/failures/3_localisation_failure.jpg", 600, 396 / 1848),
  caption("Figure 10. Failure case 3: multi-car images (green = annotated box, red = our box)."),
  H2("4.1 Root Cause & Mitigation of the Main Failure (Mathematical View)"),
  P("A closed-set classifier ends in a softmax over K = 196 classes. Its probabilities always sum to 1, so any input — flowers, or a car model it has never seen — must be assigned to some class, often with high confidence. Two mechanisms address this:"),
  N("**An explicit \"no car\" decision** from an independent, open-vocabulary model: CLIP compares the image with car and non-car text prompts.", "numbers2"),
  N("**An open-set test plus a hierarchy.** The probability is summed over coarser groups (make + model, make). The system answers at the finest level that is confident and names an outside model only when CLIP is sure.", "numbers2"),
  P("The remaining error source is unseen cars that are not flagged (about 20%): the classifier is confident in a similar trained class. Possible fixes:"),
  B("training on a larger, newer car dataset, if the organisers allow it;"),
  B("a larger catalogue with generation year ranges;"),
  B("a stronger zero-shot model (CLIP ViT-L/14 or SigLIP)."),
  H2("4.2 Limitations"),
  B("Make, model and year are learned only for the 196 classes (mostly US-market, 1991–2012). Unseen cars get no year."),
  B("About 20% of unseen models are not flagged. Regional models (e.g. Indian-market cars) are largely missing from the catalogue."),
  B("Exact year is fundamentally ambiguous between visually identical generations."),
  B("Only the primary vehicle in an image is classified."),
  B("Robustness was measured with synthetic corruptions; real night or rain photos may differ."),
  B("The vehicle gate was tested on 16 non-car probe photos, not a large benchmark."),
  B("mAP under-counts detector quality because only one car per image is annotated."),
  H2("4.3 Ethics and Responsible Use"),
  B("The system recognises vehicle types, not people or number plates, and must not be used to track individuals. Uploaded images are not stored by our code."),
  B("Dataset bias: mostly North-American cars photographed in daylight. Accuracy on other markets, modified cars and night images is lower, and the UI states the class scope."),
  B("For unfamiliar cars the app reports only what it is sure of, and labels any specific model as an unconfirmed guess."),
  B("Stanford Cars images keep their original copyrights. The dataset is not redistributed; only 8 example test images are included for the demo."),
];

const s5 = [
  H1("5. Deployment & Quickstart Guide"),
  H2("5.1 Quickstart Commands"),
  ...code([
    "# 1. Clone the repository and install dependencies",
    "git clone <your-github-repo-url>",
    "cd <repo-folder>",
    "pip install -r requirements.txt",
    "",
    "# 2. Run the web app (UI + REST API), then open http://127.0.0.1:7860",
    "python app.py",
    "",
    "# 3. Or run it with Docker",
    "docker build -t p5-cars .",
    "docker run -p 7860:7860 p5-cars",
    "",
    "# 4. Reproduce every result (Stanford Cars in ~/cvh/data/raw; ~2.5 h on an RTX 5060)",
    "python -m scripts.run_all",
    "python -m scripts.write_report --team <TeamID> --usns \"<USN1>,<USN2>,...\"",
  ]),
  new Paragraph({ children: [], spacing: { after: 160 } }),
  H2("5.2 Deployment Format"),
  ...table(["Layer", "Details"], [
    ["Backend", "FastAPI + Uvicorn (app.py): POST /api/analyze, GET /api/performance, /api/classes, /api/examples, /api/health. A single inference thread with GPU warm-up at start-up; 15 MB upload limit; readable error messages; no stack traces leaked"],
    ["Frontend", "Custom web UI \"AutoLens\" (web/: HTML, CSS, vanilla JavaScript + Motion animations). Upload, paste or camera input; loading, error, \"No car detected\", \"Unknown model\" and low-confidence states; live comparison and intermediate steps"],
    ["Packaging", "modal_app.py deploys the live demo on Modal (CPU container, 2 cores, 6 GB). The Dockerfile runs it on any Docker host. PyTorch FP16 weights; optional ONNX INT8 classifier via CVH_BACKEND=onnx"],
    ["Repository", "Large model files (models/clip_ft_fp16.pt, 165 MB) are tracked with Git LFS (.gitattributes). No secrets committed; deployment tokens stay on the local machine"],
    ["Reproducibility", "Fixed seeds (SEED = 42); stratified validation split; scripts/run_all.py runs every step in order; scripts/write_report.py regenerates the eval file"],
  ], [0.2, 0.8], { leftAll: true }),
  H2("5.3 Technology Stack"),
  B("Python 3.12, PyTorch 2.11 (CUDA 12.8), timm (ConvNeXt), open_clip (CLIP), ultralytics (YOLO11)."),
  B("OpenCV, scikit-image, scikit-learn (classical CV and ML); ONNX Runtime (INT8)."),
  B("FastAPI / Uvicorn; vanilla JS + Motion; Modal (cloud container); Docker."),
  B("Hardware: RTX 5060 Laptop GPU (8 GB)."),
];

const checklist = [
  H1("Student Submission Checklist"),
  ...table(["Item", "Status"], [
    ["Working GitHub repository link included with an updated README.md", "☑ done"],
    ["No raw code pasted into this report", "☑ done"],
    ["All table metrics match test runs reproducible via the repository (python -m scripts.run_all)", "☑ done"],
    ["All team member details accurately recorded", "☑ done"],
    ["Live hosted application URL", "☑ done"],
    ["Architecture diagram, design document, eval_P5_<TeamID>.md, ablation & failure analysis in the repository", "☑ done (update team ID)"],
  ], [0.78, 0.22], { leftAll: true }),
  H2("References"),
  B("J. Krause, M. Stark, J. Deng, L. Fei-Fei. 3D Object Representations for Fine-Grained Categorization. ICCV Workshops, 2013 (Stanford Cars)."),
  B("Z. Liu et al. A ConvNet for the 2020s (ConvNeXt). CVPR 2022."),
  B("A. Radford et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021; OpenCLIP LAION-2B weights."),
  B("Ultralytics YOLO11, 2024."),
  B("N. Dalal, B. Triggs. Histograms of Oriented Gradients for Human Detection. CVPR 2005."),
  B("T. Ojala, M. Pietikäinen, T. Mäenpää. Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. TPAMI 2002."),
  B("C. Guo et al. On Calibration of Modern Neural Networks (temperature scaling). ICML 2017."),
];

// ------------------------------------------------------------------ document
const hstyle = (size, color, before, after) => ({ run: { font: FONT, size, bold: true, color }, paragraph: { spacing: { before, after }, keepNext: true, keepLines: true } });
const doc = new Document({
  creator: "P5 team", title: "P5 Technical Solution Report", description: "Fine-grained vehicle recognition",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 52, bold: true, color: ACCENT }, paragraph: { spacing: { before: 0, after: 120 }, alignment: AlignmentType.CENTER, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, ...hstyle(32, ACCENT, 360, 160), paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, keepLines: true, outlineLevel: 0, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "B4C6E7", space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, ...hstyle(26, "2E74B5", 240, 120), paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, keepLines: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, ...hstyle(23, "404040", 200, 80), paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, keepLines: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
    { reference: "numbers2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1300, left: 1440, right: 1440 } }, titlePage: true },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [t("P5 · Fine-Grained Vehicle Recognition · Technical Report", { size: 17, color: "888888" })] })] }) },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], font: FONT, size: 17, color: "888888" })] })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children: [...cover, ...info, ...s1, ...s2, ...s3, ...s4, ...s5, ...checklist],
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length); });

