"""Generate eval_P5_<TeamID>.md from results/*.json (metric order follows the brief).

    python -m scripts.write_report --team T01 --usns "1XX22CS001,1XX22CS002"
Team / USNs can also come from env vars CVH_TEAM_ID and CVH_USNS.
"""
from __future__ import annotations

import argparse
import json
import os

from src import config as C


def pct(x):
    return "-" if x is None else f"{x * 100:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default=os.environ.get("CVH_TEAM_ID", "T80"))
    ap.add_argument("--usns", default=os.environ.get("CVH_USNS", "1RUA24CSE0200,1RUA24CSE0206,1RUA24CSE0130"))
    a = ap.parse_args()
    R = json.loads((C.RESULTS_DIR / "eval_results.json").read_text())
    O = json.loads((C.RESULTS_DIR / "optimization.json").read_text()) if (C.RESULTS_DIR / "optimization.json").exists() else None
    CL = json.loads((C.RESULTS_DIR / "classical.json").read_text())
    L = [a.usns, "", f"# Evaluation - P5 Fine-Grained Vehicle Recognition (Team {a.team})", "",
         "All numbers are on the **official Stanford Cars test split (8,041 images, never used for training "
         "or tuning)**. Hyper-parameters (fusion weight, thresholds) were tuned on a stratified 10% validation "
         "split of the training set (788 images). End-to-end = our YOLO detector localises the car, the crop is classified.", ""]

    # 1. Classification
    L += ["## 1. Classification performance (Top-1 / Top-5 accuracy)", "",
          "| Approach | Top-1 | Top-5 | Make acc. | Make+Model acc. | ECE |", "|---|---|---|---|---|---|"]
    for k, v in R["comparison"].items():
        L.append(f"| {k} | {pct(v['top1'])} | {pct(v['top5'])} | {pct(v['make_acc'])} | {pct(v['make_model_acc'])} | {v['ece']:.3f} |")
    gt = R["ablation"]["oracle: ground-truth box crop (upper bound for localisation)"]
    L += ["", f"Deep model on ground-truth box crops (oracle localisation): top-1 {pct(gt['top1'])}, top-5 {pct(gt['top5'])}.", ""]

    # 2. Localisation
    L += ["## 2. Object localisation (IoU with ground-truth box)", "",
          "| Detector | Mean IoU | Median IoU | IoU >= 0.5 | IoU >= 0.75 | No detection |", "|---|---|---|---|---|---|"]
    for d in R["detection"]:
        L.append(f"| {d['detector']} | {d['mean_iou']:.3f} | {d['median_iou']:.3f} | {pct(d['iou_ge_0_5'])} | {pct(d['iou_ge_0_75'])} | {d['missed']} |")
    # 3. Detection
    L += ["", "## 3. Object detection (mAP)", "",
          "| Detector | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|"]
    for d in R["detection"]:
        L.append(f"| {d['detector']} | {d['mAP50']:.3f} | {d['mAP50_95']:.3f} |")
    L += ["", "Class-aware mAP@0.5 over 196 classes (detection counts only if IoU >= 0.5 **and** the make/model/year is correct; "
          "score = detection conf x class conf):", "",
          "| Approach | class-aware mAP@0.5 |", "|---|---|"]
    L += [f"| {k} | {v:.3f} |" for k, v in R["class_aware_mAP50"].items()]

    # 4. Efficiency
    L += ["", "## 4. Model efficiency (size and inference time)", ""]
    if O:
        L += [f"Classifier: ConvNeXt-Tiny, {O['classifier_params_M']} M parameters, input 384x384. "
              f"Detector: {O['detector']['weights']}, {O['detector']['params_M']} M parameters, "
              f"{O['detector']['size_mb']} MB, {O['detector']['latency_gpu_ms']:.1f} ms GPU / {O['detector']['latency_cpu_ms']:.1f} ms CPU per image.", "",
              "| Classifier variant | Size (MB) | Val top-1 | Test top-1 (1,500 subset) | GPU latency (ms) | CPU latency (ms) | Note |",
              "|---|---|---|---|---|---|---|"]
        for k, v in O["variants"].items():
            g = "-" if v["latency_gpu_ms"] is None else f"{v['latency_gpu_ms']:.1f}"
            c = "-" if v["latency_cpu_ms"] is None else f"{v['latency_cpu_ms']:.1f}"
            L.append(f"| {k} | {v['size_mb']} | {pct(v['val_top1'])} | {pct(v['test_top1'])} | {g} | {c} | {v['note']} |")
        L += ["", "Latency = batch 1, mean of 20-30 runs after warm-up, RTX 5060 Laptop GPU / laptop CPU."]
        if "clip" in O:
            c = O["clip"]
            L += ["", f"Second ensemble member, CLIP ViT-B/16 fine-tune: {c['clip_ft_params_M']} M params, {c['clip_ft_size_mb']} MB (FP16), "
                  f"{c['clip_ft_ms_cuda']:.1f} ms GPU / {c['clip_ft_ms_cpu']:.0f} ms CPU with flip TTA. Zero-shot CLIP vehicle gate + "
                  f"open-vocabulary naming: {c['clip_zero_shot_visual_params_M']} M params, {c['clip_gate_and_naming_ms_cuda']:.1f} ms GPU / "
                  f"{c['clip_gate_and_naming_ms_cpu']:.0f} ms CPU."]
    L += [f"Classical feature classifier: {CL['clf_ms_per_img']:.2f} ms per image (after feature extraction, ~15 ms).", ""]

    # 5. Robustness
    if "robustness" in R:
        n = R["robustness"][0]["n"]
        L += [f"## 5. Robustness (end-to-end, {n} random test images per condition)", "",
              "| Condition | Classical top-1 | Deep top-1 | Hybrid top-1 | Hybrid mean IoU |", "|---|---|---|---|---|"]
        for r in R["robustness"]:
            L.append(f"| {r['condition']} | {pct(r['classical_top1'])} | {pct(r['deep_top1'])} | {pct(r['hybrid_top1'])} | {r['hybrid_iou']:.3f} |")
        L += ["", "Notes: the *classical* column here is fully classical end-to-end (edge/contour ROI + hand-crafted "
              "features), hence lower than the 14.5% obtained with YOLO crops. For rotation and perspective the GT box "
              "is not warped, so the lower IoU there reflects the unwarped reference, not a detector failure. With "
              "1,000 images the standard error of a top-1 value is about 0.7-1 point, so only the very-dark (+1.6) and "
              "motion-blur (+0.8) / rotation (+0.7) gains of the hybrid are larger than noise."]
    for lbl, rows in R["natural_variation"].items():
        L += ["", f"Natural variation on the full test set, by quintile of {lbl}:", "",
              "| Bin | Classical | Deep | Hybrid |", "|---|---|---|---|"]
        L += [f"| {r['bin']} | {pct(r['classical'])} | {pct(r['deep'])} | {pct(r['hybrid'])} |" for r in rows]

    # 6. Confidence
    L += ["", "## 6. Prediction confidence and threshold analysis", "",
          f"Chosen threshold (on validation: lowest t with >= 98.5% precision on accepted predictions): "
          f"**{R['hybrid_config']['conf_threshold']}**.", "",
          "| Threshold | Coverage | Accuracy on accepted |", "|---|---|---|"]
    L += [f"| {t['threshold']:.2f} | {pct(t['coverage'])} | {pct(t['accuracy_on_accepted'])} |" for t in R["threshold_analysis"][::2]]
    L += ["", "Hierarchical low-confidence handling at the chosen threshold:", "",
          "| Reported level | Fraction of test images | Accuracy at reported level | Fine-grained accuracy |", "|---|---|---|---|"]
    for k, v in R["low_confidence_handling"].items():
        L.append(f"| {k} | {pct(v['fraction'])} | {pct(v['accuracy_at_reported_level'])} | {pct(v['fine_grained_accuracy'])} |")

    # Vehicle gate + open-set
    OS = json.loads((C.RESULTS_DIR / "openset.json").read_text()) if (C.RESULTS_DIR / "openset.json").exists() else None
    if OS:
        g, o, z = OS["gate_on_stanford_test"], OS["open_set_routing"], OS["zero_shot_catalogue_test"]
        L += ["", "### Bad input: \"no car\" gate and cars outside the 196 classes", "",
              "A CLIP ViT-B/16 vehicle gate (car prompts vs. everyday non-car prompts, plus agreement with the "
              "YOLO detector for borderline scores) decides whether a car is present at all.", "",
              "| Gate check | Result |", "|---|---|",
              f"| Real cars accepted (all 8,041 test images contain a car) | {pct(g['accept_two_signal'])} |",
              f"| Non-car probe photos rejected (people, food, animals, textures, clock, flowers) | {pct(OS['gate_non_car_rejected@0.5'])} by CLIP alone; the clock (0.77) is also rejected by the detector-agreement rule |",
              "", f"Open-vocabulary naming (zero-shot CLIP over {z['catalogue_size']} catalogued models from {z['makes']} makes, "
              "no training, frame + crop): " f"make {pct(z['make'])}, make+model family {pct(z['model_family'])}, exact Stanford class {pct(z['exact_class'])} on the test set"
              + (f" (crop only: make {pct(z['crop_only']['make'])}, family {pct(z['crop_only']['model_family'])})." if "crop_only" in z else "."), "",
              "Unseen-model routing, tuned on validation (max simulated-unseen recall with at most 0.5% of known cars re-routed). "
              "Unseen cars are simulated by hiding each test image's true class from the classifier:", "",
              "| Routing metric | Value |", "|---|---|",
              f"| Simulated unseen cars flagged as 'not in training set' | {pct(o['test_simulated_unseen_recall'])} (n = {o['test_simulated_unseen_n']}) |",
              f"| ...of which CLIP names the correct make | {pct(o['test_simulated_unseen_make_correct_when_routed'])} |",
              f"| Known test cars wrongly re-routed | {pct(o['test_known_rerouted'])} |",
              f"| Hybrid top-1 counting re-routed known cars as errors (strict) | {pct(o['test_top1_strict'])} (vs {pct(o['test_top1_before_routing'])} without routing) |"]
    UK = json.loads((C.RESULTS_DIR / "unknown.json").read_text()) if (C.RESULTS_DIR / "unknown.json").exists() else None
    if UK:
        L += ["", "Responsible answers for unfamiliar cars (make first; a model is named only when CLIP is "
              f">= {UK['open_name_conf']:.0%} sure, otherwise \"<make>, model not recognised\" or \"unknown car\"). "
              f"Simulated unseen test cars, n = {UK['n_test']}, new rule vs old rule (always CLIP's top model):", "",
              "| Scenario | Correct model | Correct make only | No claim | Wrong answer |", "|---|---|---|---|---|"]
        for lab, txt in (("catalogued", "in the catalogue"), ("uncatalogued", "not in the catalogue")):
            for v in ("old", "new"):
                s = UK[f"test_{lab}_{v}"]
                L.append(f"| Unseen car {txt}, {v} | {pct(s['correct_name'])} | {pct(s['correct_make']) if v == 'new' else '-'} "
                         f"| {pct(s['unknown'])} | {pct(s['wrong'])} |")
        k = UK["test_known_cars_shown_as"]
        L += ["", f"Known test cars shown as: trained class {pct(k['trained_class'])}, make only {pct(k['make_only'])} "
              f"(make right {pct(UK['test_known_cars_make_only_accuracy'])}), unknown car {pct(k['unknown'])}, "
              f"open-vocabulary name {pct(k['open_set_named'])}."]

    # Ablation
    L += ["", "## 7. Ablation study (test top-1 / top-5)", "", "| Variant | Top-1 | Top-5 |", "|---|---|---|"]
    L += [f"| {k} | {pct(v['top1'])} | {pct(v['top5'])} |" for k, v in R["ablation"].items()]
    L += ["", "Classical feature-group ablation (validation top-1):", "", "| Setting | Val top-1 |", "|---|---|"]
    L += [f"| {k} | {pct(v)} |" for k, v in CL["feature_ablation_val_top1"].items()]
    L += ["", f"Validation search for the fusion weight w: " +
          ", ".join(f"w={k}: {pct(v)}" for k, v in list(R["val_fusion_search"].items())[:9]) + "."]

    # Errors
    E = R["error_breakdown"]
    L += ["", "## 8. Error breakdown and failure cases", "",
          f"{E['n_errors']} hybrid errors on the test set: {pct(E['same_make_model_diff_year'])} are the right make+model "
          f"with the wrong year, {pct(E['same_make_diff_model'])} the right make with a wrong model, "
          f"{pct(E['different_make'])} a different make.", "",
          "Most frequent confusions:", "", "| True | Predicted | Count |", "|---|---|---|"]
    L += [f"| {c['true']} | {c['pred']} | {c['count']} |" for c in E["top_confusions"][:10]]
    L += ["", "Failure-case galleries: `results/failures/*.jpg`; analysis in `docs/design_document.md` section 9."]
    out = C.PROJECT_ROOT / f"eval_P5_{a.team}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf8")
    print("wrote", out)


if __name__ == "__main__":
    main()
