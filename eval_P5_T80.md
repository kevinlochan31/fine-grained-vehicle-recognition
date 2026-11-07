1RUA24CSE0200,1RUA24CSE0206,1RUA24CSE0130

# Evaluation - P5 Fine-Grained Vehicle Recognition (Team T80)

All numbers are on the **official Stanford Cars test split (8,041 images, never used for training or tuning)**. Hyper-parameters (fusion weight, thresholds) were tuned on a stratified 10% validation split of the training set (788 images). End-to-end = our YOLO detector localises the car, the crop is classified.

## 1. Classification performance (Top-1 / Top-5 accuracy)

| Approach | Top-1 | Top-5 | Make acc. | Make+Model acc. | ECE |
|---|---|---|---|---|---|
| classical features (detected crop -> HOG+LBP+colour+edge+geometry -> PCA+LogReg) | 14.53% | 32.94% | 24.01% | 14.91% | 0.035 |
| deep learning (YOLO crop -> ConvNeXt) | 94.52% | 99.19% | 98.41% | 95.10% | 0.017 |
| hybrid (YOLO + adaptive enhancement views + flip TTA + ConvNeXt/CLIP ensemble) | 95.90% | 99.75% | 99.10% | 96.38% | 0.011 |

Deep model on ground-truth box crops (oracle localisation): top-1 94.64%, top-5 99.22%.

## 2. Object localisation (IoU with ground-truth box)

| Detector | Mean IoU | Median IoU | IoU >= 0.5 | IoU >= 0.75 | No detection |
|---|---|---|---|---|---|
| yolo11s_finetuned | 0.954 | 0.966 | 99.90% | 99.15% | 1 |
| yolo11n_coco_pretrained | 0.944 | 0.958 | 99.58% | 98.61% | 10 |
| classical_edge_contour_roi | 0.654 | 0.657 | 79.52% | 31.63% | 0 |

## 3. Object detection (mAP)

| Detector | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| yolo11s_finetuned | 0.989 | 0.931 |
| yolo11n_coco_pretrained | 0.854 | 0.784 |
| classical_edge_contour_roi | 0.641 | 0.225 |

Class-aware mAP@0.5 over 196 classes (detection counts only if IoU >= 0.5 **and** the make/model/year is correct; score = detection conf x class conf):

| Approach | class-aware mAP@0.5 |
|---|---|
| hybrid | 0.944 |
| deep | 0.927 |
| classical | 0.072 |

## 4. Model efficiency (size and inference time)

Classifier: ConvNeXt-Tiny, 27.97 M parameters, input 384x384. Detector: yolo_cars.pt (YOLO11s fine-tuned), 9.43 M parameters, 18.28 MB, 8.0 ms GPU / 38.5 ms CPU per image.

| Classifier variant | Size (MB) | Val top-1 | Test top-1 (1,500 subset) | GPU latency (ms) | CPU latency (ms) | Note |
|---|---|---|---|---|---|---|
| fp32 (PyTorch) | 106.76 | 95.69% | 93.87% | 8.2 | 74.2 |  |
| fp16 (PyTorch, GPU) | 53.4 | 95.69% | 93.87% | 5.5 | - | reduced precision; deployed weights file |
| fp32 (ONNX Runtime, CPU) | 106.77 | 95.69% | 93.87% | - | 46.0 |  |
| int8 static quant (ONNX Runtime, CPU) | 27.34 | 94.92% | 93.73% | - | 40.6 | 200 training crops for calibration |
| pruned 30% (global L1) + fp16 | 40.99 | 90.86% | 89.67% | - | - | size = gzip-compressed sparse weights; no fine-tuning after pruning |
| pruned 50% (global L1) + fp16 | 32.94 | 16.88% | 16.00% | - | - | size = gzip-compressed sparse weights; no fine-tuning after pruning |

Latency = batch 1, mean of 20-30 runs after warm-up, RTX 5060 Laptop GPU / laptop CPU.

Second ensemble member, CLIP ViT-B/16 fine-tune: 86.3 M params, 164.6 MB (FP16), 8.9 ms GPU / 135 ms CPU with flip TTA. Zero-shot CLIP vehicle gate + open-vocabulary naming: 86.2 M params, 6.6 ms GPU / 122 ms CPU.
Classical feature classifier: 0.02 ms per image (after feature extraction, ~15 ms).

## 5. Robustness (end-to-end, 1000 random test images per condition)

| Condition | Classical top-1 | Deep top-1 | Hybrid top-1 | Hybrid mean IoU |
|---|---|---|---|---|
| original | 5.00% | 94.70% | 94.80% | 0.947 |
| dark | 4.40% | 94.10% | 95.00% | 0.941 |
| very_dark_noisy | 2.60% | 89.70% | 93.00% | 0.923 |
| overexposed | 4.30% | 93.40% | 93.80% | 0.941 |
| low_res_x4 | 2.30% | 90.70% | 93.50% | 0.936 |
| low_res_x8 | 2.50% | 75.80% | 82.70% | 0.897 |
| occlusion_30pct | 1.80% | 88.70% | 87.50% | 0.927 |
| gaussian_noise | 3.70% | 93.20% | 94.50% | 0.927 |
| motion_blur | 2.50% | 89.40% | 90.50% | 0.920 |
| rotation_15deg | 1.70% | 92.40% | 94.80% | 0.816 |
| perspective_view | 3.70% | 94.20% | 95.20% | 0.796 |

Notes: the *classical* column here is fully classical end-to-end (edge/contour ROI + hand-crafted features), hence lower than the 14.5% obtained with YOLO crops. For rotation and perspective the GT box is not warped, so the lower IoU there reflects the unwarped reference, not a detector failure. With 1,000 images the standard error of a top-1 value is about 0.7-1 point, so only the very-dark (+1.6) and motion-blur (+0.8) / rotation (+0.7) gains of the hybrid are larger than noise.

Natural variation on the full test set, by quintile of crop brightness:

| Bin | Classical | Deep | Hybrid |
|---|---|---|---|
| (14.593, 90.85] | 12.43% | 94.47% | 95.77% |
| (90.85, 103.093] | 15.55% | 94.53% | 96.27% |
| (103.093, 112.948] | 15.05% | 94.78% | 96.02% |
| (112.948, 125.944] | 15.49% | 95.09% | 96.02% |
| (125.944, 215.004] | 14.12% | 93.72% | 95.40% |

Natural variation on the full test set, by quintile of vehicle size (sqrt box area, px):

| Bin | Classical | Deep | Hybrid |
|---|---|---|---|
| (46.475, 212.688] | 15.23% | 94.16% | 96.02% |
| (212.688, 325.555] | 14.24% | 94.40% | 95.40% |
| (325.555, 408.551] | 14.49% | 93.53% | 95.96% |
| (408.551, 543.03] | 15.36% | 94.65% | 95.71% |
| (543.03, 4739.098] | 13.31% | 95.83% | 96.39% |

## 6. Prediction confidence and threshold analysis

Chosen threshold (on validation: lowest t with >= 98.5% precision on accepted predictions): **0.67**.

| Threshold | Coverage | Accuracy on accepted |
|---|---|---|
| 0.00 | 100.00% | 95.90% |
| 0.10 | 100.00% | 95.90% |
| 0.20 | 99.99% | 95.91% |
| 0.30 | 99.83% | 96.00% |
| 0.40 | 99.59% | 96.12% |
| 0.50 | 98.74% | 96.50% |
| 0.60 | 97.21% | 97.21% |
| 0.70 | 95.03% | 97.80% |
| 0.80 | 91.85% | 98.39% |
| 0.90 | 87.76% | 98.75% |

Hierarchical low-confidence handling at the chosen threshold:

| Reported level | Fraction of test images | Accuracy at reported level | Fine-grained accuracy |
|---|---|---|---|
| confident | 95.73% | 97.62% | 97.62% |
| year-uncertain | 0.52% | 100.00% | 73.81% |
| model-uncertain | 2.67% | 98.60% | 57.21% |
| low-confidence | 1.07% | - | 48.84% |

### Bad input: "no car" gate and cars outside the 196 classes

A CLIP ViT-B/16 vehicle gate (car prompts vs. everyday non-car prompts, plus agreement with the YOLO detector for borderline scores) decides whether a car is present at all.

| Gate check | Result |
|---|---|
| Real cars accepted (all 8,041 test images contain a car) | 99.89% |
| Non-car probe photos rejected (people, food, animals, textures, clock, flowers) | 93.75% by CLIP alone; the clock (0.77) is also rejected by the detector-agreement rule |

Open-vocabulary naming (zero-shot CLIP over 954 catalogued models from 83 makes, no training, frame + crop): make 93.33%, make+model family 85.00%, exact Stanford class 79.97% on the test set (crop only: make 87.99%, family 77.90%).

Unseen-model routing, tuned on validation (max simulated-unseen recall with at most 0.5% of known cars re-routed). Unseen cars are simulated by hiding each test image's true class from the classifier:

| Routing metric | Value |
|---|---|
| Simulated unseen cars flagged as 'not in training set' | 79.97% (n = 5266) |
| ...of which CLIP names the correct make | 92.90% |
| Known test cars wrongly re-routed | 0.80% |
| Hybrid top-1 counting re-routed known cars as errors (strict) | 95.45% (vs 95.90% without routing) |

Responsible answers for unfamiliar cars (make first; a model is named only when CLIP is >= 80% sure, otherwise "<make>, model not recognised" or "unknown car"). Simulated unseen test cars, n = 5266, new rule vs old rule (always CLIP's top model):

| Scenario | Correct model | Correct make only | No claim | Wrong answer |
|---|---|---|---|---|
| Unseen car in the catalogue, old | 68.95% | - | 5.17% | 25.88% |
| Unseen car in the catalogue, new | 55.64% | 18.84% | 8.11% | 17.41% |
| Unseen car not in the catalogue, old | 5.51% | - | 28.10% | 66.39% |
| Unseen car not in the catalogue, new | 2.11% | 38.26% | 25.03% | 34.60% |

Known test cars shown as: trained class 98.47%, make only 1.08% (make right 78.16%), unknown car 0.31%, open-vocabulary name 0.14%.

## 7. Ablation study (test top-1 / top-5)

| Variant | Top-1 | Top-5 |
|---|---|---|
| full hybrid | 95.90% | 99.75% |
| - classical feature fusion | 95.90% | 99.75% |
| - flip TTA | 95.73% | 99.73% |
| - adaptive enhancement view | 95.86% | 99.75% |
| enhanced crop replaces original (instead of averaging) | 95.90% | 99.71% |
| - fusion - TTA - enhancement (= deep only) | 94.52% | 99.19% |
| - detection crop (whole image -> crop-trained CNN) | 90.32% | 97.44% |
| oracle: ground-truth box crop (upper bound for localisation) | 94.64% | 99.22% |
| - detection crop (CNN trained & tested on whole images) | 92.85% | 99.23% |
| - CLIP fine-tune (ConvNeXt-only hybrid) | 94.62% | 99.17% |
| CLIP fine-tune alone (with TTA + enhancement views) | 95.68% | 99.76% |
| CLIP fine-tune alone, single pass | 95.56% | 99.75% |

Classical feature-group ablation (validation top-1):

| Setting | Val top-1 |
|---|---|
| without_hog | 10.28% |
| only_hog | 11.42% |
| without_lbp | 13.96% |
| only_lbp | 4.06% |
| without_color | 12.94% |
| only_color | 4.82% |
| without_edge | 14.59% |
| only_edge | 2.54% |
| without_geom | 14.59% |
| only_geom | 1.27% |

Validation search for the fusion weight w: w=0.0: 97.21%, w=0.025: 97.21%, w=0.05: 97.21%, w=0.075: 97.21%, w=0.1: 97.21%, w=0.125: 97.21%, w=0.15: 97.21%, w=0.175: 97.21%, w=0.2: 97.21%.

## 8. Error breakdown and failure cases

330 hybrid errors on the test set: 11.82% are the right make+model with the wrong year, 66.36% the right make with a wrong model, 21.82% a different make.

Most frequent confusions:

| True | Predicted | Count |
|---|---|---|
| Audi TTS Coupe 2012 | Audi TT Hatchback 2011 | 17 |
| Dodge Caliber Wagon 2012 | Dodge Caliber Wagon 2007 | 13 |
| Chevrolet Express Van 2007 | Chevrolet Express Cargo Van 2007 | 13 |
| Audi S5 Coupe 2012 | Audi A5 Coupe 2012 | 12 |
| Chevrolet Express Cargo Van 2007 | Chevrolet Express Van 2007 | 10 |
| Chevrolet Silverado 1500 Regular Cab 2012 | Chevrolet Silverado 2500HD Regular Cab 2012 | 9 |
| Bentley Continental GT Coupe 2012 | Bentley Continental GT Coupe 2007 | 9 |
| Dodge Caliber Wagon 2007 | Dodge Caliber Wagon 2012 | 9 |
| Audi V8 Sedan 1994 | Audi 100 Sedan 1994 | 8 |
| Mercedes-Benz Sprinter Van 2012 | Dodge Sprinter Cargo Van 2009 | 8 |

Failure-case galleries: `results/failures/*.jpg`; analysis in `docs/design_document.md` section 9.
