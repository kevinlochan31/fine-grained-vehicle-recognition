---
title: Fine-Grained Vehicle Recognition
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# P5 - Intelligent Fine-Grained Vehicle Recognition (Stanford Cars)

Detects the vehicle in a photo, draws its bounding box and predicts its **make, model and year**
(196 Stanford Cars classes) with a confidence score. Anything that is not a car returns **"No car
detected"**, and cars outside the 196 training classes are named open-vocabulary and flagged as
**unseen models**. The system combines classical computer vision (adaptive enhancement, edges/contours,
HOG/LBP/colour/geometry features) with deep learning: a YOLO11 detector, a ConvNeXt + robust CLIP
ViT-B/16 ensemble, and a CLIP vehicle gate with zero-shot naming over 954 catalogued models.

## Results (official test split, 8,041 unseen images)

| Approach | Top-1 | Top-5 |
|---|---|---|
| Classical features (HOG + LBP + colour + edges + geometry -> PCA + LogReg) | 14.53% | 32.94% |
| Deep learning (YOLO11s crop -> ConvNeXt-T) | 94.52% | 99.19% |
| **Hybrid** (ConvNeXt + robust CLIP ensemble, enhancement views, flip TTA, calibration) | **95.90%** | **99.75%** |

Robustness, end-to-end on 1,000 images per condition (deep vs hybrid): 8x low resolution 75.8% -> 82.7%,
very dark + noisy 89.7% -> 93.0%, 15 deg rotation 92.4% -> 94.8%, 30% occlusion 88.7% -> 87.5%.

Bad input and unseen cars:
- The vehicle gate accepts 99.9% of real cars and rejected all 16 non-car probe photos.
- 80% of simulated unseen models are flagged as unseen; only 0.8% of known cars are wrongly flagged.
- Unfamiliar cars are answered make first: a model is named only when CLIP is sure, otherwise
  "<make>, model not recognised" or "Unknown car". This cuts wrong answers on simulated unseen cars from
  26% to 17% (car in the catalogue) and from 66% to 35% (car not in the catalogue).

| Detection | mean IoU | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| YOLO11s fine-tuned (ours) | 0.954 | 0.989 | 0.931 |
| YOLO11n COCO-pretrained | 0.944 | 0.854 | 0.784 |
| Classical edge/contour ROI | 0.654 | 0.641 | 0.225 |

FP16 ConvNeXt: 53 MB, 5.5 ms GPU. INT8 (ONNX): 27 MB, -0.14 points. CLIP fine-tune: 165 MB FP16, 8.9 ms GPU.
App latency: about 0.15 s per request on GPU.

Model files above 100 MB (`models/clip_ft_fp16.pt`) need Git LFS on GitHub (`.gitattributes` is
included).
Full tables: [`eval_P5_T80.md`](eval_P5_T80.md).

- **Team:** Kevin Isaac (1RUA24CSE0200), Kokila C G (1RUA24CSE0206), Dhruthishree V (1RUA24CSE0130)
- **Repository:** https://github.com/kevinlochan31/fine-grained-vehicle-recognition
- **Live demo:** https://kevinisaacbtech24--autolens.us-east.modal.run
- **Evaluation file:** [`eval_P5_T80.md`](eval_P5_T80.md)
- **Design document:** [`docs/design_document.md`](docs/design_document.md)
- **Architecture diagram:** [`docs/architecture.png`](docs/architecture.png)

![architecture](docs/architecture.png)

## World car recognition (extra layer, not part of the official metrics)

The hackathon metrics above come from models trained only on Stanford Cars (196 classes, up to 2012).
To make the app useful on today's roads, including Indian-market cars, a **separate world-car layer**
sits on the open-set path. It never changes the 196-class models, their outputs or the numbers above.

- **Catalogue:** `data_meta/world_catalogue.json` has **4,689 make / model / generation entries
  (3,668 models, 215 makes) up to 2026**. It covers India (Maruti Suzuki, Tata, Mahindra, Hyundai, Kia,
  Honda, Toyota, MG, Skoda, Renault, VW, Nissan, Citroen, Force, Hindustan…), Japan, Korea, Europe, the US
  and China (BYD, Geely, NIO, XPeng, Li Auto, Zeekr…). Built by `scripts/build_world_catalogue.py`.
- **Reference photos:** 32,629 freely licensed Wikimedia Commons photos of 800 car generations were
  downloaded (`scripts/fetch_world_photos.py`, up to 60 per generation). Round 3 added every 2012+
  generation of 34 makes sold in India (cars sold in India first; trim variants that look like their base
  car are skipped), a better category resolver (ordinal generation categories, variant models such as
  "Thar Roxx" kept apart) and a Commons file-title search fallback. Interiors, badges, scale models etc. were
  filtered out, leaving **735 generations: 23,435 reference photos and 2,149 held-out test photos**. The 510
  test photos of the first round are frozen (`test_frozen.csv`), so earlier numbers stay comparable; new
  models get their own test photos. The split is by photographer for 97.6% of models, so a test photo is
  never a re-shot of a reference car.
- **Indian street photos (round 4):** car boxes from the FGVD dataset (Khoba, Parikh, Saluja,
  Sarvadevabhatla, Jawahar, *FGVD*, ICVGIP 2022; [Zenodo 7488960](https://zenodo.org/records/7488960),
  CC BY 4.0) were cropped and mapped to the catalogue (`scripts/import_fgvd.py`): 6,214 crops of 114
  generations (Dzire, Innova, Etios, Alto 800, Indica, Swift, WagonR…). The generation is taken from the
  catalogue when only one was on sale before 2019, otherwise the current layer picks it and the crop is kept
  only if it is ≥ 60% sure. FGVD train/val crops become references (≤ 120 per generation, 3,610 used);
  FGVD test crops are a separate street-photo test set (1,135 crops), so the Commons numbers stay comparable.
- **Matching (Phase II):** SigLIP ViT-B/16 **fine-tuned end to end** (`scripts/train_world.py`) on the
  reference photos **plus the 196 Stanford classes** (train split), with a cosine classifier head. That way
  it can tell a 2017 Camry from the 2012 Camry the classifier knows. Recipe: AdamW, layer-wise lr decay 0.8,
  label smoothing 0.1, strong augmentation (random crops, colour jitter, blur, erasing), class-balanced
  sampling, YOLO crop or full frame, 14 epochs (~55 min per run on an RTX 5060 Laptop for round 3). A *tune* run holds out every
  5th reference photo and 10% of models (as unseen-car negatives) to set the temperature and thresholds;
  the *final* run uses all photos. The original SigLIP tower is kept for zero-shot naming of catalogue
  cars without photos. The layer names **make → model → generation (years)** only at p ≥ 0.95. Otherwise
  the earlier answers (zero-shot model name, "<make>, model not recognised", "Unknown car") are kept.
- **Website:** e.g. "Maruti Suzuki Swift · 4th gen (2024–present) · matched from reference photos", with
  the closest reference photo and its author / licence / source ([credits](docs/world_photo_credits.md)).

**Held-out accuracy (512 test photos, 175 models, never seen while building or tuning).** Backbone comparison
(done with the first 841 reference photos):

| Zero-shot backbone (world layer) | Make | Model | Generation | Image tower, GPU / CPU (2 threads) |
|---|---|---|---|---|
| CLIP ViT-B/16 (laion2b, used by the gate) | 77.5% | 70.3% | 68.9% | 3.9 ms / 165 ms |
| CLIP ViT-L/14 (laion2b) | 87.1% | 79.9% | 77.9% | 9.1 ms / 756 ms |
| **SigLIP ViT-B/16 (webli), adopted** | **93.0%** | **85.4%** | **84.6%** | 4.0 ms / 172 ms |
| SigLIP, text only, no reference photos | 82.4% | 36.3% | n/a | |

(Frozen linear probe over 175 world models + 196 Stanford classes, forced choice, frame + crop input.)

**Phase II: from a frozen probe to a fine-tuned SigLIP (forced choice, held-out photos):**

| World layer | Models | Ref photos | Make | Model | Generation |
|---|---|---|---|---|---|
| Frozen SigLIP + linear probe (round 1) | 175 | 2,103 | 94.1% | 88.5% | 88.1% |
| Fine-tuned SigLIP, same 512 test photos | 246 | 7,606 | **96.5%** | **93.9%** | **93.7%** |
| Fine-tuned SigLIP, all 714 test photos | 246 | 7,606 | 95.7% | 92.9% | 92.2% |
| Round 3, same 510 test photos | 735 | 23,435 | 95.9% | 92.0% | 90.2% |
| Round 3, all 2,149 test photos | 735 | 23,435 | 96.0% | 85.5% | 82.3% |
| Round 4 (+ FGVD street crops), same 510 | 746 | 27,045 | 96.3% | 92.7% | 91.0% |
| Round 4, all 2,149 test photos | 746 | 27,045 | 96.4% | 86.1% | 82.5% |

Round 3 has three times as many generations to tell apart, many with only a handful of photos, so forced-choice
accuracy drops a little. When the layer names a generation (p ≥ 0.95) it is right 93.9% of the time
(model 95.9%, make 99.8%), for 63.6% of the 2,149 test photos.

**Full app pipeline (what the user sees), same 510 frozen photos:**

| Outcome | No world layer | Round 1 (probe) | Phase II (fine-tuned) | Round 3 (735 generations) | **Round 4 (+ street crops)** |
|---|---|---|---|---|---|
| Right make + model + generation | n/a | 60.9% | 80.4% | 69.8% | 67.3% |
| Right model (generation not given or wrong) | 20.9% (any) | 7.6% | 5.9% | 4.7% | 4.1% |
| Right make only ("model not recognised" + closest models) | 24.2% | 13.7% | 6.7% | 9.2% | 8.2% |
| "Not sure" / "Unknown car" (no claim) | 40.0% | 12.7% | 4.1% | 14.9% | 19.2% |
| **Wrong** | 14.8% | 5.1% | 2.9% | 1.4% | **1.2%** |

All 2,149 test photos: round 3 right generation 59.3%, right model 2.9%, make only 13.8%, no claim 20.4%,
wrong 3.6%; round 4 58.0% / 2.8% / 11.9% / 23.1% / **4.1%** (without the world layer: 19.5% wrong).

**Indian street photos (1,135 FGVD test crops, never used for training):**

| Outcome | No world layer | Round 3 | **Round 4** |
|---|---|---|---|
| Right make + model + generation | n/a | 16.7% | **69.5%** |
| Right model | 3.6% (any) | 1.9% | 6.1% |
| Right make only | 14.7% | 13.7% | 7.1% |
| No claim | 65.3% | 60.3% | 14.4% |
| **Wrong** | 16.4% | 7.5% | **2.9%** |

Round 4 trades a little on clean Commons photos (−1.3 points right generation, +0.5 wrong) for a large gain
on real Indian roads (+52.8 points right generation, wrong 7.5% → 2.9%). The build re-picked the make
threshold as 0.95 (≥ 99% tune precision). Most remaining errors are near-twins or two names for
one car (Creta vs ix25, Jazz vs Fit, Q7 vs SQ7, 5 Series vs i5, Cyberster vs Cyberster India).

**No confident wrong answers.** Round 3 is tuned to say "not sure" rather than guess:

- *Not sure (models disagree):* when the 196-class classifier is confident but the world head puts ≥ 20% of
  its belief on cars outside the 196 classes, the app shows "Not sure · maybe <world match> or <classifier
  answer>" instead of a confident answer (e.g. a 2020+ Mahindra Thar that the classifier calls "Jeep
  Wrangler SUV 2012", 99%). The threshold doubts 0.13% of Stanford val cars and catches 98.4% of
  unseen-model negatives.
- *Not sure (text guess only):* with the world layer loaded, a car it cannot match to reference photos is no
  longer named from text alone. Those name-only guesses (CLIP open-set names and catalogue zero-shot names)
  were wrong 66 of 149 times on the test photos, so they are now shown as "Not sure · maybe <guess>". This
  halved the wrong rate (7.4% → 3.6%); right-generation answers are unchanged (59.3%), while right-model
  answers fall from 6.7% to 2.9% because the correct text guesses are now hedged too.
- *Stricter make and model levels:* "<make>, model not recognised" needs ≥ 99% held-out precision (make
  threshold 0.95 in round 4, 0.90 in round 3, was 95% / 0.35), and model-without-generation answers need p ≥ 0.99.
- *Best guess, clearly marked:* whenever no claim is made but the world layer's top match is a car outside the
  196 classes with p ≥ 0.2, the result shows it as a hedged guess ("MARUTI SUZUKI? · Maybe Grand Vitara? ·
  2022–present ?") instead of a bare "Unknown model".

If the layer is not sure of the model but is sure of the make (summed over that make's models), it
answers e.g. "Maruti Suzuki, model not recognised · closest: Grand Vitara, Fronx, Brezza". All thresholds
come from the tune run's held-out reference photos and unseen-model negatives, never from test: name 0.95
(best +1 right generation / −1 wrong model trade-off), model 0.99 and make 0.95 (lowest with ≥ 99%
precision), route 0.95 (no Stanford val car re-routed), not-sure mass 0.20 (≤ 0.5% of Stanford val doubted).

**Self-learning.** AutoLens learns from its own very confident answers; there is no manual teach step (the
Teach panel is hidden, its API is kept). When a photo is named at generation level from reference photos with
p ≥ 0.97, its SigLIP embedding (YOLO crop + frame) is imprinted into that car's row of the cosine head:
the row direction moves toward the new photo's direction by n / (n + max(n_ref, 5)) and its length is kept.
At most 15 photos per car are learnt, near-copies (cosine > 0.97) are skipped, and the state
(`~/cvh/world_cars/user_teach/auto_state.json`, outside the repository) is tied to the model version, so a
retrained model starts clean. Switch it off with `CVH_SELF_LEARN=0`. An earlier gradient refit of the row
destroyed it (a correct car fell from p 0.36 to 0.002 after one teach); imprinting fixed that. Simulation on
tune-run held-out photos: named 83.7% → 88.0%, top-1 94.0% → 95.3%, wrong-name rate 0.003% → 0.005%; unseen
cars added from 3 photos: top-1 97.6%, named 92.1%, false positives 0.64%.

**Stanford Cars safety checks:** hybrid still **95.90% top-1 / 99.75% top-5** on the 8,041 test images
(the models are untouched). Known cars wrongly re-routed to the open-set path stay at **0.796%**: with
round 3 the override threshold left 0 of the 8,041 Stanford test cars answered as a world car, with round 4
2 of 8,041 (0.02%); the fine-tuned head alone still gets 95.7% top-1 on them. The "not sure" cross-check flags
0.66% of Stanford test cars. On 500 random Stanford test images run end to end, 4 answers changed (0.8%)
and confident-correct answers go from 93.4% to 93.2% (one car now shown as "not sure"). All numbers: `results/world_cars.json`. Reproduce:

```bash
python -m scripts.build_world_catalogue
```

```bash
python -m scripts.fetch_world_photos
```

```bash
python -m scripts.build_world_gallery select
```

```bash
python -m scripts.build_world_gallery embed
```

```bash
python -m scripts.eval_world compare
```

```bash
python -m scripts.train_world crops
```

```bash
python -m scripts.train_world train --run tune
```

```bash
python -m scripts.train_world train --run final
```

```bash
python -m scripts.train_world build
```

```bash
python -m scripts.eval_world e2e
```

(`compare` and `build` need `pip install transformers sentencepiece` for the SigLIP tokenizer; the app
does not, because the text embeddings are precomputed in `models/world_text_siglip.npz`.) Set `CVH_WORLD=0` to switch the layer off.
Training needs a GPU (~1 h for both runs on 8 GB); the app only loads the result
(`models/world_siglip_ft.pt`, 177 MB fp16). Limitations: only 735 generations have reference photos. The
rest of the catalogue can only be guessed from text, so those cars are shown as "Not sure · maybe …".
Generations are labelled from Commons categories, so facelifts are not separated. Commons has few or no free
photos of the newest Indian cars (Thar 2nd gen, Harrier, Punch, Scorpio-N, Venue, Exter), so those come out
as "not sure" with a best guess. FGVD street photos were recorded in 2017–18, so they help with older and
mid-2010s Indian cars (Dzire, Innova, Alto, Indica) but not with the newest ones.

## Repository layout

```
app.py                     FastAPI server: /api/analyze, /api/performance + serves the web UI
web/                       custom frontend (index.html, styles.css, app.js; animations with Motion,
                           the vanilla-JS successor of Framer Motion, vendored in web/vendor/)
app_gradio.py              earlier Gradio UI (kept as a fallback)
start_app.bat              double-click launcher on Windows
src/
  config.py                paths & hyper-parameters
  data.py                  annotation parsing, train/val/test split, crop cache, torch dataset + augmentations
  classical.py             quality analysis, adaptive gamma/CLAHE/bilateral, Canny, contours, classical ROI,
                           HOG / LBP / HSV / edge-orientation / geometric features
  detector.py              YOLO11 wrapper with classical-ROI fallback, IoU
  classifier.py            ConvNeXt wrapper (PyTorch FP16 or ONNX INT8 backend), flip TTA
  clip_classifier.py       robust fine-tuned CLIP ViT-B/16 classifier (ensemble member)
  openvocab.py             CLIP vehicle gate ("No car detected") + zero-shot naming over 954 models
  pipeline.py              end-to-end hybrid pipeline, fusion, hierarchical low-confidence handling
  worldcars.py             world-car layer (extra): reference-photo probe + zero-shot over 3,668 models
  metrics.py               top-k, AP / mAP (COCO 101-pt), class-aware mAP, ECE
scripts/
  prepare_data.py          parse .mat annotations, split, cache crops, build YOLO dataset
  train_classifier.py      transfer learning of ConvNeXt-T (ImageNet-22k) on 196 classes
  train_clip.py            fine-tune CLIP ViT-B/16 with harsh lighting/blur/JPEG/viewpoint/occlusion augmentation
  eval_openset.py          vehicle-gate + unseen-model evaluation, tunes the open-set thresholds
  eval_unknown.py          make-first "responsible answer" rule for unfamiliar cars, tunes its thresholds
  train_detector.py        fine-tune YOLO11s on Stanford Cars boxes
  run_detection.py         IoU / mAP of fine-tuned YOLO vs COCO YOLO vs classical ROI on the test split
  train_classical.py       classical-feature classifier + feature-group ablation
  evaluate.py              comparison, ablation, robustness, thresholds, failure cases, tunes hybrid config
  optimize.py              FP16, ONNX INT8 static quantisation, global pruning - size / latency / accuracy
  write_report.py          generates eval_P5_<TeamID>.md
  run_all.py               runs every step in order
  build_world_catalogue.py / fetch_world_photos.py / build_world_gallery.py / eval_world.py
                           world-car layer: catalogue, Commons photos, embeddings, comparison + tuning
models/                    deployable weights (classifier_fp16.pt, yolo_cars.pt, classical_logreg.joblib,
                           classifier_int8.onnx, hybrid_config.json)
results/                   JSON metrics, figures, failure galleries
docs/                      design document, architecture diagram (+ script that draws it)
```

## Quick start (inference / app only)

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Then open http://127.0.0.1:7860. Or with Docker:

```bash
docker build -t p5-cars . && docker run -p 7860:7860 p5-cars
```

Set `CVH_BACKEND=onnx` to use the INT8-quantised ONNX classifier on CPU.

## Reproducing the results

1. Get the Stanford Cars dataset (Kaggle mirror `cars_train/`, `cars_test/`, `car_devkit/`) and the
   test labels file `cars_test_annos_withlabels.mat` (the Kaggle devkit has no test labels; the file is
   mirrored e.g. in the Hugging Face dataset `iharabukhouski/stanford_cars`). Extract into
   `~/cvh/data/raw/` (or set `CVH_WORK_DIR`) so that you have:
   ```
   ~/cvh/data/raw/cars_train/cars_train/*.jpg
   ~/cvh/data/raw/cars_test/cars_test/*.jpg
   ~/cvh/data/raw/car_devkit/devkit/{cars_meta,cars_train_annos,cars_test_annos_withlabels}.mat
   ```
2. Install a CUDA build of PyTorch, then the requirements:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   ```bash
   pip install -r requirements.txt
   ```
3. Run everything (about 2.5 h on an RTX 5060 Laptop GPU):
   ```bash
   python -m scripts.run_all
   ```
   ```bash
   python -m scripts.write_report --team T80 --usns "1RUA24CSE0200,1RUA24CSE0206,1RUA24CSE0130"
   ```

Seeds are fixed (`SEED=42`); the validation split is a stratified 10% of the official training set, and
the official 8,041-image test split is used only for final evaluation.

## Deploying the public URL (Modal)

The live demo runs on [Modal](https://modal.com) as a CPU container (2 cores, 6 GB RAM), defined in
`modal_app.py` (image with CPU PyTorch + requirements, CLIP weights pre-fetched at build time).

1. Install the client and sign in once (browser-based; the token is stored locally, never in the repo):
   ```bash
   pip install modal
   ```
   ```bash
   modal token new
   ```
2. Deploy (prints the public https URL; first build about 4 minutes):
   ```bash
   modal deploy modal_app.py
   ```

A warm container answers a scan in about 1-4 s. After 20 minutes without traffic it scales to zero and the
next request takes up to about 2.5 minutes to start, so keep one container warm during judging:
```bash
KEEP_WARM=1 modal deploy modal_app.py
```
(Windows PowerShell: `$env:KEEP_WARM="1"; modal deploy modal_app.py`.) The same app also runs anywhere
Docker runs (`Dockerfile`), e.g. a Hugging Face Space with `python -m scripts.deploy_hf`.

## Limitations and ethics (summary - details in the design document)

- Make, model and year are learned only for the 196 Stanford Cars classes (mostly US-market cars,
  1991-2012). Other cars are flagged as unseen (about 80% of the time) and answered with a catalogue name
  only when CLIP is sure, otherwise with the make or "Unknown car", never with a year. Unflagged unseen
  cars get the nearest known class.
- Year prediction is the weakest part: many generations look identical across years.
- One primary vehicle per image is classified; crowded scenes are out of scope.
- No number-plate recognition, person identification or tracking is done, and the system must not be
  used for surveillance of individuals. Dataset images are from the web and keep their original licences.
