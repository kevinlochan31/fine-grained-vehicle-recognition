# AutoLens: video script (3 presenters, about 8 minutes)

**Presenters**
- **KEVIN** (Kevin Isaac, 1RUA24CSE0200): hybrid pipeline, web app, deployment. Opens and closes.
- **KOKILA** (Kokila C G, 1RUA24CSE0206): dataset and classical computer vision.
- **DHRUTHI** (Dhruthishree V, 1RUA24CSE0130): detector, deep-learning classifiers, optimisation.

Swap the roles if your real split is different. Judges ask each person about *their own* part.

**Before recording**
- Have the site open (localhost:7860 or the live link https://kevinisaacbtech24--autolens.us-east.modal.run), scrolled to the top.
- Have these photos ready in a folder: a flower photo (the tulips), a 2024 Aston Martin, the older Toyota Camry front view, and one dark or low-quality car photo.
- Use a 1920×1080 screen recording at normal zoom. Move the mouse slowly; the page animations react to it.
- Each section below gives the time, the speaker, **[ON SCREEN]** for what to show, and the words to say.

---

## 0. Opening (0:00 – 0:35) — KEVIN

**[ON SCREEN]** The AutoLens home page loading: the boot screen, then the big headline flying in letter by letter. Move the mouse over the headline so the letters react.

**KEVIN:**
"Hi, we're Kevin, Kokila and Dhruthishree, and this is **AutoLens**, our solution for Problem Statement 5: intelligent fine-grained vehicle recognition.

You give it any photo. It first checks there really is a car, draws a box around it, and names its **make, model and year** from 196 classes of the Stanford Cars dataset, with a confidence score.

What makes it different is that it **knows when it doesn't know**. If it's not a car, it says so. If it's a car it was never trained on, it tells you that instead of guessing.

It combines classical computer vision with deep learning, and we'll show you each part, the numbers, and where it fails."

**[ON SCREEN]** Scroll slowly down past the stats: 95.9% top-1, 99.7% top-5, mAP 0.989, 150 ms.

---

## 1. The problem and the dataset (0:35 – 1:25) — KOKILA

**[ON SCREEN]** The scrolling ticker of car names. Then the "Feed the lens" section.

**KOKILA:**
"The dataset is **Stanford Cars**: 16,185 images, 196 classes, 49 makes, model years 1991 to 2012.

We used the official split:
- **8,041 test images**, which we never touched until the final evaluation;
- the 8,144 training images, split into **7,356 for training** and a stratified **788 for validation**.

Every threshold and weight you'll see was tuned on that validation set only.

Why is this hard? It's **fine-grained**. Many classes are almost the same car: an Audi S5 versus an A5, or a Dodge Caliber from 2007 versus 2012. And we only have about **42 training photos per class**. So a normal image classifier isn't enough. You need good localisation, robust features, and honesty about uncertainty."

---

## 2. Live demo: a normal car (1:25 – 2:30) — KEVIN, with KOKILA

**[ON SCREEN]** Click the **green Lamborghini** test image. Let the full scan play: the scanning beam, the box corners flying in, "TARGET LOCKED", the shockwave, then the result card with confetti.

**KEVIN:**
"Let's scan a test image the model has never seen. Behind the scenes, three pipelines run at once:
- **classical only**;
- **deep learning only**;
- our **hybrid**.

The hybrid found the car, drew this box, and identified a **Lamborghini Gallardo LP 570-4 Superleggera, 2012**, with very high confidence. That's above our 67% acceptance threshold, so it's marked **Confident**."

**[ON SCREEN]** Point at the top-5 list, then scroll to the image-quality telemetry on the left.

**KOKILA:**
"Before anything else, the classical side **measures the image**: brightness, contrast, a noise estimate, sharpness and resolution. Those numbers decide what enhancement runs. A dark photo gets gamma and CLAHE, and a noisy one gets edge-preserving bilateral filtering. A clean photo like this one gets nothing, so we never damage good images."

**[ON SCREEN]** Scroll to **"Classical vs deep vs hybrid, on your photo"** (the three cards).

**KEVIN:**
"Here's the brief's comparison on this one photo:
- **Classical** guessed a Spyker, which is wrong, with low confidence.
- **Deep learning** got it right.
- **Hybrid**, our final system, agrees and is the most confident."

**[ON SCREEN]** Scroll to the **"Inside the classical pipeline"** images and click one to enlarge it.

**KOKILA:**
"These are the intermediate image-processing steps the brief asks for:
- the detected region;
- the enhanced version;
- Canny edges, with thresholds set automatically from the image's median brightness;
- Sobel gradients;
- LBP texture;
- contours with our classical region of interest.

Our classical classifier uses **HOG for shape, LBP for texture, colour histograms and geometry**, then PCA and logistic regression."

---

## 3. Live demo: bad input and unseen cars (2:30 – 3:40) — KEVIN

**[ON SCREEN]** Upload the **tulips** photo.

**KEVIN:**
"Now bad input. Our first version called this photo of tulips a Lamborghini. A normal classifier *has* to pick one of its 196 classes.

We added a **vehicle gate** using CLIP, a vision-language model. It compares the image with descriptions of cars and of everyday non-car scenes, and borderline cases also need our car detector to agree. Result: **'No car detected'**.

On the test set it accepts 99.9% of real cars, and it rejected all 16 non-car photos we tried."

**[ON SCREEN]** Upload the **2024 Aston Martin**.

**KEVIN:**
"Next, a car that isn't in the dataset at all: a 2024 Aston Martin. The newest Aston Martin in Stanford Cars is from 2012.

Instead of forcing a wrong label, the system detects that it's **outside the 196 classes** and answers **make first**: **'Aston Martin, model not recognised'**. It still shows its best guess, the Vantage, but clearly marks it as unconfirmed."

**[ON SCREEN]** Upload the **older Toyota Camry** front photo.

**KEVIN:**
"One more: a Camry, but an *older generation* than the 2012 Camry in the dataset, photographed straight from the front.

The classifier's best guess is 'Camry 2012' at 54%, below our threshold. So instead of claiming the wrong year, it says **'Toyota, model not recognised'** and lists the Camry as the closest guess.

On simulated unseen cars, this make-first rule cut wrong answers from **66% to 35%** for cars outside our catalogue."

---

## 4. How it works: the deep learning (3:40 – 4:50) — DHRUTHI

**[ON SCREEN]** Scroll to the **Pipeline** section and let the timeline fill as you scroll. Then show the architecture diagram.

**DHRUTHI:**
"Here's the full pipeline. On the deep-learning side there are three trained models.

First, **detection**. We fine-tuned **YOLO11s** on the Stanford bounding boxes, so it learned to box the *main* car tightly. It reaches **mAP@0.5 of 0.989** and a mean IoU of **0.954**. The off-the-shelf COCO model only gets 0.854 mAP, because it also boxes background cars.

Second, classification on the **detected crop**, with two networks:
- **ConvNeXt-Tiny**, pre-trained on ImageNet-22k and fine-tuned at 384 pixels;
- a **CLIP ViT-B/16**, pre-trained on two billion image-text pairs. We fine-tuned it with very harsh augmentation: darkness, over-exposure, blur, JPEG damage, noise, tilted angles and blocked-out regions, so it survives real-world photos.

To stop overfitting with only 42 images per class we used:
- transfer learning;
- label smoothing and weight decay;
- layer-wise learning-rate decay, so CLIP's general knowledge isn't destroyed;
- choosing the best checkpoint on validation.

Finally, **temperature scaling** calibrates the confidence, so 90% really means about 90% correct."

**[ON SCREEN]** Point at the hybrid stages in the diagram.

**DHRUTHI:**
"The **hybrid** layer brings it together:
- the classifiers see the detected region, not the whole photo;
- the classically enhanced image is classified as a second view and averaged with the original;
- we add flip test-time augmentation;
- the two networks are combined 30/70 in a calibrated ensemble.

Then come the confidence back-off and the open-set check Kevin showed."

---

## 5. Results (4:50 – 6:00) — DHRUTHI and KOKILA

**[ON SCREEN]** Scroll to **Results**. Let the numbers flip in and the bars spring out. Hover a bar to show its tooltip.

**DHRUTHI:**
"Everything is measured on the 8,041 test images. The three approaches:
- **classical**: 14.5% top-1, about 28 times better than chance, but not enough for 196 look-alike classes;
- **deep learning**: 94.5%;
- **hybrid**: **95.9% top-1 and 99.75% top-5**.

Our ablation shows where the gain comes from. Classifying the detected crop instead of the whole image is worth over 4 points, and the robust CLIP model adds 1.3 points."

**[ON SCREEN]** Scroll to **Robustness under degradation**.

**KOKILA:**
"For robustness, we re-tested 1,000 test images under eleven damaged conditions. The hybrid beats deep-only in ten of them:
- at **8× lower resolution**: 75.8% → **82.7%**;
- **very dark and noisy**: 89.7% → **93.0%**;
- **15° rotation**: 92.4% → **94.8%**.

The one exception is 30% occlusion, about one point lower, which is within statistical noise. We report it honestly."

**[ON SCREEN]** Scroll to **Low-confidence handling** and the **optimisation** table.

**DHRUTHI:**
"For confidence, we chose the threshold on validation for 98.5% precision. At that threshold, 95.7% of test images get a full answer that's 97.6% accurate. The rest fall back to make-only or 'uncertain'.

For efficiency, we applied three optimisations:
- **FP16** halves the model to 53 MB with no accuracy loss, and that's what we deploy;
- **INT8 quantisation** makes it 3.9× smaller for only 0.14 points;
- **pruning without retraining** wasn't worth it, and we show why.

The whole request takes about 150 milliseconds on a GPU."

---

## 6. Failure cases and limits (6:00 – 7:00) — KOKILA

**[ON SCREEN]** Show the design document §9 failure table, or the `results/failures/` images.

**KOKILA:**
"We documented nine failure cases. Three important ones:
1. **Year confusion.** For example the Dodge Caliber 2007 versus 2012: the body is almost identical, and the only differences are small lamp details. That's why the system can answer 'year uncertain'.
2. **Label noise in the dataset.** Many 'confident but wrong' answers are actually *correct*, and the test label is wrong. For example, a photo labelled Jeep Compass is clearly a Grand Cherokee, which is exactly what we predicted.
3. **Unseen cars.** The dataset ends in 2012 and is mostly US cars, so about one in five unseen models still gets matched to the nearest trained class.

Our limits: year is only learned for the 196 classes, and only one main car per photo is classified. We don't do number plates or tracking, and photos are never stored."

---

## 7. Web app, deployment and close (7:00 – 8:00) — KEVIN

**[ON SCREEN]** Scroll back to the top. Show the app on a phone, or switch the browser to mobile view. Then show the GitHub repo page.

**KEVIN:**
"The app is a FastAPI backend with a custom web interface. It has loading, error, 'no car' and 'unknown model' states, it handles bad files and huge uploads safely, and it works on phones. The animations are built with Motion, the successor of Framer Motion.

It's deployed on Modal as a cloud container, and there's also a Dockerfile. The GitHub repository has the full source code, the architecture diagram, the design document, our evaluation file with all metrics in the brief's order, and the ablation and failure analysis. One command, `python -m scripts.run_all`, reproduces every number.

To sum up: AutoLens finds the car, measures and repairs the image, names it at **95.9% top-1**, and when it isn't sure, it tells you instead of guessing. Thank you!"

**[ON SCREEN]** End on the giant AUTOLENS wordmark at the bottom, moving the mouse across it so the letters hop.

---

## 3-minute version (for the live demo in the viva)

| Time | Who | What |
|---|---|---|
| 0:00–0:20 | Kevin | One-sentence intro + "it knows when it doesn't know" |
| 0:20–1:10 | Kevin | Scan the Lamborghini → box, result, confidence; show the classical vs deep vs hybrid cards |
| 1:10–1:40 | Kokila | Image-quality telemetry + intermediate classical steps |
| 1:40–2:15 | Kevin | Tulips → "No car detected"; 2024 Aston Martin → "model not recognised" |
| 2:15–2:45 | Dhruthi | Results: 95.9 / 99.75, mAP 0.989, robustness at 8× low-res 75.8 → 82.7, FP16 / INT8 |
| 2:45–3:00 | Kokila | One failure case (label noise or year confusion) + close |

---

## Likely viva questions and who answers

| Question | Who | Short answer |
|---|---|---|
| Why is classical accuracy so low? | Kokila | 196 near-identical classes; hand-crafted features capture shape and texture but not badge-level detail. HOG is the strongest group (11.4% alone). |
| How did you prevent overfitting? | Dhruthi | Transfer learning, label smoothing, weight decay, stochastic depth, layer-wise LR decay, heavy augmentation, checkpoint chosen on validation. |
| Did you use the test set for tuning? | Dhruthi | No. Everything was tuned on the 788-image validation split; the test set was only used for final numbers. |
| What is IoU / mAP? | Dhruthi | IoU = overlap ÷ union of the predicted and true box; mAP = average precision over recall levels, at IoU 0.5 and averaged over 0.5–0.95. |
| Why does the year go wrong? | Kokila | Generations look the same; the differences are tiny lamp or bumper details, so we back off to "year uncertain". |
| How does it detect "no car"? | Kevin | A CLIP vehicle gate (car vs non-car text prompts) plus agreement from the YOLO detector for borderline scores. |
| What happens with a car not in the dataset? | Kevin | The open-set check flags it; the answer is make first, and a model is named only when CLIP is at least 80% sure. |
| What optimisation did you apply? | Dhruthi | FP16 (deployed, no loss), INT8 static quantisation (3.9× smaller, −0.14 points), pruning (tested, not worth it without retraining). |
| What does the hybrid add over deep learning? | Dhruthi / Kevin | +1.4 points on clean images and much more on hard ones, calibrated confidence, the no-car gate and unseen-car handling. |
