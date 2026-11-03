# CMZ Phase 2 — Civet Behavior Classification (Active vs Inactive)

Digital Image Processing, Chiang Mai University — Phase 2 submission.

## Task

Task 1 from the course brief: **Image-based Behavior Classification**
(`Image Dataset → Labeling → CNN → Behavior Classification`), framed as
binary **Active vs Inactive** classification to match the Phase 2
submission's requested output ("active and inactive table or any
visualization").

## Dataset

The official labeled dataset named in the Phase 1 brief:
`https://drive.google.com/drive/folders/1wHgqAzrEckiyO4wlrSPLZ-hcXGjzKRtd`

| Source folder | Clips | Mapped to |
|---|---|---|
| Eating | 5 | Active |
| Locomotion | 10 | Active |
| Resting | 7 | Inactive |

22 clips total. (The raw, unlabeled camera-1–4 footage shared separately
was **not** used for training — see the notebook's intro for why.)

## Model

**MobileNetV2**, pretrained on ImageNet, fine-tuned via transfer learning
for binary classification. Full rationale is written out in the notebook
(Section 9), covering: why a CNN, why MobileNetV2 specifically, why
transfer learning given only 22 source clips, why frame-level sampling,
and why the binary active/inactive framing.

## Method summary

1. Download the 3 labeled clip folders.
2. Sample 25 evenly-spaced frames per clip → relabel into `active/` and
   `inactive/` image folders.
3. Split **by clip** (not by frame) into train/validation, 80/20, so no
   clip's frames appear in both sets — this avoids inflating validation
   accuracy with near-duplicate adjacent frames.
4. Train a MobileNetV2-based classifier: frozen-base head training, then
   a short fine-tuning pass on the top layers at a low learning rate.
5. Evaluate: confusion matrix, classification report, per-frame results
   table, and a per-clip (majority-vote) active/inactive table.

## Files

- `CMZ_Phase2_Behavior_Classification.ipynb` — the full pipeline, run top
  to bottom in Google Colab (GPU runtime recommended: Runtime > Change
  runtime type > T4 GPU). Takes roughly 10-15 minutes end to end.
- `README.md` — this file.

Running the notebook also produces (saved into the Colab session, not
included here since they only exist after you run it):
- `results_table.csv` — per-frame predictions
- `video_level_table.csv` — per-clip active/inactive table
- `cmz_behavior_classifier.keras` — the trained model

## How to run

1. Open `CMZ_Phase2_Behavior_Classification.ipynb` in Google Colab
   (upload it, or open it from GitHub once pushed).
2. `Runtime > Change runtime type` → GPU (T4).
3. `Runtime > Run all`.
4. The notebook downloads the dataset itself — no manual setup needed.
   (If Google rate-limits the automatic folder download, the notebook's
   markdown cell right after the download step explains the manual
   fallback.)

## Submission checklist

- [x] Train the model
- [x] Explain the AI/image model selected (notebook, Section 9)
- [x] Design and evaluate the output — active/inactive table +
      confusion matrix + accuracy (notebook, Sections 7-8)
- [x] Push this repo to GitHub and submit the link

## Bonus — per-behavior breakdown (beyond the minimum requirement)

Section 11 of the notebook goes further than the required active/inactive output: it
trains a second classifier on the same frames and the same held-out clips to predict
the *specific* behavior (Eating / Locomotion / Resting), not just active vs inactive.
This isn't required for the submission checklist above — it's extra scope to show
more depth and to feed a presentation dashboard. Produces, in addition to the files
above:

- `results_table_3class.csv` — per-frame behavior predictions
- `video_level_table_3class.csv` — per-clip behavior table
- `video_level_combined.csv` — both models' predictions side by side, one row per
  held-out clip (the file a dashboard should read)
- `cmz_behavior_classifier_3class.keras` — the second trained model
