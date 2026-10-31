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

22 clips total, split **by clip** (not by frame) 80/20 into train/validation
to avoid data leakage between near-duplicate adjacent frames of the same clip.

## Model

**MobileNetV2**, pretrained on ImageNet, fine-tuned via transfer learning
for binary classification (frozen-base training, then fine-tuning the top
layers at a low learning rate). Full rationale is written out in the
notebook, Section 9.

## Results (held-out validation set)

- **Frame-level accuracy: 98.00%** (98/100 frames)
- **Clip-level accuracy: 100%** (4/4 held-out clips correct, majority vote)

Confusion matrix (rows = true, cols = predicted):

|  | Predicted Active | Predicted Inactive |
|---|---|---|
| **True Active** | 75 | 0 |
| **True Inactive** | 2 | 23 |

|  | precision | recall | f1-score | support |
|---|---|---|---|---|
| active | 0.97 | 1.00 | 0.99 | 75 |
| inactive | 1.00 | 0.92 | 0.96 | 25 |

## Files

- `CMZ_Phase2_Behavior_Classification.ipynb` — full pipeline notebook, run
  top to bottom in Google Colab (GPU runtime), with all outputs/results
  already included.
- `results_table.csv` — per-frame predictions on the validation set.
- `video_level_table.csv` — per-clip active/inactive table (majority vote).
- `cmz_behavior_classifier.keras` — the trained model.
- `README.md` — this file.

## How to re-run

1. Open the notebook in Google Colab.
2. `Runtime > Change runtime type` → GPU (T4).
3. `Runtime > Run all`. Downloads the dataset itself, no manual setup.

## Submission checklist

- [x] Train the model
- [x] Explain the AI/image model selected (notebook, Section 9)
- [x] Design and evaluate the output — active/inactive table +
      confusion matrix + accuracy (above, and notebook Sections 7-8)
- [x] GitHub link — this repo
