# 08 · Preprocessing — Vision Pipeline

## Reference while you do it
- 📄 Plan: plans/eda/plans/06-preprocess-vision.md

## Section context (paste into the Todoist section)
**Description:** Build/apply the image preprocessing pipeline per the "Image Properties" decisions: resize, intensity normalization, clip/window, optional augmentation; emit a reusable config.
**Objective:** Turn preprocessing decisions into a reproducible pipeline the VAE/LDM consume.
**Goal:** A preprocessing config + a transformed-sample preview matching the decided resize/normalize/clip.
**Verify (whole leaf):** preprocessed samples match the decided resize/normalize/clip; config saved to data/<dataset>/eda/.
**▶ Recommended prompt (inferred skill — ⚠️ not installed):** `/preprocess-vision data/<dataset>` — inferred: applies the vision preprocessing pipeline (resize, normalize, clip/window, augment) and writes config.

## Tasks (one at a time)
- Apply resize / normalize / clip per the Image-Properties decisions
- Preview a batch of transformed samples
- Save a reusable preprocessing config
