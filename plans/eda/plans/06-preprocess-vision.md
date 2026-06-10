# ❌ Preprocessing — Vision Pipeline (DECOMMISSIONED 2026-06-09)

**Decommissioned:** superseded by `data-foundation/06-shared-preprocessing.md`, which now
owns the canonical shared stretch-512 preprocessing pipeline (`data/preprocess.py`) and the
reusable config the VAE/LDM consume. EDA no longer builds its own pipeline —
`03-image-properties-preprocessing.md` validates the adopted 512² contract against the data,
and `05-visualizations.md` covers transformed-sample previews. Live Todoist task
"08 · Preprocessing — Vision Pipeline" deleted.

---

_Original content (kept for history):_

## Background
Applies the preprocessing decisions from `03-image-properties-preprocessing.md`.
EDA-workflow skill: `/preprocess-vision` (⚠️ not yet installed — run manually or install).

## Description
Build and apply the image preprocessing pipeline — resize to target, intensity
normalization, clip/window, optional augmentation — and emit a reusable config the VAE/LDM
consume.

## Purpose
Turn the preprocessing decisions into a single reproducible transform, so every downstream
model sees identically prepared images.

## Goal
A preprocessing config plus a transformed-sample preview matching the decided resize/normalize/clip.

## Tasks
- [ ] ⚠️ Apply resize / normalize / clip per the Image-Properties decisions
- [ ] ⚠️ Preview a batch of transformed samples
- [ ] ⚠️ Save a reusable preprocessing config

## Engagement Instructions
```
$ /preprocess-vision data/<dataset>     # inferred skill — ⚠️ not installed; or run the equivalent manually
# expect: preprocessed samples match decided resize/normalize/clip; config saved to data/<dataset>/eda/
```
