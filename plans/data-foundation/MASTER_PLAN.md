# 📦 Data Foundation

## Mission
Acquire two image-level PNG sources — NIH ChestX-ray14 (training corpus) and
VinDr-CXR resized-PNG (held-out evaluation set) — and turn them into a clean,
model-ready 512×512 image corpus through a single shared preprocessing pipeline
(stretch-to-square 512 → per-image min–max → [-1,1]), applied identically to both
datasets so no dataset-identity cue leaks into the model. Both sources ship as
already-rendered PNGs, so no DICOM decoding or monochrome handling is needed.
(Guard: if raw VinDr DICOM is ever used instead of the resized-PNG
redistributions, MONOCHROME1 inversion + bit-depth normalization return to scope.)

## Objectives
1. Acquire NIH ChestX-ray14 (open) + VinDr-CXR resized-PNG (Kaggle rules);
   sample-first locally to validate, then bulk-download onto the RunPod volume.
2. Apply one shared preprocessing pipeline to both datasets — stretch-to-square
   512, per-image min–max, [-1,1], single channel — so intensity/aspect handling
   is identical across sources.
3. Derive image-level labels: NIH from `Data_Entry_2017.csv`; VinDr by collapsing
   radiologist bounding boxes → an image-level multi-label vector. Fold both into
   a common label space {cardiomegaly, effusion, pneumothorax, normal}.
4. Build the four-group partition (normals, disease-A-alone, disease-B-alone,
   A+B held-out co-morbid) that the PoE composition is evaluated against.
5. Verify integrity and produce a manifest (path → label(s) → source dataset)
   for downstream EDA and modeling.

## Goals
1. NIH + VinDr label files + samples acquired locally; per-pathology and
   four-group counts captured — including the uncertain VinDr pneumothorax count.
2. Shared preprocessing function validated on a sample panel from BOTH datasets
   (intensity ranges match, no aspect/inversion surprises).
3. Image-level labels extracted for both; common label-space mapping documented.
4. Four-group partition produced; A+B held-out co-morbid set confirmed viable
   (count check).
5. Integrity scan clean (corrupt/truncated = 0 or quarantined + logged); manifest written.

## Expected Outcome
A clean, intensity-consistent 512×512 PNG corpus drawn from two image-level
sources (NIH train + VinDr eval), a single validated preprocessing pipeline,
image-level labels in a common space, and a four-group partition with a viable
held-out co-morbid set — ready for VAE/LDM training, with no DICOM or monochrome
handling downstream.

## Definition of Done
1. NIH + VinDr resized-PNG acquired (sample-first local → bulk on RunPod); disk
   layout documented.
2. Shared preprocessing pipeline (stretch-512 → per-image min–max → [-1,1])
   implemented and validated on a sample panel from both datasets.
3. Image-level labels extracted — NIH from CSV, VinDr from boxes — mapped to the
   common label space; per-pathology + four-group counts recorded.
4. Four-group partition written; held-out A+B co-morbid set viable.
5. Integrity scan run; corrupt/truncated count = 0 or quarantined + logged.
6. Clean corpus written (or on-the-fly loader documented) + manifest
   (path → label(s) → source dataset).

## Sub-Scopes
(none yet — added by decompose-plan)

## Plans
- ⚠️ 00-dataset-acquisition-chain.md
- ⚠️ 01-acquisition-dicom-monochrome.md  (RETIRED — both sources are PNG; no DICOM path)
- ⚠️ 02-integrity-manifest.md  (keep; de-DICOM the wording via goal-reconcile)
- ⚠️ 03-nih-acquisition.md
- ⚠️ 04-four-group-partition.md
- ⚠️ 05-vindr-acquisition.md
- ⚠️ 06-shared-preprocessing.md
