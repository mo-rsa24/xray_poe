# Data Foundation — Todoist leaf folder

Staged by `todoist-bridge` (folder mode) from `plans/data-foundation/` after its
re-aim (DICOM → two-dataset image-level: NIH train + VinDr eval). Each leaf below
becomes one Todoist task; its plan's task lines become the "Do" bullets inside
that task. Run `/todoist-publish` on this folder to write it into Todoist —
mapping the leaves under the existing **📦 Data Foundation** parent and retiring
the DICOM-era `01 · Acquisition + DICOM/Monochrome Decoding`.

Work order: 01 → 05. Acquire both corpora, derive labels/partition, fix the
shared loader, then build the joined manifest. Excludes `00-dataset-acquisition-chain`
(meta planning) and the retired `01-acquisition-dicom-monochrome`.

## Acquisition
- [ ] [01 · NIH ChestX-ray14 acquisition](01-nih-acquisition.md) — ~112k open PNGs to data/nih/ (train corpus)
- [ ] [02 · VinDr-CXR acquisition + image-level labels](02-vindr-acquisition.md) — Kaggle resized-PNG eval set; boxes → image-level labels

## Labels & Preprocessing
- [ ] [03 · Four-group partition + counts](03-four-group-partition.md) — normal / cardio / effusion / both, from the NIH label table
- [ ] [04 · Shared stretch-512 preprocessing pipeline](04-shared-preprocessing.md) — one loader both datasets pass through

## Integrity
- [ ] [05 · Integrity scan + manifest](05-integrity-manifest.md) — readability + quarantine; manifest keyed on source_dataset
