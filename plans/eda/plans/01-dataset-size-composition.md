# 📊 Dataset Size & Composition

## Description
Quantify how big the corpus is and how it splits — total images, predefined splits,
per-label counts, and how many images carry two disease labels.

## Purpose
Establishes the sample budget; the both-disease totals drive the Exp6 floor power check.

## Recommended skill
`/data-inventory data/<dataset>` — auto-profiles size, splits, dtypes, per-label counts.

## Goal
A composition table: total N, split sizes, per-label counts, both-disease counts.

## Tasks
- [ ] ⚠️ Load the NIH ChestX-ray14 manifest (`data/nih/`); cross-check against data-foundation/plans/04 → `data/nih/four_group_counts.md`; report total images and any splits <!-- tid:6gpg2x7h258vvh62 -->
- [ ] ⚠️ Compute per-label counts across all 14 NIH labels (+ No Finding)
- [ ] ⚠️ Compute both-disease counts for candidate pairs

## Engagement Instructions
```
$ python -m eda.composition --manifest data/manifest.parquet
# expect: printed table of label -> count, split -> count, and both-disease totals
```
