# 03 · Dataset Size & Composition

## Reference while you do it
- 📄 Plan: plans/eda/plans/01-dataset-size-composition.md

## Section context (paste into the Todoist section)
**Description:** Quantify total images, splits, per-label counts, and both-disease counts from the manifest.
**Objective:** Establish the sample budget; both-disease totals drive the Exp6 floor power check.
**Goal:** A composition table: total N, split sizes, per-label counts, both-disease counts.
**Verify (whole leaf):** `python -m eda.composition --manifest data/manifest.parquet` → printed label/split/both-disease tables.

**▶ Recommended prompt:** `/data-inventory data/<dataset>` — auto-profiles size, splits, dtypes, per-label counts.

## Tasks (one at a time)
- [ ] Load the manifest; report total images and any train/val/test splits
- [ ] Compute per-label counts across all 19 labels
- [ ] Compute both-disease counts for candidate pairs
