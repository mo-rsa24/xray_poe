# 05 · Image Properties & Preprocessing

## Reference while you do it
- 📄 Plan: plans/eda/plans/03-image-properties-preprocessing.md

## Section context (paste into the Todoist section)
**Description:** Characterize resolution, intensity, and aspect ratio; decide the preprocessing pipeline (resize, normalize, clip).
**Objective:** Fix a justified preprocessing contract for the VAE/LDM from data, not guesswork.
**Goal:** Documented preprocessing decisions backed by property histograms.
**Verify (whole leaf):** `python -m eda.properties --manifest data/manifest.parquet --out figures/properties.png` → histograms + a PREPROCESSING.md.

**▶ Recommended prompt:** `/data-distributions data/<dataset>` — intensity / resolution / aspect distributions to inform the decisions.

## Tasks (one at a time)
- [ ] Histogram resolutions, aspect ratios, and intensity ranges
- [ ] Decide resize target (256² with 128² fallback) and normalization/clipping; document rationale
