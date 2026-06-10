# 07 · EDA Visualization Set

## Reference while you do it
- 📄 Plan: plans/eda/plans/05-visualizations.md

## Section context (paste into the Todoist section)
**Description:** Produce the EDA figure set — distributions, ringed 19×19 heatmap, sample grids, intensity histograms.
**Objective:** Sanity-check the data and seed the paper's data section.
**Goal:** A saved figure set (distributions, heatmap, sample grids, intensity hists).
**Verify (whole leaf):** `python -m eda.visualize --manifest data/manifest.parquet --out figures/eda/` → figures/eda/ populated.

**▶ Recommended prompt:** `/visualize-data-samples` + `/data-distributions` — sample grids + distribution figures.

## Tasks (one at a time)
- [ ] Distribution plots (label frequency, co-occurrence)
- [ ] 19×19 heatmap with treatment + control pairs ringed
- [ ] Sample-image grid spanning normal / treatment / control conditions
- [ ] Intensity histograms (pre/post preprocessing)
