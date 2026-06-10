# 🎨 EDA Visualization Set

## Description
Produce the set of EDA figures — label-frequency + co-occurrence distributions, a 19×19
heatmap with treatment/control pairs ringed, sample-image grids, and intensity histograms.

## Purpose
The figures both sanity-check the data and seed the paper's data section. (Candidate to
`decompose-plan` into an `eda/visualizations/` sub-scope if it grows.)

## Recommended skill
`/visualize-data-samples` + `/data-distributions` — sample grids + distribution figures.

## Goal
A saved figure set covering distributions, the ringed heatmap, sample grids, and intensity histograms.

## Tasks
- [ ] ⚠️ Distribution plots (label frequency, co-occurrence)
- [ ] ⚠️ 19×19 heatmap with treatment + control pairs ringed
- [ ] ⚠️ Sample-image grid spanning normal / treatment / control conditions
- [ ] ⚠️ Intensity histograms (pre/post preprocessing)

## Engagement Instructions
```
$ python -m eda.visualize --manifest data/manifest.parquet --out figures/eda/
# expect: figures/eda/ populated with distributions.png, heatmap.png, samples.png, intensity.png
```
