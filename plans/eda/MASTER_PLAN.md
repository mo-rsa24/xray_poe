# 🔍 Exploratory Data Analysis

## Mission
Characterize the corpus before modeling — dataset size, label distributions and
imbalance, image properties, validation of the adopted preprocessing contract, and the
label co-occurrence (correlation) matrix that gates the project and selects the
treatment and control pairs — through a set of targeted visualizations.

## Objectives
1. Quantify dataset size and composition — counts, splits, per-label and both-disease counts.
2. Measure data imbalance — label frequency, rare-class tail, both-disease N per candidate pair.
3. Characterize image properties — resolution(s), intensity distributions, aspect ratios —
   and validate the adopted preprocessing contract (512²/f=4 stretch + per-image min–max,
   owned by `data-foundation/06`) against the data.
4. Compute the correlation matrix (Exp1) — φ / odds-ratio co-occurrence; select
   treatment (strong) and control (≈ 0) pairs.
5. Produce the visualization set — several kinds (distributions, heatmap, sample
   grids, intensity histograms).

## Goals
1. Size + composition reported.
2. Imbalance quantified; both-disease N per candidate pair reported (power check).
3. Adopted preprocessing contract validated against the data (or flagged where challenged).
4. Correlation matrix done; treatment + control pairs locked; go/no-go gate recorded.
5. Visualization set saved.

## Expected Outcome
A clear empirical picture of the corpus — how big, how imbalanced, whether the
adopted preprocessing holds — plus the correlation matrix that gates the project and
fixes the two pairs, all backed by saved figures.

## Definition of Done
1. Dataset size + composition table (counts, splits, per-label, both-disease).
2. Imbalance analysis: label-frequency plot, rare-class tail, both-disease N per candidate pair.
3. Image-property analysis: resolution / intensity / aspect histograms; the adopted 512²
   stretch / per-image min–max contract validated against the data (flagged if challenged).
4. φ / odds-ratio matrix + 19×19 heatmap; treatment + control pairs ringed; go/no-go recorded.
5. Visualization set saved (distributions, heatmap, sample grids, intensity histograms).

## Sub-Scopes
(none yet — added by decompose-plan)

## Plans
- ⚠️ 01-dataset-size-composition.md
- ⚠️ 02-imbalance.md
- ⚠️ 03-image-properties-preprocessing.md
- ⚠️ 04-correlation-matrix.md
- ⚠️ 05-visualizations.md
- ❌ 06-preprocess-vision.md  (DECOMMISSIONED — canonical pipeline now in data-foundation/06; Todoist task deleted)
- ⚠️ 07-feature-engineering-preview.md
