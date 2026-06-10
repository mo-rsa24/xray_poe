# 🔗 Correlation Matrix — Pair Selection & Gate

## Background
This is Experiment 1 — labels-only, no GPU, and the project's go/no-go gate.

## Description
Compute the 19×19 label co-occurrence matrix (φ-coefficient / odds ratio) and select the
treatment pair (strong correlation) and control pair (≈ 0) from it.

## Purpose
Triple duty — picks treatment, picks control, and gates the whole project. If no strongly
correlated pair *and* no near-zero pair both exist, the controlled comparison is impossible.

## Goal
A φ/odds-ratio matrix plus a locked treatment pair and control pair, with the go/no-go
decision recorded and both-disease N reported per chosen pair.

## Tasks
- [ ] ⚠️ Compute pairwise φ-coefficient (and odds ratio) over the 19 labels from the label table
- [ ] ⚠️ Identify the strongest pair (treatment) and a near-zero pair (control); confirm cardiomegaly+effusion is strong and verify the control is ≈ 0 in *this* data
- [ ] ⚠️ Record the go/no-go gate decision; report both-disease N per chosen pair

## Engagement Instructions
```
$ python -m eda.correlation --manifest data/manifest.parquet --out figures/correlation_heatmap.png
# expect: heatmap saved; prints top-5 and bottom-5 pairs by |phi|; treatment + control pairs named
# GATE: a strong pair AND a near-zero pair both exist -> go; else pivot
```
