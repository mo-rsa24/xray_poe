# 🧮 Floor Baseline (PoE vs overlay)

## Background
Experiment 8 — sanity.

## Description
Compare PoE-composed images to a naive overlay (the average of two single-disease images)
on presence and FID.

## Purpose
If PoE can't beat trivial mixing, reconsider the method before any joint claim.

## Goal
A decision that PoE beats overlay on both presence and FID (beyond CIs), or that it doesn't.

## Tasks
- [ ] ⚠️ Build the naive overlay baseline (average two single-disease images)
- [ ] ⚠️ Compare PoE vs overlay on presence + FID

## Engagement Instructions
```
$ python -m experiments.overlay_baseline
# expect: PoE beats overlay on presence AND FID beyond the confidence interval
```
