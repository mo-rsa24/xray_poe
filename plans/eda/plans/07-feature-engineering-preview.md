# 🧪 Feature-Engineering Preview

## Background
EDA-workflow skill: `/feature-engineering-preview` (⚠️ not yet installed). Precursor to the
heart-size & blunting extractors built in `metrics-extractors`.

## Description
Preview candidate engineered features derivable from the images before committing — feature
ideas plus quick distributions.

## Purpose
De-risk the feature extractors by checking what is reliably measurable on this data before
building them.

## Goal
A shortlist of candidate features with quick distributions, flagging those that will become
metrics-extractors extractors.

## Tasks
- [ ] ⚠️ List candidate features measurable from the images (e.g. heart-size proxy, costophrenic-angle/blunting proxy)
- [ ] ⚠️ Compute quick distributions for each candidate
- [ ] ⚠️ Flag the candidates that will become metrics-extractors extractors

## Engagement Instructions
```
$ /feature-engineering-preview data/<dataset>   # inferred skill — ⚠️ not installed; or do manually
# expect: candidate-feature preview with distributions saved to figures/eda/
```
