# 🚪 Marginals Check (H1 gate)

## Background
Experiment 5 — the gate before the headline. Tests H1.

## Description
Compare single-disease composed images vs real single-disease images on presence rate and
the single-disease feature (heart size or blunting); also validate the extractor here.

## Purpose
If marginals aren't preserved, Exp6 can't be read as a joint result — the experts are too weak.

## Goal
H1 decision — presence within 5 points and two-sample ≤ 0.60 for both diseases.

## Tasks
- [ ] ⚠️ Generate ≥ 2000 samples/disease; measure presence rate + single-disease feature
- [ ] ⚠️ Compare to real single-disease; compute two-sample with CIs
- [ ] ⚠️ Record H1 decision (supported / gate trips / inconclusive)

## Engagement Instructions
```
$ python -m experiments.marginals --n 2000
# expect: presence within 5 pts of real, two-sample <= 0.60 -> H1 supported
```
