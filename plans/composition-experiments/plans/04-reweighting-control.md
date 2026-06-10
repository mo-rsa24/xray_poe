# 🎛️ Reweighting Control (H3)

## Background
Experiment 7 — structural, or just a knob?

## Description
Vary the two composition weights over a grid and pick the best on a validation split,
testing whether any weighting closes the Exp6 gap.

## Purpose
Weights can rescale how loud each disease is but can't invent the interaction features
neither expert ever saw; if the gap survives the best weights, it is structural.

## Goal
H3 decision — best weights still ≥ 0.65 (structural) or ≤ 0.55 (just imbalance).

## Tasks
- [ ] ⚠️ Sweep (w₁, w₂) over {0.5, 0.75, 1.0, 1.5, 2.0}²; choose best on validation
- [ ] ⚠️ Re-test the joint at the best weights; record the H3 decision

## Engagement Instructions
```
$ python -m experiments.reweight --grid 0.5,0.75,1.0,1.5,2.0
# expect: best-on-val weights; joint score still >= 0.65 -> H3 structural
```
