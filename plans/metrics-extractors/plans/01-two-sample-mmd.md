# 🔍 Two-Sample (C2ST) + MMD

## Description
Implement the two-sample test (a small classifier reporting C2ST AUC) and MMD distribution
distance — the core "are these two image sets the same distribution" metrics.

## Purpose
These power every composition claim (Exp5/6/7): 0.5 = indistinguishable (good), 1.0 =
trivially separable (bad).

## Goal
C2ST AUC and MMD implemented and sanity-checked on known-identical and known-separable pairs.

## Tasks
- [ ] ⚠️ Implement C2ST (train classifier generated-vs-real, report held-out AUC)
- [ ] ⚠️ Implement MMD on a feature space
- [ ] ⚠️ Sanity-check: ~0.5 on identical sets, ~1.0 on obviously different sets

## Engagement Instructions
```
$ python -m metrics.c2st --a real_a/ --b real_a/   # expect AUC ~0.5
$ python -m metrics.c2st --a real/ --b noise/      # expect AUC ~1.0
```
