# ✅ Extractor Validation (real vs generated)

## Background
This is the validation the Exp5 marginals gate requires.

## Description
Confirm the heart-size and blunting extractors behave identically on real and generated
images.

## Purpose
If an extractor reads generated images differently from real ones, every joint number in
Exp6 is confounded.

## Goal
Validation showing extractor agreement on real vs generated (two-sample ≤ threshold), with a figure.

## Tasks
- [ ] ⚠️ Run extractors on matched real and generated single-disease sets
- [ ] ⚠️ Compare distributions; require agreement (two-sample ≤ threshold)
- [ ] ⚠️ Save a validation figure

## Engagement Instructions
```
$ python -m metrics.validate_extractors --real real/ --gen gen/
# expect: extractor outputs agree (two-sample <= 0.60); figures/extractor_validation.png
```
