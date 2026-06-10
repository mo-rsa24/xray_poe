# 23 · Extractor Validation

## Reference while you do it
- 📄 Plan: plans/metrics-extractors/plans/05-extractor-validation.md

## Section context (paste into the Todoist section)
**Description:** Confirm the heart-size and blunting extractors behave identically on real and generated images (the Exp5 validation).
**Objective:** Remove the confound — an extractor reading generated differently from real would corrupt every joint number.
**Goal:** Validated agreement on real vs generated (two-sample ≤ threshold), with a figure.
**Verify (whole leaf):** `python -m metrics.validate_extractors --real real/ --gen gen/` → two-sample ≤ 0.60; figures/extractor_validation.png.

## Tasks (one at a time)
- [ ] Run extractors on matched real and generated single-disease sets
- [ ] Compare distributions; require agreement (two-sample ≤ threshold)
- [ ] Save a validation figure
