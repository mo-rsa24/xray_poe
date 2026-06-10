# 19 · Two-Sample (C2ST) + MMD

## Reference while you do it
- 📄 Plan: plans/metrics-extractors/plans/01-two-sample-mmd.md

## Section context (paste into the Todoist section)
**Description:** Implement the two-sample test (C2ST AUC classifier) and MMD distribution distance — the core "same distribution?" metrics.
**Objective:** Power every composition claim (Exp5/6/7): 0.5 = indistinguishable, 1.0 = trivially separable.
**Goal:** C2ST + MMD implemented and sanity-checked on identical and separable pairs.
**Verify (whole leaf):** `python -m metrics.c2st --a real_a/ --b real_a/` → AUC ~0.5; `--a real/ --b noise/` → AUC ~1.0.

## Tasks (one at a time)
- [ ] Implement C2ST (train classifier generated-vs-real, report held-out AUC)
- [ ] Implement MMD on a feature space
- [ ] Sanity-check: ~0.5 on identical sets, ~1.0 on obviously different sets
