# 20 · FID

## Reference while you do it
- 📄 Plan: plans/metrics-extractors/plans/02-fid.md

## Section context (paste into the Todoist section)
**Description:** Implement FID for single-disease and overlay comparisons (Exp4 gate, Exp8 baseline).
**Objective:** Provide a standard realism distance as a second angle alongside C2ST/MMD.
**Goal:** FID implemented and sanity-checked.
**Verify (whole leaf):** `python -m metrics.fid --a real/ --b real/` → near 0; `--a real/ --b noise/` → large.

## Tasks (one at a time)
- [ ] Implement FID (Inception features)
- [ ] Sanity-check on real-vs-real (low) and real-vs-noise (high)
