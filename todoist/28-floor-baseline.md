# 28 · Floor Baseline (Exp8)

## Reference while you do it
- 📄 Plan: plans/composition-experiments/plans/05-floor-baseline.md

## Section context (paste into the Todoist section)
**Description:** Compare PoE-composed to a naive overlay (average of two single-disease images) on presence and FID. **Experiment 8, sanity.**
**Objective:** Confirm PoE beats trivial mixing before any joint claim.
**Goal:** A decision that PoE beats overlay on both presence and FID (beyond CIs), or doesn't.
**Verify (whole leaf):** `python -m experiments.overlay_baseline` → PoE beats overlay on presence AND FID beyond the CI.

## Tasks (one at a time)
- [ ] Build the naive overlay baseline (average two single-disease images)
- [ ] Compare PoE vs overlay on presence + FID
