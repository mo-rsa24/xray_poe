# 27 · Reweighting Control (Exp7 / H3)

## Reference while you do it
- 📄 Plan: plans/composition-experiments/plans/04-reweighting-control.md

## Section context (paste into the Todoist section)
**Description:** Sweep the two composition weights over a grid, best-on-validation, to test whether any weighting closes the Exp6 gap. **Experiment 7, tests H3.**
**Objective:** Show the gap is structural — weights rescale loudness but can't invent unseen interaction features.
**Goal:** H3 decision — best weights still ≥ 0.65 (structural) or ≤ 0.55 (imbalance).
**Verify (whole leaf):** `python -m experiments.reweight --grid 0.5,0.75,1.0,1.5,2.0` → best-on-val weights; joint still ≥ 0.65 ⇒ structural.

## Tasks (one at a time)
- [ ] Sweep (w₁, w₂) over {0.5, 0.75, 1.0, 1.5, 2.0}²; choose best on validation
- [ ] Re-test the joint at the best weights; record the H3 decision
