# 25 · Marginals Check (Exp5 / H1 gate)

## Reference while you do it
- 📄 Plan: plans/composition-experiments/plans/02-marginals-check.md

## Section context (paste into the Todoist section)
**Description:** Compare single-disease composed vs real single-disease on presence rate and the single-disease feature; validate the extractor here. **Experiment 5, tests H1.**
**Objective:** Gate the headline — if marginals aren't preserved, Exp6 can't be read as a joint result.
**Goal:** H1 decision — presence within 5 points and two-sample ≤ 0.60 for both diseases.
**Verify (whole leaf):** `python -m experiments.marginals --n 2000` → presence within 5 pts of real, two-sample ≤ 0.60 ⇒ H1 supported.

## Tasks (one at a time)
- [ ] Generate ≥ 2000 samples/disease; measure presence rate + single-disease feature
- [ ] Compare to real single-disease; compute two-sample with CIs
- [ ] Record H1 decision (supported / gate trips / inconclusive)
