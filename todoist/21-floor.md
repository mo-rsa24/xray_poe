# 21 · The Floor

## Reference while you do it
- 📄 Plan: plans/metrics-extractors/plans/03-floor.md

## Section context (paste into the Todoist section)
**Description:** Split the real both-disease set in half and measure the halves against each other (real-vs-real) with CIs — the smallest gap measurable.
**Objective:** Provide the reference all Exp6/7 scores are judged against.
**Goal:** A floor value + 95% bounds per pair; a power flag if N is small.
**Verify (whole leaf):** `python -m metrics.floor --pair cardiomegaly,effusion` → floor ~0.5x with 95% upper bound; N + power flag.

## Tasks (one at a time)
- [ ] Split real both-disease set in half; compute two-sample + MMD between halves
- [ ] Bootstrap to get 95% bounds; flag if N small
