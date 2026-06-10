# 31 · Pre-Registration Outcomes

## Reference while you do it
- 📄 Plan: plans/documentation/plans/03-prereg-outcomes.md

## Section context (paste into the Todoist section)
**Description:** Record each experiment's result against the pre-registration table — claim holds / wrong / anchoring / in-between — with no post-hoc reinterpretation.
**Objective:** Keep the study honest; results can't be narrated after the fact.
**Goal:** A filled pre-registration outcome table, one row per experiment.
**Verify (whole leaf):** `test -f documentation/PREREG.md && grep -A4 -i 'pre-registration' documentation/PREREG.md` → each experiment mapped to one outcome.

## Tasks (one at a time)
- [ ] Copy the pre-registration table from EXPERIMENTS.md; fill the observed outcome per experiment
- [ ] Flag any "in-between" result for rerun, not reinterpretation
