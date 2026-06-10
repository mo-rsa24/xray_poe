# 22 · Heart-Size & Blunting Extractors

## Reference while you do it
- 📄 Plan: plans/metrics-extractors/plans/04-extractors.md

## Section context (paste into the Todoist section)
**Description:** Build extractors measuring heart size (CTR-like) and pleural blunting/fluid — the features whose *joint* Exp6 tests.
**Objective:** Make the coupling (heart size and fluid rising together with severity) measurable; Exp6 is untestable without it.
**Goal:** Two scalar extractors per image + a named definition of the coupling.
**Verify (whole leaf):** `python -m metrics.extractors --image <real both-disease>` → prints (heart_size, blunting) in sensible ranges.

## Tasks (one at a time)
- [ ] Implement the heart-size extractor
- [ ] Implement the pleural-blunting/fluid extractor
- [ ] Name the coupling the joint must capture (heart size and fluid rising together with severity)
