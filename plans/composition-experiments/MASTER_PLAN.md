# 🧪 Composition Experiments

## Mission
Implement PoE composition with both anchors and run the inference-only
experiments (marginals, joint-structure, reweighting, floor baseline) that test
H1, H2, H2-control, H3, and H-anchoring against the floor — the paper's headline.

## Objectives
1. Implement PoE composition (score addition) with `∅` and `normal` anchors and weighting.
2. Exp5 marginals check (H1 gate) — single-disease composed vs real.
3. Exp6 joint-structure test (H2 · H2-control · H-anchoring) — three arms vs the floor.
4. Exp7 reweighting control (H3) — weight grid.
5. Exp8 floor baseline (sanity) — PoE vs naive overlay.

## Goals
1. PoE implementation produces composed samples for both pairs and both anchors.
2. H1 gate cleared (presence within 5 points, two-sample ≤ 0.60).
3. H2 / H2-control / H-anchoring decisions recorded vs the floor.
4. H3 decision recorded; Exp8 PoE-beats-overlay decision recorded.

## Expected Outcome
The headline result — treatment joint broken, control preserved, anchoring effect
reported, structural confirmation — all filed against the pre-registration table.

## Definition of Done
1. PoE composition implemented and unit-checked (score addition = product); both
   anchors selectable.
2. Exp5 run with ≥ 2000 samples/disease; H1 decision with CIs.
3. Exp6 three arms run (treatment/`∅`, control/`∅`, treatment/`normal`); two-sample +
   distribution distance on the joint; decisions vs the floor's 95% upper bound.
4. Exp7 weight grid {0.5, 0.75, 1.0, 1.5, 2.0}² run; best-on-validation; H3 decision.
5. Exp8 PoE vs overlay (presence + FID); decision.
6. Corroboration / anti-corroboration figures saved; every outcome filed against
   the pre-registration table.

## Sub-Scopes
(none yet — added by decompose-plan)

## Plans
- ⚠️ 01-poe-implementation.md
- ⚠️ 02-marginals-check.md
- ⚠️ 03-joint-structure-test.md
- ⚠️ 04-reweighting-control.md
- ⚠️ 05-floor-baseline.md
