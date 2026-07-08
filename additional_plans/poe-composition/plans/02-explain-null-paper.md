# 📝 Explain-the-null — the paper's lead section (toy control + qualitative overlays + framing)

## Description
Turn the co-presence null into the paper's lead section: a pre-registered composition investigation carried by a
**toy 2×2 positive control** and **paired quantitative + qualitative (overlay) evidence**. Model: the shipped
null is **arm1 (`gtr6kqpk`, yield 0.125)** — the section's numbers and overlays are arm1's; the toy is
model-free. Runs REGARDLESS of the O1 diagnosis (this is the paper whether or not anything improves).

## Purpose
Serves O2 and DoD-2. This is the paper's headline. Its outputs feed Track A's `plans/03-figure-table-paper-ready.md`.

## Goal
A drafted section with: the toy 2×2 (ancestral fails / corrected recovers a known product), a qualitative figure
pack of ≥6 overlay callouts pinned to the CTR×CP scatter clusters, and the framing (flat-sweep-is-multiplicative,
four-locus subtraction, pre-registered two-abstract fork).

## Tasks
- [ ] ⚠️ **Toy 2×2 positive control** — Du et al. `reduce_reuse_recycle` toy re-skinned to the failure geometry
  (factor A = "x large" heart, factor B = "y large" fluid; AND-target = top-right). Two one-factor score models,
  composed with the exact `ε_null + w(ε_A−ε_null) + w(ε_B−ε_null)` operator, sampled ancestrally: show collapse
  to the axes ("A or B") = "big heart or fluid" in a system where the true product is known.
  - [ ] ⚠️ Remedy: if the re-skinned toy doesn't reproduce the axis-collapse, fall back to Du et al.'s original
    8-Gaussian ring setup verbatim and state the analogy gap (existence proof, not anatomy).
- [ ] ⚠️ **Qualitative figure pack** — PSPNet CTR + costophrenic-angle overlays (`scripts/ctr_overlay.py`) and
  GradCAM callouts (`metrics/grad_cam_utils.py`) pinned to CTR×CP scatter clusters (the abstain pile; the "big
  heart, no fluid" corner). Each exemplar carries a reading AND a competing reading (falsifiable annotation).
  - [ ] ⚠️ Remedy: if GradCAM localizes poorly on synthetics (known: detector attends off-heart), demote it to
    "illustration-of-phenotype" per Saporta et al. and let the CTR/CP overlays + numbers carry the mechanism.
- [ ] ⚠️ **Framing prose** — (a) why the sweep is flat (joint yield ≈ cardio × effusion × composition; a
  near-zero composition factor makes w_cardio sweeps flat); (b) the four-locus subtraction (components ruled out
  by Exp-09, operator + sampler adjudicated by the toy/O3, support the residual); (c) the pre-registered
  two-abstract fork (Abstract-N support-limited / Abstract-M sampler-fixable), reading rule fixed in advance.
- [ ] ⚠️ **File the section into Track A** — hand the figure pack + section draft to
  `plans/03-figure-table-paper-ready.md` with canonical paths; run `/plain-speak --in-place` once on the draft.

## Engagement Instructions
```bash
# toy: small 2D score-model pair + the exact CFG-sum operator; ancestral vs corrected (see plan 03)
# figure pack: scripts/ctr_overlay.py (CTR+CP overlays) + metrics/grad_cam_utils.py, pinned to
#   results/vae_v2/downstream_eval/exp03/ctr_cp_scatter.png clusters → results/poe_composition/exhibits/
```
Verify: the toy shows ancestral-collapse + corrected-recovery of the known product; the scatter carries ≥6
annotated overlay callouts each with a competing reading; the section states the pre-registered fork and reads as
a subtraction, not a menu; the pack is referenced from `plans/03-figure-table-paper-ready.md`.

## Recommended skill
▶ `/design-figure` ✅ — for the overlay-callout figure design (form, faithfulness check) before rendering.
   alt: `/plain-speak --in-place` for the section prose; the toy + overlays themselves are custom scripts.
