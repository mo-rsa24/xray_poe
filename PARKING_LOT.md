# Parking Lot

Non-authoritative waiting room for ideas not yet routed to the plan tree.
No completion state, no checkboxes-as-tracked-work — *doing* lives in the plan tree and Todoist.
Updated by `/triage-plan`. Re-runs update entries in place; promoted items are struck through and kept.

## Refine — promising but not plan-ready (run the prompt, then re-triage)

- ~~**Heart-size instrument selection (cardiomegaly).**~~ **PROMOTED 2026-06-21** → in-plan
  **decision-gate task** inside `plans/02-pre-evaluation/04-visual-inspection.md`, sequenced after
  the manual go/no-go (it's evidence-dependent — can't be decided before the plan produces the
  evidence). The `/socratic` below is its engagement instruction; outcome may reopen the locked
  Grad-CAM scalar in `06-metrics-extractors/04-extractors.md`.
  - ▶ `/socratic which heart-size instrument to trust for cardiomegaly — locked Grad-CAM bbox proxy vs HybridGNet CTR vs MedSAM-cut-and-measure — given Grad-CAM smears across the mediastinum; weigh raw-size simplicity vs CTR's thorax normalization`
  - *source: triage 2026-06-21 → promoted to in-plan task same day*

- ~~**Paper significance of presence-determination.**~~ **PROMOTED 2026-06-21** → framing/writing
  task in `plans/08-documentation/` feeding `32-methods-results-draft`. The `/socratic` below is its
  engagement instruction.
  - ▶ `/socratic why determining cardiomegaly + effusion presence in composed images is imperative to the Paper3 argument, and how the visual inspection harness earns its place in the paper`
  - *source: triage 2026-06-21 → promoted to in-plan task same day*

## Defer — good, but not now / blocked

- **MedSAM3 text-prompting** ("pleural effusion" / "enlarged heart" as concept phrases) instead of
  box-prompting. Validated only on infection opacity (COVID-QU-Ex); weights freshness unknown.
  Revisit once MedSAM3 weights are confirmed released, and after the MedSAM box-prompt gate (C)
  clears on real images.
  - *source: augment compile 2026-06-20*

- **CTR printed as a number on generated images.** Only once HybridGNet contours are trusted on
  synthetic anatomy (gated on the leakage matrix E + the method-selection refine J). Until then,
  lines-only on LDM/PoE rows.
  - *source: augment/unpack 2026-06-20*

## Drop

(none — no candidate was judged both not-worth and not-load-bearing)
