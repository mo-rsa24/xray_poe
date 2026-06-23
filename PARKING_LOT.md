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

- ~~**Treatment-only C2ST (real-vs-composed, cardio+effusion).**~~ **PROMOTED 2026-06-23** → in-plan
  update to `plans/06-metrics-extractors/01-two-sample-mmd.md`. The single-arm version (can a
  classifier tell real both-disease from PoE-composed?) needs only the treatment pair + existing
  checkpoints + the existing floor (`eda/floor_power_check.py`), so it fits the hard date. The
  three-arm (treatment-vs-control) version stays deferred below.
  - *source: triage 2026-06-23 → promoted same day*

## Defer — good, but not now / blocked

### Deadline-deferred (date 2026-06-27 is hard; these are the registered-but-won't-fit upgrade)

- **5-class joint LDM + independent-control arm.** The MASTER_PLAN's control pair
  (emphysema×infiltration, φ≈0) and the 5-class model that would produce its single-disease experts.
  Un-defers H2-control (DoD #10), the three-arm Exp6, and the three-arm C2ST. Parked as **abortable
  bonus** — run only if the Tier-1 shipping work finishes with window to spare; otherwise it is the
  paper's stated future work. *Reason: doubles the eval layer (second pair) + new training-loop code;
  does not fit the hard date.*
  - *source: triage 2026-06-23 (date-is-hard verdict)*

- **Optimal-model training grafts.** Non-leaking augmentation-conditioning, symmetric ~5k quota
  sampler, FID-plateau + val-loss-knee stop-gates, post-hoc EMA sweep, BPA noise priors, Patch
  Diffusion. *Reason: marginal quality gain for new code debugged on the clock; the existing proven
  loop ships the treatment arm. Revisit with the 5-class cluster above.*
  - *source: augment-training compile 2026-06-23*

- **VAE fine-tune from 25k (CONDITIONAL).** Fires **only if** the rFID ceiling gate on
  `vae_step0025000` fails. If the gate passes, the current codec ships and this never runs.
  *Reason: gated on a measurement, not scheduled.*
  - *source: augment-training compile 2026-06-23*

- **Three-arm C2ST (treatment vs control).** The stronger joint-structure test across all arms vs the
  floor. *Reason: meaningless without the parked control pair above. Floor already exists.*
  - *source: triage 2026-06-23*

### Instrument-blocked

- **Q5 — MedSAM lesion boxes** and **Q6 — CTR via HybridGNet.** Both blocked on the heart-size
  instrument decision (the promoted refine task in `02-pre-evaluation/04-visual-inspection.md`).
  Until that decision lands on real evidence, lines-only on LDM/PoE rows; no auto effusion masks
  dressed as ground truth. *Reason: blocked, not unworthy.*
  - *source: triage 2026-06-23*

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
