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
  sampler, FID-plateau + val-loss-knee stop-gates, ~~post-hoc EMA sweep~~, BPA noise priors, Patch
  Diffusion. *Reason: marginal quality gain for new code debugged on the clock; the existing proven
  loop ships the treatment arm. Revisit with the 5-class cluster above.*
  - **EMA GRADUATED 2026-06-27 (augment-triage):** post-hoc EMA splits into → (a) **SWA checkpoint-averaging do-now** on arm1 (`model_step080000/090000/100000` → sample → compare CTR/border vs single ckpt; true EMA can't be reconstructed post-hoc, this is the proxy), integrate into `plans/rebalanced-cardio-retrain/`; and (b) **real-EMA** as a training setting in the active `vindr-transfer` (Scope 1) run.
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

### VinDr-transfer scope — Scope 1 (ACTIVE; renamed from padchest-transfer → `additional_plans/vindr-transfer/`)

Parked candidates from the augment compile (PadChest-pretrain → NIH-finetune cardio lever),
triaged 2026-06-26. The spine (download → VAE-gate → preprocess/projection → data-investigation
→ stage-1 pretrain → stage-2 finetune → eval) routed to the new scope; these are its
fallbacks / gated extras.

> **DEMOTED 2026-06-26 (route-B re-point):** the WHOLE two-stage transfer is now the **escalation** — runs only if the cheaper single-stage **clean-cardio augmentation** (now leaves in `plans/rebalanced-cardio-retrain/`) plateaus. The `additional_plans/padchest-transfer/` scope stays dormant; the items below remain its sub-fallbacks.
>
> **RE-TRIAGED 2026-06-27 (post-sync):** confirmed **defer** — not load-bearing for the WACV paper (headline rests on the clean-cardio Arm-1 lever, not transfer). Real status: only the VinDr data foundation exists, and that was acquired by the sibling Arm-1 scope; plans 02–08 entirely unstarted. **User is re-authoring this scope into a new set of plans via `/integrate-plans`** — that re-authoring is the owner of its disposition; triage does not pull it into the paper-critical tree. Stays the escalation lever.
>
> **ACTIVATED + RENAMED 2026-06-27 (augment-triage, supersedes the defer above):** arm1 clean-cardio augmentation (`gtr6kqpk`) **plateaued at the diagnostic line** (CTR flat 0.499 from 50k, presence 0.20, yield 0.125) → the escalation condition the defer was waiting on has **FIRED**. Scope promoted to **ACTIVE Scope 1** and **renamed `additional_plans/padchest-transfer/` → `additional_plans/vindr-transfer/`**. Extend via `/integrate-plans`; **mission/guardrails unchanged** — vanilla class-conditional, composition-only, keep-our-VAE, external-base-only, NIH-frozen-eval. Headline question: *is clean VinDr+NIH data enough for a vanilla conditional LDM?* PadChest stays this scope's internal headroom escalation. A new **Scope 2** (`additional_plans/spatial-conditioning/`) carries heart-mask + haze-map + decorrelation as the follow-up, **gated on Scope 1 failing the yield gate** — so the bottleneck (data vs. conditioning) is attributable to exactly one change.

- ~~**Clean-unconditional-ε anchor test.**~~ **PROMOTED 2026-07-08 (triage → poe-composition scope)** — routed
  as the first (cheapest) operator step in `additional_plans/poe-composition/` (O4), run before SuperDiff/MCMC
  because if the ~75%-diseased `null` anchor is the problem, the fancy operators would be fixing the wrong thing.
  Pairs with `scripts/cfg_anchor_sweep.py` (Part 1). Source: Unconditional Priors Matter (arXiv 2503.20240).
  - *source: augment compile 2026-06-26 → triage 2026-06-26 → promoted 2026-07-08*

- **LoRA + latent-replay (stage-2 fallback).** Low-rank adapter + effusion/normal replay instead
  of plain finetune. *Reason: fires only if S6 plain low-LR finetune visibly overfits the 854 or
  regresses effusion. Plain is the default; this is the escalation.* Source: latent replay (2509.10529).
  - *source: augment compile 2026-06-26 → triage 2026-06-26*

- **Cascaded-LDM shortcut (stage-1 replacement).** Cascaded latent diffusion for hi-res CXR
  (arXiv 2303.11224) as a ready base, skipping our own PadChest pretrain. *Reason: blocked on
  verification — are its weights openly downloadable AND its corpus external to NIH? If
  MIMIC-trained it is gated like RoentGen. Verify before it can replace S5.*
  - *source: augment compile 2026-06-26 → triage 2026-06-26*

- **Synthetic-CXR pretrain (stage-1 fallback base).** Pretrain on SynthCheX-75K (CheXGenBench) —
  a zero-patient-leakage base. *Reason: fires only if PadChest licensing/download snags (S1).
  PadChest is the default base.*
  - *source: augment compile 2026-06-26 → triage 2026-06-26*

- **Memorization probe (privacy hygiene).** CheXGenBench / Dutt memorization metrics on the
  finetuned model. *Reason: NOT load-bearing for the headline — NIH test patients aren't in
  PadChest, so the model cannot memorize them; S4's data-side disjointness check carries the
  clean-test guarantee. This protects PadChest privacy, not Paper3 validity. Optional insurance.*
  Source: De-Identification Traces (arXiv 2502.07516).
  - *source: augment compile 2026-06-26 → triage 2026-06-26*

- **Foreign-VAE path (+ Exp-2 re-run) — CONDITIONAL.** Adopt a foreign VAE and re-run the codec
  ceiling. *Fires ONLY if the S2 VAE round-trip gate fails (our VAE doesn't preserve VinDr
  lower-zone detail). If S2 passes, our VAE ships and this never runs. Mirrors the conditional
  VAE-finetune item above.*
  - **ABSORBS (2026-06-27 augment-triage):** the *"continuous-latent VAE retrain / borrow-SOTA / mix VinBig+NIH"* question from the augment compile → **REFINE, not scheduled.** Conflicts with Exp-2 (VAE recorded faithful & reusable, |ΔCTR| 0.011) and `vindr-transfer` guardrail #2 (keep our VAE = one latent space so Exp-2 carries). Decision belongs to `02-vae-roundtrip-gate` on evidence; this is its home — do **not** front-load a VAE retrain.
  - *source: triage 2026-06-26 → absorbed 2026-06-27*

### Clean-cardio augmentation (in plans/rebalanced-cardio-retrain/)

Parked extras from the clean-cardio augmentation compile (triaged 2026-06-26 → integrated into the
rebalanced-retrain scope as an **ablation ladder**, leaves 06–11). **GRADUATED into leaves:** InvDiff
→ leaf 10 (Arm 2); soft down-weight → leaf 08 (the <854-floor fallback); diversity preservation
→ leaf 08 (the diversity check). The two items below stay **GATED** — promoted to ablation arms by
leaf 11 only if the cheap arms (1/2) don't clear the Exp-09 gate (~0.65) / Gate-3 (30%).

- **InvDiff invariant-guidance (inference-time haze steer).** Steer `cfg_compose` sampling toward
  the heart, away from basal opacity — no retrain. *Reason: fallback if the clean diet + the
  generated-haze monitor still show haze; an inference-time nudge, not load-bearing.* Source:
  InvDiff (arXiv 2412.08480).
  - *source: augment compile 2026-06-26 → triage 2026-06-26*

- ~~**Explicit haze micro-conditioning.**~~ **PROMOTED 2026-06-27 (augment-triage)** → graduates into the new **Scope 2 (`additional_plans/spatial-conditioning/`)** as the `haze-map` conditioning *input* — now a spatial regional-opacity map (RALE-style instrument, escalate to per-pixel if too coarse), **not** just a scalar — alongside heart-mask conditioning + a distance-correlation decorrelation loss. Stays gated: fires only if the renamed `vindr-transfer` vanilla-conditional run fails the yield gate. Mask-concat is spatial conditioning (does NOT break the "no text-conditioning" guardrail) but DOES change "composition is the only contribution" → which is exactly why it lives in a separate Scope 2, not in Scope 1.
  - *source: augment compile 2026-06-26 → promoted 2026-06-27 augment-triage*

- **Soft down-weight (per-sample loss weighting).** Keep hazy cardio at low weight instead of hard
  drop. *Reason: the fallback BELOW the 854 floor — fires only if the hard basal-clean filter
  leaves < 854 cardio or collapses diversity. Needs a train-loop change.*
  - *source: augment compile 2026-06-26 → triage 2026-06-26*

- **Diversity preservation.** Keep some heart-shape variation if the hard filter narrows to a
  "textbook heart". *Reason: a caution, fires only if the kept cardio distribution collapses
  (Ambient-Omni / Slight-Corruption).* Source: arXiv 2506.10038 / 2405.20494.
  - *source: augment compile 2026-06-26 → triage 2026-06-26*

### Communication / narrative artifacts

- **Combined illuminate readout (supervisor narrative).** Merge the two existing illuminate runs —
  `illuminate-output/paper3-trust-chain/` + `illuminate-output/cardio-retrain-readout/` — into one
  combined illuminated artifact for the paper/supervisor narrative. *Reason: a strong comms aid but
  NOT a WACV deliverable; and premature — results are still partly on the contaminated before-model,
  so a "combined readout" would harden numbers that change once the after-model (rebalanced / Arm-1)
  results land. Revisit after the after-model utility numbers are in.* Goal link: serves the
  supervisor-narrative track, parallel to the paper, not the submission itself.
  - *source: triage 2026-06-27 (post-sync)*

### PoE-composition scope — Scope (NEW 2026-07-08, routed to `additional_plans/poe-composition/`)

Defers from the three-augment triage (1a explain-null, 1b bottleneck, 1c alt-operators). The 8 keeps routed to
the new scope; these four are its gated fallbacks — each fires only on a named condition, so none is scheduled.

- **Concept-algebra cardio-direction injection.** The only *signal-injector* (adds heart signal the composition
  can't produce), estimated from real single-cardio NIH films. *Reason: not plan-ready — the real single-cardio
  pool isn't built/sized, high variance under the squashed latent, circularity risk if seeded from generated
  cardio. Fires as the empty-ingredient fallback, gated on the #5 cardio-alone ceiling probe (near-empty) + an
  adequate pool.* Source: Concept Algebra (arXiv 2302.03693); CASL SAE (arXiv 2601.15441).
  - ▶ `/augment build a stable real-single-cardio direction for concept-algebra injection on v4-60k — confirm the NIH cardio-only pool is large/clean enough to estimate a low-variance score-space direction, and design the anti-circularity check (estimate from real, never from generated cardio-alone)`
  - *source: 1c augment compile → triage 2026-07-08*

- **HMC on a composed energy.** Better mixing than ULA, but needs an energy distillation/parameterization of
  v4-60k (retrain-flavored) + leapfrog evals per step. *Reason: fires only if the #7 no-retrain corrected
  samplers (ULA + approximate-energy MH) give ambiguous diagnostics that need hardening.* Source: Du et al.,
  Reduce/Reuse/Recycle (arXiv 2302.11552).
  - *source: 1a/1c augment compiles → triage 2026-07-08*

- **VAE re-disentangle retrain.** Retrain the codec for a disease-separable latent. *Reason: expensive, breaks
  the keep-our-VAE / one-latent-space guardrail, and conflicts with Exp-2 (VAE recorded faithful & reusable,
  |ΔCTR| 0.011). Fires ONLY if the #2 separability/CAV probe shows collapse specifically at `z` with CAV mass in
  the known dead channels — i.e. a probe demands it. Otherwise never runs.*
  - *source: 1b augment compile → triage 2026-07-08*

- **Beam / greedy search over best-of-N.** Beats plain best-of-N at equal budget. *Reason: 1c's own read — adds
  table width, not signal. Fires only if best-of-N proves competitive and we then want to optimize it.* Source:
  inference-time-scaling literature (2025).
  - *source: 1c augment compile → triage 2026-07-08*

## Drop

(none — no candidate was judged both not-worth and not-load-bearing)
