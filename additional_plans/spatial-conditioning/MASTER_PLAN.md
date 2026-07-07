# Spatial-Conditioning Escalation (Scope 2)

## Mission
The escalation lever for the cardio under-rendering, gated on Scope 1 (`additional_plans/vindr-transfer/`) failing the Gate-3 yield. If a **vanilla class-conditional** LDM on clean VinDr+NIH data still cannot render a co-morbid cardiomegaly, the bottleneck is the conditioning, not the data — and this scope tests that directly by adding **spatial conditioning** to the *same* UNet: a PSPNet-derived **heart-mask** (forces heart size/location), a regional **haze-map** (the Scope-1 opacity instrument as an *input*, so basal haze stops substituting for the heart), and a **distance-correlation decorrelation** loss (breaks the size↔haze substitution while allowing residual realistic correlation). Changed one variable at a time: **heart-mask alone first** (does the UNet even obey a mask?), then haze-map + decorrelation. **PoE is preserved** — mask and haze are CFG-droppable concat inputs inside `c_cardio` only, so PoE(cardio, effusion) stays the symmetric score-sum and effusion-alone/unconditional still sample.

> **Guardrails:** (1) keep PoE = score-sum; the conditioning rides inside `c_cardio`, never a separate stage. (2) Keep **our VAE** and the Scope-1 pretrained UNet (warm-start, not from-scratch). (3) Mask/haze are **CFG-droppable** (spatial null = zeroed channels) — effusion-alone and unconditional must still sample. (4) **Final eval on the NIH frozen separable set**, NIH-calibrated instrument. (5) Do **not** wire the PSPNet judge in as a differentiable loss — masks are an upstream label source, CTR stays the judge.

## Objectives
1. Add mask-grounded conditioning to the Scope-1 pretrained UNet without breaking PoE (mask/haze as CFG-droppable concat inputs in `c_cardio`).
2. Test **heart-mask conditioning alone** before anything else — establish whether the UNet obeys a spatial mask at all.
3. Disentangle heart-size from basal-haze via haze-map conditioning + a decorrelation loss — break the substitution, keep residual realistic correlation.
4. Re-measure yield and the *honest* evidence (border crispness, the comorbid PoE composite, the independent DenseNet classifier) vs Scope 1 and from-scratch; attribute the bottleneck.

## Goals
1. **Mask-listened** — heart-mask-alone shifts gen-cardio CTR toward the mask target *and* lifts `border_frac` materially above Scope 1's; a mask-on/mask-off ablation proves the UNet obeys it.
2. **PoE intact** — effusion-alone + unconditional still sample with mask/haze zeroed; PoE(cardio,effusion) score-sum verified unchanged from the Scope-1 composition path.
3. **Disentangled** — with haze-map + decorrelation, a haze-off ablation **holds CTR** (heart, not fog, carries the gain); residual size↔haze correlation stays realistic, not forced to zero.
4. **Yield re-measured** — Gate-3 yield on the NIH frozen set vs Scope-1/from-scratch; the independent classifier reads cardio; decision recorded.

## Expected Outcome
A mask-grounded conditional checkpoint whose NIH PoE composites clear Gate-3 where the vanilla model couldn't — attributing the bottleneck to the conditioning, not the data — **or** a clean null (spatial conditioning didn't help either, pointing the bottleneck deeper). Either way the mask-alone ablation answers "does the UNet listen to spatial conditioning," and PoE math is confirmed untouched. The headline evidence is border crispness + the comorbid composite + the independent classifier — **not** CTR (which, once you condition on a CTR-targeted mask, only shows the model followed the mask).

## Definition of Done
1. **Gate fired** — Scope 1 recorded a Gate-3 null/inconclusive (this scope runs only if so).
2. **Heart-mask conditioning implemented** — `conv_in` widened to concat `heart_mask` at latent res; masks PSPNet-derived, scaled to a *sampled* CTR distribution (not a fixed template); CFG-droppable (zeroed channels) inside `c_cardio` only.
3. **Mask-alone run** — trained warm-started from a Scope-1 checkpoint; mask-on/mask-off ablation shows CTR shift + `border_frac` lift; recorded.
4. **PoE preservation check** — effusion-alone + unconditional sample with mask/haze zeroed; PoE score-sum verified equal to the Scope-1 composition path.
5. **Haze-map + decorrelation added** — haze-map reuses Scope-1's RALE opacity instrument as the *input*; haze-off ablation in place; distance-correlation loss between heart-size and peri-cardiac-opacity, modest weight.
6. **Yield + honest evidence re-measured** — Gate-3 on the NIH frozen set vs Scope-1/from-scratch; independent DenseNet classifier read; decision filed (recovered/null) + bottleneck attribution.

## Sub-Scopes
(none)

## Plans
Dependency order: `01 → 02 → 03 → 04 → 05` (mask-alone-first; haze waits until 04; the Scope-1-null gate is 01's opening task).
- ⚠️ 01-heart-mask-conditioning.md — conv_in widen + PSPNet sampled-CTR mask pipeline + CFG-droppable (DoD 1–2)
- ⚠️ 02-mask-alone-ablation.md — warm-start train + mask-on/off; does the UNet obey the mask? (DoD 3)
- ⚠️ 03-poe-preservation-check.md — effusion-alone/unconditional + score-sum == Scope-1 (DoD 4)
- ⚠️ 04-haze-map-decorrelation.md — haze-map input + distance-corr loss + haze-off ablation (DoD 5)
- ⚠️ 05-yield-remeasure.md — Gate-3 + independent classifier vs Scope-1/from-scratch; decision (DoD 6)
