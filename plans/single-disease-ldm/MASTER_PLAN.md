# 🎲 Single-Disease LDM

## Mission
Train one conditional latent diffusion model on single-disease latents only —
never seeing a both-disease image — with CFG dropout so a single model supports
both the `∅` and `normal` composition anchors at inference.

## Objectives
1. Overfit-sanity: memorize a handful of single-disease latents and regenerate them.
2. Prepare single-disease latents from the frozen VAE; define the condition set
   (normal, cardiomegaly, effusion, …).
3. Train the conditional LDM with CFG label dropout (learns ε(z, ∅)); both-disease
   images strictly held out.
4. Evaluate single-disease sample quality (FID vs real single-disease).

## Goals
1. Overfit sanity: regenerates the memorized latents.
2. Single-disease FID convincing (gate).
3. Both anchors (`∅` and `normal`) usable at inference without retraining.

## Expected Outcome
A conditional LDM checkpoint that generates convincing single-disease X-rays and
supports both composition anchors, with the both-disease hold-out intact so
composition is a genuine test.

## Definition of Done
1. Overfit sanity passed and logged.
2. Single-disease latents prepared; condition set defined; both-disease hold-out
   verified (zero both-disease images in training).
3. LDM trained with CFG dropout; config (≤128ch UNet, batch ≤8 + grad-accum, bf16)
   recorded; checkpoint in `ckpts/`.
4. Single-disease FID reported; passes gate.
5. Sample-grid figure saved; both `∅` and `normal` nulls confirmed callable at inference.

## Sub-Scopes
(none yet — added by decompose-plan)

## Plans
- ⚠️ 01-overfit-sanity.md
- ⚠️ 02-latent-prep-conditions.md
- ⚠️ 03-train-ldm-cfg.md
- ⚠️ 04-fid-eval.md
