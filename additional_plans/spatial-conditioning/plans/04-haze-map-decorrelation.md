# 🌫️ Haze-Map Conditioning + Decorrelation

## Description
Add the regional haze-map (Scope-1's RALE opacity instrument) as a second CFG-droppable input channel, plus a distance-correlation loss between heart-size and peri-cardiac-opacity, so haze stops substituting for the heart. Add only after mask-alone is shown to work.

## Purpose
Serves Objective 3 and DoD item 5 — the disentanglement layer.

## Goal
A checkpoint conditioned on (heart-mask, haze-map) with the decorrelation loss; a haze-off ablation that HOLDS CTR (heart, not fog, carries the gain).

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks
- [ ] ⚠️ **Add the haze-map channel (reuse the Scope-1 opacity instrument)**
  - `▶ prompt:` `Label each training film with metrics/opacity_score.py (built in the vindr-transfer/plans/opacity-instrument sub-scope) → a regional haze-map at latent res; widen conv_in 5→6; concat; CFG-droppable like the mask. Inside c_cardio only.`
  - `✓ verify:` model consumes [B,6,128,128]; haze channel droppable.
  - `⊳ note:` blocks on the `vindr-transfer/plans/opacity-instrument/` sub-scope having shipped `metrics/opacity_score.py`.
- [ ] ⚠️ **Add the distance-correlation decorrelation loss**
  - `▶ prompt:` `Add a distance-correlation penalty between per-sample heart-size (mask/CTR) and peri-cardiac-opacity over the batch, modest weight (tunable); log it. Target: break the substitution, NOT zero the correlation (keep residual realistic).`
  - `✓ verify:` loss logged; weight tunable; training stable.
- [ ] ⚠️ **Train + haze-off ablation**
  - `$ run:` warm-start finetune with mask+haze+decorrelation; then sample with haze set low vs realistic.
  - `✓ verify:` haze-off HOLDS CTR (border-visible variant the instrument can read) → proves the heart, not the fog, carries the gain.

## Recommended skill
— custom; core training-loop code. alt: `training-analyst` ✅ for the offline ablation.

## Engagement Instructions
```bash
$ grep -iE "haze|decorr|dcor|ctr.?hold|ablation" results/spatial_cond/haze_decorr.md
# expect: haze-off holds CTR; decorrelation logged with a modest (non-maximal) weight
```
