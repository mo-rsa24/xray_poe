# 🔬 Mask-Alone Ablation (does the UNet obey?)

## Description
Warm-start from a Scope-1 checkpoint, finetune with heart-mask conditioning ONLY (no haze, no decorrelation), then run a mask-on vs mask-off ablation to prove the UNet uses the mask.

## Purpose
Serves Objective 2 and DoD item 3 — the cheapest, decisive test before building the rest. If the UNet ignores the mask here, the larger build is moot.

## Goal
A mask-conditioned checkpoint + an ablation showing mask-on shifts CTR toward the mask target and lifts `border_frac` materially above Scope-1; mask-off reverts.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks
- [ ] ⚠️ **Warm-start finetune with mask conditioning only**
  - `$ run:` `CONFIG=configs/ldm_maskcond_stage.yaml WANDB_NAME=maskalone_$(date +%Y%m%d) bash scripts/launch_retrain.sh` — `--resume-init <Scope-1 ckpt>`, mask channel on, haze OFF, decorrelation OFF, low LR.
  - `✓ verify:` ckpts written; W&B loss falls; mask channel active in the batch.
- [ ] ⚠️ **Mask-on / mask-off ablation**
  - `▶ prompt:` `Sample cardio with the heart-mask supplied (scaled to CTR target) vs with the mask channel zeroed, N≥200 each; score gen CTR + border_frac with scripts/ctr_overlay.py → results/spatial_cond/mask_ablation.md.`
  - `✓ verify:` mask-on CTR tracks the target + border_frac ≫ Scope-1; mask-off reverts to Scope-1-like → the UNet obeys the mask.

## Recommended skill
— custom; `scripts/train_ldm.py`. alt: `training-analyst` ✅ — offline ablation sweep (never alongside training).

## Engagement Instructions
```bash
$ grep -iE "mask.?on|mask.?off|ctr|border" results/spatial_cond/mask_ablation.md
# expect: mask-on CTR≈target & border_frac up; mask-off ≈ Scope-1 → mask is obeyed (or NOT → stop, UNet ignores spatial cond)
```
