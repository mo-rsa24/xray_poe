# 🏋️ Stage-1 — Class-Conditional Pretrain on VinDr

> ## ⛔ SUPERSEDED (2026-07-01)
> The two-stage **transfer** lever (Stage-1 VinDr pretrain → Stage-2 NIH finetune) is
> retired. The VinDr cardio it was meant to inject is now delivered by the **vae-v2
> joint-from-scratch** LDM, which trains one model on the union of NIH + VinDr μ-latents
> in a single VAE-v2 codec (see `additional_plans/vae-v2/plans/06-ldm-retrain.md`,
> `results/vae_v2/ldm_data_decision.md`). This plan never ran (no Stage-1 checkpoint).
> Its Todoist task was deleted 2026-07-01. The VinDr **data-prep** it depended on
> (plans 01–04, 09) is done and feeds the vae-v2 diet; the **composition-operators**
> sub-scope survives as standalone eval tooling. Kept for provenance only — do not execute.

## Background
This step targets the actual ceiling. VinDr's **1,952 deconfounded cardio** films (vs NIH 854, 2.3×) plus 10,606 normal let the cardio conditional learn lower-chest rendering the from-scratch run collapsed to mush — effusion already proved the same UNet+VAE renders the lower zone given enough data. Same conditional UNet (3 classes + null, CFG dropout), VinDr latents, our VAE space → PoE math intact.

## Description
Pretrain the conditional LDM on the VinDr latent cache, on hippo, under project discipline (sequential, `df` guard, W&B, divergence guard).

## Purpose
Serves Objective 3 and DoD item 5 — the prior the NIH adaptation builds on.

## Goal
A Stage-1 checkpoint that loads and renders a border-readable, enlargeable heart (CTR shifted toward real).

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python` · `CKPTS=additional_plans/vindr-transfer/ckpts`

## Tasks

- [ ] ⚠️ **Author the Stage-1 config from the rebalanced template**
  - `▶ prompt:` `Copy configs/ldm_rebalanced.yaml → configs/ldm_vindr_stage1.yaml. Point train data at data/latents/vindr/; keep build_unet(num_classes=3) + null token + cfg_dropout_p; set class_mult / balance_classes from results/vindr_transfer/eda/REPORT.md's distribution finding; set ckpt_dir to additional_plans/vindr-transfer/ckpts; keep batch_size 16 × grad_accum 1, bf16 train + fp32 VAE decode. Do a 1-step dry run confirming it reads VinDr latents + a label per sample.`
  - `✓ verify:` config loads; dry run reads a VinDr latent + label.
  - [ ] **Enable real-EMA in the config.**
    - `▶ prompt:` `Add weight-EMA to scripts/train_ldm.py + configs/ldm_vindr_stage1.yaml: ema_decay ~0.9999, tracked from step 0, EMA shadow weights saved alongside each ckpt (this is the cheapest unused lever — never run before; arm1 had none). Confirm sampling can load from the EMA weights.`
    - `✓ verify:` a stage1 ckpt has EMA shadow weights; sampling from EMA loads.

- [ ] ⚠️ **Launch the pretrain on hippo (sequential, guarded, W&B)**
  - `$ run:`
    ```bash
    CKPT_DIR=additional_plans/vindr-transfer/ckpts \
    CONFIG=configs/ldm_vindr_stage1.yaml \
    WANDB_NAME=vindr_stage1_$(date +%Y%m%d) \
    bash scripts/launch_retrain.sh   # df-guarded wrapper around scripts/train_ldm.py; checkpoints every 10k
    ```
  - `✓ verify:` ≥1 `stage1_step*.safetensors` written; W&B loss falls cleanly; **never run a heavy eval concurrently (single-box OOM)** [[rebalanced-retrain-operational]].

- [ ] ⚠️ **Confirm the prior renders an enlargeable heart**
  - `$ run:` (offline, on a saved checkpoint, GPU free)
    ```bash
    $PY scripts/monitor_retrain.py --latest --ckpt-dir $CKPTS --out results/vindr_transfer/stage1/cardio_render.md
    ```
  - `✓ verify:` `border_frac` ≫ the from-scratch 0.41 and cardio CTR shifted up — the lower-chest mush is gone.
  - `⊳ note:` parked alt base **Cascaded-LDM shortcut** (PARKING_LOT.md) only if its weights are verified open + external.

## Recommended skill
— custom; `scripts/train_ldm.py` on hippo.
   alt: `training-analyst` agent ✅ — offline W&B sweep (never alongside training).

## Engagement Instructions
```bash
$ ls -t additional_plans/vindr-transfer/ckpts/stage1_step*.safetensors | head -1
$ $PY -c "from safetensors.torch import load_file; import glob; load_file(sorted(glob.glob('additional_plans/vindr-transfer/ckpts/stage1_step*.safetensors'))[-1]); print('loads OK')"
$ grep -iE "border_frac|ctr|frac" results/vindr_transfer/stage1/cardio_render.md
# expect: loads OK; border_frac ≫ 0.41; cardio CTR shifted toward real 0.535
```
