# 🎯 Stage-2 — NIH Appearance Adaptation

> ## ⛔ SUPERSEDED (2026-07-01)
> Retired with Stage-1 — the **transfer** approach is replaced by the **vae-v2
> joint-from-scratch** LDM (one model on NIH + VinDr μ-latents; see
> `additional_plans/vae-v2/plans/06-ldm-retrain.md`). Never ran (no Stage-2 checkpoint);
> Todoist task deleted 2026-07-01. The in-training 3-way composition grid referenced here
> lives on in the **composition-operators** sub-scope (standalone eval tooling, still
> planned). Kept for provenance only — do not execute.

## Background
The Stage-1 prior renders cardiomegaly, but in VinDr's *look* (Vietnamese cohort, different scanners, PA-only, different CTR norms). The NIH-calibrated instrument + frozen test are NIH, so the final model must speak NIH appearance. Because Stage-1 already carries the rendering competence, this is light **domain adaptation** — a plain low-LR finetune on the clean NIH latents + replay so effusion doesn't regress. LoRA stays a parked fallback (a strong prior adapts appearance fully; LoRA would cap how far it can move).

## Description
Finetune the Stage-1 checkpoint on the clean NIH cardio/effusion deconfounded latents at a low LR, with effusion/normal replay + early-stop. Log the in-training 3-way composition grid so the best composition operator is chosen as the model converges.

## Purpose
Serves Objective 4 and DoD item 6 — the NIH-appearance model the Gate-3 eval judges.

## Goal
A Stage-2 checkpoint that loads, renders NIH-appearance cardio, holds effusion, no new leakage.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python` · `CKPTS=additional_plans/vindr-transfer/ckpts`

## Tasks

- [ ] ⚠️ **Decide the adaptation recipe: two-stage finetune (default) vs pooled**
  - `⊳ decide:` **Default = two-stage**: finetune the Stage-1 ckpt on NIH latents only (ends cleanly on NIH appearance). **Alternative = pooled**: one train on VinDr+NIH with NIH up-weighted *within each class* (simpler, but the cardio class skews heavily VinDr → NIH look underrepresented). Pick two-stage unless leaf 04's REPORT shows the VinDr↔NIH CTR gap is small. Record the call + why in `results/vindr_transfer/stage2/recipe.md`.
  - `✓ verify:` `recipe.md` names the chosen recipe.
  - `⊳ DATA PREREQUISITE (Gap-A):` the replay below mixes NIH latents for the **chosen 2nd disease** (leaf 09). Effusion latents exist (`data/latents/rebalanced/`); **pleural-thickening latents do NOT** — if 09 picked thickening, build the NIH thickening deconfounded split + latent cache FIRST (mirror `plans/rebalanced-cardio-retrain/02-split-wiring`), or the replay step has nothing to replay.
  - `✓ verify:` if 2nd disease = thickening, `data/latents/<thickening-split>/` exists before the finetune launches.

- [ ] ⚠️ **Plain low-LR finetune on NIH + 2nd-disease/normal replay + early-stop**
  Three core-training changes to `scripts/train_ldm.py` — real code, not glue, so do them as explicit steps:
  - [ ] **Add `--resume-init` (weights-only warm start).**
    - `▶ prompt:` `In scripts/train_ldm.py add --resume-init <ckpt>: load_state_dict the UNet + class_embed from the Stage-1 safetensors but do NOT restore optimizer / global_step (carry the EMA shadow if present). Verify with a 1-step run that global_step starts at 0 with warm (non-random) weights.`
    - `✓ verify:` 1-step run logs step=0 with a low initial loss (warm weights, not from-scratch).
  - [ ] **Add effusion/normal replay to the batch sampler.**
    - `▶ prompt:` `Give the sampler a replay knob: every batch mixes NIH cardio with a fixed fraction (~0.5) of effusion+normal latents so effusion can't be forgotten. Reuse balance_classes/class_mult if it already supports per-class fractions; else add replay_frac to the config. Log the realized per-class batch composition.`
    - `✓ verify:` logged batch composition matches the configured fractions; effusion present in every batch.
  - [ ] **Author `configs/ldm_vindr_stage2.yaml` + launch.**
    - `$ run:` `CONFIG=configs/ldm_vindr_stage2.yaml WANDB_NAME=vindr_stage2_$(date +%Y%m%d) bash scripts/launch_retrain.sh` — set `--resume-init <stage1 ckpt>`, train=`data/latents/rebalanced/`, LR ~1/5 of Stage-1, replay on, EMA inherited from Stage-1, ckpt every 2k for early-stop.
    - `✓ verify:` `stage2_step*.safetensors` written; W&B finetune curve; LR materially below Stage-1.

- [ ] ⚠️ **In-training 3-way composition grid (PoE / MCMC-corrected / SuperDiff-AND)**  → decomposed: see `plans/composition-operators/MASTER_PLAN.md`

- [ ] ⚠️ **Confirm effusion HOLDS + no new leakage (offline, on saved ckpts)**
  - `$ run:`
    ```bash
    for c in $(ls $CKPTS/stage2_step*.safetensors); do
      $PY scripts/monitor_retrain.py --ckpt "$c" --out results/vindr_transfer/stage2/hold_check.md --append
    done
    ```
  - `✓ verify:` effusion blunt-rate held ~36%, effusion CTR low, cardio→effusion leak ≤ 34% floor; pick the NIH-like-but-effusion-holding step (before overfit) and record it.
  - `⊳ note:` if it overfits the 854 / effusion regresses → escalate to the parked **LoRA + latent-replay** fallback (PARKING_LOT.md).

## Recommended skill
— custom; `scripts/train_ldm.py` low-LR resume-as-init.
   alt: `training-analyst` agent ✅ — offline effusion-HOLD + leak sweep.

## Engagement Instructions
```bash
$ $PY -c "from safetensors.torch import load_file; import glob; load_file(sorted(glob.glob('additional_plans/vindr-transfer/ckpts/stage2_step*.safetensors'))[-1]); print('loads OK')"
$ grep -iE "blunt|effusion|leak|chosen|step" results/vindr_transfer/stage2/hold_check.md
# expect: loads OK; effusion blunt-rate held ~36%; leak ≤ 34% floor; a chosen step
```
