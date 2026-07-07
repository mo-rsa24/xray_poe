#!/usr/bin/env bash
# Resume the rebalanced-cardio LDM retrain that the OLD divergence-guard killed
# at step 10 294 (W&B run pcp9jo36). The guard is now EMA-smoothed + persistent
# (train_ldm.py), so per-step DDPM loss noise can no longer false-trip it.
#
# Resumes from ckpts_rebalanced/model_step0010000.* (model + optimizer + LR state)
# and continues the SAME W&B run to the full max_steps (100k) in the config.
#
# Run inside tmux:  tmux new -s retrain  then  bash scripts/relaunch_retrain.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/molefe/micromamba/envs/jaxstack/bin/python
RESUME_RUN_ID="${RESUME_RUN_ID:-pcp9jo36}"

# --- sanity: at least one checkpoint (model + training_state) must exist --------
# --resume loads the LATEST checkpoint via _find_latest_checkpoint, not step 10k.
LATEST_MODEL=$(ls -t ckpts_rebalanced/model_step*.safetensors 2>/dev/null | head -1)
[ -n "$LATEST_MODEL" ] || { echo "ABORT — no model_step*.safetensors in ckpts_rebalanced/"; exit 1; }
LATEST_STEP=$(echo "$LATEST_MODEL" | grep -oE 'step[0-9]+' | tr -dc '0-9')
STATE="ckpts_rebalanced/training_state_step${LATEST_STEP}.pt"
[ -f "$STATE" ] || { echo "ABORT — missing $STATE (optimizer/LR state for the latest model)"; exit 1; }

# --- disk guard (abort before writing if the checkpoint disk is nearly full) ----
USED=$(df --output=pcent . | tail -1 | tr -dc '0-9')
if [ "$USED" -ge 90 ]; then echo "ABORT — disk ${USED}% full"; exit 1; fi
echo "disk ${USED}% full — OK · resuming W&B run=${RESUME_RUN_ID} from step ${LATEST_STEP}"

exec "$PY" scripts/train_ldm.py \
    --config configs/ldm_rebalanced.yaml \
    --latent-cache data/latents/rebalanced \
    --vae-decode-ckpt ckpts/vae_step0025000.pt \
    --resume "$RESUME_RUN_ID"
