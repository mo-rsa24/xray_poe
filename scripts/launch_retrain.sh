#!/usr/bin/env bash
# Rebalanced-cardio LDM retrain — disk-guarded launch.
# Run inside tmux:  tmux new -s retrain  then  bash scripts/launch_retrain.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/molefe/micromamba/envs/jaxstack/bin/python
RUN_NAME="${RUN_NAME:-exp5_retrain_s01_$(date +%Y%m%d)}"

# --- disk guard (abort before writing if the checkpoint disk is nearly full) ---
USED=$(df --output=pcent . | tail -1 | tr -dc '0-9')
if [ "$USED" -ge 90 ]; then echo "ABORT — disk ${USED}% full"; exit 1; fi
echo "disk ${USED}% full — OK · run=${RUN_NAME}"

# batch 8 / num_workers 8 / wandb / regular logging / artifacts all come from the config
exec "$PY" scripts/train_ldm.py \
    --config configs/ldm_rebalanced.yaml \
    --latent-cache data/latents/rebalanced \
    --vae-decode-ckpt ckpts/vae_step0025000.pt \
    --run-name "$RUN_NAME"
