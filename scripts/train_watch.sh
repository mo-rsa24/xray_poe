#!/usr/bin/env bash
# Plan 05 — launch a VAE training run and tail its log + GPU memory side by side.
# Usage: bash scripts/train_watch.sh [-- <vae.train args>]
#   e.g. bash scripts/train_watch.sh -- --data noise --res 512 --steps 2000
set -euo pipefail

PYTHON="${VAE_PYTHON:-/home/molef/micromamba/envs/jaxstack/bin/python}"
LOGDIR="logs"
mkdir -p "$LOGDIR"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG="$LOGDIR/train_${STAMP}.log"

# strip a leading "--" so the rest passes through to vae.train
[[ "${1:-}" == "--" ]] && shift
ARGS=("$@")

echo "launching: $PYTHON -m vae.train ${ARGS[*]}"
echo "log: $LOG"
"$PYTHON" -m vae.train "${ARGS[@]}" 2>&1 | tee "$LOG" &
TRAIN_PID=$!

# tail GPU memory alongside until training exits
if command -v nvidia-smi >/dev/null 2>&1; then
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | \
      sed "s/^/[gpu] /"
    sleep 5
  done
fi
wait "$TRAIN_PID"
