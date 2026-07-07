#!/usr/bin/env bash
# Combined retrain watch — run in a SECOND terminal while the train runs.
#   • every cycle: prints step / velocity / ETA   (how fast)
#   • on each new checkpoint: runs the recovery instrument + W&B histograms  (how well)
# Usage:  bash scripts/watch_retrain.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/home/molefe/micromamba/envs/jaxstack/bin/python
CKPT_GLOB="ckpts_rebalanced/model_step*.safetensors"
INTERVAL="${INTERVAL:-300}"     # seconds between cycles
SEEN=""

echo "watching retrain — progress every ${INTERVAL}s, recovery on each new checkpoint"
while true; do
  # how fast (velocity + ETA from W&B)
  "$PY" scripts/monitor_progress.py --once 2>/dev/null | grep -E "step|run" || echo "  (run not visible yet)"

  # how well (instrument recovery on a freshly-saved checkpoint)
  LATEST=$(ls -t $CKPT_GLOB 2>/dev/null | head -1)
  if [ -n "$LATEST" ] && [ "$LATEST" != "$SEEN" ]; then
    SEEN="$LATEST"
    echo "=== new checkpoint $LATEST → recovery check (cardio↑ / effusion-hold / yield) ==="
    "$PY" scripts/monitor_retrain.py --ckpt "$LATEST" --wandb 2>/dev/null \
      | grep -E "CARDIO|EFFUSION|PoE|W&B" || echo "  (monitor_retrain produced no summary)"
  fi
  sleep "$INTERVAL"
done
