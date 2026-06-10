#!/usr/bin/env bash
# Plan 05 — tail GPU memory + utilization from the terminal.
# Usage: bash scripts/monitor_gpu.sh [interval_seconds]
set -euo pipefail
INTERVAL="${1:-2}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found — no GPU to monitor." >&2
  exit 1
fi

echo "monitoring GPU every ${INTERVAL}s (Ctrl-C to stop)"
echo "time                 mem_used/total(MiB)   util%   temp"
while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  read -r used total util temp < <(nvidia-smi \
    --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits | head -1 | tr -d ',')
  printf '%s   %6s/%-6s          %3s%%    %sC\n' "$ts" "$used" "$total" "$util" "$temp"
  sleep "$INTERVAL"
done
