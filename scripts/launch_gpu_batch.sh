#!/bin/bash
# Combined GPU batch — runs the two deferred jobs back-to-back the moment the GPU is free.
# Guards ONCE (disk + free GPU), then:
#   1) Exp 9  — individual-disease body-or-no-body gate on the BEFORE model (fast)
#   2) Exp 6  — Oracle/Starved arms + detector + headroom verdict (heavy)
#
# Exp 9 first: it is the cheap gate that decides whether the synthetic track even has a
# body, and it is independent of Exp 6. Both only need a free GPU (NOT concurrent with
# the rebalanced retrain — that OOMs the ~33 GB host).
#
#   bash scripts/launch_gpu_batch.sh
#   ONLY_EXP9=1 bash scripts/launch_gpu_batch.sh    # just the Exp 9 gate
#   AFTER_LDM=ckpts_rebalanced/model_step0050000.safetensors bash scripts/launch_gpu_batch.sh
#       ^ also run the Exp 9 AFTER-model gate on the given rebalanced checkpoint
set -euo pipefail

PY=/home/molefe/micromamba/envs/jaxstack/bin/python
ROOT=/home/molefe/PhD/Paper3
CKPT_DIR=${CKPT_DIR:-$ROOT/ckpts}
cd "$ROOT"

# ── guards (once) ────────────────────────────────────────────────────────────
USED=$(df --output=pcent "$CKPT_DIR" | tail -1 | tr -dc '0-9')
if [ "${USED:-100}" -ge 90 ]; then
  echo "ABORT: $CKPT_DIR is ${USED}% full — checkpointing will fail." >&2; exit 1
fi
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
if [ "${GPU_USED:-99999}" -ge 2000 ]; then
  echo "ABORT: GPU has ${GPU_USED} MiB in use — the rebalanced retrain is likely live." >&2
  echo "       These jobs OOM the box if run concurrently. Wait for a free GPU." >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
  exit 1
fi
echo "guards OK (disk ${USED}%, GPU ${GPU_USED} MiB free) — starting GPU batch"

# ── 1) Exp 9 — individual-disease gate, BEFORE model (runnable on the 40k ckpt) ──
echo "── [1/2] Exp 9: individual-disease gate (before-model) ──"
$PY scripts/validate_individual_synthetics.py \
    --ldm ckpts/model_step0040000.safetensors --tag before

# optional AFTER-model gate if a rebalanced checkpoint is named
if [ -n "${AFTER_LDM:-}" ]; then
  echo "── Exp 9: individual-disease gate (after-model: ${AFTER_LDM}) ──"
  $PY scripts/validate_individual_synthetics.py --ldm "$AFTER_LDM" --tag after
fi

if [ "${ONLY_EXP9:-0}" = "1" ]; then
  echo "ONLY_EXP9 set — stopping after the Exp 9 gate. See results/exp09_individual/verdict.md"
  exit 0
fi

# ── 2) Exp 6 — oracle/starved arms (guards already passed; skip its re-gate) ──
echo "── [2/2] Exp 6: detector + Oracle×3 + Starved×3 + headroom verdict ──"
EXP06_SKIP_GUARD=1 bash scripts/launch_exp06.sh

echo "GPU batch complete — Exp 9: results/exp09_individual/verdict.md · Exp 6: results/exp06_oracle/verdict.md"
