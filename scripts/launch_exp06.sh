#!/bin/bash
# Exp 6 — Oracle baseline launcher. Runs the GPU-heavy arms IN ORDER, but only on a
# FREE GPU (the rebalanced retrain must NOT be running — concurrent training OOMs the
# ~33 GB host, per the operational memory) and behind the df disk guard.
#
#   bash scripts/launch_exp06.sh
#
# Sequence: detector (CTR-label proof) -> Oracle x3 -> Starved x3 -> GradCAM Gate1 -> verdict.
set -euo pipefail

PY=/home/molefe/micromamba/envs/jaxstack/bin/python
ROOT=/home/molefe/PhD/Paper3
CKPT_DIR=${CKPT_DIR:-$ROOT/ckpts}
DATE=$(date +%Y%m%d)
cd "$ROOT"

# ── guards (disk + free GPU) — skippable when a parent launcher already gated ──
if [ "${EXP06_SKIP_GUARD:-0}" != "1" ]; then
  USED=$(df --output=pcent "$CKPT_DIR" | tail -1 | tr -dc '0-9')
  if [ "${USED:-100}" -ge 90 ]; then
    echo "ABORT: $CKPT_DIR is ${USED}% full — checkpointing will fail." >&2; exit 1
  fi
  GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  if [ "${GPU_USED:-99999}" -ge 2000 ]; then
    echo "ABORT: GPU has ${GPU_USED} MiB in use — a training/eval job (the rebalanced retrain?)" >&2
    echo "       is likely live. Running Exp 6 concurrently OOMs the box. Wait for a free GPU." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
    exit 1
  fi
  echo "guards OK (disk ${USED}%, GPU ${GPU_USED} MiB free-ish) — launching Exp 6"
fi

# ── manifests (idempotent; cheap) ────────────────────────────────────────────
$PY scripts/exp06_build_manifests.py

# ── task 1: shared two-sigmoid detector on CTR-derived labels ────────────────
$PY scripts/train_detector.py --both-n all --seed 1 \
    --out ckpts/detector_ctr_labels.pt --auc-csv results/exp06_oracle/detector_auc.csv \
    --wandb-name "exp6_detector_s01_${DATE}"

# task-1 verify: both sigmoids can exceed 0.5 on a known both-image (co-presence property)
$PY - <<'PY'
import torch, pandas as pd
from metrics.grad_cam_utils import load_model, preprocess
m, ci, ei = load_model("ckpts/detector_ctr_labels.pt",
                       "cuda" if torch.cuda.is_available() else "cpu")
# a both_separable test film the instrument labelled [1,1]
t = pd.read_csv("data/splits/exp06_manifests/test.csv")
both = t[(t.stratum=="both_separable") & (t.cardio==1) & (t.effusion==1)].iloc[0]
x, _ = preprocess("/home/molefe/PhD/Paper3/" + both["path"])
with torch.no_grad():
    p = torch.sigmoid(m(x.to(next(m.parameters()).device)))[0].cpu().tolist()
print(f"[co-presence check] {both['Image Index']}: P(cardio)={p[0]:.2f} P(effusion)={p[1]:.2f}")
print("CO-PRESENCE PROPERTY:", "HOLDS" if p[0] > 0.5 and p[1] > 0.5 else "NOT YET (both must exceed 0.5)")
PY

# ── tasks 4 + 5: Oracle (abundant) and Starved (scarce=50) arms, 3 seeds each ─
for SEED in 1 2 3; do
  $PY scripts/train_detector.py --both-n all --seed "$SEED" \
      --out "ckpts/oracle_classifier_s0${SEED}.pt" \
      --auc-csv results/exp06_oracle/oracle_auc.csv \
      --wandb-name "exp6_oracle_s0${SEED}_${DATE}"
done
for SEED in 1 2 3; do
  $PY scripts/train_detector.py --both-n 50 --seed "$SEED" \
      --out "ckpts/starved_classifier_s0${SEED}.pt" \
      --auc-csv results/exp06_oracle/starved_auc.csv \
      --wandb-name "exp6_starved_s0${SEED}_${DATE}"
done

# ── task 2: Gate-1 pointing-game on the detector ─────────────────────────────
$PY scripts/exp06_gradcam_pointing.py --ckpt ckpts/detector_ctr_labels.pt --n 20

# ── task 6: headroom verdict + figure ────────────────────────────────────────
$PY scripts/exp06_headroom_verdict.py

echo "Exp 6 complete — see results/exp06_oracle/verdict.md"
