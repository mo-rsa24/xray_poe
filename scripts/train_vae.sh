#!/usr/bin/env bash
# VAE training wrapper for RunPod.
#
# Usage:
#   bash scripts/train_vae.sh --overfit              # real-data overfit gate (~30 min)
#   bash scripts/train_vae.sh --config configs/vae.yaml  # full training run
#   bash scripts/train_vae.sh --profile              # measure img/s + VRAM first
#
# The --overfit gate MUST pass before launching the full run.
# Logs go to runs/vae_train_<timestamp>.log and to W&B (if WANDB_API_KEY is set).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${VAE_PYTHON:-python}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
mkdir -p runs ckpts figures logs

log()  { echo "[train_vae] $*"; }
die()  { echo "[train_vae] ERROR: $*" >&2; exit 1; }

MODE=""
CONFIG_FILE=""

for arg in "$@"; do
    case "$arg" in
        --overfit) MODE="overfit" ;;
        --profile) MODE="profile" ;;
        --config)  shift; CONFIG_FILE="$1" ;;
        --config=*) CONFIG_FILE="${arg#--config=}" ;;
    esac
done

# ── helpers ─────────────────────────────────────────────────────────────────

parse_yaml() {
    # minimal key: value parser — no arrays, no anchors
    local key="$1" file="$2"
    grep -E "^${key}:" "$file" | head -1 | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"'\''#' | awk '{print $1}'
}

# ── profile mode: measure img/s before committing to the full run ────────────

if [ "$MODE" = "profile" ]; then
    log "profiling 512² bf16 grad-checkpoint..."
    LOG="logs/vae_profile_${STAMP}.log"
    $PYTHON -m vae.profile \
        --res 512 --precision bf16 --grad-checkpoint --sweep \
        2>&1 | tee "$LOG"
    log "profile complete → $LOG"
    log "update runs/pod_provision.md and configs/vae.yaml (batch) before the full run"
    exit 0
fi

# ── overfit gate: real-data correctness check ────────────────────────────────

if [ "$MODE" = "overfit" ]; then
    log "=== real-data overfit gate ==="
    LOG="runs/vae_overfit_${STAMP}.log"
    $PYTHON -m vae.train \
        --overfit \
        --data real \
        --data-dir data/nih/images \
        --csv data/nih/Data_Entry_2017.csv \
        --res 512 \
        --batch 4 \
        --steps 500 \
        --lr 1e-4 \
        --log-every 50 \
        --log-images-every 100 \
        --figure "figures/vae_overfit_real.png" \
        --wandb-project "${WANDB_PROJECT:-paper3-vae}" \
        2>&1 | tee "$LOG"

    # check that recon loss decreased (last printed recon < 0.5 is a reasonable gate)
    LAST_RECON=$(grep -oP 'recon \K[0-9.]+' "$LOG" | tail -1)
    FIRST_RECON=$(grep -oP 'recon \K[0-9.]+' "$LOG" | head -1)
    log "overfit gate: recon  $FIRST_RECON → $LAST_RECON"
    python - "$FIRST_RECON" "$LAST_RECON" <<'PYEOF'
import sys; f,l = float(sys.argv[1]), float(sys.argv[2])
if l >= f * 0.9:
    print("GATE FAILED — recon did not decrease by >10%. Do NOT proceed to full train.")
    sys.exit(1)
print("GATE PASSED — recon decreased. Proceed to full train.")
PYEOF
    exit 0
fi

# ── full training run ────────────────────────────────────────────────────────

[ -z "$CONFIG_FILE" ] && die "pass --overfit, --profile, or --config <file>"
[ -f "$CONFIG_FILE" ] || die "config not found: $CONFIG_FILE"

log "=== full VAE training run ==="
log "config: $CONFIG_FILE"

# read config fields
DATA_DIR=$(parse_yaml data_dir "$CONFIG_FILE")
CSV=$(parse_yaml csv "$CONFIG_FILE")
RES=$(parse_yaml res "$CONFIG_FILE")
BATCH=$(parse_yaml batch "$CONFIG_FILE")
STEPS=$(parse_yaml steps "$CONFIG_FILE")
LR=$(parse_yaml lr "$CONFIG_FILE")
WEIGHT_DECAY=$(parse_yaml weight_decay "$CONFIG_FILE")
GRAD_CKPT=$(parse_yaml grad_checkpoint "$CONFIG_FILE")
EMA=$(parse_yaml ema "$CONFIG_FILE")
EMA_DECAY=$(parse_yaml ema_decay "$CONFIG_FILE")
VAL_FRAC=$(parse_yaml val_fraction "$CONFIG_FILE")
NUM_WORKERS=$(parse_yaml num_workers "$CONFIG_FILE")
CKPT_DIR=$(parse_yaml ckpt_dir "$CONFIG_FILE")
CKPT_EVERY=$(parse_yaml ckpt_every "$CONFIG_FILE")
CKPT_FINAL=$(parse_yaml ckpt "$CONFIG_FILE")
LOG_EVERY=$(parse_yaml log_every "$CONFIG_FILE")
LOG_IMAGES=$(parse_yaml log_images_every "$CONFIG_FILE")
MANIFOLD=$(parse_yaml manifold_every "$CONFIG_FILE")
WANDB_PROJ=$(parse_yaml wandb_project "$CONFIG_FILE")
FIGURE=$(parse_yaml figure "$CONFIG_FILE")

TRAIN_LOG="runs/vae_train_${STAMP}.log"
log "log → $TRAIN_LOG"
log "W&B project → ${WANDB_PROJ}"
log "steps=${STEPS}  batch=${BATCH}  lr=${LR}  res=${RES}"

EXTRA_FLAGS=""
[ "$GRAD_CKPT" = "true" ] && EXTRA_FLAGS="$EXTRA_FLAGS --grad-checkpoint"
[ "$EMA"       = "true" ] && EXTRA_FLAGS="$EXTRA_FLAGS --ema --ema-decay $EMA_DECAY"

# resume from latest checkpoint if one exists
LATEST_CKPT=$(ls -t "${CKPT_DIR}"/vae_step*.pt 2>/dev/null | head -1 || true)
[ -n "$LATEST_CKPT" ] && EXTRA_FLAGS="$EXTRA_FLAGS --resume $LATEST_CKPT" \
    && log "resuming from $LATEST_CKPT"

$PYTHON -m vae.train \
    --data real \
    --data-dir "$DATA_DIR" \
    --csv "$CSV" \
    --val-fraction "$VAL_FRAC" \
    --num-workers "$NUM_WORKERS" \
    --res "$RES" \
    --batch "$BATCH" \
    --steps "$STEPS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --ckpt-dir "$CKPT_DIR" \
    --ckpt-every "$CKPT_EVERY" \
    --ckpt "$CKPT_FINAL" \
    --log-every "$LOG_EVERY" \
    --log-images-every "$LOG_IMAGES" \
    --manifold-every "$MANIFOLD" \
    --wandb-project "$WANDB_PROJ" \
    --figure "$FIGURE" \
    $EXTRA_FLAGS \
    2>&1 | tee "$TRAIN_LOG"

log "training complete → $TRAIN_LOG"
log "final checkpoint → $CKPT_FINAL"
