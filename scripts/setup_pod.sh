#!/usr/bin/env bash
# Bootstrap script for a fresh RunPod instance.
# Run ONCE after git clone: bash scripts/setup_pod.sh
# Blocks until NIH ChestX-ray14 is fully downloaded.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[setup] $*"; }

# ── 1. CUDA / GPU sanity ─────────────────────────────────────────────────────
log "checking CUDA..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PYEOF'
import torch, sys
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "none"
mem  = torch.cuda.get_device_properties(0).total_memory/1e9 if ok else 0
print(f"torch {torch.__version__}  cuda={ok}  gpu={name}  mem={mem:.1f}GB")
if not ok:
    print("ERROR: no CUDA — check pod config"); sys.exit(1)
PYEOF

# ── 2. Python deps ───────────────────────────────────────────────────────────
log "installing Python dependencies..."
pip install --quiet --no-input \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
pip install --quiet --no-input -r requirements.txt
python -c "import monai, pytorch_msssim, lpips, wandb; print('deps OK')"

# ── 3. W&B login ─────────────────────────────────────────────────────────────
log "W&B login..."
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY" --relogin
    log "W&B: logged in via WANDB_API_KEY env var"
else
    log "WANDB_API_KEY not set — prompting interactively (or set it before running)"
    wandb login
fi

# ── 4. Kaggle credentials ─────────────────────────────────────────────────────
log "Kaggle credentials..."
KAGGLE_DIR="$HOME/.kaggle"
mkdir -p "$KAGGLE_DIR" && chmod 700 "$KAGGLE_DIR"

if [ -f "$KAGGLE_DIR/kaggle.json" ]; then
    log "kaggle.json already present"
elif [ -n "${KAGGLE_KEY:-}" ] && [ -n "${KAGGLE_USERNAME:-}" ]; then
    echo "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" \
        > "$KAGGLE_DIR/kaggle.json"
    chmod 600 "$KAGGLE_DIR/kaggle.json"
    log "kaggle.json written from env vars"
else
    log "ERROR: no Kaggle credentials found."
    log "  Option A: set KAGGLE_USERNAME + KAGGLE_KEY env vars before running this script."
    log "  Option B: copy kaggle.json to $KAGGLE_DIR/kaggle.json manually."
    exit 1
fi

# ── 5. Download NIH ChestX-ray14 (blocking) ──────────────────────────────────
DATA_DIR="$REPO_ROOT/data/nih"
if [ -d "$DATA_DIR/images" ] && [ "$(ls -1 "$DATA_DIR/images" | wc -l)" -gt 100000 ]; then
    log "NIH images already present ($(ls -1 "$DATA_DIR/images" | wc -l) files) — skipping download"
else
    log "downloading NIH ChestX-ray14 (~45 GB, this will take a while)..."
    bash scripts/download_nih.sh
    log "NIH download complete: $(ls -1 "$DATA_DIR/images" | wc -l) images"
fi

# ── 6. Create required directories ───────────────────────────────────────────
mkdir -p ckpts runs figures logs
log "directories: ckpts/ runs/ figures/ logs/ ready"

# ── 7. Smoke test: vae.sanity (architecture gate) ────────────────────────────
log "running vae.sanity (architecture gate)..."
python -m vae.sanity
log "vae.sanity: PASSED"

# ── 8. Record provisioning facts ─────────────────────────────────────────────
PROVISION_LOG="runs/pod_provision.md"
{
    echo "# Pod Provisioning Record"
    echo "date: $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo ""
    echo "## Hardware"
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader | awk -F',' '{printf "- GPU: %s\n- VRAM: %s\n- driver: %s\n", $1, $2, $3}'
    echo ""
    echo "## Environment"
    python - <<'PYEOF'
import torch, monai
print(f"- torch: {torch.__version__}")
print(f"- monai: {monai.__version__}")
PYEOF
    echo ""
    echo "## Status"
    echo "- NIH images: $(ls -1 data/nih/images | wc -l)"
    echo "- vae.sanity: PASSED"
    echo ""
    echo "## Measured throughput (fill after 00-profile step)"
    echo "- img/s @512 bf16 grad-ckpt: ⟨measure⟩"
    echo "- peak VRAM @512 bf16 grad-ckpt: ⟨measure⟩ GB"
    echo "- max batch @512 bf16 grad-ckpt: ⟨measure⟩"
} > "$PROVISION_LOG"
log "provisioning record → $PROVISION_LOG"

log "================================================================"
log "setup complete — ready for scripts/train_vae.sh"
log "================================================================"
