#!/usr/bin/env bash
# Waits for the wedged FUSE/CUDA process (zombie holding GPU mem) to release the
# GPU, then launches the recompile-enabled precompute run with logging.
set -u
cd /workspace/Paper3

echo "[relaunch] waiting for GPU memory to free (stuck MooseFS FUSE call)..."
while true; do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l)
    if [ "$apps" -eq 0 ]; then
        break
    fi
    sleep 10
done
echo "[relaunch] GPU free at $(date -u +%H:%M:%S)UTC — launching run"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/precompute_latents_fast.py \
    --csv        data/nih/Data_Entry_2017.csv \
    --image-dir  data/nih/images \
    --vae-ckpt   ckpts/vae_step0025000.pt \
    --out-dir    /workspace/Paper3/data/latents \
    --batch-size 48 --num-workers 16 --write-workers 16 \
    --compile --compile-mode default
