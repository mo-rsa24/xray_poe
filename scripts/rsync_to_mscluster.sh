#!/usr/bin/env bash
# Move the git-EXCLUDED heavy state from hippo → mscluster.
#
# Git carries code + plans (additional_plans/) + docs. This script moves everything else:
#   - Precomputed latents and canonical checkpoints (expensive to recompute)
#   - data/splits, data/vae_v2_diet, data/vindr512 manifests (gitignored small CSVs)
#   - .claude/ (skills, memory, commands — gitignored per .gitignore comment)
# Raw NIH/VinDr images are NOT moved — reuse /datasets/mmolefe or download on-node with
# scripts/download_{nih,vindr}.sh.
#
# IMPORTANT: hippo cannot reach mscluster directly. Two ways to run this:
#   MODE=push — run from a box that CAN ssh to mscluster (e.g. your laptop with the repo),
#               pushing local files to mscluster.
#   MODE=pull — run ON mscluster, pulling from hippo (needs mscluster→hippo ssh to work).
#
# Dataset store on mscluster is /datasets/mmolefe (NOT the project dir). After transfer,
# symlink the repo's data/ and ckpts/ at it (printed at the end) so relative paths resolve.
#
# Usage:
#   MODE=push DEST_SSH=mmolefe@mscluster            bash scripts/rsync_to_mscluster.sh   # DRY-RUN
#   MODE=push DEST_SSH=mmolefe@mscluster DRY=0      bash scripts/rsync_to_mscluster.sh   # execute
#   MODE=pull SRC_SSH=molefe@hippo   DRY=0          bash scripts/rsync_to_mscluster.sh   # run ON mscluster
#   SNAPSHOT=1 …                                    also move results/outputs/figures (bigger)
set -euo pipefail

MODE="${MODE:-push}"
DRY="${DRY:-1}"                              # 1 = dry-run (rsync -n); 0 = execute
DEST_SSH="${DEST_SSH:-mmolefe@mscluster}"    # for MODE=push
SRC_SSH="${SRC_SSH:-}"                        # for MODE=pull (user@hippo)
SRC_REPO="${SRC_REPO:-PhD/Paper3}"           # repo path on the SOURCE box (relative to its home)
DATA_DEST="${DATA_DEST:-/datasets/mmolefe/data}"
CKPT_DEST="${CKPT_DEST:-/datasets/mmolefe/checkpoints}"
PROJ_DEST="${PROJ_DEST:-/home-mscluster/mmolefe/Playground/PhD/Paper3}"
SNAPSHOT="${SNAPSHOT:-0}"                     # 1 = also move results/outputs/figures
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RFLAGS="-avP --human-readable"
if [ "$DRY" = "1" ]; then RFLAGS="$RFLAGS -n"; echo "*** DRY-RUN — set DRY=0 to actually transfer ***"; fi

# What MUST move (edit if your checkpoint paths differ):
LATENTS=(data/latents_v2 data/latents)
MANIFESTS=(data/splits data/vae_v2_diet data/vindr512)   # small CSVs; gitignored but needed by scripts
CKPTS=(
  ckpts/vae_step0025000.pt                                     # v1 VAE codec
  ckpts/vae_v2                                                 # v2 VAE codec
  ckpts/vae_v2_ldm ckpts/vae_v2_ldm_v2 ckpts/vae_v2_ldm_v3 ckpts/vae_v2_ldm_v4  # LDM v1/v2/v3(+v4) on v2 codec
  ckpts/presence_classifier_finetuned.pt ckpts_arm1_cleancardio)
# Loose root files = the v1-VAE-codec LDM set (model + optimizer state, all steps)
CKPT_GLOBS=('ckpts/model_step*.safetensors' 'ckpts/training_state_step*.pt')
SNAP=(results outputs figures plans illuminate-output learning-captures briefs paper)

run() { echo "+ rsync $RFLAGS $*"; rsync $RFLAGS "$@"; }

case "$MODE" in
  push)
    echo "MODE=push  →  $DEST_SSH   (data→$DATA_DEST  ckpts→$CKPT_DEST)"
    [ "$DRY" = "0" ] && ssh "$DEST_SSH" "mkdir -p '$DATA_DEST' '$CKPT_DEST' '$PROJ_DEST'"
    for p in "${LATENTS[@]}";   do [ -e "$REPO/$p" ] && run "$REPO/$p/" "$DEST_SSH:$DATA_DEST/$(basename "$p")/"   || echo "  skip (absent): $p"; done
    for p in "${MANIFESTS[@]}"; do [ -e "$REPO/$p" ] && run "$REPO/$p/" "$DEST_SSH:$DATA_DEST/$(basename "$p")/"   || echo "  skip (absent): $p"; done
    for p in "${CKPTS[@]}";     do [ -e "$REPO/$p" ] && run "$REPO/$p"  "$DEST_SSH:$CKPT_DEST/"                    || echo "  skip (absent): $p"; done
    for g in "${CKPT_GLOBS[@]}"; do for f in $REPO/$g; do [ -e "$f" ] && run "$f" "$DEST_SSH:$CKPT_DEST/"; done; done
    [ -e "$REPO/.claude" ] && run "$REPO/.claude/" "$DEST_SSH:$PROJ_DEST/.claude/"
    if [ "$SNAPSHOT" = "1" ]; then for p in "${SNAP[@]}"; do [ -e "$REPO/$p" ] && run "$REPO/$p" "$DEST_SSH:$PROJ_DEST/"; done; fi
    ;;
  pull)
    [ -n "$SRC_SSH" ] || { echo "MODE=pull needs SRC_SSH=user@hippo"; exit 2; }
    echo "MODE=pull  ←  $SRC_SSH:$SRC_REPO   (data→$DATA_DEST  ckpts→$CKPT_DEST)"
    [ "$DRY" = "0" ] && mkdir -p "$DATA_DEST" "$CKPT_DEST" "$PROJ_DEST"
    for p in "${LATENTS[@]}";   do run "$SRC_SSH:$SRC_REPO/$p/" "$DATA_DEST/$(basename "$p")/"; done
    for p in "${MANIFESTS[@]}"; do run "$SRC_SSH:$SRC_REPO/$p/" "$DATA_DEST/$(basename "$p")/"; done
    for p in "${CKPTS[@]}";     do run "$SRC_SSH:$SRC_REPO/$p"  "$CKPT_DEST/"; done
    for g in "${CKPT_GLOBS[@]}"; do run "$SRC_SSH:$SRC_REPO/$g" "$CKPT_DEST/"; done   # remote shell expands the glob
    run "$SRC_SSH:$SRC_REPO/.claude/" "$PROJ_DEST/.claude/"
    if [ "$SNAPSHOT" = "1" ]; then for p in "${SNAP[@]}"; do run "$SRC_SSH:$SRC_REPO/$p" "$PROJ_DEST/"; done; fi
    ;;
  *) echo "MODE must be 'push' or 'pull'"; exit 2 ;;
esac

cat <<EOF

── next, ON mscluster: wire the repo's relative paths to the dataset store ──
  cd $PROJ_DEST
  ln -sfn $DATA_DEST data      # if /datasets/mmolefe/data holds nih/ + latents_v2/ ; else symlink subdirs
  ln -sfn $CKPT_DEST ckpts
  ls data/latents_v2 && ls ckpts/vae_v2_ldm_v4   # verify the symlinks resolve
EOF
