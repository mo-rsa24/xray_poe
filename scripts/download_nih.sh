#!/usr/bin/env bash
# data-foundation plan 03 — download NIH ChestX-ray14 (training corpus).
# Open access, no credentialing, no DUA.
#
# Usage:
#   bash scripts/download_nih.sh [DEST]            # full ~45 GB set (Kaggle)
#   bash scripts/download_nih.sh [DEST] --only N   # sample-first: just archive N (images_00N.zip)
#
# Prereq:  pip install kaggle ; token at ~/.kaggle/access_token (chmod 600)  [or kaggle.json / KAGGLE_USERNAME+KAGGLE_KEY]
# Source:  Kaggle nih-chest-xrays/data   (confirm file names: kaggle datasets files nih-chest-xrays/data)
#          alt, no account: NIH Box https://nihcc.app.box.com/v/ChestXray-NIHCC (per-archive, ships batch_download_zips.py)
set -euo pipefail

DEST="${1:-data/nih}"; shift || true
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="${2:?--only needs an archive number 1-12}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

command -v kaggle >/dev/null 2>&1 || { echo "kaggle CLI not found — run: pip install kaggle" >&2; exit 1; }
[[ -f "$HOME/.kaggle/access_token" || -f "$HOME/.kaggle/kaggle.json" || -n "${KAGGLE_USERNAME:-}" ]] \
  || { echo "no Kaggle auth — expected ~/.kaggle/access_token (or kaggle.json, or KAGGLE_USERNAME+KAGGLE_KEY)" >&2; exit 1; }

SLUG="nih-chest-xrays/data"
mkdir -p "$DEST/images"

# Label table (small) — always.
kaggle datasets download -d "$SLUG" -f Data_Entry_2017.csv -p "$DEST"
[[ -f "$DEST/Data_Entry_2017.csv.zip" ]] && { unzip -o -q "$DEST/Data_Entry_2017.csv.zip" -d "$DEST"; rm -f "$DEST/Data_Entry_2017.csv.zip"; }

if [[ -n "$ONLY" ]]; then
  ARCH="images_$(printf '%03d' "$ONLY").zip"   # verify exact name: kaggle datasets files nih-chest-xrays/data
  echo "sample-first: $ARCH only"
  kaggle datasets download -d "$SLUG" -f "$ARCH" -p "$DEST"
  unzip -o -q "$DEST/$ARCH" -d "$DEST"
  rm -f "$DEST/$ARCH"
else
  echo "full download (~45 GB) — this takes a while"
  kaggle datasets download -d "$SLUG" -p "$DEST"
  unzip -o -q "$DEST/data.zip" -d "$DEST"
  rm -f "$DEST/data.zip"
fi

# Flatten any images_XXX/images/*.png into $DEST/images/
find "$DEST" -path '*/images/*.png' ! -path "$DEST/images/*" -exec mv -t "$DEST/images" {} + 2>/dev/null || true

echo "done — images in $DEST/images: $(ls "$DEST/images" 2>/dev/null | wc -l)"
echo "      labels: $DEST/Data_Entry_2017.csv"
