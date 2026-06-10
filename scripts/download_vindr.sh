#!/usr/bin/env bash
# data-foundation plan 05 — download VinDr-CXR (held-out eval corpus).
# Resized-PNG redistribution (no DICOM) + competition box-annotation labels.
#
# Usage:
#   bash scripts/download_vindr.sh [DEST]             # full: 1024px PNG images + train.csv
#   bash scripts/download_vindr.sh [DEST] --sample N  # labels + extract first N images for a quick look
#
# Prereq:  pip install kaggle ; token ~/.kaggle/access_token (chmod 600) [or kaggle.json / env] ;
#          ACCEPT competition rules ONCE in-browser:
#          https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/rules
# Sources (verify slugs on the page):
#   images  -> xhlulu/vinbigdata-chest-xray-resized-png-1024x1024  (alt 512px: xhlulu/vinbigdata-chest-xray-png-512px-original-ratio)
#   labels  -> competition vinbigdata-chest-xray-abnormalities-detection (train.csv)
set -euo pipefail

DEST="${1:-data/vindr}"; shift || true
SAMPLE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample) SAMPLE="${2:?--sample needs an image count}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

command -v kaggle >/dev/null 2>&1 || { echo "kaggle CLI not found — run: pip install kaggle" >&2; exit 1; }
[[ -f "$HOME/.kaggle/access_token" || -f "$HOME/.kaggle/kaggle.json" || -n "${KAGGLE_USERNAME:-}" ]] \
  || { echo "no Kaggle auth — expected ~/.kaggle/access_token (or kaggle.json, or KAGGLE_USERNAME+KAGGLE_KEY)" >&2; exit 1; }

IMG_SLUG="xhlulu/vinbigdata-chest-xray-resized-png-1024x1024"
COMP="vinbigdata-chest-xray-abnormalities-detection"
mkdir -p "$DEST/images"

# Box-annotation labels (small) — needs the competition rules accepted.
kaggle competitions download -c "$COMP" -f train.csv -p "$DEST" \
  || { echo "label download failed — accept the competition rules in-browser first (see header URL)" >&2; exit 1; }
[[ -f "$DEST/train.csv.zip" ]] && { unzip -o -q "$DEST/train.csv.zip" -d "$DEST"; rm -f "$DEST/train.csv.zip"; }

# Images (single archive — can't partial-download; --sample only limits extraction)
kaggle datasets download -d "$IMG_SLUG" -p "$DEST"
ZIP="$DEST/$(basename "$IMG_SLUG").zip"
if [[ -n "$SAMPLE" ]]; then
  echo "sample: extracting first $SAMPLE images (download was full — the resized-PNG set is one archive)"
  mapfile -t NAMES < <(unzip -Z1 "$ZIP" | grep -i '\.png$' | head -n "$SAMPLE")
  [[ ${#NAMES[@]} -gt 0 ]] && unzip -o -q -j "$ZIP" "${NAMES[@]}" -d "$DEST/images"
else
  unzip -o -q -j "$ZIP" '*.png' -d "$DEST/images"
fi
rm -f "$ZIP"

echo "done — images in $DEST/images: $(ls "$DEST/images" 2>/dev/null | wc -l)"
echo "      labels: $DEST/train.csv"
