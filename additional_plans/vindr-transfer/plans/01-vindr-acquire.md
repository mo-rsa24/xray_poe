# 📥 Acquire VinDr-CXR (512) — External Base

## Background
The transfer lever needs an external, NIH-disjoint cohort with cardio. **VinDr-CXR / VinBigData** (Vietnamese, Vingroup) is the pragmatic base: 15,000 **PA** films, 3-radiologist bounding-box labels, including Cardiomegaly, Pleural effusion, **and Pleural thickening** (the two effusion/thickening options are both candidate 2nd-diseases — decided in leaf 09). Crucially it ships as **512² PNG** on Kaggle (~4 GB) — no 1 TB download, no resolution compromise, no PA/AP split. Counts (from `train.csv`, box-level): **Cardiomegaly 5,427 · Pleural thickening 4,842 · Pleural effusion 2,476** (thickening is ~2× effusion); image-level deconfounded **cardio∧¬effusion 1,952** (2.3× NIH's 854), effusion∧¬cardio 684, **No-finding 10,606**. It was dropped from the project earlier purely for compute (pre-hippo); that constraint is gone. PadChest (10k cardio) is the headroom escalation if VinDr's boost is insufficient.

## Description
Pull a VinDr-512 Kaggle mirror (images + `train.csv` annotations) onto hippo, derive presence labels from the bboxes, verify counts.

## Purpose
Serves Objective 1 and DoD item 1 — the clean external base, at 512 and ~4 GB.

## Goal
VinDr 512² PNGs + `train.csv` on disk; per-class presence counts reported and reconciled.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python` · `KG="$PY -m kaggle"` (auth'd on hippo)

## Tasks

- [x] ✅ **Pull the VinDr-512 mirror + annotations** — ✓ verified (`/home/molefe/data/vindr512/vinbigdata/train.csv` + `train/` PNGs on disk)
  - `$ run:`
    ```bash
    DEST=/home/molefe/data/vindr512; mkdir -p "$DEST"
    USED=$(df --output=pcent / | tail -1 | tr -dc '0-9'); echo "disk ${USED}%"; [ "$USED" -ge 90 ] && { echo ABORT; exit 1; }
    $KG datasets download -d awsaf49/vinbigdata-512-image-dataset -p "$DEST" --unzip
    ls "$DEST" | head; find "$DEST" -name train.csv
    ```
  - `✓ verify:` `train.csv` present AND **~15,000 train PNGs physically on disk and a sample opens** (assert the count + open a file — not just "dir non-empty"); `df` < 90% (it's ~4 GB).
  - `⊳ note:` alt mirrors if that ref 403s: `zstusnoopy/vinbigdata-chest-xray-resized-png-512x512`, `robertlangdonvinci/vinbigdata-resized-image-512`. Original-res PNG (bigger): `corochann/vinbigdata-chest-xray-original-png`.

- [x] ✅ **Derive per-image presence labels (ALL candidate diseases)** — ✓ verified (`scripts/vindr_presence_labels.py` written + run; `data/vindr512/presence_manifest.csv` 15,000 rows; cardio∧¬eff=1,952 ✓, normal=10,606 ✓, thickening counts printed for both pairs)
  - `▶ prompt:` `Write scripts/vindr_presence_labels.py: from train.csv (cols image_id, class_name, rad_id, bbox…) build a per-image presence table with a flag PER candidate disease: cardio=1 if ANY radiologist drew a Cardiomegaly box, effusion=1 if any Pleural effusion box, thickening=1 if any Pleural thickening box (majority-of-3 optional flag each), normal if only "No finding". Write data/vindr512/presence_manifest.csv (image_id, cardio, effusion, thickening). DO NOT hardcode a single 2nd disease — the class integer {normal:0, cardio:1, <2nd>:2} is assigned in leaf 03 AFTER leaf 09 picks effusion vs thickening. Print deconfounded counts for BOTH candidate pairs: cardio∧¬effusion, cardio∧¬thickening, effusion∧¬cardio, thickening∧¬cardio, normal.`
  - `✓ verify:` manifest carries cardio + effusion + thickening presence flags; cardio∧¬eff ≈ 1,952 (≫ NIH 854), normal ≈ 10,606; thickening counts reported so leaf 09 can choose on real numbers.

- [x] ✅ **Spot-check a few films open + are PA frontal** — ✓ verified (5 sampled; all (512, 512) L — correct grayscale CXRs)
  - `$ run:` `$PY -c "from PIL import Image; import glob; [print(p, Image.open(p).size, Image.open(p).mode) for p in glob.glob('/home/molefe/data/vindr512/**/*.png', recursive=True)[:5]]"`
  - `✓ verify:` images are 512², valid grayscale CXRs.

## Recommended skill
— custom; Kaggle download on hippo + label derivation.
   alt: `/data-inventory /home/molefe/data/vindr512` ✅ — profile the pull (feeds leaf 04).

## Engagement Instructions
```bash
# Gap-B: the data is ACTUALLY there — assert the image count + open one, not just "dir non-empty"
$ find /home/molefe/data/vindr512 -name train.csv
$ N=$(find /home/molefe/data/vindr512 -path '*/train/*.png' | wc -l); echo "train PNGs: $N"   # expect ~15,000
$ $PY -c "from PIL import Image,glob; p=glob.glob('/home/molefe/data/vindr512/**/train/*.png',recursive=True)[0]; im=Image.open(p); print('opens', im.size, im.mode)"
# presence manifest carries all three candidate diseases
$ $PY -c "import pandas as pd; d=pd.read_csv('data/vindr512/presence_manifest.csv'); print([c for c in d.columns if c in ('cardio','effusion','thickening')]); print('cardio∧¬eff', ((d.cardio==1)&(d.effusion==0)).sum(), '| thickening', (d.thickening==1).sum())"
# expect: ~15,000 train PNGs on disk + opens; manifest has cardio/effusion/thickening; cardio∧¬eff ≈ 1,952
```
> ⚠️ Headroom escalation (more cardio): the parked **PadChest** route (1 TB full-res / stream-filter to 512) in `PARKING_LOT.md` — only if VinDr's ~1.3k border-visible cardio plateaus.
