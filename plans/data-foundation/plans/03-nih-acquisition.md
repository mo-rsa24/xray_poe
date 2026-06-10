# 📥 NIH ChestX-ray14 Acquisition (image-level PNG)

## Background
The chosen feasibility corpus is NIH ChestX-ray14 — already-rendered 8-bit PNGs,
not DICOM. This plan supersedes the download half of `01-acquisition-dicom-monochrome.md`
(retained until `goal-reconcile` retires it).

## Description
Download and extract NIH ChestX-ray14 to local disk and confirm the corpus is the
expected shape before any partitioning or modeling.

1. Source is open — no credentialing, no DUA. **Kaggle** `nih-chest-xrays/data`
   (~45 GB whole set; needs `pip install kaggle` + a `~/.kaggle/access_token` token)
   OR the **NIH Box** mirror <https://nihcc.app.box.com/v/ChestXray-NIHCC>
   (12 `images_0XX.tar.gz` + `Data_Entry_2017.csv`, no account, per-archive — supports
   sample-first; ships `batch_download_zips.py`). HF mirror:
   `alkzar90/NIH-Chest-X-ray-dataset`. Verify the Kaggle slug on the page before scripting.
2. ~42 GB total (under the 50 GB disk budget), so the whole set is pulled and
   filtered locally. NIH offers no CSV-filtered partial download: images are tarred
   by arbitrary index, not by label.
3. Images are 1024×1024, 8-bit grayscale, frontal-only (PA/AP). No DICOM decoding
   and no MONOCHROME1/2 handling is required.

## Purpose
The single-disease experts train on, and the held-out both-disease set is tested
against, images that must physically exist locally and fit the disk budget before
any modeling. Serves Objective 1 and Definition-of-Done #1.

## Goal
NIH ChestX-ray14 extracted to `data/nih/` — ~112,120 PNGs plus
`Data_Entry_2017.csv` — with the disk layout documented in the repo and the corpus
shape sanity-checked (image count, resolution, bit depth, frontal-only).

## Tasks
- [ ] ⚠️ Prereq (Kaggle route): `pip install kaggle`; token → `~/.kaggle/access_token` (chmod 600)
- [ ] ⚠️ Download to `data/nih/`: Kaggle `kaggle datasets download -d nih-chest-xrays/data` (whole ~45 GB) OR NIH Box <https://nihcc.app.box.com/v/ChestXray-NIHCC> (`images_001.tar.gz` first for sample-first) + `Data_Entry_2017.csv`
- [ ] ⚠️ Extract all archives into `data/nih/images/`; document the disk layout in the repo
- [ ] ⚠️ Sanity-check the corpus: image count ≈ 112,120, resolution 1024×1024, 8-bit grayscale
- [ ] ⚠️ Confirm `Data_Entry_2017.csv` loads and that `View Position` is frontal-only (PA/AP)

## Recommended skill
— custom; no skill fits (download + extract is project-specific shell work).

## Engagement Instructions
```
# DO THIS — run the download+extract bash script (the local deliverable; reruns
#   unchanged on RunPod after `git clone` + `bash`)
$ bash scripts/download_nih.sh data/nih      # pulls archives + Data_Entry_2017.csv → extracts to data/nih/images/
# scripts/download_nih.sh (to be written) wraps one of:
#   Kaggle (whole):  pip install kaggle; kaggle datasets download -d nih-chest-xrays/data -p data/nih; (cd data/nih && unzip -q data.zip)
#   NIH Box (sample): download images_001.tar.gz from https://nihcc.app.box.com/v/ChestXray-NIHCC; tar -xzf images_001.tar.gz -C data/nih/images

# Local smoke-test first — one archive (~4 GB, well under the ~50 GB test budget)
#   exercises the script end-to-end before committing to the full ~42 GB pull:
$ bash scripts/download_nih.sh data/nih --only 1

# GET THAT — corpus present and the right shape
$ ls data/nih/images | wc -l            # expect ~112120 PNG (or one archive's worth on the smoke test)
$ python -c "from PIL import Image; im=Image.open('data/nih/images/00000001_000.png'); print(im.size, im.mode)"
# expect: (1024, 1024) L   (8-bit grayscale)
$ python -c "import pandas as pd; d=pd.read_csv('data/nih/Data_Entry_2017.csv'); print(d.shape); print(d['View Position'].value_counts())"
# expect: ~112120 rows; only PA and AP present (frontal-only)
```
