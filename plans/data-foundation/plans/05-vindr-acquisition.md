# 📥 VinDr-CXR Resized-PNG Acquisition + Image-Level Labels (eval set)

## Background
VinDr-CXR is the held-out **evaluation** set — expert labels from 3 radiologists,
the clean complement to NIH's noisy NLP-derived labels (`03-nih-acquisition.md`).
The raw release is DICOM (~200 GB, MONOCHROME1 handling); instead we use the
community **resized-PNG redistributions** on Kaggle (a few GB, already rendered),
so no DICOM decoding is needed here. Labels ship as bounding boxes; image-level
labels are derived by collapsing the boxes.

## Description
Download the VinDr-CXR resized-PNG corpus, derive image-level labels from the box
annotations, and partition it into the four hypothesis groups — emphasising the
held-out both-disease set the PoE composition is tested against.

1. Source (needs `pip install kaggle` + a `~/.kaggle/access_token` token, and accepting
   the competition rules once in-browser): images from a Kaggle resized-PNG
   redistribution — `xhlulu/vinbigdata-chest-xray-resized-png-1024x1024` (1024px),
   `xhlulu/vinbigdata-chest-xray-png-512px-original-ratio` (512px), or
   `corochann/vinbigdata-chest-xray-original-png` (full-res) — plus box-annotation
   labels via the competition `train.csv` from
   `vinbigdata-chest-xray-abnormalities-detection`. No credentialing/DUA (the DICOM
   original on PhysioNet does; not used here). Verify slugs on the page.
2. ~few GB at 1024px (18,000 images), well under the disk budget → pull the whole
   set, no partial download needed. Sample-first locally to validate the script,
   then bulk-pull onto the RunPod volume.
3. Labels are bounding boxes from 3 radiologists; collapse to an image-level
   multi-label vector (a finding is present if *any* radiologist boxed it). Map to
   the common label space {cardiomegaly, effusion, pneumothorax, normal} —
   `No finding` → normal. NIH's `Finding Labels` already use these names, so the
   space is shared across both datasets.
4. Assign each image to the same four groups as NIH (normal / cardio-only /
   effusion-only / both), recording per-pathology counts — especially the uncertain
   pneumothorax count flagged in the dataset research.

## Purpose
Produce the clean, expert-labeled evaluation corpus (and the held-out co-morbid
set) the single-disease experts' PoE composition is tested against — physically
present locally and fit to the disk budget before any modeling. Serves Objectives
1 & 3 and Definition-of-Done #1 (VinDr), #3 (VinDr labels + common space), #4
(VinDr groups).

## Goal
VinDr-CXR resized PNGs extracted to `data/vindr/` (+ `train.csv`), an image-level
label table (path → {cardiomegaly, effusion, pneumothorax, normal}) derived from
the boxes and folded into the common label space, plus
`data/vindr/four_group_counts.md` recording per-pathology and four-group counts
and the pneumothorax-count viability check.

## Tasks
- [ ] ⚠️ Prereq: `pip install kaggle`; token → `~/.kaggle/access_token` (chmod 600); accept the `vinbigdata-chest-xray-abnormalities-detection` competition rules in-browser
- [ ] ⚠️ Images: `kaggle datasets download -d xhlulu/vinbigdata-chest-xray-resized-png-1024x1024 -p data/vindr` (alt 512px: `xhlulu/vinbigdata-chest-xray-png-512px-original-ratio`); Labels: `kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection -f train.csv -p data/vindr`; smoke-test on a ~200-image sample first
- [ ] ⚠️ Bulk-pull onto the RunPod volume (same script after `git clone`); extract to `data/vindr/images/`; document the disk layout
- [ ] ⚠️ Collapse box annotations → image-level multi-label (present if any of the 3 radiologists boxed it); map to common space {cardiomegaly, effusion, pneumothorax, normal}, `No finding` → normal
- [ ] ⚠️ Assign each VinDr image to the four groups (normal / cardio-only / effusion-only / both); emit a partition index (path → group)
- [ ] ⚠️ Record per-pathology + four-group counts in `data/vindr/four_group_counts.md`; explicitly check the pneumothorax count and flag if the held-out co-morbid group is too thin for eval

## Recommended skill
▶ `/data-inventory data/vindr` ✅ — profiles size + per-label counts once labels are derived.
   alt: `/data-distributions` for the label co-occurrence view; the box→image-level collapse + Kaggle download are custom.

## Engagement Instructions
```
# DO THIS — Kaggle download (sample-first), then box→image-level label build
$ bash scripts/download_vindr.sh data/vindr            # kaggle download of the resized-PNG redistribution + train.csv → data/vindr/images/
# scripts/download_vindr.sh (to be written) wraps:
#   pip install kaggle; token → ~/.kaggle/access_token; accept competition rules in-browser
#   Images: kaggle datasets download -d xhlulu/vinbigdata-chest-xray-resized-png-1024x1024 -p data/vindr; (cd data/vindr && unzip -q '*.zip')
#   Labels: kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection -f train.csv -p data/vindr
$ bash scripts/download_vindr.sh data/vindr --sample 200   # local smoke-test before the full (~few GB) pull

# GET THAT — corpus present and the right shape
$ ls data/vindr/images | wc -l        # expect ~18000 PNG (or 200 on the sample)
$ python -c "import glob; from PIL import Image; im=Image.open(sorted(glob.glob('data/vindr/images/*.png'))[0]); print(im.size, im.mode)"
# expect a square-ish PNG (resized redistribution), mode 'L' or 'RGB'

# Build image-level labels from boxes + four-group assignment
$ python -m data.vindr_labels --boxes data/vindr/train.csv --out data/vindr/partition.parquet
# expect: prints per-pathology counts (incl. pneumothorax) and four-group counts (normal / cardio_only / effusion_only / both)
$ cat data/vindr/four_group_counts.md
# expect: per-pathology + four-group counts, the pneumothorax-count flag, and the held-out co-morbid viability note
```
