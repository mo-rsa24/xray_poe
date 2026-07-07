# 🧹 Preprocess + Photometric Normalization + Border Filter

## Background
VinDr films must enter the *same* pipeline the NIH latents went through, or Stage-1 learns a different manifold than Stage-2 adapts from. VinDr is **PA-only @512²** (no AP split needed — AP inflates apparent heart size, and there is none here), but it ships **mixed photometric interpretation** (MONOCHROME1 vs MONOCHROME2 — some films are white-background / black-skeleton inverted) which must be normalized to NIH convention before any intensity-based instrument runs. Keep only **border-visible** films (the cardiac border must be readable or composites are untestable) [[rebalanced-retrain-operational]].

## Description
Standardize VinDr images to NIH preprocessing (including photometric-interpretation normalization), confirm PA-only and tag projection, apply the `_border_visible` filter on the original PNGs, and produce a border-visible, projection-tagged, class-mapped manifest ready for VAE-v2 diet assembly.

## Purpose
Serves Objective 1/3 and DoD item 2 — a clean, projection-tagged, border-visible latent cache for Stage-1.

## Goal
`data/latents/vindr/` of border-visible films with per-class N reported.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks

- [x] ✅ **Standardize images to the NIH preprocessing**  ✓ verified (`data/vindr512/std/train/` 13,242 files; histograms at `results/vindr_transfer/histograms/vindr_vs_nih_histograms.png`)
  - [x] ✅ **Normalize photometric interpretation (MONOCHROME1→MONOCHROME2).**  ✓ verified (calibration panel `results/vindr_transfer/histograms/vindr_polarity_flip_panel.png`; 0 genuine MONO1 inversions detected in this batch; threshold=40 documented in `scripts/vindr_preprocess.py`)

- [x] ✅ **Confirm PA-only + tag projection**  ✓ verified (`data/latents/vindr/projection_split.csv` exists; 15,000 rows; all tagged `PA`)

- [x] ✅ **Apply the border-visible filter on the original PNGs + map labels → classes**  ✓ verified (`data/latents/vindr/border_visible_manifest.csv` 6,671 rows; class 0/1/2 present; second_disease=`effusion+thickening` per `results/vindr_transfer/criteria/second_disease.md`)
  - [x] ✅ **Compute cardio border-visible rate and gate Stage-1.**  ✓ verified (1,136 / 1,952 = **58.2% PASS** ≥ 50%; Stage-1 cleared)

- ~~[ ] ❌ **Encode the kept set to latents through our VAE**~~
  > **Blocked** — VinDr latents must be encoded under the frozen **VAE-v2** codec, not
  > `vae_step0025000.pt`. Encoding now would produce latents in the wrong space that
  > are thrown away once VAE-v2 is trained. The re-encode is handled by
  > `additional_plans/vae-v2/plans/05-ldm-reencode.md` (VinDr task already present
  > at that plan's last task block). Prerequisite: `border_visible_manifest.csv` ✅ done.

## Recommended skill
`/visualize-data-samples /home/molefe/data/vindr512` ✅ — eyeball photometric + border-filter results.
   alt: `/data-inventory data/latents/vindr` ✅.

## Engagement Instructions
```bash
$ ls data/latents/vindr/*.pt | wc -l
$ $PY -c "import pandas as pd; d=pd.read_csv('data/latents/vindr/border_visible_manifest.csv'); print(d.groupby('class').size())"
$ $PY -c "import torch,glob; print(torch.load(sorted(glob.glob('data/latents/vindr/*.pt'))[0]).shape)"
# expect: cardio ≫ NIH 854; latent shape (4,128,128)

# ── cardio rate gate ──────────────────────────────────────────────────────────
$ $PY -c "
import pandas as pd
d = pd.read_csv('data/latents/vindr/border_visible_manifest.csv')
kept = (d['class']==1).sum(); print(f'cardio kept {kept}/1952 = {kept/1952:.1%}')
"
# expect: ≥ 50% (≥ 976 films); below → PadChest escalation before Stage-1
```
