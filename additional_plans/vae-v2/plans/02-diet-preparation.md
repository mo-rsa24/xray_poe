# 🥗 Diet Preparation — NIH + VinDr Border-Visible Combined Manifest

## Background
The current VAE was trained on ~112k NIH CXRs with no border-visibility filter. The perceptual fine-tune needs a high-quality, anatomically clean subset: films where the cardiac border is unambiguously visible, so the discriminator has clean signal to sharpen against. We have two sources: NIH border=Y (filtered by `scripts/wire_border_visible_cache.py`, ~4k films) and VinDr-512 border=Y (~1.47k films, manifest at `data/latents/vindr/border_visible_manifest.csv` ✅ done). This plan assembles the combined **image-path** manifest for VAE fine-tuning, confirms patient-disjointness from the NIH frozen test, and produces the augmentation config. Latent encoding happens after VAE-v2 is trained (plan 05), not here — the VAE trains on raw pixels.

## Description
Write `scripts/build_vae_diet.py` that reads the NIH border-visible manifest and the VinDr border-visible manifest, merges them, removes any NIH films whose `Patient ID` appears in `data/splits/test_patient_ids.txt`, and writes a combined training manifest. Then configure the flip + random-crop ±5% augmentation in the VAE training config.

## Purpose
Serves vae-v2 Objective 2 and DoD item 2. Unlocks leaf 03 (VAE retrain). Must be done before any training starts.

## Goal
A combined training manifest `data/vae_v2_diet/train_manifest.csv` (image_path, source, label, border_visible=True) + a VAE training config update specifying the augmentation; N per source reported and patient-disjoint check logged. No `latent_path` column — latents don't exist yet and the VAE trains on images.

## Tasks

- [x] ✅ **Confirm NIH border=Y image paths**  ✓ verified (`data/vae_v2_diet/audit_nih.txt`; N_nih=10,409 border-visible: normal=4,766 / cardio=880 / effusion=4,763; 10,409/10,409 files exist on disk; plan expected ~4k — actual is larger because all labels are included, not cardio-only; N_nih ≥ 3,500 ✓)

- [x] ✅ **Confirm VinDr border=Y image paths**  ✓ verified (`data/latents/vindr/border_visible_manifest.csv` exists; 6,671 rows; class 0=4,774 / 1=1,136 / 2=761; all images at `data/vindr512/vinbigdata/train/`; second_disease=effusion+thickening per `results/vindr_transfer/criteria/second_disease.md`)
  > VinDr latents will be encoded under frozen VAE-v2 by plan 05 — not here.

- [x] ✅ **Write `scripts/build_vae_diet.py`**  ✓ verified (`scripts/build_vae_diet.py`; `data/vae_v2_diet/train_manifest.csv` 17,080 rows; N_nih=10,409 N_vindr=6,671 N_excluded_leaked=0; columns: image_path/source/label/border_visible; no latent_path; sanity checks passed)
  - **Objective:** a reproducible script that builds the combined manifest from the two sources.
  - **Inputs:** NIH border-visible manifest CSV, VinDr border-visible manifest CSV, `data/splits/test_patient_ids.txt`.
  - **Method:**
    1. Load NIH border=Y manifest; subtract rows whose `Patient ID` ∈ `test_patient_ids.txt` (patient-disjoint guard).
    2. Load VinDr border=Y manifest (VinDr is external — no NIH patient-ID overlap by construction; log this assertion).
    3. Concatenate; add `source` column (`nih` / `vindr`); shuffle with fixed seed 42.
    4. Write `data/vae_v2_diet/train_manifest.csv` with columns: `image_path, source, label, border_visible`.
    5. Print: `N_nih=XXX N_vindr=XXX N_total=XXX N_excluded_leaked=XXX`.
    > **No `latent_path` column** — latent paths are populated by plan 05 after VAE-v2 is trained.
  - **Saved artifact:** `scripts/build_vae_diet.py`, `data/vae_v2_diet/train_manifest.csv`
  - **Verify:** script runs without error; CSV row count == N_nih_clean + N_vindr; `source` column has only `nih`/`vindr`; no Patient ID from `test_patient_ids.txt` appears in the NIH rows.

- [x] ✅ **Patient-disjoint check — log result**  ✓ verified (`data/vae_v2_diet/leakage_check.txt`; build-time check N_excluded=0 + post-build re-verification from manifest: 6,454 unique NIH patient IDs ∩ 317 reserved = 0; `grep "intersection=0"` passes)
  - **Objective:** explicit logged assertion that the diet does not contaminate the frozen test.
  - **Method:** after `build_vae_diet.py`, re-verify: load `train_manifest.csv`, filter source==nih, check Patient IDs against `test_patient_ids.txt`; intersection must be 0. Write result to `data/vae_v2_diet/leakage_check.txt`.
  - **Saved artifact:** `data/vae_v2_diet/leakage_check.txt` (`intersection=0` or list of offenders)
  - **Verify:** `grep "intersection=0" data/vae_v2_diet/leakage_check.txt`

- [x] ✅ **Configure flip + random-crop augmentation in VAE training config**  ✓ verified (`configs/vae_v2.yaml`; augmentation.hflip=true / hflip_prob=0.5 / random_crop scale=[0.95,1.0] ratio=[0.99,1.01]; train_manifest=data/vae_v2_diet/train_manifest.csv; batch smoke-test: shape (4,1,512,512) pixel-range [-1.000,1.000] ✓)
  - **Objective:** effective dataset size lifted from ~5.5k to ~44k passes/epoch.
  - **Method:** update (or create) `configs/vae_v2.yaml` (inheriting from `configs/vae_base.yaml` or equivalent) with:
    - `augmentation.hflip: true` (prob 0.5)
    - `augmentation.random_crop: {scale: [0.95, 1.0], ratio: [0.99, 1.01]}` (±5% scale, near-square crops only)
    - `train_manifest: data/vae_v2_diet/train_manifest.csv`
  - **Rationale note in config:** "hflip is CXR-safe for structural sharpening (VAE is a codec, not an anatomy model); random_crop adds scale variation without geometry distortion."
  - **Saved artifact:** `configs/vae_v2.yaml`
  - **Verify:** config loads without error; augmentation keys present; inspect one augmented batch (print shape + pixel-range).

## Recommended skill
— custom; no existing skill fits. Script is straightforward data plumbing.

## Engagement Instructions

```bash
# ── env ───────────────────────────────────────────────────────────────────────
# On hippo: micromamba activate jaxstack

# ── Step 1: NIH border-visible cache ─────────────────────────────────────────
# wire_border_visible_cache.py has no CLI args; reads hardcoded deconfounded split CSVs
# (data/splits/cardio_not_effusion.csv + effusion_not_cardio.csv), scores border_visible
# once (cached → results/exp05_decision/split_border_visible.csv), writes symlinked
# latents to data/latents/rebalanced/.
$ python scripts/wire_border_visible_cache.py --score-limit 20   # smoke (20 films)
$ python scripts/wire_border_visible_cache.py                    # full (~15 min)
# expect output: "border-visible fraction label 1 (cardio): XX%  (854/...)"
#   and "TOTAL train XXXX · patient-disjoint ✓ · border-visible only ✓"
# border=Y latent stems now under data/latents/rebalanced/{0,1,2}/train/

# ── Step 2: VinDr border=Y image paths (manifest already exists ✅) ───────────
# data/latents/vindr/border_visible_manifest.csv written by vindr-transfer leaf 03
# columns: image_id, class, class_name, projection, border_visible, ...
# images live at data/vindr512/vinbigdata/train/{image_id}.png
$ python3 -c "
import pandas as pd
m = pd.read_csv('data/latents/vindr/border_visible_manifest.csv')
print('VinDr border=Y:', len(m))
print(m.groupby('class_name').size().to_dict())
"
# expect: ~6671 total; class 0=normal ~4774, class 1=cardio ~1136, class 2=disease2 ~761
# VinDr latents will be encoded under VAE-v2 by plan 05 — not needed here.

# ── Step 3: build combined manifest ──────────────────────────────────────────
$ python scripts/build_vae_diet.py
# expect: N_nih=XXX N_vindr=XXX N_total=XXX N_excluded_leaked=XX
# (script to be written as part of this leaf's tasks)

$ wc -l data/vae_v2_diet/train_manifest.csv
$ python3 -c "
import pandas as pd
m = pd.read_csv('data/vae_v2_diet/train_manifest.csv')
print('sources:', m.source.value_counts().to_dict())
print('total:', len(m))
assert 'image_path' in m.columns, 'missing image_path column'
assert 'source' in m.columns, 'missing source column'
assert 'latent_path' not in m.columns, 'latent_path should not be here (plan 05 concern)'
"

# ── Step 4: patient-disjoint check ───────────────────────────────────────────
$ grep "intersection=0" data/vae_v2_diet/leakage_check.txt
# expect: intersection=0 (NIH rows only; VinDr is external, no NIH patient IDs)

# ── Step 5: config smoke-check ───────────────────────────────────────────────
$ python3 -c "
import yaml
c = yaml.safe_load(open('configs/vae_v2.yaml'))
assert c.get('augmentation',{}).get('hflip') == True, 'hflip missing'
assert 'random_crop' in c.get('augmentation',{}), 'random_crop missing'
print('config OK:', c['augmentation'])
print('train_manifest:', c.get('train_manifest'))
"
```
