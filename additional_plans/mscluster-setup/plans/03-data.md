# 📦 Data: Verify and Transfer

## Description
Verify what is already in `/datasets/mmolefe` (NIH splits, VinDr zip, latents), unzip VinDr, then
rsync from hippo via `rsync_to_mscluster.sh`. The script transfers latents AND the gitignored data
manifests (`data/splits/`, `data/vae_v2_diet/`, `data/vindr512/`) that scripts need at runtime.
NIH raw images are NOT rsynced — reuse `/datasets/mmolefe/data/nih` or download on-node.

## Purpose
O3 data. The scripts/generate.py and effusion_blunt_check.py both need latents and the NIH manifest.
Reusing `/datasets/mmolefe` avoids a full re-download if the data is already there.

## Goal
`/datasets/mmolefe/data` contains NIH splits (Data_Entry*.csv visible, images_0* dirs present) and
latents (`latents_v2/joint_perch/` present); VinDr unzipped. Manifest counts match hippo originals.

## Tasks
- [x] ✅ Inspect what is already on the cluster: ✓ verified (`data/` has nih, latents, latents_v2, splits, vae_v2_diet, vindr512)
  ```
  ls -la /datasets/mmolefe/data /datasets/mmolefe/checkpoints
  find /datasets/mmolefe -maxdepth 3 \( -iname "Data_Entry*" -o -iname "images_0*" \)
  ```
- [x] ✅ Unzip VinDr (skip if already extracted): ✓ verified (already extracted at `/datasets/mmolefe/vinbigdata/`: train/, train.csv present)
  ```
  unzip -n /datasets/mmolefe/vinbigdata-chest-xray-abnormalities-detection.zip -d /datasets/mmolefe/vinbigdata
  ```
- [x] ✅ Rsync latents: Not needed — 162,004 `.pt` latents already present in `/datasets/mmolefe/data/latents_v2` (≥74k expected). Original command below left for reference:
  ```
  MODE=push DEST_SSH=mmolefe@mscluster DRY=0 bash scripts/rsync_to_mscluster.sh
  ```
- [x] ✅ Remedy (NIH absent): DONE — kaggle CLI works; `download_nih.sh` finished: **112,120** images extracted to `/datasets/mmolefe/data/nih/images/` (full ChestX-ray14 set), `Data_Entry_2017.csv` present, `data.zip` auto-removed. Empty `images_00{1..12}` shell dirs remain (harmless; safe to `rmdir`). Note: `kaggle whoami` is invalid in kaggle 2.2.3 — use `kaggle config view` instead.
  ```bash
  pip install kaggle
  mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
  # copy kaggle.json, or write it from env vars:
  # echo '{"username":"'$KAGGLE_USERNAME'","key":"'$KAGGLE_KEY'"}' > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
  kaggle whoami   # must pass before proceeding
  bash scripts/download_nih.sh /datasets/mmolefe/data/nih
  ```
- [x] ✅ Spot-check latent count: ✓ verified 162,004 `.pt` files (≥74k expected).
  `find /datasets/mmolefe/data/latents_v2 -name "*.pt" | wc -l` — expect
  ~74k+ (NIH 70,613 train + 3,715 val + VinDr 6,671).
- [x] ✅ Verify manifests arrived: ✓ verified — `splits/` includes `cardio_not_effusion.csv`,
  `effusion_not_cardio.csv`, `test_patient_ids.txt`; `vae_v2_diet/` and `vindr512/` present.

## Engagement Instructions
```bash
find /datasets/mmolefe/data/latents_v2 -name "*.pt" | wc -l
ls /datasets/mmolefe/data/latents_v2/joint_perch/scale_factor.pt
ls /datasets/mmolefe/data/splits/cardio_not_effusion.csv \
   /datasets/mmolefe/data/splits/effusion_not_cardio.csv \
   /datasets/mmolefe/data/splits/test_patient_ids.txt
ls /datasets/mmolefe/data/ | head -20
```
Expected: latent count ≥74k; `scale_factor.pt` present; all three splits CSVs present;
`nih/`, `latents_v2/`, `splits/` visible in data/ listing. DoD-3 met.

## Recommended skill
— custom; no skill fits (project-specific rsync and dataset verification).
