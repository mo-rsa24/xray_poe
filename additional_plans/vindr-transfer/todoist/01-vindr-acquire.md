# Acquire VinDr-CXR (512)

**Description:** Pull the VinDr-512 Kaggle mirror onto hippo, derive presence labels for ALL candidate diseases from the bboxes, and verify the images are actually on disk.
**Objective:** Establish the clean external base at 512²/~4 GB (DoD 1).
**Goal:** VinDr PNGs + train.csv on disk; per-disease presence counts (cardio/effusion/thickening) reconciled.
**Verify:** ~15,000 train PNGs physically present AND a sample opens (not just "dir non-empty"); manifest carries cardio/effusion/thickening; cardio∧¬eff ≈ 1,952.
**▶ Recommended prompt:** — custom; Kaggle download on hippo. alt: /data-inventory /home/molefe/data/vindr512

## Tasks
- [x] Pull the VinDr-512 mirror + annotations  (✅ done — 15,000 PNGs present + opens verified)
- [x] Derive per-image presence labels for ALL candidate diseases (cardio, effusion, pleural thickening) — ✅ done (`data/vindr512/presence_manifest.csv` 15,000 rows; cardio∧¬eff=1,952 ✓)
- [x] Spot-check a few films open + are PA frontal — ✅ done (5×(512,512) L ✓)
