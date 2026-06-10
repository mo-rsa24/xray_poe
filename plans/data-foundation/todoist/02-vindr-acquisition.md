# 02 · VinDr-CXR acquisition + image-level labels

[⌂ Index](00-INDEX.md) · [← prev 01](01-nih-acquisition.md) · [next → 03](03-four-group-partition.md)

## Reference while you do it
- 📄 Plan: plans/data-foundation/plans/05-vindr-acquisition.md
- 🔗 Images: Kaggle `xhlulu/vinbigdata-chest-xray-resized-png-1024x1024` (alt 512px `xhlulu/vinbigdata-chest-xray-png-512px-original-ratio`) · Labels: competition `vinbigdata-chest-xray-abnormalities-detection`

## Section context (paste into the Todoist section)
**Description:** VinDr-CXR is the held-out evaluation set — expert labels from 3 radiologists, the clean complement to NIH's NLP-derived labels. Use the Kaggle resized-PNG redistributions (a few GB, already rendered — no DICOM path); labels ship as boxes and are collapsed to image-level.
**Objective:** Produce the clean, expert-labeled eval corpus (and held-out co-morbid set) the PoE composition is tested against.
**Goal:** VinDr resized PNGs in `data/vindr/` (+ `train.csv`), an image-level label table folded into the common space {cardiomegaly, effusion, pneumothorax, normal}, and `data/vindr/four_group_counts.md` with per-pathology + four-group counts incl. the pneumothorax viability check.
**Verify:** `ls data/vindr/images | wc -l` ≈ 18000; `python -m data.vindr_labels --boxes data/vindr/train.csv --out data/vindr/partition.parquet` prints per-pathology counts (incl. pneumothorax) + four-group counts; `cat data/vindr/four_group_counts.md` shows the held-out co-morbid viability note.
**▶ Recommended prompt:** `/data-inventory data/vindr` ✅ — profiles size + per-label counts once labels are derived. alt: `/data-distributions` for label co-occurrence; the box→image-level collapse + Kaggle download are custom.

**Do (strike through as you finish):**
- Prereq: `pip install kaggle`; token → `~/.kaggle/access_token` (chmod 600); accept the `vinbigdata-chest-xray-abnormalities-detection` competition rules in-browser
- Images: `kaggle datasets download -d xhlulu/vinbigdata-chest-xray-resized-png-1024x1024 -p data/vindr` (alt 512px: `xhlulu/vinbigdata-chest-xray-png-512px-original-ratio`) — verify slug on the page
- Labels: `kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection -f train.csv -p data/vindr`; unzip into `data/vindr/`; smoke-test on a ~200-image sample first
- Bulk-pull onto the RunPod volume (same commands after `git clone`); extract to `data/vindr/images/`; document the disk layout
- Collapse box annotations → image-level multi-label (present if any of the 3 radiologists boxed it); map to common space {cardiomegaly, effusion, pneumothorax, normal}, `No finding` → normal
- Assign each VinDr image to the four groups (normal / cardio-only / effusion-only / both); emit a partition index (path → group)
- Record per-pathology + four-group counts in `data/vindr/four_group_counts.md`; explicitly check the pneumothorax count and flag if the held-out co-morbid group is too thin for eval
