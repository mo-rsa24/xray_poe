# 01 · NIH ChestX-ray14 acquisition

[⌂ Index](00-INDEX.md) · [next → 02](02-vindr-acquisition.md)

## Reference while you do it
- 📄 Plan: plans/data-foundation/plans/03-nih-acquisition.md
- 🔗 Kaggle `nih-chest-xrays/data` · NIH Box https://nihcc.app.box.com/v/ChestXray-NIHCC · HF `alkzar90/NIH-Chest-X-ray-dataset`

## Section context (paste into the Todoist section)
**Description:** NIH ChestX-ray14 is the training corpus — already-rendered 8-bit PNGs (no DICOM, no MONOCHROME handling), open access, no credentialing. Download + extract to local disk and confirm the corpus shape before any partitioning or modeling.
**Objective:** The single-disease experts must train on images that physically exist locally and fit the disk budget.
**Goal:** NIH extracted to `data/nih/` — ~112,120 PNGs + `Data_Entry_2017.csv` — disk layout documented, corpus shape sanity-checked (count, resolution, bit depth, frontal-only).
**Verify:** `ls data/nih/images | wc -l` ≈ 112120; `python -c "from PIL import Image; im=Image.open('data/nih/images/00000001_000.png'); print(im.size, im.mode)"` → `(1024, 1024) L`; `Data_Entry_2017.csv` loads with `View Position` ∈ {PA, AP} only.
**▶ Recommended prompt:** — custom; no skill fits (download + extract is project-specific shell work).

**Do (strike through as you finish):**
- Prereq (Kaggle route): `pip install kaggle`; token → `~/.kaggle/access_token` (chmod 600)
- Download to `data/nih/`: Kaggle `kaggle datasets download -d nih-chest-xrays/data` (whole ~45 GB) OR NIH Box https://nihcc.app.box.com/v/ChestXray-NIHCC (grab `images_001.tar.gz` first for sample-first) + `Data_Entry_2017.csv` — verify the Kaggle slug on the page
- Extract all archives into `data/nih/images/`; document the disk layout in the repo
- Sanity-check the corpus: image count ≈ 112,120, resolution 1024×1024, 8-bit grayscale
- Confirm `Data_Entry_2017.csv` loads and that `View Position` is frontal-only (PA/AP)
