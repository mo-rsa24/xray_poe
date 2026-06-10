# 05 · Integrity scan + manifest

[⌂ Index](00-INDEX.md) · [← prev 04](04-shared-preprocessing.md)

## Reference while you do it
- 📄 Plan: plans/data-foundation/plans/02-integrity-manifest.md

## Section context (paste into the Todoist section)
**Description:** Scan every PNG (NIH + VinDr) for readability, quarantine corrupt or truncated files, and emit a manifest linking each clean image to its label(s), source dataset, and properties.
**Objective:** Produce the single source of truth every later scope joins against — EDA, VAE curation, LDM latent prep, and the floor.
**Goal:** A parquet manifest — one row per readable image (path, labels, `source_dataset` ∈ {nih, vindr}, resolution, group) — plus a quarantine log of failures.
**Verify (whole leaf):** `python -m data.build_manifest --root data/ --out data/manifest.parquet` prints `readable=N, quarantined=M` (M small) broken out by source_dataset; `python -c "import pandas as pd; df=pd.read_parquet('data/manifest.parquet'); print(df.shape, df.columns.tolist()); print(df['source_dataset'].value_counts())"` shows the source_dataset column + counts for both datasets.

## Tasks (one at a time)
- [ ] Iterate all PNGs (NIH + VinDr); attempt decode; record failures with reason; quarantine corrupt/truncated
- [ ] Join images to the label table; emit manifest (path → label(s) + `source_dataset` + properties)
- [ ] Report counts: total, readable, quarantined — broken out by source dataset
