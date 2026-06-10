# 🧾 Integrity Scan + Manifest

## Description
Scan every PNG for readability, quarantine corrupt or truncated files, and emit a
manifest linking each clean image to its label(s), **source dataset**, and image
properties — across both NIH (train) and VinDr (eval).

## Purpose
A clean, indexed corpus with a manifest is the single source of truth every later scope
reads from — EDA, VAE curation, LDM latent prep, and the floor all join against it.

## Goal
A manifest (parquet) with one row per readable image — path, labels, `source_dataset`
(`nih` | `vindr`), resolution, group — plus a quarantine log of failures.

## Tasks
- [ ] ⚠️ Iterate all PNGs (NIH + VinDr); attempt decode; record failures with reason; quarantine corrupt/truncated
- [ ] ⚠️ Join images to the label table; emit manifest (path → label(s) + `source_dataset` + properties)
- [ ] ⚠️ Report counts: total, readable, quarantined — broken out by source dataset

## Engagement Instructions
```
$ python -m data.build_manifest --root data/ --out data/manifest.parquet
# expect: prints "readable=N, quarantined=M" (M small), broken out by source_dataset
$ python -c "import pandas as pd; df=pd.read_parquet('data/manifest.parquet'); print(df.shape, df.columns.tolist()); print(df['source_dataset'].value_counts())"
# expect: columns include source_dataset (nih|vindr) + labels + resolution + group; counts for both datasets
```
