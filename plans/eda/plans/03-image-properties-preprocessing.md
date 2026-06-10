# 🖼️ Image Properties & Preprocessing Decisions

## Description
Characterize resolution, intensity distribution, and aspect ratio across the corpus, and
decide the preprocessing pipeline (resize target, normalization, clipping).

## Purpose
The preprocessing contract is now adopted (512²/f=4 stretch + per-image min–max, owned by
`data-foundation/06-shared-preprocessing` and the latent-shape decision); this validates it
against the data rather than re-deciding from scratch.

## Recommended skill
`/data-distributions data/<dataset>` — intensity / resolution / aspect distributions to validate the adopted contract. (Canonical pipeline → `data-foundation/06-shared-preprocessing.md`.)

## Goal
Property histograms that confirm (or challenge) the adopted 512² stretch / per-image min–max
contract.

## Tasks
- [ ] ⚠️ Histogram resolutions, aspect ratios, and intensity ranges
- [ ] ⚠️ Validate the adopted 512²/f=4 stretch + per-image min–max (owned by `data-foundation/06`) against the histograms; flag if the data argues against it

## Engagement Instructions
```
$ python -m eda.properties --manifest data/manifest.parquet --out figures/properties.png
# expect: histograms saved; a PREPROCESSING.md noting chosen resize/normalize/clip
```
