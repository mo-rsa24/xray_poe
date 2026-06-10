# ⚖️ Class Imbalance

## Description
Measure label frequency and the rare-class tail, and report the both-disease N for each
candidate treatment/control pair as a power check.

## Purpose
Imbalance shapes sampling and training; a thin both-disease N would widen the Exp6 floor
and threaten the headline test on sample size alone.

## Recommended skill
`/data-distributions data/<dataset>` — label-frequency + imbalance distributions.

## Goal
An imbalance report plus the both-disease N per candidate pair, flagged if any pair is underpowered.

## Tasks
- [ ] ⚠️ Plot label-frequency (sorted) and quantify the rare-class tail
- [ ] ⚠️ For each candidate pair, report both-disease N and flag if only a few hundred

## Engagement Instructions
```
$ python -m eda.imbalance --manifest data/manifest.parquet --out figures/imbalance.png
# expect: figures/imbalance.png + printed per-pair both-disease N with a power flag
```
