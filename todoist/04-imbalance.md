# 04 · Class Imbalance

## Reference while you do it
- 📄 Plan: plans/eda/plans/02-imbalance.md

## Section context (paste into the Todoist section)
**Description:** Measure label frequency and the rare-class tail; report both-disease N per candidate pair as a power check.
**Objective:** Catch a thin both-disease N that would widen the Exp6 floor and threaten the headline.
**Goal:** An imbalance report + per-pair both-disease N, flagged if underpowered.
**Verify (whole leaf):** `python -m eda.imbalance --manifest data/manifest.parquet --out figures/imbalance.png` → figure + per-pair N with power flag.

**▶ Recommended prompt:** `/data-distributions data/<dataset>` — label-frequency + imbalance distributions.

## Tasks (one at a time)
- [ ] Plot label-frequency (sorted) and quantify the rare-class tail
- [ ] For each candidate pair, report both-disease N and flag if only a few hundred
