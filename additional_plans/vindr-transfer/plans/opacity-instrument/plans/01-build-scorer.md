# 🩺 Build the 4-Quadrant Opacity Scorer

## Description
Write `metrics/opacity_score.py`: a RALE-style regional lung-opacity score (extent × density per quadrant) computed over the PSPNet lung mask, returning both a per-film scalar and a 4-quadrant breakdown.

## Purpose
Serves Objective 1 and DoD item 1 — the scorer all downstream validation and both consuming scopes depend on.

## Goal
`metrics/opacity_score.py` runs over the PSPNet lung mask on ≥50 real films and emits a 4-quadrant extent×density score.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks
- [ ] ⚠️ **Write the 4-quadrant RALE-style scorer**
  - `▶ prompt:` `Write metrics/opacity_score.py: take a CXR + the PSPNet lung mask (reuse scripts/ctr_overlay.py seg), split each lung into upper/lower quadrants (4 total), score extent (fraction opacified) × density (intensity over an aerated baseline) per quadrant, return a per-film scalar + the 4-quadrant vector. Geometry/mask-grounded, NOT a raw brightness mean. Add a --selftest.`
  - `✓ verify:` `--selftest` prints 4 quadrant scores + the aggregate on a sample film.
- [ ] ⚠️ **Smoke-run on ≥50 real films**
  - `$ run:` `$PY metrics/opacity_score.py --manifest data/vindr512/presence_manifest.csv --n 50 --out results/vindr_transfer/opacity/scores.csv`
  - `✓ verify:` 50 rows, each with 4 quadrant scores + aggregate; no NaNs on valid masks.

## Recommended skill
— custom; core instrument code (`metrics/opacity_score.py`).

## Engagement Instructions
```bash
$ $PY metrics/opacity_score.py --selftest          # prints 4 quadrant scores + aggregate
$ $PY -c "import pandas as pd; d=pd.read_csv('results/vindr_transfer/opacity/scores.csv'); print(d.shape, d.columns.tolist())"
# expect: ~50 rows; columns include 4 quadrant scores + an aggregate
```
