# 📊 Yield Re-Measure + Bottleneck Attribution

## Description
Re-run the Gate-3 yield on the NIH frozen set with the spatially-conditioned model, read the independent DenseNet classifier, and compare head-to-head vs Scope-1 and from-scratch to attribute the bottleneck.

## Purpose
Serves Objective 4 and DoD item 6 — the verdict.

## Goal
Gate-3 yield (CI) + classifier read vs Scope-1/from-scratch; a filed decision (conditioning recovered it / null) with the bottleneck attribution.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks
- [ ] ⚠️ **Gate-3 yield on the NIH frozen set**
  - `$ run:` `$PY scripts/gate3_yield.py --ldm <mask+haze ckpt> --vae ckpts/vae_step0025000.pt --anchor null normal --n 2000 --floor-dir outputs/exp04_leakage/single/no_finding --out results/spatial_cond/eval/`
  - `✓ verify:` ≥2000 samples/anchor; floor-corrected yield + CI.
- [ ] ⚠️ **Independent classifier read (the honest evidence)**
  - `▶ prompt:` `Run the DenseNet presence classifier (NOT conditioned on) over the composites; report cardio presence + the comorbid read. This is the headline evidence — CTR alone only shows mask-following.`
  - `✓ verify:` classifier cardio/comorbid presence reported.
- [ ] ⚠️ **File the decision + attribution**
  - `⊳ decide:` compare yield + classifier vs Scope-1 and from-scratch 9.3%. **recovered** → conditioning was the bottleneck (clean data wasn't enough); **null** → bottleneck is deeper than conditioning. Write `results/spatial_cond/eval/decision.md`.
  - `✓ verify:` decision.md cites yield+CI, classifier read, head-to-head, and the bottleneck attribution.

## Recommended skill
— custom; `scripts/gate3_yield.py`. alt: `/analyze-figure` ✅ on the yield figure.

## Engagement Instructions
```bash
$ $PY -c "import pandas as pd; print(pd.read_csv('results/spatial_cond/eval/post_yield.csv').groupby('anchor')['gate3_pass'].mean())"
$ grep -iE "recovered|null|classifier|attribution|winner" results/spatial_cond/eval/decision.md
# expect: yield vs Scope-1/9.3%; classifier read; a recovered/null call + bottleneck attribution
```
