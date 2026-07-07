# 🔍 Validate the Instrument on Real Films

## Description
Validate `opacity_score` against the trusted references — correlate with `_peri_cardiac_opacity` on border-clear cases, agree with the basal-haze overlay on hazy cases — and quantify that it beats the exposure-sensitive `basal_clarity`.

## Purpose
Serves Objective 2 and DoD items 2–3 — the evidence the instrument is trustworthy at CTR's bar before either scope relies on it.

## Goal
A validation doc with correlation/agreement numbers vs the geometric references and a robustness comparison vs `basal_clarity`.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks
- [ ] ⚠️ **Correlate vs the geometric references**
  - `▶ prompt:` `On ≥50 real films, correlate opacity_score with _peri_cardiac_opacity on border-clear cases AND with the basal-haze overlay (scripts/basal_clean_score.py) on hazy cases; report Pearson/Spearman → results/vindr_transfer/opacity/validation.md.`
  - `✓ verify:` validation.md cites correlation numbers on both case sets.
- [ ] ⚠️ **Quantify robustness vs basal_clarity**
  - `▶ prompt:` `Compare opacity_score vs the brightness-only basal_clarity under an exposure/intensity shift (e.g. gamma/scale perturbation of the same films); show opacity_score is materially more stable. Append to validation.md.`
  - `✓ verify:` validation.md shows opacity_score's lower variance under the exposure shift.

## Recommended skill
— custom; analysis code. alt: `/analyze-figure` ✅ on the agreement scatter.

## Engagement Instructions
```bash
$ grep -iE "pearson|spearman|corr|robust|basal_clarity|exposure" results/vindr_transfer/opacity/validation.md
# expect: correlation vs the geometric references + a robustness win over basal_clarity
```
