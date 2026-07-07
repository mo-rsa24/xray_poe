# 🩻 Morbidity Criteria + Opacity Instrument (Manual Inspection · 2nd-Disease Pick)

## Background
Before cleaning and encoding, two things must be settled on *real* films, by eye: (a) what cardiomegaly, pleural effusion, and pleural thickening look like in VinDr at 512², and (b) which second disease (effusion vs pleural thickening) pairs best with cardiomegaly for a *pixel-verifiable* composition — the bar Exp 1–3 set on NIH. The brightness-based basal descriptors (`basal_clarity`, `_peri_cardiac_opacity`) are not trusted at CTR's bar; a validated regional-opacity instrument is needed for cleaning + inspection, and is reused by the conditioning follow-up (Scope 2, `additional_plans/spatial-conditioning/`). This manual-inspection rigor is what lets a later vanilla-LDM failure be read as "the model, not the data."

## Description
Hand-inspect 10–15 VinDr films per morbidity with ctr_overlay + basal-haze overlays, write explicit keep/clean criteria, decide effusion-vs-pleural-thickening as the 2nd disease, and build + validate a regional-opacity (RALE-style) instrument against real.

## Purpose
Serves Objective 1 + DoD items 2–3; gates the 03 class map and the 03/04 cleaning. Sequenced `01 → 09 → {02,03,04}`.

## Goal
A criteria doc (per-morbidity keep/clean rules), a recorded 2nd-disease decision feeding the 03 class map, and a validated opacity instrument with a real-correlation number.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks

- [ ] ⚠️ **Manual inspection — 10–15 films × {cardiomegaly, pleural effusion, pleural thickening}**
  - `▶ prompt:` `Select 10–15 VinDr films per morbidity from data/vindr512/presence_manifest.csv (pleural thickening from the VinDr "Pleural thickening" boxes in train.csv). Run scripts/ctr_overlay.py + the basal-haze overlay (scripts/basal_clean_score.py) on each → results/vindr_transfer/criteria/overlays/. Eyeball border visibility, basal whiteout, laterality; write results/vindr_transfer/criteria/CRITERIA.md with explicit per-morbidity keep/clean rules.`
  - `✓ verify:` CRITERIA.md has per-morbidity rules + an overlay panel per morbidity.
  - alt `$ run:` `/visualize-data-samples /home/molefe/data/vindr512`

- [ ] ⚠️ **Decide the 2nd disease: pleural effusion vs pleural thickening**
  - `⊳ decide:` From the inspection + counts (VinDr box-level: effusion 2,476 vs **pleural thickening 4,842 ≈ 2× more**; image-level effusion∧¬cardio 684 vs thickening∧¬cardio from leaf 01's manifest), pick the disease that is (a) pixel-verifiable with our instruments and (b) sufficiently sampled. Record in `results/vindr_transfer/criteria/second_disease.md`. **THIS FEEDS the 03 class map** `{normal:0, cardio:1, <chosen>:2}`.
  - `⊳ NIH-side cost (decide with eyes open):` effusion already has NIH deconfounded latents (`data/latents/rebalanced/`); **pleural thickening does NOT** — picking it triggers a new NIH pleural-thickening latent build (raw NIH labels have `Pleural_Thickening`, but no split/cache exists). Weigh VinDr's thickening abundance against this NIH prep cost.
  - `✓ verify:` second_disease.md names the pick + why + (if thickening) flags the NIH-latent-build it triggers; the 03 class-map task references it.

- [ ] ⚠️ **Build + validate the regional-opacity (RALE-style) instrument**  → decomposed: see `plans/opacity-instrument/MASTER_PLAN.md`

## Recommended skill
`/visualize-data-samples` ✅ + `/data-integrity-check` ✅

## Engagement Instructions
```bash
$ ls results/vindr_transfer/criteria/overlays/*.png | wc -l   # ≥30 (10–15 × 3 morbidities)
$ grep -iE "keep|clean|criteria" results/vindr_transfer/criteria/CRITERIA.md | head
$ grep -iE "effusion|thickening|pick|chosen" results/vindr_transfer/criteria/second_disease.md
$ $PY metrics/opacity_score.py --selftest   # prints the real-correlation number
# expect: per-morbidity criteria; a named 2nd disease; opacity instrument validated on real
```
