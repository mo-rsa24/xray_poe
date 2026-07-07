# 🔌 Consumer API + Per-Pixel Escalation Call

## Description
Expose `opacity_score` as a stable instrument both scopes call — a per-film score (Scope 1 cleaning/inspection) and a latent-res haze-map (Scope 2 conditioning input) — and record whether the 4-quadrant version suffices or escalates to a per-pixel segmenter.

## Purpose
Serves Objective 3 and DoD items 4–5 — the released, documented instrument + the escalation decision.

## Goal
A documented API (score + latent-res map) and a recorded escalation call.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks
- [ ] ⚠️ **Expose + document the consumer API**
  - `▶ prompt:` `Add to metrics/opacity_score.py two documented entry points: score_film(img,mask)→float (Scope-1 cleaning/inspection) and haze_map(img,mask)→[1,128,128] latent-res map (Scope-2 conditioning input). Write a short API note in results/vindr_transfer/opacity/API.md.`
  - `✓ verify:` both entry points importable; the latent-res map is (1,128,128); API note exists.
- [ ] ⚠️ **Record the per-pixel escalation call**
  - `⊳ decide:` from `02`'s validation — if the 4-quadrant resolution is too coarse to separate the structure that matters (e.g. it can't distinguish border-occluding basal haze from diffuse fog), record FIRE → per-pixel opacity segmenter; else NOT-FIRED. Write to `results/vindr_transfer/opacity/escalation.md`.
  - `✓ verify:` escalation.md states FIRED/NOT-FIRED with the reason from the validation numbers.

## Recommended skill
— custom; instrument API + a decision note.

## Engagement Instructions
```bash
$ $PY -c "from metrics.opacity_score import score_film, haze_map; print('API importable')"
$ grep -iE "fired|not.?fired|per.?pixel" results/vindr_transfer/opacity/escalation.md
# expect: both entry points import; an escalation FIRED/NOT-FIRED call with its reason
```
