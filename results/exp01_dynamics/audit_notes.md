# Exp01 — Audit notes

**Date:** 2026-06-25 · **Reviewer:** researcher (visual) · **Instrument:** `scripts/ctr_overlay.py`

## What was reviewed
- `failure_sheets/entangled_whiteout_around_heart.png` — 16 worst-by-opacity entangled films
- `failure_sheets/borderline_ctr_0p48_0p52.png` — 16 films at the cardiomegaly CTR boundary
- `failure_sheets/separable_sample.png` — 16 random separable films
- `failure_sheets/entangled_low_conf.png` — the 2 segmentation failures

## Findings
- **Whiteout/entangled calls correct:** flagged films genuinely show large opacities
  obscuring the cardiac border (silhouette sign). No false entanglements spotted.
- **Separable calls correct:** cardiac border traceable, ≥1 costophrenic recess readable
  (sharp or blunted both acceptable — the requirement is *visibility*, not normality).
- **CP read matches the eye:** orange (blunted) circles land on whited-out recesses,
  green (sharp) on preserved angles; unilateral effusions correctly read per-side.
- Reviewer verdict: **"accurate and clean"** → thresholds accepted as locked.

## Corrections / systematic errors
- One residual: occasional PSPNet under-call of cardiac silhouette width on borderline
  films (CTR slightly low). **Does not change the separable/entangled class** (which keys
  on peri-cardiac opacity + CP readability, not the CTR magnitude), so no reclassification
  was triggered and the automation was **not** re-run. To be revisited only if CTR values
  themselves (not the separability flag) are used as a downstream quantitative claim.

## Seg failures (2 films, reason `seg_failed`)
Logged as entangled (no usable heart/lung mask). Acceptable: these are not part of the
pixel-verifiable set and are recorded in `entangled_scoped_out.csv`, not deleted.
