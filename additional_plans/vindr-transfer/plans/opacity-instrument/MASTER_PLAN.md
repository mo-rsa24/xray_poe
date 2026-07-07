# Regional Opacity Instrument (RALE-style, shared)

## Mission
A validated, segmentation-grounded regional lung-opacity instrument for chest X-rays (RALE-style, computed over the PSPNet lung mask), replacing the exposure-sensitive brightness descriptors (`basal_clarity`, raw `_peri_cardiac_opacity` thresholds) the project does not trust at CTR's bar. It is a **shared** instrument: Scope 1 (`additional_plans/vindr-transfer/`) consumes it as a per-film cleaning/inspection score; Scope 2 (`additional_plans/spatial-conditioning/`, plan `04-haze-map-decorrelation`) consumes it as a latent-res haze-map conditioning input. It must be validated against the trusted geometric references on real films before either scope relies on it.

> **Why its own scope:** two scopes depend on it (Scope 1 cleaning + Scope 2 conditioning), and it is a multi-step deliverable with its own validation. Decomposed 2026-06-27 from `vindr-transfer/plans/09-morbidity-criteria-instrument.md` task 3 so both scopes depend on a clean released instrument, not a buried task line.

## Objectives
1. Build a 4-quadrant RALE-style regional opacity scorer (extent × density per quadrant) over the PSPNet lung mask.
2. Validate it on real films against the trusted geometric references and quantify its advantage over the brightness-only `basal_clarity`.
3. Expose a stable, documented instrument both scopes can call (per-film score + latent-res map), with an escalation path to per-pixel.

## Goals
1. **Scorer built** — `metrics/opacity_score.py` runs over the PSPNet lung mask on ≥50 real films, emits a 4-quadrant extent×density score.
2. **Validated** — correlation vs `_peri_cardiac_opacity` (border-clear cases) + agreement vs the basal-haze overlay (hazy cases); shown more robust than `basal_clarity`.
3. **Consumable** — documented API: a per-film score (Scope 1 cleaning/inspection) + a latent-res haze-map (Scope 2 conditioning input).
4. **Escalation decided** — per-pixel segmenter path recorded as fired / not.

## Expected Outcome
A trustworthy, segmentation-grounded opacity instrument that both Scope 1 (cleaning/inspection) and Scope 2 (haze-map conditioning input) depend on — exposure-invariant where `basal_clarity` is not — or, if the 4-quadrant version proves too coarse, the recorded escalation to a per-pixel opacity segmenter. Either way, validated on real before either scope relies on it.

## Definition of Done
1. `metrics/opacity_score.py` built — 4-quadrant RALE-style score (extent × density per quadrant) over the PSPNet lung mask; runs on ≥50 real films.
2. Validated on real: correlation with `_peri_cardiac_opacity` on border-clear cases + agreement with the basal-haze overlay on hazy cases; reported.
3. Agreement vs brightness-only `basal_clarity` quantified — the new instrument shown more robust (exposure-invariant).
4. Consumer API documented: a per-film score (Scope 1) + a latent-res haze-map (Scope 2).
5. Escalation call recorded — per-pixel opacity segmenter only if the 4-quadrant proves too coarse (else not-fired).

## Sub-Scopes
(none)

## Plans
Dependency order: `01 → 02 → 03`.
- ⚠️ 01-build-scorer.md — write `metrics/opacity_score.py` (4-quadrant RALE over the PSPNet lung mask) (DoD 1)
- ⚠️ 02-validate-on-real.md — correlate vs the geometric references + robustness vs `basal_clarity` (DoD 2–3)
- ⚠️ 03-api-and-escalation.md — consumer API (score + latent-map) + per-pixel escalation call (DoD 4–5)
