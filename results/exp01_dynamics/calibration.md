# Exp01 — Instrument calibration record

**Date:** 2026-06-25 · **Instrument:** `scripts/ctr_overlay.py` (xrv ChestX-Det PSPNet)

## Locked decision thresholds
| Constant | Value | Meaning |
|---|---|---|
| `TAU_OPACITY` | 0.55 | peri-cardiac opacity ≥ this → cardiac border erased (entangled) |
| `TAU_HEART_CONF` | 0.45 | PSPNet mean heart-mask confidence floor |
| `CP_BLUNT_DEG` | 80° | costophrenic wedge angle > this → blunted |
| `CP_SEV_BINS` | 60/90/120° | sharp \| mild \| moderate \| severe |

`separable = border_visible AND ≥1 costophrenic recess readable`, where
`border_visible = (peri_cardiac_opacity < 0.55) AND (heart_conf ≥ 0.45)`.

## Validation method
Rather than de-novo radiological hand-labelling, the instrument auto-classified all
1063 both-disease films and the **researcher reviewed the failure-mode contact sheets**
(`failure_sheets/`: whiteout worst-first, borderline CTR 0.48–0.52, separable sample)
and signed off as "accurate and clean" (2026-06-25). The whiteout/entangled call is a
tractable yes/no by eye; the separable sample showed traceable cardiac borders with
readable CP corners. No threshold change was required.

## Sensitivity — is the verdict robust to the threshold choice?
PA separable fraction (N=525) vs the binding `TAU_OPACITY`:

| tau_opacity | PA separable | fraction |
|---|---|---|
| 0.45 | 225 | 42.9% |
| 0.50 | 302 | 57.5% |
| **0.55 (locked)** | **377–378** | **71.8%** |
| 0.60 | 438 | 83.4% |
| 0.65 | 471 | 89.7% |

`TAU_HEART_CONF` is **non-binding** — sweeping 0.35→0.55 leaves the count at 377
(opacity is always the deciding constraint for these films).

**Conclusion:** the exact separable fraction is opacity-threshold-sensitive, but the
**Gate-0 verdict (support, ≥30%) holds across the entire plausible threshold range**
(42.9%–89.7%, all ≫ 30%). Thresholds locked at the values above.

## Known residual (carried to audit)
PSPNet occasionally under-calls the cardiac silhouette width on borderline films,
nudging CTR slightly low. Does not affect the separable/entangled call (which keys on
opacity + CP readability, not the CTR value); flagged for per-film audit correction.
