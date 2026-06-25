# Exp01 — Gate-0 verdict (population-exists)

**VERDICT: support** — a pixel-verifiable separable population exists.

- Primary metric: PA separable fraction = **378/525 = 72.0%** (Wilson 95% CI 68.0%–75.7%).
- Pre-registered rule: support ≥30% · inconclusive 10–30% · null <10% → 72.0% ⇒ **support** (≥30% by a wide margin; AP-threshold recovery not needed).
- Overall (PA+AP) separable = 570/1063 = 53.6%; AP separable 192/538 = 35.7% (logged separately — AP-magnified CTR).
- Robust to threshold: PA separable stays 43%–90% across opacity 0.45–0.65, always ≫ 30% (see calibration.md).

**Frozen test set:** data/splits/pixel_verifiable_separable_test.csv (378 PA films, 317 patients reserved).
**Caveat (downstream):** 235 of the 317 reserved test patients are present in the current train latents — the existing train set predates this hold-out and must subtract data/splits/test_patient_ids.txt on the next retrain (data/splits/leaked_patient_ids.txt).

Figure: figures/exp01_separable_entangled_bar.png
