# Exp 6 — Oracle headroom verdict (Gate 6, DoD 6)

**VERDICT: null** — both-separable oracle−starved ΔAUC (mean of heads) = **-0.012**.

Pre-registered rule: support ≥ 0.05 · inconclusive 0.02–0.05 · null < 0.02.

## ΔAUC on the both-separable stratum (the headline)

| head | oracle AUC | starved AUC | ΔAUC | 95% bootstrap CI |
|---|---|---|---|---|
| cardio | 0.721 | 0.731 | -0.010 | [-0.018, -0.002] |
| effusion | 0.760 | 0.774 | -0.013 | [-0.027, -0.000] |

## Sensitivity / specificity on both-separable (D9, at each arm's Youden threshold)

| arm | head | sensitivity | specificity | threshold |
|---|---|---|---|---|
| oracle | cardio | 0.773 | 0.500 | 0.626 |
| oracle | effusion | 0.675 | 0.784 | 0.437 |
| starved | cardio | 0.783 | 0.522 | 0.607 |
| starved | effusion | 0.675 | 0.784 | 0.474 |

## Role — the null-interpretability guard (D10)

The oracle is **not** a utility competitor; it is the headroom-guard that makes an Exp 7 null *interpretable*. If real headroom exists here (support), a PoE-≤-overlay result in Exp 7 is a **clean, publishable null** rather than an artifact of a dead test set. If the test set is saturated (null verdict here), an Exp 7 null is uninterpretable and the test set / stratum must be revisited before Exp 7 is read.

**Headroom bound (known-unknown 3).** The both-disease pool is capped by NIH (~1063 both-labelled films; 506 fall to the 317 reserved test patients, leaving an abundant oracle pool of ~555). Within the 378-film both-separable test stratum the instrument labels only ~36 films as co-present [1,1] (cardio 286+/effusion 40+), so the effusion-head AUC rests on few positives — thin headroom weakens the null's interpretability and is reported honestly, not hidden.

Inputs: results/exp06_oracle/preds/ (oracle 3 seeds, starved 3 seeds). Figure: exp06_oracle_vs_starved_auc.png.
