---
name: Metrics
kind: reference
grounded: metrics/presence_classifier.py:155-161
confidence: verified
---
# Metrics & distributions — what we report and why

**Gist.** Sensitivity is the operative gate metric; AUC/ROC is the threshold-free headline; the
decision threshold is Youden-optimal; intervals are bootstrap; evaluation is stratified.

**Explainer.**
- **Sensitivity (recall)** = TP/(TP+FN) — of real diseased cases, how many we catch; the operative
  metric for the Exp09 body-gate, because the question is "does a synthetic disease image *read as
  diseased*". **Specificity** = TN/(TN+FP); **FPR** = 1 − specificity. Real operating point: cardio
  sens 0.79 / spec 0.82, effusion sens 0.77 / spec 0.84.
- **AUC / ROC** — threshold-free ranking quality, stored per head (`val_auc_cardio/effusion`), the
  headline for the Table-2 augmentation arms (ΔAUC on the comorbid stratum).
- **Youden threshold** — max(sens + spec − 1), stored in the checkpoint, not hardcoded.
- **Confidence intervals** — bootstrap, 1000 resamples, 2.5/97.5 percentiles.
- **Stratification** — normal / cardio_only / effusion_only / both_separable; the comorbid
  `both_separable` stratum is the actual claim criterion, and real both-disease is scarce (~1063
  films — the reason the whole synthetic-augmentation question exists).

**Deep-dive.** Bootstrap CI `presence_classifier.py:155-161`; Youden `:178-179`.

Related: [[presence-classifier]], [[cardiomegaly-readout]]
