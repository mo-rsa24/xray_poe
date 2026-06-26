# Table 1 — pretrained classifier: accuracy / sensitivity / specificity (D4, D9)

Frozen DenseNet judge, Youden thresholds cardio>0.616 effusion>0.516. Both-disease held out of training (struggles on comorbidity by design). 95% bootstrap CIs.

**Leakage-clean (D9):** real single-disease = the classifier's held-out val split (never trained on); comorbid = held-out both-disease test; synth = generated. No real eval film was in the classifier's training set.

## Per-head sens / spec / acc

| source | n | head | sensitivity | specificity | accuracy |
|---|---|---|---|---|---|
| real | 1278 | cardio | 0.792 [0.76,0.82] | 0.822 [0.79,0.85] | 0.808 |
| real | 1278 | effusion | 0.771 [0.73,0.81] | 0.837 [0.81,0.86] | 0.819 |
| synth-before | 360 | cardio | 0.083 [0.05,0.12] | 0.950 [0.91,0.98] | 0.372 |
| synth-before | 360 | effusion | 0.833 [0.79,0.88] | 0.242 [0.17,0.32] | 0.636 |
| synth-after | 360 | cardio | 0.196 [0.15,0.25] | 0.917 [0.86,0.96] | 0.436 |
| synth-after | 360 | effusion | 0.863 [0.82,0.90] | 0.292 [0.22,0.38] | 0.672 |

## §7.6 comorbidity breakdown — head positive-rate by category

| source | head | normal | cardio | effusion | comorbid |
|---|---|---|---|---|---|
| real | cardio | 0.123 | 0.787 | 0.147 | 0.714 |
| real | effusion | 0.123 | 0.187 | 0.790 | 0.225 |
| synth-before | cardio | — | 0.092 | 0.050 | 0.075 |
| synth-before | effusion | — | 0.758 | 0.867 | 0.800 |
| synth-after | cardio | — | 0.233 | 0.083 | 0.158 |
| synth-after | effusion | — | 0.708 | 0.908 | 0.817 |

_Read: if the cardio head's positive-rate drops from the `cardio` column to the `comorbid` column, the classifier under-detects cardiomegaly when effusion co-occurs (the comorbidity weakness §7.4/7.6 targets)._
