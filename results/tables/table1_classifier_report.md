# Table 1 — pretrained classifier: accuracy / sensitivity / specificity (D4, D9)

Frozen DenseNet judge, Youden thresholds cardio>0.616 effusion>0.516. Both-disease held out of training (struggles on comorbidity by design). 95% bootstrap CIs.

## Per-head sens / spec / acc

| source | n | head | sensitivity | specificity | accuracy |
|---|---|---|---|---|---|
| real | 1413 | cardio | 0.806 [0.77,0.84] | 0.830 [0.81,0.85] | 0.823 |
| real | 1413 | effusion | 0.770 [0.73,0.81] | 0.867 [0.84,0.89] | 0.837 |

## §7.6 comorbidity breakdown — head positive-rate by category

| source | head | normal | cardio | effusion | comorbid |
|---|---|---|---|---|---|
| real | cardio | 0.156 | 0.827 | 0.122 | 0.714 |
| real | effusion | 0.109 | 0.110 | 0.784 | 0.225 |

_Read: if the cardio head's positive-rate drops from the `cardio` column to the `comorbid` column, the classifier under-detects cardiomegaly when effusion co-occurs (the comorbidity weakness §7.4/7.6 targets)._
