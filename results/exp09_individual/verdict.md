# Exp 09 · Individual-Disease Validation — Body-or-No-Body Gate (D2)

Supervisor gate: *"if individual synthetics fail, paper reports 'diffusion-not-composition.'"* Per-disease PASS = synth sensitivity ≥ real-baseline sensitivity − margin. Baseline = Exp-4 deconfounded real single-disease (may include classifier-train images → an optimistic reference; the synth-vs-real delta is the signal).

<!-- TAG:before -->
## tag = `before` — `ckpts/model_step0040000.safetensors`

Generation: w=1.0, anchor=null, n=200/disease, steps=50. Classifier thresholds cardio>0.616, effusion>0.516. margin=0.15.

| disease | source | acc | sens | spec | PASS/FAIL |
|---|---|---|---|---|---|
| cardio | synth | 0.525 | 0.100 | 0.950 | **FAIL** |
| cardio | real | 0.848 | 0.800 | 0.895 | — |
| effusion | synth | 0.575 | 0.905 | 0.245 | **PASS** |
| effusion | real | 0.823 | 0.790 | 0.855 | — |

**Gate consequence (before):** ❌ gate fires → paper reports 'diffusion-not-composition' for cardio (§4 confirmatory of the null on this channel)

Artifact: `results/exp09_individual/validate_individual_before.csv`.

<!-- /TAG:before -->

<!-- TAG:after -->
## tag = `after` — `ckpts_rebalanced/model_step0050000.safetensors`

Generation: w=1.0, anchor=null, n=200/disease, steps=50. Classifier thresholds cardio>0.616, effusion>0.516. margin=0.15.

| disease | source | acc | sens | spec | PASS/FAIL |
|---|---|---|---|---|---|
| cardio | synth | 0.608 | 0.250 | 0.965 | **FAIL** |
| cardio | real | 0.848 | 0.800 | 0.895 | — |
| effusion | synth | 0.630 | 0.960 | 0.300 | **PASS** |
| effusion | real | 0.823 | 0.790 | 0.855 | — |

**Gate consequence (after):** ❌ gate fires → paper reports 'diffusion-not-composition' for cardio (§4 confirmatory of the null on this channel)

Artifact: `results/exp09_individual/validate_individual_after.csv`.

<!-- /TAG:after -->
