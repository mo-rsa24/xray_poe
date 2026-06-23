# Exp1 floor-power check — how to run the dataset decision

`script: eda/floor_power_check.py` · `serves: data-foundation dataset choice + plans/04-eda/04-correlation-matrix` · `labels-only, no GPU`

Settles **NIH ChestX-ray14 vs VinBigData (VinDr-CXR)** on the number that governs it:
the both-disease (cardiomegaly ∧ effusion) count **N_AB**, which sets the Exp6 *floor* —
the smallest gap the two-sample test can resolve.

## 1 · See it now (no downloads)
```bash
python eda/floor_power_check.py --selftest                 # SYNTHETIC counts, shows report shape
python eda/floor_power_check.py --selftest --c2st-inflation 0.08   # realistic floor → verdict shifts
```

## 2 · Get the labels (a few MB — images NOT needed for this gate)
```bash
# NIH ChestX-ray14 — wide CSV, one row/image, pipe-separated 'Finding Labels'
kaggle datasets download -d nih-chest-xrays/data -f Data_Entry_2017_v2020.csv -p data/nih --unzip
# VinDr/VinBigData — long CSV, one row per box per radiologist
kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection -f train.csv -p data/vindr
#   (the xhlulu ...resized-png-1024x1024 mirror also bundles a copy of train.csv)
```

## 3 · Run for real
```bash
python eda/floor_power_check.py \
    --nih   data/nih/Data_Entry_2017_v2020.csv \
    --vindr data/vindr/train.csv \
    --vindr-agree 1 \
    --c2st-inflation 0.06 \
    --out eda/out/floor_power_report.md
```

## Knobs that change the verdict (set them deliberately)
| flag | default | why it matters |
|---|---|---|
| `--vindr-agree` | 1 | findings present iff ≥k of 3 radiologists marked them. `1`=any (more N), `2`=majority (cleaner, less N). **Report both** — it moves N_AB. |
| `--c2st-inflation` | 0.0 | additive realism margin on the floor. **0 = optimistic rank-null** (narrowest possible floor); a learned C2ST overfits above it at small N. Try 0.05–0.10. |
| `--eval-frac` | 1.0 | fraction of each both-disease half used for evaluation. <1.0 if the floor procedure holds out a test split → shrinks m → widens the floor. |

## What it reports
1. **Counts** — N, normal, A/B present, A-only/B-only (the LDM single-disease conditions), **A∧B = N_AB** (the floor set), and "pure" exact-set variants.
2. **Association** — phi, odds ratio + 95% CI, chi-square p for the treatment pair (this is Exp1, restricted to A×B).
3. **Floor power** — floor 95% upper bound (analytic + MC cross-check), the required N_AB to clear each bar, and the floor-vs-N curve with each dataset marked.
4. **Verdict** — POWERED / MARGINAL / UNDERPOWERED vs the Exp6 prereg bars (real-gap 0.65; expected treatment ~0.78, control ~0.57).

## Honest caveats (don't over-read a green verdict)
- The floor uses the **rank/Mann–Whitney AUC null, which is distribution-free** — that's what makes a labels-only floor legitimate. But a *learned* C2ST classifier can beat the rank null by overfitting, so the floor here is the **optimistic (narrowest) case**. The real floor is ≥ what this prints — hence `--c2st-inflation`.
- `--c2st-inflation` is modelled as an **additive constant** (a conservative sensitivity knob). Real overfit inflation *decays* with m, so at small N this is pessimistic and at large N near-exact. If a constant inflation pushes required-N to `-1`, that means "no N clears it under this pessimistic model" → switch to a two-sample test whose null stays near the rank null (or report the inflation-0 floor and validate the C2ST null empirically once you have features).
- This decides the **treatment** pair only (cardio ∧ effusion). The **control** pair (a near-zero φ pair) still needs the full co-occurrence matrix — same machinery, extended to all classes. VinDr's 14 classes don't include a traumatic finding (no "Fracture"), so re-pick the control pair from what VinDr actually has (e.g. Pneumothorax, Nodule/Mass) and confirm φ ≈ 0 there.
