# 📊 Eval — Gate-3 Yield on the NIH Frozen Set

> ## ⛔ SUPERSEDED (2026-07-01)
> This evaluated the Stage-2 transfer checkpoint, which is no longer produced. The
> equivalent downstream gate (synth-cardio Exp-09 sensitivity + PoE Gate-3 yield on the
> NIH frozen set) is now carried by **`additional_plans/vae-v2/plans/07-downstream-eval.md`**
> against the vae-v2 joint-from-scratch LDM. Never ran; Todoist task deleted 2026-07-01.
> Kept for provenance only — do not execute.

## Background
The transfer lever only earns its compute if PoE composition with the Stage-2 checkpoint clears the Gate-3 co-presence yield on the **same NIH frozen separable-both test** the from-scratch lever was judged on — NIH-calibrated CTR/CP, floor-corrected, abstain-margined. This is the head-to-head.

## Description
Sample ≥2000 composites (null + normal anchor sweep) with the composition operator chosen in leaf 06's 3-way grid, measure the floor-corrected Gate-3 yield + CI vs from-scratch ~9.3%, plot the CTR shift, file the decision.

## Purpose
Serves Objective 4 and DoD items 7–8 — the recovery number and the filed decision.

## Goal
Gate-3 yield (CI) vs from-scratch 9.3%, a CTR-shift figure, a recorded decision; winner released to Exp 6/7.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks

- [ ] ⚠️ **Compose ≥2000 samples — null + normal anchor sweep**
  - `$ run:` (pick the chosen Stage-2 ckpt from leaf 06; use the composition operator chosen by leaf 06's 3-way grid — PoE / MCMC / SuperDiff-AND)
    ```bash
    CKPT=$(ls -t additional_plans/vindr-transfer/ckpts/stage2_step*.safetensors | head -1)
    for ANCHOR in null normal; do
      $PY scripts/gate3_yield.py --ldm "$CKPT" --vae ckpts/vae_step0025000.pt \
        --anchor $ANCHOR --w-cardio 2.0 --w-effusion 1.0 --n 2000 \
        --floor-dir outputs/exp04_leakage/single/no_finding \
        --out results/vindr_transfer/eval/
    done
    ```
  - `✓ verify:` ≥2000 samples per anchor scored; floor-corrected yield + CI printed to `post_transfer_yield.csv`.

- [ ] ⚠️ **Report the cardio-CTR shift vs baselines**
  - `▶ prompt:` `Plot the composed-sample cardio CTR distribution from results/vindr_transfer/eval/ against baselines gen-from-scratch 0.395 / plateau 0.47 / real 0.547 → figures/vindr_transfer_ctr_shift.png; report the median to results/vindr_transfer/eval/cardio_ctr_shift.csv.`
  - `✓ verify:` distribution plotted with baselines marked; median reported.

- [ ] ⚠️ **File the decision + head-to-head vs from-scratch**
  - `⊳ decide:` Run `$PY scripts/exp03_yield_verdict.py results/vindr_transfer/eval/post_transfer_yield.csv`. **recovered** if yield ≥30% → release the Stage-2 ckpt for Exp 6/7; **null** if <10% → transfer didn't beat from-scratch, record honestly (this is the pre-registered trigger for **Scope 2**, the spatial-conditioning follow-up — clean data wasn't enough); **inconclusive** 10–30% → small w_c/w_e adjustment + more samples. Compare head-to-head with the from-scratch ~9.3% and name the winner in `results/vindr_transfer/eval/transfer_decision.md`.
  - `✓ verify:` `transfer_decision.md` cites yield + CI, the recovered/null/inconclusive call vs 30%/10%, and the head-to-head winner.

## Recommended skill
— custom; `scripts/gate3_yield.py` + `scripts/exp03_yield_verdict.py`.
   alt: `/analyze-figure figures/vindr_transfer_ctr_shift.png` ✅.

## Engagement Instructions
```bash
$ $PY -c "import pandas as pd; d=pd.read_csv('results/vindr_transfer/eval/post_transfer_yield.csv'); print(d.groupby('anchor')['gate3_pass'].mean())"
$ grep -E "recovered|null|inconclusive|winner" results/vindr_transfer/eval/transfer_decision.md
# expect: floor-corrected yield per anchor vs 9.3%; one decision line + head-to-head call
```
