# 📊 In-Training 3-Way Grid · Operator Choice · Expose

## Description
Wire the in-training hook that composes cardio∧<2nd-disease> three ways and logs a grid + per-operator yield; choose the best operator on the back-half evidence; wire it into `08` and expose the module for both scopes.

## Purpose
Serves Objectives 2–3 and DoD items 3–5 — the diagnostic, the decision, and the shared release.

## Goal
A logged 3-way grid + per-operator yields, a recorded best-operator choice wired into `08`, and a module both scopes import with PoE score-sum confirmed untouched.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks
- [ ] ⚠️ **Add the in-training 3-way composition hook**
  - `▶ prompt:` `Every 10–15k Stage-2 steps, compose cardio∧<2nd-disease> three ways (PoE / MCMC / SuperDiff-AND-if-built) and log a side-by-side grid + per-operator floor-corrected Gate-3 yield to the W&B run.`
  - `✓ verify:` a 3-panel grid + per-operator yields logged at ≥2 checkpoints; back-half panels are the meaningful ones.
- [ ] ⚠️ **Choose the best operator + wire it into `08`**
  - `⊳ decide:` from the back-half yields, pick the operator that maximizes floor-corrected Gate-3 yield while effusion HOLDS; record in `results/vindr_transfer/compose_ops/operator_choice.md`; point `08-eval-gate3-yield` at it.
  - `✓ verify:` operator_choice.md names the pick + numbers; `08` uses it.
- [ ] ⚠️ **Expose the module for both scopes + confirm PoE untouched**
  - `▶ prompt:` `Ensure the operators are importable as a clean module by both Scope 1 (Stage-2 in-training) and Scope 2 (spatial-conditioning eval). Confirm eval-only: the trained model + the PoE score-sum baseline are unchanged.`
  - `✓ verify:` import works from both scopes; a check shows the score-sum operator is byte-identical to the pre-existing `cfg_compose`.

## Recommended skill
— custom; composition + eval-hook code. alt: `/analyze-figure` ✅ on the 3-way grid.

## Engagement Instructions
```bash
$ grep -iE "pick|chosen|operator|yield" results/vindr_transfer/compose_ops/operator_choice.md
$ $PY -c "from src.inference.cfg import compose; print('importable by both scopes')"
# expect: a recorded operator choice wired into 08; module importable; PoE score-sum unchanged
```
