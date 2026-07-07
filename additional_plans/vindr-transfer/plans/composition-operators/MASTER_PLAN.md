# Alternative AND-Composition Operators (shared)

## Mission
A shared module of alternative AND-composition operators for the PoE pipeline. Beside the existing `cfg_compose` score-sum (the PoE baseline), implement **MCMC-corrected composition** ([Du et al. arXiv 2302.11552] — a Metropolis/Langevin correction on the summed score) and **SuperDiff-AND** ([Skreta et al. arXiv 2412.17762] — Itô-density auto-reweighting), plus an in-training/eval hook that composes cardio∧2nd-disease three ways and logs a grid + per-operator floor-corrected Gate-3 yield. It is **eval-only** — it picks the best composition operator without changing the trained model and without altering PoE's score-sum. Shared: Scope 1 (`additional_plans/vindr-transfer/`) runs it as a Stage-2 in-training diagnostic; Scope 2 (`additional_plans/spatial-conditioning/`) imports the same operators.

> **Why its own scope:** two non-trivial published algorithms to implement, each with its own verification, imported by two scopes. Decomposed 2026-06-27 from `vindr-transfer/plans/06-stage2-nih-finetune.md`. **SuperDiff-AND is conditional** — it may never be built if MCMC-corrected alone clears/decides the operator choice.

## Objectives
1. Implement the alternative AND-operators as eval samplers beside the PoE score-sum, without touching the trained model.
2. Provide an in-training/eval hook that composes 3 ways and logs grid + per-operator yield.
3. Pick the best operator on evidence; carry it into the `08` eval; expose the module for both scopes.

## Goals
1. **MCMC-corrected built** — composes cardio∧2nd-disease and returns a sample on a saved checkpoint.
2. **SuperDiff-AND built (conditional)** — implemented only if MCMC alone doesn't decide; else its omission recorded.
3. **Hook live** — every 10–15k steps logs a 3-way grid + 3 floor-corrected Gate-3 yields.
4. **Operator chosen** — best operator recorded, wired into `08`; module importable by both scopes.

## Expected Outcome
A small validated composition-operators module (PoE score-sum + MCMC-corrected + optional SuperDiff-AND) whose in-training diagnostic names the best AND-operator for cardio∧2nd-disease, reused by both scopes, with PoE's score-sum and the trained model untouched.

## Definition of Done
1. MCMC-corrected composition implemented (Metropolis/Langevin correction on the summed score, Du 2302.11552); composes cardio∧2nd-disease and returns a sample on a saved checkpoint.
2. SuperDiff-AND implemented IF needed (Itô-density auto-reweighting, Skreta 2412.17762) — or its omission recorded as "MCMC sufficed."
3. In-training hook logs a 3-way grid + per-operator floor-corrected Gate-3 yield every 10–15k steps to W&B.
4. Best operator chosen on the back-half evidence; recorded; wired into the `08-eval-gate3-yield` composition path.
5. Module importable by both scopes (Scope 1 Stage-2 in-training + Scope 2 inheritance); PoE score-sum confirmed untouched (eval-only, model unchanged).

## Sub-Scopes
(none)

## Plans
Dependency order: `01 → 02`.
- ⚠️ 01-operators.md — implement MCMC-corrected; SuperDiff-AND gated on MCMC not deciding (DoD 1–2)
- ⚠️ 02-grid-choice-expose.md — in-training 3-way hook + operator choice (→ `08`) + expose module, PoE untouched (DoD 3–5)
