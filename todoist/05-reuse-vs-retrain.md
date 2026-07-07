# 🚦 05 · Reuse-vs-Retrain

## Reference while you do it
- 📄 Plan: plans/05-reuse-vs-retrain.md
- 🪆 Sub-scope (dormant): plans/rebalanced-cardio-retrain/MASTER_PLAN.md
- 🧭 Authority: EXPERIMENTS.md (EXP-05) · docs/EVAL_PHILOSOPHY.md (reuse-or-retrain)

## Section context (paste into the Todoist section)
**Description:** Read the Exp 2 ceiling verdict and the Exp 3 yield verdict, apply the pre-committed rule, and record the decision. The expensive retrain is decomposed into its own sub-scope and only fires if the rule returns `retrain` — do NOT bridge it until then.
**Objective:** Decide which models feed every downstream arm, by rule not by vibe (`local:reuse-or-retrain`, DoD 5).
**Goal:** A written decision (`reuse` | `retrain`) against the rule, logging which gate forced any retrain.
**Verify (whole leaf):** `grep -E "DECISION: (reuse|retrain)" results/exp05_decision/decision.md` → one decision citing the ceiling + yield verdicts.
**▶ Recommended prompt:** — custom; no skill fits.

## Tasks (one at a time)
- [ ] ⚖️ Apply the pre-committed decision rule + record it — **reuse iff** (ceiling high) AND (yield ≥ 30%); else **retrain** (low ceiling→VAE, low yield→cardio LDM). Inconclusive upstream → resolve upstream first. Verify: `decision.md` states rule inputs + the call.
- [ ] 🪆 (Conditional) Rebalanced-cardio retrain + yield recovery — decomposed into `plans/rebalanced-cardio-retrain/`. Stays dormant unless the rule above returns `retrain`; bridge/run that sub-scope only when fired. Verify: sub-scope's `recovery_decision.md` exists only if retrain ran.
