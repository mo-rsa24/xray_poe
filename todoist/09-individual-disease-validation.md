# 09 · 🚦 Individual-Disease Validation — the Body-or-No-Body Gate (D2)

## Reference while you do it
- 📄 Plan: plans/09-individual-disease-validation.md
- 🧭 Authority: docs/SUPERVISOR_RECONCILE.md (D2) · wraps scripts/generate.py + the DenseNet classifier (scripts/finetune_classifier.py / ckpts/presence_classifier_finetuned.pt)

## Section context (paste into the Todoist section)
**Description:** Build `scripts/validate_individual_synthetics.py`: generate synth-only cardio & synth-only effusion, run each through the classifier, report accuracy/sens/spec vs the real single-disease baseline, emit explicit pass/fail per disease. **Runnable NOW on the before-model** — does not wait on the retrain.
**Objective:** Prove single-disease synthetics classify *before* composition — the gate on whether the paper has a synthetic body.
**Goal:** Per-disease pass/fail (cardio, effusion) on before + after, with the body/no-body gate consequence recorded.
**Verify (whole leaf):** `grep -iE "PASS|FAIL|diffusion-not-composition|body|no-body" results/exp09_individual/verdict.md` → per-disease PASS/FAIL + the gate branch per checkpoint.
**▶ Recommended prompt:** `/analyze results/exp09_individual/verdict.md`

## Tasks (one at a time)
- [x] ✅ Validate individual-disease synthetics through the classifier (the gate) — DONE (verdict.md cardio FAIL/effusion PASS, before+after). — generate synth-only cardio & effusion; acc/sens/spec vs real single-disease baseline; explicit pass/fail per disease. **Supervisor (D2, 1.1/7.2/7.7):** "individual-disease validation gate." Script: new `scripts/validate_individual_synthetics.py` (wraps generate.py + classifier) · Checkpoint: **BEFORE** `ckpts/model_step0040000` (contaminated — *ideal* exposure read) + **AFTER** rebalanced · **Accept (verbatim):** "Synth-only cardio & synth-only effusion each run through classifier; accuracy/sens/spec vs real single-disease baseline; explicit pass/fail per disease. Gate: if individual synthetics fail, paper reports 'diffusion-not-composition.'" Verify: per-disease PASS/FAIL printed + in verdict.md. **RUNNABLE NOW on the before-model.**
- [x] ✅ Record the gate consequence (body / no-body, before + after) — if either disease fails on the after-model → pivot to "diffusion-quality-not-composition," §4 becomes confirmatory of the null; if both pass → the synthetic body proceeds. **Supervisor (D2 gate):** "if individual synthetics fail, paper reports 'diffusion-not-composition.'" Script: `results/exp09_individual/verdict.md` · Checkpoint: before gates NOW, after = clean restatement · Accept: gate branch documented per checkpoint. Verify: verdict.md states per-disease pass/fail on both ckpts + the body/no-body branch taken.
