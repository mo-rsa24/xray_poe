# 08 · 🧾 Methods — System Description (D1)

## Reference while you do it
- 📄 Plan: plans/08-methods-system-description.md
- 🧭 Authority: docs/SUPERVISOR_RECONCILE.md (D1, D9) · grounded in scripts/finetune_classifier.py + scripts/ctr_overlay.py + EXPERIMENTS.md (ckpt inventory)

## Section context (paste into the Todoist section)
**Description:** One source-of-truth methods doc (`docs/SYSTEM_DESCRIPTION.md`) stating classifier I/O, a full model inventory with reuse/retrain status, and the DenseNet-classifier-vs-PSPNet-instrument disambiguation. Also carries the D9 no-leak / sens-spec statement.
**Objective:** Make the supervisor's three §0 questions answerable without reading code.
**Goal:** A committed `docs/SYSTEM_DESCRIPTION.md` answering §0 + the explicit no-leak/sens-spec statement.
**Verify (whole leaf):** `grep -iE "224|sigmoid|PSPNet|DenseNet|patient-disjoint|sensitivity" docs/SYSTEM_DESCRIPTION.md` → all hit; a §0-answer checklist closes the doc.
**▶ Recommended prompt:** — custom; no skill fits.

## Tasks (one at a time)
- [x] 🧾 Write the classifier + model-inventory system description — state 224² input; 2 sigmoid logits [cardio,effusion]; model inventory w/ reuse/retrain status; what each model outputs; DenseNet-vs-PSPNet disambiguated; held-out-both-disease choice. **Supervisor (D1, §0 questions):** "§0 questions." Script: new `docs/SYSTEM_DESCRIPTION.md` · Checkpoint: none · **Accept (verbatim):** "Classifier I/O stated (224² input; 2 sigmoid logits [cardio,effusion]); full model inventory with reuse/retrain status; what each model outputs; DenseNet-classifier vs PSPNet-instrument disambiguated; held-out-both-disease training choice stated. Supervisor can answer all three §0 questions from it." Verify: grep 224/sigmoid/PSPNet/DenseNet all hit.
- [x] 🔒 State the no-leak / sens-spec rule (D9 — statement half) — patient-disjoint + test-on-real-only + sens/spec everywhere; name `cross_leakage_check.py` / `filter_test_patients.py` as the verification path (computation half = leaf 11). **Supervisor (D9, 7.5/7.10/7.1/7.3):** "Sens/spec everywhere; explicit no-leak statement (patient-disjoint, test-on-real-only) in methods + verified." Script: `docs/SYSTEM_DESCRIPTION.md` · Checkpoint: none · **Accept (verbatim):** as quoted. Verify: grep patient-disjoint/test-on-real/sensitivity/specificity.
