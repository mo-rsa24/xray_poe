# 11 · 📊 Table 1 — Classifier Report + Shared Metric Module (D4 + D9)

## Reference while you do it
- 📄 Plan: plans/11-tables-classifier-report.md
- 🧭 Authority: docs/SUPERVISOR_RECONCILE.md (D4, D9) · new metrics/classifier_report.py — the shared module reused by leaf 15 (Table 2 / D7)

## Section context (paste into the Todoist section)
**Description:** Build `metrics/classifier_report.py` (shared accuracy/sens/spec + bootstrap CIs) and produce Table 1 **eval-only** on the pretrained classifier — real vs synth columns, synth half on both checkpoints.
**Objective:** The face-validity table (D4) + the D9 sens/spec computation, computed identically before and after fine-tuning.
**Goal:** `metrics/classifier_report.py` + `results/exp11_tables/table1_{before,after}.csv` (acc/sens/spec per category, real vs synth, bootstrap CIs).
**Verify (whole leaf):** `python -c "import metrics.classifier_report"` + `ls results/exp11_tables/table1_{before,after}.csv` + pandas shows sens/spec columns.
**▶ Recommended prompt:** `/analyze results/exp11_tables/table1_after.csv`

## Tasks (one at a time)
- [ ] 🧮 Build the shared accuracy/sens/spec metric module (D9 — computation half) — per-category accuracy/sens/spec + bootstrap CIs, real-vs-synth columns; reused by Table 2 (leaf 15). **Supervisor (D9, 7.5/7.10):** "Sens/spec everywhere." Script: new `metrics/classifier_report.py` · Checkpoint: module is checkpoint-agnostic · **Accept:** returns accuracy + sensitivity + specificity per category with bootstrap CIs; same call signature for Table 1 & Table 2. Verify: import + smoke table.
- [ ] 📊 Produce Table 1 — pretrained classifier, real vs synth, eval-only (D4) — run the pretrained classifier per category, real & synth, via the module; report acc + sens + spec + CIs; **no training**. **Supervisor (D4, Table 1):** "pretrained classifier accuracy real vs synth." Script: `metrics/classifier_report.py` on `ckpts/presence_classifier_finetuned.pt` · Checkpoint: synth half **BEFORE** `model_step0040000` (contaminated) + **AFTER** rebalanced · **Accept (verbatim):** "Eval-only, no training. Accuracy + sensitivity + specificity per category (normal/cardio/effusion/comorbid), real vs synth columns, bootstrap CIs." Verify: `table1_{before,after}.csv` carry sens+spec columns per category.
- [ ] 🫀 Real-image disease-pair breakdown — cardio alone vs cardio+effusion (D9 / 7.6) — accuracy/sens/spec for the cardio head on (a) real cardio-alone vs (b) real cardio-in-presence-of-effusion; the (a)→(b) drop is the inherent-comorbidity-struggle signal behind the "we're not there yet" lead. **Supervisor (7.6):** "Accuracy on cardiomegaly alone (real images) [vs] cardiomegaly in presence of effusion (real images)." Script: reuse `metrics/classifier_report.py` (real cardio-only split vs real both-separable) · Checkpoint: none (real, eval-only) · Accept: cardio acc alone vs with-effusion (real) + Δ, sens/spec included. Verify: `results/exp11_tables/real_disease_pair_breakdown.csv` reports the pair + Δ.
