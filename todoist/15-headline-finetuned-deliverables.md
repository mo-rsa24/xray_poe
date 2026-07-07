# 15 · 📈 Headline — Fine-Tuned GradCAM (Fig 3) + Table 2 (D6 + D7)

## Reference while you do it
- 📄 Plan: plans/07-headline-auc.md (§ "Supervisor deliverables — fine-tuned figures + tables")
- 🧭 Authority: docs/SUPERVISOR_RECONCILE.md (D6, D7, D13) · reuses leaf 10 (scripts/fig1_gradcam_panel.py) + leaf 11 (metrics/classifier_report.py)
- ⏳ Gated on: the Exp 7 arms (plan 07) + the after-model (rebalanced) checkpoint

## Section context (paste into the Todoist section)
**Description:** The §4 after-fine-tuning deliverables — reuse the D3/D4 harnesses on the +PoE fine-tuned classifier + Exp 7 arms. Every figure/table is a before/after pair; before-model utility numbers carry the contamination label.
**Objective:** Turn the headline into the "after fine-tuning" story that supports the gain or the clean null.
**Goal:** `figures/fig3_gradcam_finetuned.png` + `results/exp07_headline/table2_finetuned.csv`, before/after-labelled.
**Verify (whole leaf):** `ls figures/fig3_gradcam_finetuned.png results/exp07_headline/table2_finetuned.csv` + the contamination label present on before-model numbers.
**▶ Recommended prompt:** `/analyze figures/exp07_auc_by_stratum.png`

## Tasks (one at a time)
- [ ] 📈 Fig 3 — fine-tuned GradCAM panel (D6, §4 + Fig 3) — reuse `fig1_gradcam_panel.py` on the +PoE fine-tuned classifier; place beside Fig 1; one claim, glanceable. **Supervisor (D6, §4 + Fig 3):** "fine-tuned GradCAM." Script: reuse `scripts/fig1_gradcam_panel.py` (leaf 10) on `ckpts/poe_classifier_s0X.pt` · Checkpoint: after (rebalanced) + Exp 7 arms · **Accept (verbatim):** "Same panel as Fig 1 on +PoE fine-tuned model; comparison to Fig 1 visible." Verify: `fig3_gradcam_finetuned.png` renders; Fig1↔Fig3 comparison composited.
- [ ] 📊 Table 2 — post-fine-tune accuracy/sens/spec (D7) — reuse `classifier_report.py` on the Exp 7 fine-tuned arms; report ΔAUC + Δsens/spec on the comorbid stratum w/ bootstrap CIs. **Supervisor (D7, Table 2):** "post-fine-tune accuracy/sens/spec." Script: reuse `metrics/classifier_report.py` (leaf 11) + `arm_auc.csv` · Checkpoint: after + Exp 7 arms; **before-model numbers labelled "pre-retrain (contaminated) — illustrative of the gap, not clean utility evidence"** · **Accept (verbatim):** "Same metric table after fine-tuning; ΔAUC + Δsens/spec on comorbid stratum, bootstrap CIs; supports the gain or the clean null." Verify: `table2_finetuned.csv` carries comorbid ΔAUC + Δsens/spec.
- [ ] 🔀 Apply the before/after contamination label across synthetic outputs (D13) — every figure/table a before/after pair; before-model (`model_step0040000`) utility numbers stamped contaminated; the clean number comes only from the after-model. **Supervisor (D13):** "Every synthetic figure rendered as a before/after pair; before-model utility numbers carry the contamination label." Script: labelling pass over fig3 + table2 · Checkpoint: before `model_step0040000` (contaminated) + after rebalanced · Accept: every synthetic fig/table paired; before-model utility labelled contaminated. Verify: pairs labelled; before-model utility carries the caption.
