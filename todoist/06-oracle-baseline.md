# 🧭 06 · Oracle Baseline

## Reference while you do it
- 📄 Plan: plans/06-oracle-baseline.md
- 🧭 Authority: EXPERIMENTS.md (EXP-06, Gate 6) · docs/EVAL_PHILOSOPHY.md (oracle-headroom)
- 💻 Extend: scripts/finetune_classifier.py + presence_classifier_finetuned.pt · grad_cam.py (Gate 1)

## Section context (paste into the Todoist section)
**Description:** Build the two-independent-sigmoid detector (NOT softmax — softmax forbids "both"), retrain on CTR-derived labels, then train the Oracle (abundant real both) and Starved (scarce real both) reference arms and report the oracle − starved headroom. The detector harness is reused by every Exp 7 arm.
**Objective:** The Gate-6 headroom check that makes "as good as having the data" a real comparison (`local:oracle-headroom`, DoD 6).
**Goal:** Trained detector + Oracle + Starved (≥ 3 seeds) + oracle − starved ΔAUC on both-separable with the headroom verdict.
**Verify (whole leaf):** `grep -E "support|inconclusive|null" results/exp06_oracle/verdict.md` → oracle − starved ΔAUC ≥ 0.05 on both-separable → support.
**▶ Recommended prompt:** — custom; no skill fits.

## Tasks (one at a time)
- [ ] ❌ Build the two-sigmoid detector + retrain on CTR-derived labels — extend `finetune_classifier.py`; two independent sigmoid heads (cardio←CTR, effusion←CP). Verify (§7 ex 4): both sigmoids exceed 0.5 on a hand-picked both-image.
- [ ] ❌ Validate detector construct + criterion agreement (Gate 1 + Gate 2) — GradCAM pointing-game ≥ 80% in the heart box; criterion agreement via GradCAM + radiologist, never CTR again. Verify: `detector_construct_validity.md` logs pointing-game ≥ 0.80.
- [x] 🛡️ Exclude reserved test patients from oracle/starved training (patient-leakage guard) — drop films whose `Patient ID` ∈ `test_patient_ids.txt`; assert empty intersection BEFORE training (fail fast). Verify: `patient_disjoint_check.md` shows 0 shared.
- [x] ✅ Train the Oracle arm (abundant real both) — reuse the harness, ≥ 3 seeds, W&B `exp6_oracle_s0X_<date>` → `oracle_auc.csv`. Verify: ≥ 3 seed ckpts; AUC-by-stratum logged.
- [x] ✅ Train the Starved arm (scarce real both) — the headroom reference, ≥ 3 seeds → `starved_auc.csv`. Verify: ≥ 3 seed ckpts; AUC-by-stratum logged.
- [x] ✅ Compute oracle − starved headroom + figure + verdict — ΔAUC on both-separable + bootstrap CIs; apply 0.05/0.02 rule. Verify: `verdict.md` cites the both-separable ΔAUC + the headroom call.
