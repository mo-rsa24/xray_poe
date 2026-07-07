# ⚙️ 14 · Monolithic Joint-Condition Baseline (D12)

## Reference while you do it
- 📄 Plan: plans/14-monolithic-baseline-deferred.md
- 🧭 Authority: docs/SUPERVISOR_RECONCILE.md (D12) — **promoted from deferred → ACTIVE 2026-06-26** (Devon lists it 3× as a result to include)
- 💻 Reuse: scripts/train_ldm.py (LDM stack, joint-condition conditioning) · scripts/ctr_overlay.py (Gate-3 yield/CTR/CP)

## Section context (paste into the Todoist section)
**Description:** Build the comorbid training set, train a joint-condition diffusion model directly on comorbid images (NOT composition), and evaluate it head-to-head against the PoE composition on the frozen both-separable stratum (Gate-3 yield + CTR/CP fidelity; downstream if arms reach it).
**Objective:** Answer the reviewer's "why compose at all instead of training one model on joint conditions?" — by *showing* it, not asserting it (`local:monolithic-vs-composition`).
**Goal:** A trained joint-condition diffusion model + a recorded monolithic-vs-PoE comparison on both-separable, with the verdict on whether composition is justified.
**Verify (whole leaf):** `grep -iE "composition-justified|monolithic-wins|tie" results/exp14_monolithic/verdict.md` → monolithic vs PoE on both-separable + the call.
**▶ Recommended prompt:** `/decompose-plan` — only if the monolithic train balloons into its own retrain saga. alt: `/analyze figures/exp14_monolithic_vs_poe.png`.

## Tasks (one at a time)
- [ ] 🗃️ Build the comorbid joint-condition training set — assemble real both-disease films with the joint label; drop any `Patient ID` ∈ `test_patient_ids.txt` (assert empty intersection); report N. **Supervisor (D12 / 7.8):** "Train a diffusion model directly on comorbid images (not compositional)." Script: new builder over `data/nih/Data_Entry_2017.csv` + `data/splits/test_patient_ids.txt` · Checkpoint: none · Accept: patient-disjoint set with N reported. Verify: `results/exp14_monolithic/patient_disjoint_check.md` shows 0 shared with the test reserve.
- [ ] 🏋️ Train the joint-condition diffusion model (hippo, `df` guard) — condition directly on the comorbid label (not two experts); train to convergence; W&B `exp14_monolithic_<date>`. **Supervisor (D12 / 1.2, 1.3):** the monolithic joint-condition diffusion baseline. Script: reuse `scripts/train_ldm.py` on `data/splits/monolithic_comorbid_train.csv` · Checkpoint: produces `ckpts/monolithic_joint_step*.safetensors` · Accept: ckpt exists; sampling produces comorbid films. Verify: ckpt present; sampled comorbid grid logged.
- [ ] 📊 Compare monolithic vs +PoE on the both-separable stratum + verdict — Gate-3 co-presence yield + CTR/CP fidelity of monolithic vs PoE; if Exp 7 arms run, also a +Monolithic augmentation arm vs +PoE. **Supervisor (D12 / 7.8):** "Shows composition is superior to monolithic" — or refutes it. Script: reuse `ctr_overlay.py` + the arm-AUC harness · Checkpoint: D13 before/after where synthetic · Accept: monolithic-vs-PoE numbers on both-separable + composition-justified/refuted call. Verify: `results/exp14_monolithic/verdict.md` + `figures/exp14_monolithic_vs_poe.png`.
