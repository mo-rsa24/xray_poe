# ⭐ 07 · Headline AUC

## Reference while you do it
- 📄 Plan: plans/07-headline-auc.md
- 🧭 Authority: EXPERIMENTS.md (EXP-07 ⭐, Gate 6) · docs/EVAL_PHILOSOPHY.md (headline-utility, §5 pre-reg)
- 💻 Reuse: gate filter (ctr_overlay + grad_cam.py + metrics/fid.py) · naive_overlay.py · Exp-6 detector

## Section context (paste into the Todoist section)
**Description:** Train five classifiers — Starved / +Individual / +Overlay / +PoE / Oracle — and compare AUC on every stratum. Every synthetic image passes the identical Gate 0–5 filter before entering its arm (symmetry rule). The claim wins only if PoE beats both overlay and starved on the both-separable stratum; +Individual (single-disease synthetics only) is the composition control isolating whether a +PoE gain is composition or just more single-disease data.
**Objective:** The full thesis — gated PoE data improves the starved classifier, beats overlay, closes toward the oracle (`local:headline-utility`, DoD 7).
**Goal:** A 5-arm AUC-by-stratum figure + the single pre-registered verdict + the ΔAUC(+PoE−+Individual) composition diagnostic.
**Verify (whole leaf):** `grep -E "support|inconclusive|null" results/exp07_headline/verdict.md` → ΔAUC(PoE−overlay) ≥ 0.03 AND ΔAUC(PoE−starved) ≥ 0.05 on both-separable, disjoint CIs, lumpy → support.
**▶ Recommended prompt:** — custom; no skill fits. alt: `/analyze figures/exp07_auc_by_stratum.png`.

## Tasks (one at a time)
- [ ] 🚪 Build the unified synthetic Gate 0–5 filter (symmetry rule) — reuse `ctr_overlay.py` (Gate 3) + `grad_cam.py` (Gate 1) + `metrics/fid.py` (Gate 4); image flows 0→1→3→4→5, failures discarded. Verify: known-good passes, known-bad (heart not enlarged) discarded at Gate 3.
- [ ] 🎲 Generate + gate the PoE augmentation set — Exp-5-chosen LDM + `cfg_compose` → `data/synthetic/poe_gated/`. Verify: every kept image has a Gate-3 pass logged.
- [ ] 🩹 Generate + gate the overlay augmentation set — `naive_overlay.py` through the IDENTICAL filter → `data/synthetic/overlay_gated/`. Verify: same filter version/hash as PoE logged in both.
- [ ] ✋ Radiologist gate (Gate 5) on both synthetic sets (MANUAL, blinded) — calibrate rubric → ≥90% self-consistency → full blinded read → audit rejects → freeze accepted into `*_final/`. Verify: review CSV covers every gated image, blinded; accept-rates per source.
- [ ] 🛡️ Assert patient-disjoint training for every arm (patient-leakage guard) — every real training film's `Patient ID` ∉ `test_patient_ids.txt` (synthetic images exempt). Verify: `patient_disjoint_check.md` shows 0 shared across arms.
- [ ] 🏋️ Train the +Overlay and +PoE arms — reuse the Exp-6 detector harness, ≥ 3–5 seeds, W&B `exp7_{overlay,poe}_s0X_<date>`. Verify: ≥ 3 seeds/arm; every synthetic image traces to a Gate 0–5 + radiologist pass.
- [ ] 🧩 Generate + gate + train the +Individual-synthetic arm (composition control · reconcile 2026-06-26) — reuse plan 09's synthetic cardio + effusion (mixed, single-disease, NEVER composed) through the IDENTICAL Gate 0–5 filter; train ≥ 3–5 seeds, W&B `exp7_individual_s0X_<date>`. **Supervisor (7.11-step-4 / §4):** "Test if augmenting with individual synthetics helps." D13: before-model now / after-model on retrain. Verify: ≥ 3 seeds; same filter version as PoE; images single-disease (not composed).
- [ ] 📊 Compute stratified ΔAUC + figure + evaluate the headline rule — AUC by stratum + bootstrap CIs; apply the single headline rule; file against §5 (no post-hoc reinterpretation). **+ Diagnostic (reported, NOT a gate):** also report ΔAUC(+PoE−+Individual) on both-separable w/ CI — the value of composition specifically (vs just more single-disease data); the pre-registered rule is unchanged. Verify: `verdict.md` states all three ΔAUCs, CI disjointness, lumpiness, the call.
