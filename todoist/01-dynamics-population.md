# 👯 01 · Dynamics Population

> ✅ **DONE (2026-06-25) — Gate-0 = support.** PA separable 378/525 = 72.0% (CI 68.0–75.7%). Test set frozen, 317 patients reserved. Open carry-forwards (in `plans/01-dynamics-population.md`): (1) 235 reserved patients leak into current train → next retrain subtracts `test_patient_ids.txt`; (2) `test_both` latent re-filter is a manifest, not yet materialized. Guard already wired into Exp 4 + rebalanced-cardio-retrain.

## Reference while you do it
- 📄 Plan: plans/01-dynamics-population.md
- 🧭 Authority: docs/EVAL_PHILOSOPHY.md (Gate 0) · EXPERIMENTS.md (EXP-01)
- 💻 Instrument: scripts/ctr_overlay.py (PSPNet CTR + CP) · data/nih/Data_Entry_2017.csv

## Section context (paste into the Todoist section)
**Description:** Go film-by-film through every NIH `Cardiomegaly ∧ Effusion` film, run the PSPNet segmentation CTR + costophrenic read, and classify each as separable (both signs independently readable) or entangled. The separable count *is* the pixel-verifiable test set.
**Objective:** Define the study population and the size of the gap PoE must fill — Gate 0, the paper's go/no-go (`local:population-exists`).
**Goal:** A frozen, audited separable test set + a reserved patient hold-out, with N / fraction / CI reported and the instrument validated ≥ 90% against hand labels.
**Verify (whole leaf):** `grep -E "support|inconclusive|null" results/exp01_dynamics/verdict.md` → one verdict citing the separable fraction vs the 30% / 10% thresholds.
**▶ Recommended prompt:** — custom; no skill fits. alt: `/visualize-data-samples data/nih` for the calibrate eyeball.

## Tasks (one at a time)
- [x] 📏 Build + sanity-check the CTR + CP instrument — reuse `ctr_overlay.py` (PSPNet); demote the circular `extractors.py` proxy. Verify: 3 known films (cardiomegaly/effusion/normal) match the eye.
    - [x] Confirm the red/blue overlay + numeric CTR
    - [x] Add the CP-angle read off the lung mask if absent
    - [x] Print `CTR=…, border_visible=…, CP_blunted=…` on one sample
- [x] 🗂️ Pull the both-disease film list + split PA/AP → `results/exp01_dynamics/both_film_list.csv`. Verify: rows == printed N; indices resolve under `data/nih/images*/`.
    - [x] Filter `Finding Labels` ⊇ {Cardiomegaly, Effusion}
    - [x] Split by `View Position` → PA / AP
    - [x] Write CSV + print N total/PA/AP
- [x] ✋ Calibrate — hand-label ~30 PA films (MANUAL) then validate the instrument ≥ 90%. PSPNet CTR for the separable call — never classifier+GradCAM. Verify: agreement ≥ 0.90 logged.
    - [x] Label ~30 (separable = border traceable AND a CP corner assessable; entangled = whiteout hides border → CTR unmeasurable)
    - [x] Run `ctr_overlay.py` on the same ~30; require ≥ 90% agreement
    - [x] On < 90% → inspect, fix threshold/rule, re-validate (do not proceed)
- [x] ⚙️ Automate — run the validated instrument on all ~1063 → `per_film_ctr_cp.csv`. Verify: rows == N total; no NaN CTR where border_visible=True.
    - [x] Run over all films from `both_film_list.csv`
    - [x] Write per-film CSV with the separable flag + AP note
    - [x] Print N separable / N entangled / fraction
- [x] 🔍 Audit — random 30 + every borderline (CTR 0.48–0.52 / ambiguous CP) → `audit_notes.md`. Verify: review + corrections recorded.
    - [x] Re-open a random 30 calls + all borderline rows with the overlay
    - [x] Correct systematic errors; re-run automate if a rule changed
    - [x] Write `audit_notes.md`
- [x] 🧊 Freeze the separable set as the pixel-verifiable test set; log entangled → `data/splits/pixel_verifiable_separable_test.csv` + `entangled_scoped_out.csv`. Verify: `wc -l` == N separable; entangled referenced, not deleted.
    - [x] Write the separable subset → frozen CSV
    - [x] Re-filter `data/latents/test_both` to separable-only (OLD-scope caution)
    - [x] Log the entangled subset (not deleted)
- [x] 🛡️ Reserve the test patients (patient-leakage guard) → `data/splits/test_patient_ids.txt`. NIH has multiple films per patient; every training builder subtracts this list. Verify: intersection with any training split == 0.
    - [x] Extract `Patient ID`s for every separable test film from `Data_Entry_2017.csv`
    - [x] Write the unique set → `test_patient_ids.txt`
    - [x] Print `test patients reserved: K unique Patient IDs across N films`
- [x] 📊 Figure + file verdict against pre-registration → `figures/exp01_separable_entangled_bar.png` + `results/exp01_dynamics/verdict.md`. Verify: verdict line — support ≥30% · null <10% · inconclusive 10–30%.
    - [x] Plot the stacked PA/AP separable-vs-entangled bar (separable slice called out)
    - [x] Apply the falsify rule (check AP-threshold recovery first if 10–30%)
    - [x] Write `verdict.md` with fraction + CI + the call
