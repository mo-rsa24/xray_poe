# 🧪 02 · Deconfounded Splits

> ✅ **DONE (2026-06-25).** Splits built 1713/12254 → patient-filtered 1395/11671 → materialized 1323/11094 latents. Cross-leakage @100k: **floor-corrected SUPPORT** (clean experts; raw "null" was a 34.1% generated-image instrument artifact). Carry-forwards (in `plans/02-deconfounded-splits.md`): (1) cardio expert under-renders (4.2% vs 49.1% real) → reuse effusion / retrain cardio [Exp 3/5]; (2) instrument has a generated-image domain shift → floor-subtract for generated scoring [Exp 3/7].

## Reference while you do it
- 📄 Plan: plans/02-deconfounded-splits.md
- 🧭 Authority: EXPERIMENTS.md (EXP-04) · docs/EVAL_PHILOSOPHY.md (clean experts)
- 💻 Reuse: scripts/extract_single_disease.py · materialize_split.py · generate.py · ctr_overlay.py

## Section context (paste into the Todoist section)
**Description:** Construct `cardio ∧ ¬effusion` and `effusion ∧ ¬cardio` training sets, materialize them, then run a cross-leakage check: does the cardio-conditioned generator render effusion (CP blunting), and vice-versa? Keeps composition the only place the two diseases meet.
**Objective:** Clean single-disease experts so "composition" actually tests composition (`local:clean-experts`, DoD 2).
**Goal:** Two materialized deconfounded splits (N each) + a cross-leakage measurement ≤ the real single-disease base-rate + margin.
**Verify (whole leaf):** `grep -E "support|inconclusive|null" results/exp04_leakage/verdict.md` → cross-presence ≤ base-rate → support.
**▶ Recommended prompt:** — custom; no skill fits.

## Tasks (one at a time)
- [x] 🧱 Build the deconfounded split file lists — extend `extract_single_disease.py` to the strict `∧ ¬other` predicate → `data/splits/cardio_not_effusion.csv` + `effusion_not_cardio.csv`. Verify: label sets disjoint; no Effusion in the cardio split.
- [x] 🛡️ Exclude reserved test patients from the splits (patient-leakage guard) — drop films whose `Patient ID` ∈ `data/splits/test_patient_ids.txt` (from leaf 01). Verify: intersection == 0.
- [x] 📦 Materialize the splits into trainable latents — reuse `materialize_split.py` over `data/latents/train` → `*_latents.txt` manifests. Verify: every latent path resolves.
- [x] 🔬 Build the cross-leakage check harness — reuse `generate.py` + `ctr_overlay.py`; sample ≥1000 per condition, tally the *other* disease's presence → `results/exp04_leakage/cross_presence.csv`. Verify: ≥1000 per condition, rates in [0,1].
- [x] 📊 Run the leakage check + figure + verdict — bar of cross-presence (deconfounded vs mixed); apply support/null rule. Verify: `verdict.md` cites cross-presence vs base-rate.
