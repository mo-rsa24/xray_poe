# 13 · 🧹 Data-Cleaning Bias Audit (D8)

## Reference while you do it
- 📄 Plan: plans/13-data-cleaning-bias-audit.md
- 🧭 Authority: docs/SUPERVISOR_RECONCILE.md (D8, §7.9) · uses the Exp 1 logged-not-deleted entangled set (data/splits/entangled_scoped_out.csv)

## Section context (paste into the Todoist section)
**Description:** Build `scripts/excluded_set_audit.py` over the Exp 1 logged-not-deleted entangled set: count the films excluded by the `border_visible`/entangled filter (N, % of both-labelled), run the classifier on the excluded set, report as an explicit limitation.
**Objective:** Quantify + probe what the pixel-verifiable filter excluded (supervisor 7.9) — an explicit limitation, not a disclaimer.
**Goal:** `results/exp13_cleaning_bias/excluded_audit.md` reporting N + % excluded + the classifier's reads, framed as a limitation.
**Verify (whole leaf):** `ls results/exp13_cleaning_bias/{excluded_audit.md,excluded_classifier_reads.csv}` + `grep -iE "excluded|%|limitation" results/exp13_cleaning_bias/excluded_audit.md`.
**▶ Recommended prompt:** `/analyze results/exp13_cleaning_bias/excluded_audit.md`

## Tasks (one at a time)
- [ ] 🧹 Audit the excluded (entangled / border-filtered) set (D8) — count excluded by `border_visible`/entangled filter (N, % of both-labelled); run the classifier on the excluded set; write up as an explicit limitation. **Supervisor (D8, 7.9):** "data-cleaning bias." Script: new `scripts/excluded_set_audit.py` (uses `data/splits/entangled_scoped_out.csv` + `results/exp01_dynamics/both_film_list.csv` + `ckpts/presence_classifier_finetuned.pt`) · Checkpoint: none (real-data audit; classifier-under-test only) · **Accept (verbatim):** "Counts excluded by border_visible/entangled filter (N, % of both-labelled); classifier run on excluded set; reported as explicit limitation, not a disclaimer." Verify: `excluded_audit.md` states N + % excluded + classifier reads, framed as a stated limitation.
