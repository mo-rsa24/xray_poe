# 12 · 📈 Figure 2 — CTR Scatter, Real vs Synth (D5)

## Reference while you do it
- 📄 Plan: plans/12-figure2-ctr-scatter.md
- 🧭 Authority: docs/SUPERVISOR_RECONCILE.md (D5, Fig 2) · consumes scripts/ctr_overlay.py (PSPNet heart/lung widths)

## Section context (paste into the Todoist section)
**Description:** Build `scripts/fig2_ctr_scatter.py` consuming `ctr_overlay.py`: heart-width (Y) vs lung-width (X), real vs synth colour-coded, CTR=0.5 iso-line; single-disease overlap vs comorbid divergence. Real layer now; synth before/after.
**Objective:** The geometric "not there yet" plot — synth tracks real on single-disease, diverges on comorbid.
**Goal:** `figures/fig2_ctr_scatter_{real,before,after}.png` with the iso-line + comorbid divergence visible at a glance.
**Verify (whole leaf):** `ls figures/fig2_ctr_scatter_{real,before,after}.png` + `python scripts/fig2_ctr_scatter.py --help`.
**▶ Recommended prompt:** `/design-figure figures/fig2_ctr_scatter_beforeafter.png`

## Tasks (one at a time)
- [ ] 📈 Build the CTR scatter (heart-width Y vs lung-width X, real vs synth) — extract widths via `ctr_overlay.py`; scatter real vs synth colour-coded; draw CTR=0.5 iso-line; surface single-disease overlap vs comorbid divergence. **Supervisor (D5, Fig 2):** "CTR scatter heart-width(Y) vs lung-width(X), real vs synth." Script: new `scripts/fig2_ctr_scatter.py` (consumes `ctr_overlay.py`) · Checkpoint: **real** now (ckpt-independent); synth **BEFORE** `model_step0040000` + **AFTER** rebalanced · **Accept (verbatim):** "Scatter, real vs synth color-coded, CTR=0.5 iso-line; synth tracks real on single-disease, diverges on comorbid (the 'not there yet' visual)." Verify: real/before/after scatters present; iso-line drawn.
- [ ] 🔀 Render before/after synth pair + label contamination (D13) — before/after synth layers against the same real backdrop; caption the before layer contaminated; one claim, glanceable. **Supervisor (D13):** "Every synthetic figure rendered as a before/after pair; before-model utility numbers carry the contamination label." Script: compositing → `figures/fig2_ctr_scatter_beforeafter.png` · Checkpoint: before `model_step0040000` (contaminated) + after rebalanced · Accept: before/after vs same real backdrop, before captioned contaminated. Verify: paired against the same real backdrop, before captioned.
