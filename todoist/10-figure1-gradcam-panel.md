# 10 · 🖼️ Figure 1 — GradCAM Panel, Real vs Synth (D3)

## Reference while you do it
- 📄 Plan: plans/10-figure1-gradcam-panel.md
- 🧭 Authority: docs/SUPERVISOR_RECONCILE.md (D3, Fig 1) · extends scripts/grad_cam.py + corner-artifact-fixed metrics/grad_cam_utils.py

## Section context (paste into the Todoist section)
**Description:** Extend `grad_cam.py` → `scripts/fig1_gradcam_panel.py`: Normal/Effusion/Cardio/Comorbid × {2 correct, 2 incorrect}, real & synth side-by-side, corner-artifact-fixed saliency. Synth columns rendered on both checkpoints.
**Objective:** Qualitative saliency — the classifier attends the right anatomy on correct cases, the failure geometry on incorrect ones.
**Goal:** `figures/fig1_gradcam_panel_{before,after}.png` matching the supervisor layout, corner-artifact-free, tidy thumbnails.
**Verify (whole leaf):** `ls figures/fig1_gradcam_panel_before.png figures/fig1_gradcam_panel_after.png` + `python -c "from metrics.grad_cam_utils import *"`.
**▶ Recommended prompt:** `/design-figure figures/fig1_gradcam_panel_beforeafter.png`

## Tasks (one at a time)
- [ ] 🖼️ Build the Fig 1 GradCAM panel (real vs synth, correct + incorrect) — per category 2 correctly + 2 incorrectly classified; real & synth side-by-side; one saliency map each; tidy small thumbnails. **Supervisor (D3, Fig 1):** "GradCAM 4-col × (2 correct+2 incorrect), real vs synth." Script: `scripts/grad_cam.py` → new `scripts/fig1_gradcam_panel.py` (saliency via corner-fixed `metrics/grad_cam_utils.py`) · Checkpoint: synth cols **BEFORE** `model_step0040000` + **AFTER** rebalanced · **Accept (verbatim):** "Normal/Effusion/Cardio/Comorbid × {2 correct, 2 incorrect}, real & synth side-by-side; corner-artifact-fixed grad_cam_utils.py; saliency on correct anatomy for correct cases. Tidy thumbnails." Verify: both panels render; corner-artifact check passes.
- [ ] 🔀 Render before/after pair + label contamination (D13 + figure principle) — present before/after synth columns as a pair; caption the before column "pre-retrain (contaminated)"; one claim, side-by-side, glanceable. **Supervisor (D13):** "Every synthetic figure rendered as a before/after pair." Script: compositing → `figures/fig1_gradcam_panel_beforeafter.png` · Checkpoint: before `model_step0040000` (contaminated) + after rebalanced · Accept: paired, before captioned contaminated, reads as one claim at a glance. Verify: figure shows before/after side-by-side, before captioned contaminated.
