# Opacity Instrument (subtask)

**Description:** A validated regional lung-opacity instrument (RALE-style, over the PSPNet lung mask) replacing brightness-based basal_clarity; shared by Scope 1 (cleaning) + Scope 2 (haze-map input).
**Objective:** a trustworthy, segmentation-grounded opacity instrument at CTR's bar.
**Goal:** built + validated scorer with a real-correlation number + a documented API.
**Verify:** --selftest prints quadrant scores; validation.md cites correlation + robustness win.

## Tasks
- Build the 4-quadrant scorer (metrics/opacity_score.py)
- Validate on real vs geometric refs + robustness vs basal_clarity
- Expose consumer API + record per-pixel escalation call
