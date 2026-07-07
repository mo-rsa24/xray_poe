# 🔁 03 · VAE Ceiling

> ✅ **DONE (2026-06-25).** Codec ceiling **boundary-refined INCONCLUSIVE — but NOT low; VAE faithful & reusable** (median |ΔCTR|=0.011, local LPIPS ≈ control; genuine sign-flips 6.1%/6.9% « 10% once boundary jitter removed). Carry-forwards (in `plans/03-vae-ceiling.md`): (1) abstain margin ±0.03 CTR / ±10° CP → Exp 3/7; (2) VAE reusable, ceiling not a retrain trigger → Exp 5.

## Reference while you do it
- 📄 Plan: plans/03-vae-ceiling.md
- 🧭 Authority: EXPERIMENTS.md (EXP-02) · docs/EVAL_PHILOSOPHY.md (codec ceiling)
- 💻 Extend: scripts/vae_recon_pilot.py · Reuse: ckpts/vae_step0025000.pt · ctr_overlay.py

## Section context (paste into the Todoist section)
**Description:** Push the frozen separable-both films through the VAE round-trip and check the heart edge and the fluid corner still read the same — localized heart-border SSIM/LPIPS, CP sharpness, |ΔCTR|, CP-class flip-rate. Read *locally*, not whole-image.
**Objective:** Rule out the codec as a scapegoat so a later composition gap is genuinely a composition failure (`local:codec-ceiling`, DoD 3).
**Goal:** A localized recon-fidelity report on the separable set + matched control, with the ceiling verdict (high/low/inconclusive).
**Verify (whole leaf):** `grep -E "high|low|inconclusive" results/exp02_ceiling/verdict.md` → |ΔCTR| ≤ 0.02 and CP preserved ≥ 95% → high.
**▶ Recommended prompt:** — custom; no skill fits.

## Tasks (one at a time)
- [x] 🛠️ Extend the recon pilot with localized witness metrics — add heart-border / CP local LPIPS + |ΔCTR| + CP-flip to `vae_recon_pilot.py` (currently whole-image). Verify: finite values + printed before/after CTR on one film.
- [x] 🗃️ Assemble the recon set — frozen separable set (leaf 01) + a size-matched single-disease control → `results/exp02_ceiling/recon_set_manifest.csv`. Verify: both N reported; control films carry one disease.
- [x] ✋ Eyeball ~20 real-vs-recon pairs (MANUAL) — confirm the metric's "preserved" verdict matches what you see (you can still trace the border / read the corner). Verify: `eyeball_notes.md` confirms agreement (or records the witness-box fix).
- [x] 📊 Run the round-trip on all films + figure + ceiling verdict — tally |ΔCTR|, CP-flip-rate, local LPIPS vs control; apply the tolerance rule. Verify: `verdict.md` states |ΔCTR|, CP-flip, the high/low call.
