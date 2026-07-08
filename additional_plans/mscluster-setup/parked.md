# Execution halted — blunt-rate gate FAILED (real result, needs your diagnosis)

Branch: `auto/exec-plans-20260626`
Scope: `additional_plans/mscluster-setup`
Date: 2026-07-08

The environment is fully stood up and the smoke test passes. The loop ran every
verifiable task through plans 03–07. It stopped at the one genuine scientific gate:
the effusion blunt-rate came in at **32%** (pass mark ≥80%). Per plan 07's own remedy,
that number is recorded honestly and NOT forced to a pass — it is your call to diagnose.

## Executed and verified (this branch)

- ✅ **03 data** — 162,004 latents (≥74k expected), splits present, VinDr extracted.
- ✅ **04 checkpoints** — VAE-v2 + LDM v4-60k both present, non-zero.
- ✅ **05 paths** — `data/` + `ckpts/` symlinks resolve; fixed a broken `scale_factor.pt`
  sub-symlink (pointed at a dead hippo path → repointed to local `scale_factor_perch.pt`).
- ✅ **06 smoke test** — Blackwell RTX PRO 6000 (sm_120) confirmed; 4 cardiomegaly films
  generated in 1.7s, mean pixel 123–161 (non-black). **DoD-6 met.**
- ✅ **07 blunt-check ran** — after fixing an instrument bug (see below).

## Errors found and fixed autonomously

1. **Env / container bind (the big one).** The pytorch module is an Apptainer container
   (`apptainer exec --nv`) that binds `$HOME` but NOT `/datasets`. Every `/datasets/...`
   path and the repo symlinks into it were invisible inside the container, so safetensors
   raised a misleading `FileNotFoundError` on files that exist on the host.
   Fix: `export APPTAINER_BIND=/datasets` before running the module python. Also: never
   pipe `module load ... | tail` (subshell drops the env). Saved to project memory.
2. **Broken `scale_factor.pt` symlink** → repointed to the local real file.
3. **`effusion_blunt_check.py` API mismatch.** It read `effusion_present`/`cp_max_angle`
   straight off `score_decoded_array`, which returns only the geometry dict (those keys come
   from `_row_from_score`). Unpatched → `valid=0/200`, `nan%`. Patched to route each score
   through `EC._row_from_score` (the canonical two-step). Post-fix: `valid=200/200`.

## Parked — needs you

- **07 diagnosis (blocking the close)** · blunt-rate **32%**, median CP angle **70.5°**
  (just under the 80° cutoff), n=200, w=1.0. The v4 effusion *classifier* evidence was strong
  (Exp-09 synth sensitivity 0.905→0.960 PASS), but the *geometric* CP-blunting instrument reads
  low. Is it (a) a real rendering shortfall, (b) a CFG-weight effect (measured at w=1.0 vs the
  smoke test's w=4.0), or (c) instrument calibration (80° cutoff vs 70.5° median)?
  Inspect `results/vae_v2/ldm_sanity/v4_60k_final/effusion_panel.png` to judge.
  Do NOT close plan 06 until root cause is clear.
- **05 W&B login** · secret · `WANDB_API_KEY` not set. Set it yourself and `wandb login`.
  Not required by the smoke test or blunt-check.
- **07 close-out (blocked on the diagnosis)** · promote plan 06 → ✅, file DoD-6 in
  `docs/DECISION_TIMELINE.md`, `/progress-track vae-v2`, `/sync-plan-tree`.

## Not gate-blocking

- **NIH raw images** · `data.zip` (24GB) downloaded to `/datasets/mmolefe/data/nih`; images not
  yet extracted. Not needed for the smoke test or blunt-check. `kaggle whoami` is not a valid
  command in kaggle 2.2.3 — use `kaggle config view`.

## How compute must be run on this node (for reference)

```
source /usr/share/lmod/lmod/init/bash
module load pytorch/pytorch-cuda13.3-ubuntu26.04-00
export APPTAINER_BIND=/datasets
python scripts/generate.py ...
```
