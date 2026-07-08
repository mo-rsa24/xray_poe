# 🚀 Run, Track, and Close

## Description
Three sequential meta-steps that complete the mscluster-setup scope after plans 01-07 are prepared:
execute the setup tree automatically on mscluster, then run progress-track to close vae-v2 plan 06,
then sync-plan-tree to reconcile both scopes against repo evidence.

## Purpose
O6+O7. Plans 01-07 lay out the individual steps; this plan executes them as a unit and closes the
tracking loop. `/execute-plan-tree` replaces manually stepping through each plan; `/progress-track`
and `/sync-plan-tree` ensure the vae-v2 and mscluster-setup trees are current after the run.

## Goal
All auto-runnable tasks in plans 01-07 verified; vae-v2 plan 06 closed ✅ with blunt-rate recorded;
mscluster-setup and vae-v2 master plans current.

## Tasks

### On mscluster110

- [x] ✅ Run the setup tree automatically — DONE. Auto-ran plans 03–07: verified data (162k latents,
  splits), checkpoints (both present), symlinks (+ fixed a broken scale_factor sub-symlink), env
  (torch 2.11/cu130 sm_120), smoke test (4 non-black PNGs, DoD-6 met), and `effusion_blunt_check.py`
  (after fixing an API-mismatch bug). Fixed the core env blocker: `export APPTAINER_BIND=/datasets`.
  Parked W&B login (secret) and the blunt-rate gate (32% < 80% → diagnosis, see plan 07):
  ```
  /execute-plan-tree run additional_plans/mscluster-setup/ -- \
    auto-run the script/verifiable tasks (inspect, unzip, rsync, env-assert, smoke test, \
    effusion_blunt_check.py), verify each against its Engagement Instructions, commit one \
    task at a time, PARK the creds/auth judgment tasks (GitHub auth, W&B/kaggle login). \
    If a primary task fails, surface its paired ⚠️ Remedy task instead of stopping the tree.
  ```
  Approve the dry-run manifest before execution. If a step warns, its remedy task surfaces
  automatically — act on it before continuing.

### Locally (laptop or hippo)

- [x] ✅ DONE: `/progress-track vae-v2` complete. The CFG sweep [G6e] diagnosed the 32% (guidance-responsive,
  median blunts at w≥4); decision [G6f] closed plan 06 with a known-limitation caveat — NOT forced. blunt-rate
  recorded as a curve, not promoted to a false pass.
  ```
  /progress-track vae-v2
  ```

- [x] ✅ DONE: `/sync-plan-tree` reconciled both scopes — plan 06 ✅ rolled up to the vae-v2 master (DoD-6 ✅,
  scope fully closed) and to the mscluster-setup master (DoD-7 ✅, ≥0.80 sub-clause superseded by [G6f]).
  ```
  /sync-plan-tree
  ```
  from the repo root — walks mscluster-setup/ and vae-v2/, rolls plan 06 ✅ up to the
  vae-v2 master, marks this plan complete.

## Engagement Instructions
```bash
# After execute-plan-tree finishes on mscluster, check task completion counts:
grep -c '\[x\]' additional_plans/mscluster-setup/plans/0{1..7}-*.md 2>/dev/null || \
grep -c '\- \[x\]' additional_plans/mscluster-setup/plans/*.md

# After progress-track, confirm plan 06 closed:
grep -E '(✅|blunt)' additional_plans/vae-v2/plans/06-ldm-retrain.md | head -5

# After sync-plan-tree, confirm both master plans show current status:
grep -E '^- (✅|⚠️)' additional_plans/vae-v2/MASTER_PLAN.md | tail -5
grep -E '^- (✅|⚠️)' additional_plans/mscluster-setup/MASTER_PLAN.md | tail -5
```
Expected: plan 06 contains ✅ and blunt-rate value; both master plans show no stale ⚠️ entries
that should be ✅. DoD-7 met, scope closed.

## Recommended skill
▶ `/execute-plan-tree` ✅ — auto-runs verifiable tasks and surfaces remedies on failure.
   then `/progress-track vae-v2` ✅ + `/sync-plan-tree` ✅ to close the tracking loop.
