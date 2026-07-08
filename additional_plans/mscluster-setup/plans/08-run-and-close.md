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

- [ ] ⚠️ Run the setup tree automatically — inspects, unzips, rsyncs, asserts env, runs smoke test
  and `effusion_blunt_check.py`; verifies each step against its Engagement Instructions; parks the
  judgment tasks (GitHub auth, W&B/kaggle login) for manual resolution:
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

- [ ] ⚠️ Resume the blocked progress-track pass now that the blunt-check has run on Blackwell —
  promote the blunt-rate, close vae-v2 plan 06 + DoD-6, append the gate to
  `docs/DECISION_TIMELINE.md`:
  ```
  /progress-track vae-v2
  ```
  Gate: blunt-rate ≥0.80 → plan 06 ✅. If blunt-rate <0.80, do NOT force a pass — record
  the number and open a diagnosis task instead (see plan 07 remedy).

- [ ] ⚠️ Reconcile both scopes against repo evidence and confirm the tree is current:
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
