# mscluster-setup: Stand Up Paper3 on mscluster110

## Mission
Resume Paper3 on mscluster110's Blackwell GPU (sm_120) exactly where hippo left off.
Dataset store at /datasets/mmolefe (separate from project dir at
/home-mscluster/mmolefe/Playground/PhD/Paper3). Seven objectives take the node from
bare access to a passing smoke test and a closed vae-v2 plan 06.

## Objectives
1. **O1 cluster-access+git** — clone `git@github.com:mo-rsa24/xray_poe.git`,
   branch `auto/exec-plans-20260626`. Blocker: GitHub auth on cluster. Remedy
   ladder: ssh -A agent-forward → deploy key → HTTPS PAT.
2. **O2 blackwell-env** — micromamba env `jaxstack`; torch>=2.7 built for cu128
   (pinned 2.5.1/cu124 will not run on sm_120). Blocker: Blackwell wheels.
   Remedy: stable cu128 → nightly cu128 → verify driver CUDA>=12.8 / module load;
   gate on `assert torch.cuda.get_device_capability()[0]>=12`.
3. **O3 data** — reuse `/datasets/mmolefe` first (VinDr zip + data/ + checkpoints/
   already present); rsync latents via `scripts/rsync_to_mscluster.sh`. Blocker:
   missing NIH / kaggle creds. Remedy: `download_nih.sh` into
   `/datasets/mmolefe/data/nih`; set `KAGGLE_USERNAME` / `KAGGLE_KEY`.
4. **O4 checkpoints** — reuse `/datasets/mmolefe/checkpoints` or
   `rsync_to_mscluster.sh`. Blocker: hippo and mscluster not mutually reachable.
   Remedy: MODE=pull on mscluster, or two-hop via laptop.
5. **O5 path-align+creds** — symlink `Paper3/data -> /datasets/mmolefe/data` and
   `Paper3/ckpts -> /datasets/mmolefe/checkpoints`; `WANDB_API_KEY` set;
   `~/.kaggle/kaggle.json` in place. Blocker: dataset layout mismatch. Remedy:
   symlink specific subdirs instead of top-level.
6. **O6 verify+smoke** — `nvidia-smi` confirms Blackwell GPU; 1-film generate
   completes without error.
7. **O7 finish-progress-track** — run `scripts/effusion_blunt_check.py` (unblocked
   on Blackwell), record blunt-rate; close vae-v2 plan 06 + DoD-6; then
   `/progress-track vae-v2` + `/sync-plan-tree`.

## Goals
1. **Git access confirmed** — repo cloned at
   `/home-mscluster/mmolefe/Playground/PhD/Paper3`, branch
   `auto/exec-plans-20260626` checked out, `git log --oneline -3` verified.
2. **Environment ready** — `micromamba activate jaxstack` succeeds; `python -c
   "import torch; assert torch.cuda.get_device_capability()[0]>=12"` passes;
   torch version >=2.7 logged.
3. **Data present** — `/datasets/mmolefe/data` accessible with NIH splits and
   VinDr latents; manifest counts match hippo originals.
4. **Checkpoints present** — VAE-v2 (`vae_v2_step0075000.pt`) and LDM v4-60k
   (`4vg2h5i4`) accessible via symlink or direct path; `ls -lh` output logged.
5. **Paths and creds aligned** — symlinks resolve; `WANDB_API_KEY` accepted by
   W&B CLI; `kaggle whoami` succeeds.
6. **Smoke test passed** — `nvidia-smi` shows Blackwell GPU (sm_120); 1-film
   `generate.py` run completes and produces a non-black PNG.
7. **progress-track complete** — `effusion_blunt_check.py` output recorded (blunt
   rate >=0.80 = pass); plan 06 body updated to ✅; DoD-6 filed; decision timeline
   entry appended; `/progress-track vae-v2` and `/sync-plan-tree` both complete.

## Expected Outcome
A fully operational Paper3 environment on mscluster110 with the Blackwell GPU
verified, all data and checkpoints accessible, and a passing 1-film smoke test.
Effusion blunt-rate >=0.80 recorded, vae-v2 plan 06 closed ✅, DoD-6 filed in
DECISION_TIMELINE.md, and the vae-v2 progress tree up to date. After this scope,
the Blackwell node can serve as the primary compute for any remaining Paper3 work.

## Definition of Done
1. ✅ Repo cloned and branch confirmed on mscluster110.
2. ✅ `jaxstack` env active with torch >=2.7 on cu128; capability >=12 asserted.
   (met via the pytorch container module: torch 2.11.0+cu130, capability (12,0))
3. ✅ Data manifest verified (NIH splits + VinDr latents count matches hippo).
   (162,004 latents; NIH 112,120 images + splits; VinDr extracted)
4. ✅ VAE-v2 + LDM v4 checkpoint files accessible on mscluster.
5. ✅ All symlinks resolve; WANDB and kaggle creds accepted.
6. ✅ 1-film smoke test produces a valid PNG on the Blackwell GPU.
7. ⚠️ `effusion_blunt_check.py` blunt-rate >=0.80 recorded AND plan 06 ✅ AND
   timeline entry appended AND `/progress-track vae-v2` + `/sync-plan-tree` done.
   (blunt-rate measured 32% < 0.80 → FAIL; diagnosis parked, plan 06 not closed)

## Sub-Scopes
(none)

## Plans
Dependency order: `01 → 02 → 03 → 04 → 05 → 06 → 07`
- ✅ 01-cluster-access-git.md — clone repo + checkout branch on mscluster110 (Goal 1)
- ✅ 02-blackwell-env.md — micromamba `jaxstack` env with torch >=2.7 / cu128; capability >=12 gate (Goal 2)
- ✅ 03-data.md — verified `/datasets/mmolefe/data` (162k latents, splits, VinDr extracted, NIH 112,120 images) (Goal 3)
- ✅ 04-checkpoints.md — both VAE-v2 + LDM v4-60k checkpoints present and non-zero (Goal 4)
- ✅ 05-path-align-creds.md — symlinks resolve (+ fixed broken scale_factor), kaggle works, WANDB login accepted (Goal 5)
- ✅ 06-verify-smoke.md — Blackwell sm_120 confirmed; 4-film generate produced non-black PNGs; DoD-6 met (Goal 6)
- ⚠️ 07-finish-progress-track.md — blunt-check RAN (bug fixed) but blunt-rate 32% < 80% → FAIL; diagnosis parked; plan 06 stays open (Goal 7)
- ⚠️ 08-run-and-close.md — auto-run of 03–07 DONE; `/progress-track` + `/sync-plan-tree` parked pending the plan 07 diagnosis (Goals 6+7)
