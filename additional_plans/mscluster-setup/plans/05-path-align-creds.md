# 🔗 Path Alignment and Credentials

## Description
Symlink the repo's `data/` and `ckpts/` into `/datasets/mmolefe` so all scripts resolve paths without
modification. Set `WANDB_API_KEY` and ensure `~/.kaggle/kaggle.json` is in place.

## Purpose
O5 path-align+creds. Scripts reference `data/latents_v2` and `ckpts/vae_v2` relative to the repo root.
The symlinks make those paths resolve to the shared dataset store without touching any script.

## Goal
`ls data/latents_v2` and `ls ckpts/vae_v2_ldm_v4` both resolve from the repo root.
`wandb login` succeeds. `kaggle whoami` succeeds.

## Tasks
- [x] ✅ Create top-level symlinks: ✓ verified — `data -> /datasets/mmolefe/data`, `ckpts -> /datasets/mmolefe/checkpoints` both resolve. Also FIXED a broken sub-symlink: `latents_v2/joint_perch/scale_factor.pt` pointed at a dead hippo path; repointed to local `../scale_factor_perch.pt`.
  ```
  cd /home-mscluster/mmolefe/Playground/PhD/Paper3
  ln -sfn /datasets/mmolefe/data data
  ln -sfn /datasets/mmolefe/checkpoints ckpts
  ls data/latents_v2 && ls ckpts/vae_v2_ldm_v4
  ```
- [x] ✅ Remedy (layout mismatch): Not needed — top-level layout matched; only the one broken sub-symlink above needed repointing.
  ```
  mkdir -p data ckpts
  ln -sfn /datasets/mmolefe/data/nih data/nih
  ln -sfn /datasets/mmolefe/data/latents_v2 data/latents_v2
  ln -sfn /datasets/mmolefe/checkpoints/vae_v2 ckpts/vae_v2
  ln -sfn /datasets/mmolefe/checkpoints/vae_v2_ldm_v4 ckpts/vae_v2_ldm_v4
  ```
- [x] ✅ Set `WANDB_API_KEY` and log in: ✓ confirmed — key set in environment and `wandb login` accepted (user-verified). Not used by the smoke test or blunt-check, but ready for training/eval runs. Original steps below (add to `~/.bashrc` for persistence):
  ```bash
  export WANDB_API_KEY=<your-key>        # get from wandb.ai/settings
  echo "export WANDB_API_KEY=$WANDB_API_KEY" >> ~/.bashrc
  wandb login "$WANDB_API_KEY" --relogin
  ```
- [x] ✅ Verify kaggle creds: ✓ working — kaggle CLI (2.2.3, in `jaxstack`) successfully pulled the 24GB NIH `data.zip`, so auth is valid. Note: `kaggle whoami` is not a command in 2.2.3; use `kaggle config view`.

## Engagement Instructions
From the repo root on mscluster:
```
readlink -f data/latents_v2
readlink -f ckpts/vae_v2_ldm_v4
wandb status
kaggle whoami
```
Expected: both paths resolve under `/datasets/mmolefe/`, W&B shows logged-in user,
kaggle prints your username. DoD-5 met.

## Recommended skill
— custom; no skill fits (project-specific symlink setup and credential wiring).
