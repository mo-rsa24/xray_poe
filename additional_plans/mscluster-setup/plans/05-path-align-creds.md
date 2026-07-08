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
- [ ] ⚠️ Create top-level symlinks:
  ```
  cd /home-mscluster/mmolefe/Playground/PhD/Paper3
  ln -sfn /datasets/mmolefe/data data
  ln -sfn /datasets/mmolefe/checkpoints ckpts
  ls data/latents_v2 && ls ckpts/vae_v2_ldm_v4
  ```
- [ ] ⚠️ Remedy (layout mismatch — top-level names differ): symlink specific subdirs instead:
  ```
  mkdir -p data ckpts
  ln -sfn /datasets/mmolefe/data/nih data/nih
  ln -sfn /datasets/mmolefe/data/latents_v2 data/latents_v2
  ln -sfn /datasets/mmolefe/checkpoints/vae_v2 ckpts/vae_v2
  ln -sfn /datasets/mmolefe/checkpoints/vae_v2_ldm_v4 ckpts/vae_v2_ldm_v4
  ```
- [ ] Set `WANDB_API_KEY` and log in (add to `~/.bashrc` for persistence):
  ```bash
  export WANDB_API_KEY=<your-key>        # get from wandb.ai/settings
  echo "export WANDB_API_KEY=$WANDB_API_KEY" >> ~/.bashrc
  wandb login "$WANDB_API_KEY" --relogin
  ```
- [ ] Verify kaggle creds (needed for plan 03 NIH remedy and future downloads):
  `ls -l ~/.kaggle/kaggle.json` (expect chmod 600); `kaggle whoami`.

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
