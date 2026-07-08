# 🗂️ Checkpoints: Verify or Transfer

## Description
Confirm that the canonical VAE-v2 and LDM v4-60k checkpoints are accessible in `/datasets/mmolefe/checkpoints`.
These are already carried by `rsync_to_mscluster.sh` if plan 03 ran — this plan just verifies and handles
the fallback when hippo and mscluster cannot reach each other directly.

## Purpose
O4 checkpoints. `generate.py` and `effusion_blunt_check.py` both load `vae_v2_step0075000.pt` and
`model_ema_step0060000.safetensors`. If they are missing, smoke test and progress-track are blocked.

## Goal
`find` confirms both `vae_v2_step0075000.pt` and `model_ema_step0060000.safetensors` (LDM v4-60k,
run `4vg2h5i4`) are accessible under `/datasets/mmolefe/checkpoints`. File sizes non-zero.

## Tasks
- [x] ✅ Check if checkpoints are already present: ✓ verified — both `vae_v2/vae_v2_step0075000.pt` and `vae_v2_ldm_v4/model_ema_step0060000.safetensors` present under `/datasets/mmolefe/checkpoints`.
  ```
  find /datasets/mmolefe/checkpoints -iname "*model_ema_step0060000*" -o -iname "*vae_v2*"
  ```
- [x] ✅ Remedy (transfer): Not needed — checkpoints already present.
  ```
  MODE=pull SRC_SSH=molefe@hippo DRY=0 bash scripts/rsync_to_mscluster.sh
  ```
  If hippo is unreachable from mscluster too: scp from laptop as two-hop relay
  (`scp -J mscluster molefe@hippo:/path/to/ckpt /datasets/mmolefe/checkpoints/`).
- [x] ✅ Log sizes: ✓ verified — LDM `model_ema_step0060000.safetensors` = 65,739,320 bytes (~63M);
  `vae_v2_step0075000.pt` present, non-zero. DoD-4 met.

## Engagement Instructions
```
find /datasets/mmolefe/checkpoints -iname "*model_ema_step0060000*" -o -iname "*vae_v2_step0075000*"
```
Expected: both files appear with non-zero sizes. DoD-4 met.

## Recommended skill
— custom; no skill fits (project-specific checkpoint transfer and verification).
