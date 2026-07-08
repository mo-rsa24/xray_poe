# 💨 Verify GPU and Smoke Test

## Description
Confirm the Blackwell GPU is visible and functional, then run a 4-film generate to prove the full
model stack (VAE-v2 + LDM v4-60k) loads and produces non-black images on sm_120.

## Purpose
O6 verify+smoke. This is the end-to-end gate before running the longer `effusion_blunt_check.py` in plan 07.
A black or errored image means the env (plan 02) or paths (plan 05) need fixing first.

## Goal
`nvidia-smi` shows a Blackwell GPU; the 4-film generate run completes and `/tmp/smoke/` contains
non-black PNG files.

## Tasks
- [ ] Check GPU:
  ```
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  ```
- [ ] ⚠️ Run the smoke generate:
  ```
  python scripts/generate.py \
    --ckpt ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors \
    --vae-ckpt ckpts/vae_v2/vae_v2_step0075000.pt \
    --scale-factor data/latents_v2/joint_perch/scale_factor.pt \
    --disease cardiomegaly --n 4 --w 4 -o /tmp/smoke
  ```
- [ ] ⚠️ Remedy (sm_120 kernel error / CUDA error): back to plan 02 — re-check env and torch capability;
  check `python -c "import torch; print(torch.cuda.get_device_capability())"`.
- [ ] ⚠️ Remedy (FileNotFoundError on ckpt or latent path): back to plan 05 — check symlinks with
  `readlink -f ckpts/vae_v2_ldm_v4` and `readlink -f data/latents_v2`.
- [ ] Inspect output: `ls -lh /tmp/smoke/` and open one PNG to confirm it is not all-black.

## Engagement Instructions
```
ls -lh /tmp/smoke/
python -c "
from PIL import Image
import glob, sys
files = glob.glob('/tmp/smoke/*.png')
assert files, 'no output files'
for f in files:
    img = Image.open(f).convert('L')
    mean = sum(img.getdata()) / len(img.getdata())
    print(f, 'mean pixel:', round(mean, 1))
    assert mean > 10, f'black image: {f}'
print('PASS')
"
```
Expected: 4 PNG files, mean pixel > 10 for each (not all-black). DoD-6 met.

## Recommended skill
— custom; no skill fits (project-specific GPU and generate smoke test).
