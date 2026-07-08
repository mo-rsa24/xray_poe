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
- [x] ✅ Check GPU: ✓ verified — `NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB`, capability (12, 0) = sm_120.
  ```
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  ```
- [x] ✅ Run the smoke generate: ✓ verified — 4 cardiomegaly films in 1.7s → `/tmp/smoke/single/cardiomegaly/`. Required `module load pytorch/pytorch-cuda13.3-ubuntu26.04-00` (torch 2.11.0+cu130) + `export APPTAINER_BIND=/datasets` (the module is an Apptainer container that does not bind /datasets by default).
  ```
  python scripts/generate.py \
    --ckpt ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors \
    --vae-ckpt ckpts/vae_v2/vae_v2_step0075000.pt \
    --scale-factor data/latents_v2/joint_perch/scale_factor.pt \
    --disease cardiomegaly --n 4 --w 4 -o /tmp/smoke
  ```
- [x] ✅ Remedy (sm_120 kernel error / CUDA error): Not needed — no CUDA/kernel error; generation ran clean on sm_120.
- [x] ✅ Remedy (FileNotFoundError on ckpt or latent path): HIT and RESOLVED. Root cause was NOT the
  symlinks — the pytorch module runs inside an Apptainer container that binds `$HOME` but not
  `/datasets`, so every `/datasets/...` path (and the repo symlinks into it) was invisible and
  safetensors raised a misleading `FileNotFoundError`. Fix: `export APPTAINER_BIND=/datasets` before
  running the module python. Also fixed the broken `scale_factor.pt` sub-symlink (see plan 05).
- [x] ✅ Inspect output: ✓ verified — 4 PNGs, mean pixel 123.5–161.3 (all > 10, not black). DoD-6 met.

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
