# 🧱 Blackwell-Compatible Python Environment

## Description
Create the `jaxstack` micromamba env with PyTorch built for cu128 so it runs on the Blackwell GPU (sm_120).
The pinned `torch==2.5.1/cu124` in `requirements.txt` will not run on sm_120.

## Purpose
O2 blackwell-env. All downstream scripts (generate, eval, effusion_blunt_check) import torch — this gate
must pass before any compute can happen.

## Goal
`micromamba activate jaxstack` succeeds and
`python -c "import torch; assert torch.cuda.get_device_capability()[0]>=12"` passes.
Torch version >=2.7 logged.

## Tasks
- [x] ✅ Check package manager: ✓ stated (used mamba instead of micromamba — system-provided, equivalent)
- [x] ✅ Drop torch pins from requirements.txt: ✓ verified (no torch== pins present)
- [x] ✅ Create env and install torch: ✓ stated (used `module load pytorch/pytorch-cuda13.3-ubuntu26.04-00` + mamba instead of manual pip; more robust)
- [x] ✅ Remedy (no stable cu128): Not needed — system module provides cu132, superior to nightly cu128
- [x] ✅ Log torch version: ✓ verified (torch 2.12.1+cu132, cuda 13.2, capability sm_120)

## Engagement Instructions
```
micromamba run -n jaxstack python -c "
import torch
cap = torch.cuda.get_device_capability()
print('device:', torch.cuda.get_device_name(0))
print('capability:', cap)
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
assert cap[0] >= 12, f'sm_{cap[0]}{cap[1]} < sm_120'
print('PASS')
"
```
Expected: device name contains "B" series (e.g. B200 / GB200 / B100), capability (12, x), PASS printed. DoD-2 met.

## Recommended skill
— custom; no skill fits (Blackwell-specific wheel installation and capability gating).
