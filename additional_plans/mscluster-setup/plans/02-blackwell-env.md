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
- [ ] Check micromamba is available; install if absent:
  ```bash
  which micromamba || {
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj bin/micromamba
    mkdir -p ~/.local/bin && mv bin/micromamba ~/.local/bin/
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
  }
  micromamba --version
  ```
- [ ] Drop the `torch==2.5.1` / `cu124` pins from `requirements.txt` (exact command):
  ```bash
  sed -i '/^torch==/d; /^torchvision==/d' requirements.txt
  grep "torch" requirements.txt   # should return nothing
  ```
- [ ] ⚠️ Create env and install torch for cu128:
  ```
  micromamba create -n jaxstack python=3.11 -y
  micromamba activate jaxstack
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  python -c "import torch;assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0]>=12, torch.cuda.get_device_name(0)"
  ```
- [ ] ⚠️ Remedy (no stable cu128 wheel available): use nightly:
  ```
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
  If the driver is too old (`nvidia-smi` must show CUDA>=12.8): `module avail cuda` then `module load cuda/12.8`
  (or whichever >=12.8 version the cluster offers); re-run the capability assert.
- [ ] Log torch version: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.

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
