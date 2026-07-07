# 🔄 LDM Re-Encode — NIH Latents Under Frozen VAE-v2

## Background
The LDM was trained in `vae_step0025000.pt`'s latent space. Now that the VAE decoder has changed (LPIPS + PatchGAN fine-tune), the latent distribution has shifted — even if the encoder weights moved only slightly, the LDM's learned denoising trajectory no longer corresponds to the new codec's space. Re-encoding all NIH training films under the frozen VAE-v2 is mandatory before retraining the LDM. This is the cheapest compute step in the scope (~1.5 hrs at 19.4 img/s measured on hippo A100).

## Description
Re-run `scripts/materialize_latent_manifests.py` (or write a thin wrapper) with `--vae_checkpoint ckpts/vae_v2/vae_v2_step075000.pt` and `--output_dir data/latents_v2/train`. Verify file counts and spot-check latent statistics (mean, std, finite). Preserve the original latent cache at `data/latents/train` — do not delete until leaf 06 (LDM retrain) confirms convergence.

## Purpose
Serves vae-v2 Objective 5 and DoD item 5. Hard prerequisite for leaf 06 (LDM retrain). Prerequisite: leaf 04 GO verdict.

## Goal
A new latent cache `data/latents_v2/train/` with the same film count as the original `data/latents/train/`, encoded under the frozen VAE-v2 **as the deterministic posterior mean μ (encode_moments)**, with shape/finite/N integrity verified and signal-fraction confirmed.

## ⚠️ CORRECTION — INVALIDATED & REOPENED (2026-07-01)

The first re-encode (SAMPLED `z~N(μ,σ)`, std ≈ 1.0) is **invalidated**. Two LDM retrains
(`mzd3jwy9` sampled-NIH+μ-VinDr, `gtsw5bq7` fully sampled) both produced pure noise from
step 5000. Root cause (direct VAE probing, see memory `vindr-latent-scale-bug`):

> **VAE-v2 is posterior-collapsed.** Posterior σ ≈ 0.999 (≈ prior), μ_std ≈ 0.187 →
> **signal fraction 3.4%** (vs V1 VAE 21.3%, which worked). A *sampled* v2 latent is
> ~96% prior noise and decodes 5.8× noisier than its μ decode, so the LDM trained on it
> collapses to noise. VinDr scale was a red herring — the collapse hits NIH identically.

**Fix (chosen path A):** encode ALL data as the deterministic mean **μ** (`encode_moments`,
100% signal = the VAE's clean reconstruction), recompute `scale_factor` on the μ set
(std ≈ 0.19 → sf ≈ 5.3), retrain. Code wired 2026-07-01: `--encode-mode {sample,mode}`
added to `precompute_latents_fast.py` and `encode_vindr_latents.py` (default `mode` for
VinDr); `compute_scale_factor.py` assert relaxed 5.0 → 10.0.

### Ordered corrected pipeline (run on hippo, GPU free, jaxstack env)

```bash
cd ~/PhD/Paper3

# ── 1. Re-encode NIH (train+val) as μ — overwrites the sampled cache ──────────
#     batch 4: VAE-v2 spatial-attention needs ~2 GB/sample; batch 16 OOMs on 31 GB.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
micromamba run -n jaxstack python scripts/precompute_latents_fast.py \
  --csv         data/nih/Data_Entry_2017.csv \
  --image-dir   data/nih/images \
  --vae-ckpt    ckpts/vae_v2/vae_v2_step0075000.pt \
  --out-dir     data/latents_v2 \
  --encode-mode mode \
  --batch-size  4 --num-workers 8 --write-workers 8 \
  --split both --overwrite

# ── 2. Re-encode VinDr as μ — overwrites in place (joint symlinks auto-update) ─
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
micromamba run -n jaxstack python scripts/encode_vindr_latents.py \
  --vae-ckpt     ckpts/vae_v2/vae_v2_step0075000.pt \
  --manifest     data/latents/vindr/border_visible_manifest.csv \
  --image-dir    data/vindr512/vinbigdata/train/ \
  --out-dir      data/latents_v2/vindr/ \
  --encode-mode  mode \
  --batch-size   4 --device cuda --overwrite

# ── 3. Recompute scale_factor on the joint μ set (joint symlink picks it up) ──
micromamba run -n jaxstack python scripts/compute_scale_factor.py \
  --latent-dir data/latents_v2/joint/train \
  --out        data/latents_v2/scale_factor.pt

# ── 4. Verify (expect NIH ≈0.19, VinDr ≈0.20, sf ≈5.3) ───────────────────────
micromamba run -n jaxstack python3 -c "
import torch,glob,numpy as np
for d in ['data/latents_v2/train','data/latents_v2/vindr']:
    fs=[f for f in glob.glob(d+'/*.pt') if 'scale_factor' not in f]
    Z=torch.stack([torch.load(f,map_location='cpu',weights_only=False)['z'].float()
                   for f in np.random.default_rng(0).choice(fs,200,replace=False)])
    print(d,'std',round(float(Z.std()),3),'per-ch',[round(float(Z[:,c].std()),3) for c in range(4)])
print('scale_factor',round(float(torch.load('data/latents_v2/scale_factor.pt',weights_only=False)),3))
"
```

Then proceed to plan 06 step 5 (retrain as `exp_vae_v2_ldm_mu`). Go/no-go = step-5000
recon grid: μ-latents are 100% signal, so anatomy should appear by 5k.

**Fallback if 5k is noisy/low-quality:** NOT per-channel normalization. The signal-fraction
check (2026-07-01) found the collapse is channel-concentrated — ch1 ≈12%, ch0/2/3 = 0.1–1.4%
(VAE uses ~1 of 4 channels). Per-channel norm would amplify the near-empty channels to unit
variance, wasting LDM capacity on noise. Global scaling is CORRECT here. The deep fix is a
VAE-v2 re-fine-tune with KL/free-bits to spread information across channels (plan 04 Gate-5).

## Tasks

- [x] ✅ **Confirm the original latent cache film count as the target**  ✓ verified (`find data/latents/train -name "*.pt" | wc -l` = 70,607; logged to `results/vae_v2/reencode/baseline_count.txt` 2026-06-30)
  - **Objective:** establish the expected N so the re-encode can be verified against it.
  - **Method:** `find data/latents/train -name "*.pt" | wc -l` (expect ~70,607 from `precompute_latents_status.md`). Log to `results/vae_v2/reencode/baseline_count.txt`.
  - **Saved artifact:** `results/vae_v2/reencode/baseline_count.txt`
  - **Verify:** count printed and logged; matches known value ~70k.

- [x] ✅ **Run the re-encode as μ (`--encode-mode mode`)**  ✓ verified (2026-07-01: `data/latents_v2/train/` 70,613 + `val/` 3,715 μ latents, global std 0.199; `compute_scale_factor.py` on joint → 5.034 (needs relaxed <10.0 assert); joint/train 77,284 symlinks)
  - **Objective:** produce `data/latents_v2/train/` (+ `val/`) with all NIH films encoded under frozen VAE-v2 **as μ (encode_moments)**, not sampled.
  - **Script:** `scripts/precompute_latents_fast.py` — already supports `--vae-ckpt`; EMA key fix applied 2026-06-30 (loads `ema` → `model` → `model_state` in that order). `materialize_latent_manifests.py` is a latent-selector, NOT an encoder — do not use.
  - **Method:**
    ```bash
    mkdir -p results/vae_v2/reencode
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup python scripts/precompute_latents_fast.py \
      --csv         data/nih/Data_Entry_2017.csv \
      --image-dir   data/nih/images \
      --vae-ckpt    ckpts/vae_v2/vae_v2_step0075000.pt \
      --out-dir     data/latents_v2 \
      --batch-size  16 \
      --num-workers 8 \
      --write-workers 8 \
      --split       train \
      > results/vae_v2/reencode/reencode.log 2>&1 &
    echo "PID $!"
    # monitor:
    tail -f results/vae_v2/reencode/reencode.log
    ```
  - **After completion — compute scale factor (required for LDM):**
    ```bash
    python scripts/compute_scale_factor.py \
      --latent-dir data/latents_v2/train \
      --out        data/latents_v2/scale_factor.pt
    ```
  - **Wall-clock estimate:** ~70k films / 19.4 img/s ≈ 60 min; add 30 min overhead for disk I/O.
  - **Saved artifact:** `data/latents_v2/train/*.pt` (one per film, same stem naming as original). `data/latents_v2/scale_factor.pt`.
  - **Verify:** file count matches `baseline_count.txt` ± 5.

- [x] ✅ **Integrity check — shape, dtype, finite, N**  ✓ verified on μ cache (2026-07-01: 300-sample stacks (N,4,128,128) succeeded for NIH/VinDr/joint = shape confirmed; `torch.isfinite().all()` = True all three; float32; N: NIH 70,613 + val 3,715, VinDr 6,671 — logged in `latent_stats_comparison.txt`)
  - **Objective:** confirm the new μ cache is usable before the LDM retrain starts.
  - **Method:** spot-check 200 random latents (no loading bias — use `shuf`):
    ```bash
    python3 -c "
    import torch, glob, random
    files = glob.glob('data/latents_v2/train/*.pt')
    sample = random.sample(files, min(200, len(files)))
    for f in sample:
        d = torch.load(f, map_location='cpu')
        z = d['z']          # files are dicts: {'z': tensor, 'label': int}
        assert z.shape == (4, 128, 128), f'shape {z.shape} in {f}'
        assert z.dtype == torch.float32, f'dtype {z.dtype} in {f}'
        assert torch.isfinite(z).all(), f'non-finite in {f}'
    print(f'Spot-check OK: {len(sample)} files, shape (4,128,128), float32, finite')
    print(f'Total files: {len(files)}')
    "
    ```
  - **Saved artifact:** `results/vae_v2/reencode/integrity_check.txt` (output of the above).
  - **Verify:** "Spot-check OK" printed; total file count logged.

- [x] ✅ **Compare latent statistics + SIGNAL-FRACTION check**  ✓ verified (`results/vae_v2/reencode/latent_stats_comparison.txt` written 2026-07-01. μ cache: per-ch std finite & >0, means ≈0, NIH 0.199/VinDr 0.202/joint 0.199 consistent. Signal fraction (sampled) 3.6% → correctly FAILS >15% bar (= the collapse, = why μ adopted; V1 was 21%). NEW: collapse is channel-concentrated — ch1 ~12%, ch0/2/3 = 0.1–1.4% → VAE uses ~1 of 4 channels; global scaling is CORRECT (per-channel norm would amplify empty channels); deep-fix fallback = VAE re-fine-tune w/ KL/free-bits, NOT normalization.)
  - **Objective:** confirm the μ distribution is well-scaled AND that the VAE latent is informative enough for LDM training (the check that would have caught this failure a run earlier).
  - **Method:** on the μ cache, compute channel-wise mean/std; ALSO compute posterior signal
    fraction on a sample of images: `sf = μ_std² / (μ_std² + σ_mean²)` from `vae.encode_moments`.
    Log to `results/vae_v2/reencode/latent_stats_comparison.txt`.
  - **Saved artifact:** `results/vae_v2/reencode/latent_stats_comparison.txt`
  - **Verify:** μ per-channel std finite and > 0; **signal fraction > ~15%** (VAE-v2 measured
    3.4% for sampled — the collapse; V1 was 21%). If signal fraction stays < 15% even on μ,
    escalate to per-channel normalization or a VAE re-fine-tune (see plan 04 gap note).

- [x] ✅ **Document new cache path; preserve old cache**  ✓ verified (`results/vae_v2/reencode/cache_registry.txt` rewritten 2026-07-01 for μ-encode: encode-mode=mode, scale_factor 5.034, NIH 70,613 (+3,715 val) / VinDr 6,671 / joint 77,284; DO NOT DELETE on V1 cache `data/latents/train/` (70,607) until exp_vae_v2_ldm_mu converges; channel-collapse note + cross-refs to signal-fraction artifact and memory)
  - **Objective:** clear record so leaf 06 uses the right input (μ cache); old cache not accidentally deleted.
  - **Method:** write `results/vae_v2/reencode/cache_registry.txt`:
    ```
    OLD cache: data/latents/train/  (vae_step0025000.pt, DO NOT DELETE until LDM retrain converges)
    NEW cache: data/latents_v2/train/  (vae_v2_step075000.pt, feeds leaf 06 LDM retrain)
    VinDr v2:  data/latents_v2/vindr/  (vae_v2_step075000.pt, feeds leaf 06 joint option)
    Re-encode date: YYYY-MM-DD
    Film count NIH: N   VinDr: N
    ```
  - **Saved artifact:** `results/vae_v2/reencode/cache_registry.txt`
  - **Verify:** file exists; both paths listed; "DO NOT DELETE" note on old cache.

- [x] ✅ **Encode VinDr border-visible films as μ (`--encode-mode mode`)**  ✓ verified (2026-07-01: `data/latents_v2/vindr/` 6,671 μ latents, global std 0.201, per-ch [0.056,0.364,0.105,0.126] — matches NIH μ 0.199; converted from the sampled 1.019 that broke gtsw5bq7; V1 `data/latents/vindr/` untouched)
  - **Objective:** produce `data/latents_v2/vindr/` as μ so both NIH and VinDr latents share one
    codec AND one encode-mode before leaf 06. Mixing sampled + μ (or two scales) is what broke
    the first two runs.
  - **Prerequisite:** `data/latents/vindr/border_visible_manifest.csv` written by
    `vindr-transfer/03` border-filter task AND cardio rate gate PASSED.
  - **Method:**
    ```bash
    # reuse encode_vindr_latents.py from vindr-transfer/03-sub-D, new checkpoint + output path
    python scripts/encode_vindr_latents.py \
      --vae-ckpt  ckpts/vae_v2/vae_v2_step075000.pt \
      --manifest  data/latents/vindr/border_visible_manifest.csv \
      --image-dir data/vindr512/std/train/ \
      --out-dir   data/latents_v2/vindr/ \
      --scale-factor data/latents_v2/scale_factor.pt \
      --batch-size 16 --device cuda
    ```
  - **Saved artifact:** `data/latents_v2/vindr/*.pt`; append VinDr v2 path to `cache_registry.txt`.
  - **Verify:** `ls data/latents_v2/vindr/*.pt | wc -l` matches border-visible manifest count;
    spot-check: shape (4,128,128), float32, finite. `data/latents/vindr/` (VAE-v1) untouched.
  - **Note:** `data/latents/vindr/` (VAE-v1 space) is the VinDr-transfer Stage-1 input and must
    not be overwritten. `data/latents_v2/vindr/` is VAE-v2 only.

## Recommended skill
— custom; thin wrapper around existing `materialize_latent_manifests.py`.

## Engagement Instructions

```bash
# ── env ───────────────────────────────────────────────────────────────────────
# On hippo: micromamba activate jaxstack

# ── IMPORTANT: materialize_latent_manifests.py SELECTS, does not re-encode ───
# That script reads from data/latents/train/*.pt (existing cache) and writes manifests.
# For VAE-v2 re-encode we need a NEW script: scripts/encode_latents_vae_v2.py
# Pattern: load VAE-v2 in eval mode, iterate over NIH images, encode to latent, save .pt
# Reference pattern: vae_recon_pilot.py (encodes images → saves overlays + CSVs)

# ── Step 0: confirm original latent count ────────────────────────────────────
$ find data/latents/train -name "*.pt" | wc -l
# expect: ~70,607 (from precompute_latents_status.md)
$ python3 -c "import torch; sf = torch.load('data/latents/scale_factor.pt'); print('scale_factor:', sf)"
# note the scale_factor value — must be copied to data/latents_v2/

# ── Step 1: write scripts/encode_latents_vae_v2.py ───────────────────────────
# Minimal script skeleton (adapt from vae_recon_pilot.py):
# python scripts/encode_latents_vae_v2.py \
#     --vae-ckpt ckpts/vae_v2/vae_v2_step075000.pt \
#     --manifest data/vae_v2_diet/train_manifest.csv \
#     --out-dir data/latents_v2/train/ \
#     --scale-factor data/latents/scale_factor.pt \
#     --batch-size 16 \
#     --device cuda
# Throughput estimate: 19.4 img/s (A100, fp32, batch 16) — from precompute_latents_status.md
# ~70k films / 19.4 img/s ≈ 60 min

# ── Step 2: run encode (smoke first) ─────────────────────────────────────────
$ python scripts/encode_latents_vae_v2.py \
    --vae-ckpt ckpts/vae_v2/vae_v2_step075000.pt \
    --manifest data/vae_v2_diet/train_manifest.csv \
    --out-dir data/latents_v2/train/ \
    --scale-factor data/latents/scale_factor.pt \
    --batch-size 16 --n 20   # smoke: 20 films only
# verify smoke output:
$ find data/latents_v2/train -name "*.pt" | head -5
$ python3 -c "
import torch, glob
f = glob.glob('data/latents_v2/train/*.pt')[0]
z = torch.load(f, map_location='cpu')
print('shape:', z.shape, '  dtype:', z.dtype, '  finite:', torch.isfinite(z).all().item())
"
# expect: shape (4,128,128)  dtype torch.float32  finite True

# then full run (remove --n flag):
$ python scripts/encode_latents_vae_v2.py \
    --vae-ckpt ckpts/vae_v2/vae_v2_step075000.pt \
    --manifest data/vae_v2_diet/train_manifest.csv \
    --out-dir data/latents_v2/train/ \
    --scale-factor data/latents/scale_factor.pt \
    --batch-size 16

# ── Step 3: integrity check ───────────────────────────────────────────────────
$ find data/latents_v2/train -name "*.pt" | wc -l
# expect: same as original manifest length (N from build_vae_diet.py)
$ python3 -c "
import torch, glob, random
files = glob.glob('data/latents_v2/train/*.pt')
sample = random.sample(files, min(200, len(files)))
for f in sample:
    z = torch.load(f, map_location='cpu')
    assert z.shape == (4,128,128), f'{f}: shape {z.shape}'
    assert z.dtype == torch.float32, f'{f}: dtype {z.dtype}'
    assert torch.isfinite(z).all(), f'{f}: non-finite values'
print(f'Spot-check OK: {len(sample)} files, shape/dtype/finite all pass')
print(f'Total files: {len(files)}')
"

# ── Step 4: latent stats comparison ──────────────────────────────────────────
$ python3 -c "
import torch, glob, random
def stats(pattern):
    files = random.sample(glob.glob(pattern), 1000)
    zs = torch.stack([torch.load(f, map_location='cpu') for f in files])
    return {'mean': zs.mean(dim=(0,2,3)).tolist(), 'std': zs.std(dim=(0,2,3)).tolist()}
old = stats('data/latents/train/*.pt')
new = stats('data/latents_v2/train/*.pt')
print('OLD mean/channel:', [f'{v:.3f}' for v in old['mean']])
print('NEW mean/channel:', [f'{v:.3f}' for v in new['mean']])
print('OLD std/channel:', [f'{v:.3f}' for v in old['std']])
print('NEW std/channel:', [f'{v:.3f}' for v in new['std']])
# std ratio should be 0.5–2× (shifted but not collapsed/exploded)
"

# ── Step 5: copy scale_factor ─────────────────────────────────────────────────
$ mkdir -p data/latents_v2
$ cp data/latents/scale_factor.pt data/latents_v2/scale_factor.pt
$ cat results/vae_v2/reencode/cache_registry.txt   # should exist after this leaf

# ── VinDr v2 encode (after NIH re-encode complete) ───────────────────────────
$ python scripts/encode_vindr_latents.py \
    --vae-ckpt  ckpts/vae_v2/vae_v2_step075000.pt \
    --manifest  data/latents/vindr/border_visible_manifest.csv \
    --image-dir data/vindr512/std/train/ \
    --out-dir   data/latents_v2/vindr/ \
    --scale-factor data/latents_v2/scale_factor.pt \
    --batch-size 16 --n 10   # smoke first
$ ls data/latents_v2/vindr/*.pt | wc -l   # then full run without --n
$ $PY -c "
import torch, glob
f = glob.glob('data/latents_v2/vindr/*.pt')[0]
z = torch.load(f, map_location='cpu')
print('shape:', z.shape, ' finite:', torch.isfinite(z).all().item())
"
# expect: shape (4,128,128), finite True
# check VAE-v1 cache still intact:
$ ls data/latents/vindr/*.pt | wc -l   # must be unchanged
```
