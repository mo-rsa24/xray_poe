# NIH Latent Cache Pipeline

Pre-encodes NIH ChestX-ray14 PNGs through a frozen VAE so the LDM training
loop never has to run the encoder on the GPU — cutting VRAM usage and
eliminating per-step encoding overhead.

---

## Why a latent cache?

Training the LDM online (VAE encode every batch) wastes GPU memory and compute
on a model that is already fully trained.  Precomputing latents once lets the
LDM training loop load `(z, label)` pairs directly from disk, keeping the full
GPU budget for the UNet.

---

## Files Created

### `src/data/__init__.py`
Empty package marker so Python treats `src/data/` as a module.

---

### `src/data/latent_dataset.py` — `LatentDataset`

Loads pre-encoded latents from `{cache_dir}/{split}/*.pt`.

**What it does:**
- Scans the split directory for `.pt` files on construction.
- Caches all labels in memory at init time so `make_sampler()` is instant.
- `__getitem__` reads a single `.pt` dict `{"z": Tensor, "label": int}` and
  returns `(z, label)` as `((4,128,128) float32, int64 scalar)`.
- `make_sampler(effusion_weight)` returns a `WeightedRandomSampler` that
  up-weights effusion samples to compensate for class imbalance.

**Why separate from `RealCXRDataset`:**  
The training loop selects one path at startup (`--latent-cache` vs
`--vae-ckpt`) and both datasets expose the same `(x, labels)` tuple contract,
so the rest of the loop is identical in both modes.

---

### `src/data/real_cxr_dataset.py` — `RealCXRDataset`

Online dataset over raw NIH PNGs — used when no latent cache exists yet.

**What it does:**
- Parses `Data_Entry_2017.csv` directly (no dependency on `vae/data.py`'s
  path-only `make_splits`) to produce `(path, label)` pairs.
- Applies the same transform as the VAE training set:
  `PIL.LANCZOS resize → float32 / 127.5 − 1.0 → unsqueeze(0)`
  yielding `(1, 512, 512) float32` in `[-1, 1]`.
- Gracefully skips corrupt/truncated PNGs by cycling forward.
- `make_sampler(effusion_weight)` mirrors `LatentDataset` exactly.

**Label mapping:**

| Label | Group |
|-------|-------|
| 0 | No Finding (normal) |
| 1 | Cardiomegaly only |
| 2 | Effusion only |
| — | Both / Other (excluded) |

`both` and `other` groups are excluded to keep the 3-class label space clean
for CFG conditioning.

---

### `scripts/precompute_latents.py`

Batch-encodes the NIH dataset through a frozen VAE and writes the cache.

**What it does:**
1. Loads the frozen VAE from a checkpoint.
2. Parses `Data_Entry_2017.csv` with the same frontal-only, stratified-split
   logic as `RealCXRDataset`.
3. Encodes batches of images: `vae.encode(x)` → `(B, 4, 128, 128) float32`.
4. Writes one `.pt` dict per image: `{"z": (4,128,128) float32, "label": int}`.
5. Prints a class-count summary and the exact `compute_scale_factor.py`
   command to run next.

**Why per-file `.pt` dicts (not HDF5 or a single tensor file):**
- `compute_scale_factor.py` already handles this dict format (`"z"` key).
- DataLoader workers can read individual files without seeking in a shared
  file, avoiding contention.
- Individual files are trivial to inspect and delete selectively.

---

### Bug fixed — `scripts/train_ldm.py` line 364

```python
# Before (broken): vae.encode() returns a tensor, not a distribution object
return vae_model.encode(x).latent_dist.sample()

# After (correct):
return vae_model.encode(x)
```

`VAE.encode()` already returns a sampled latent `z ~ N(μ, σ²)` — calling
`.latent_dist.sample()` on the returned tensor would raise `AttributeError`
at runtime.

---

## Cache Layout

```
data/latents/
├── scale_factor.pt          # scalar tensor — computed by compute_scale_factor.py
├── train/
│   ├── 00000001_000.pt      # {"z": (4,128,128) float32, "label": 0}
│   ├── 00000002_000.pt      # {"z": (4,128,128) float32, "label": 1}
│   └── ...
└── val/
    ├── 00000003_000.pt
    └── ...
```

---

## Run Order

### 1. Precompute latents

Encode all NIH frontal PNGs through the trained VAE.  Adjust `--batch-size`
to fit available VRAM (16 works on 24 GB, reduce to 8 if needed).

```bash
python scripts/precompute_latents.py \
    --csv        data/nih/Data_Entry_2017.csv \
    --image-dir  data/nih/images \
    --vae-ckpt   ckpts/vae/best.pt \
    --out-dir    data/latents \
    --batch-size 16 \
    --device     cuda
```

### 2. Compute the scale factor

Samples 512 random training latents and saves `1 / std(latents)` as a scalar
tensor.  This normalises the diffusion input to unit variance.

```bash
python scripts/compute_scale_factor.py \
    --latent-dir data/latents/train \
    --out        data/latents/scale_factor.pt
```

Expected output:
```
Latent std  : ~0.18
scale_factor: ~5.5
```
The script asserts `0.5 < scale_factor < 5.0` — if it fails, the latents are
not VAE samples (check the VAE checkpoint).

### 3. Train the LDM

Pass `--latent-cache` to skip online VAE encoding entirely.

```bash
python scripts/train_ldm.py \
    --config       configs/ldm_full.yaml \
    --latent-cache data/latents
```

The training loop will:
- Load `data/latents/scale_factor.pt` automatically.
- Use `LatentDataset` for both train and val splits.
- Apply `make_sampler(effusion_weight=2.0)` for class-balanced training.

---

## Tensor Contracts (quick reference)

| Stage | Shape | Dtype | Range |
|-------|-------|-------|-------|
| NIH PNG (raw) | `(1, 512, 512)` | uint8 | `[0, 255]` |
| After normalisation | `(1, 512, 512)` | float32 | `[-1, 1]` |
| VAE encode output | `(4, 128, 128)` | float32 | unbounded |
| After × scale_factor | `(4, 128, 128)` | float32 | ~`[-2, 2]` |
| UNet input (noised) | `(4, 128, 128)` | float32 | ~`[-3, 3]` |
| Labels | `()` scalar | int64 | `{0, 1, 2}` |
