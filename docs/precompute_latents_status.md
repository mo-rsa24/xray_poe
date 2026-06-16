# Latent Cache Precompute — Status & Next Steps

_Last updated: 2026-06-16_

## TL;DR
The previously-computed VAE latents were **lost** (written to the pod's ephemeral
disk, not the persistent volume). We are **regenerating them on-pod** rather than
re-uploading the 18 GB copy from the local machine, because regeneration needs
**no internet transfer** and is faster. A new, parallelized precompute script is
ready and verified; the **val split is done**, the **train split still needs to be
run**.

---

## What we found

### 1. The old latents are gone
- `/workspace/Paper3/data/latents/` was **empty** (only an empty placeholder dir,
  `Links: 2` → no `train/`/`val/` subdirs ever written there).
- Filesystem-wide search found **only the 5 VAE checkpoints** (`ckpts/vae_step00*.pt`);
  no latent `.pt` files anywhere.
- Root cause: `/workspace` is a **persistent RunPod network volume** (MooseFS,
  `mfs#us-md-1.runpod.net`), but `/` is an **ephemeral 20 GB overlay** disk. The
  earlier run almost certainly wrote its `--out-dir` to a path on the ephemeral
  disk, which was wiped when that pod closed. (`/root/.bash_history` was also
  wiped — consistent with ephemeral-disk loss.)

### 2. The dataset is ~74k images, not ~7k
- `_collect_samples` yields **70,607 train + 3,714 val = 74,321** images
  (frontal PA/AP, 3 clean classes: normal=0, cardio=1, effusion=2).
- Each latent is `(4,128,128)` float32 = **256 KB** → 74,321 × 256 KB ≈ **19 GB**
  (matches the ~18 GB local copy).

### 3. Why the original script was slow (~6–9 hr)
Measured throughput of the original single-threaded `precompute_latents.py`:
| Stage | Original | Bottleneck |
|---|---|---|
| CSV scan (stat 112k rows over MooseFS) | ~40 s (one-time) | network FS latency |
| PNG decode + LANCZOS resize | **3.4 img/s** | single-threaded |
| `torch.save` per file | **11 files/s** | serial small-file writes to MooseFS |
| GPU encode (A100, fp32, B=16) | 19.4 img/s | — |

### 4. bf16 is OFF the table
User reported the VAE produces **black reconstructions in bf16** (decoder issue).
Encode-path risk is lower but non-zero, and these latents become LDM training
targets — so we **encode in fp32**. No autocast.

---

## What we did

### New script: `scripts/precompute_latents_fast.py`
Functionally identical output to the original, but pipelined:
- **Parallel decode** via `DataLoader(num_workers=16)` → ~400 img/s (was 3.4).
- **Parallel writes** via `ThreadPoolExecutor(16)` → ~490 files/s (was 11).
- **fp32 encode**, batch 16 (~38 GB GPU; B=64 OOMs on the attention layer).
- **Resumable**: skips images whose output `.pt` already exists, so a disconnect
  only costs the in-flight batch. `--overwrite` to force.
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid fragmentation OOM.

### Verified on the val split (DONE ✅)
- All **3,714 val latents** written to `data/latents/val/`.
- Validation: shape `(4,128,128)`, float32, all finite, per-sample std ≈ 1.0
  (→ `scale_factor` will be ≈ 1.0, inside the expected 0.5–5.0 range). Labels {0,2} seen.

### ⚠️ Observed infrastructure stalls
During the val run, steady-state was the expected **~18 img/s**, but the MooseFS
volume **froze twice for ~9.5 min** (read-side stalls — GPU idle waiting for the
DataLoader). Effective val rate was dragged to **2.8 img/s** (22 min for 3.7k).
- This is a **shared network-volume latency issue**, not our code; it would hit
  any reader of the 70k PNGs.
- If these stalls recur at the same frequency on the train split, train could take
  far longer than the ~65 min steady-state estimate. To be watched.

---

## What still needs to be done

### 1. Run the train split (NOT yet started)
Launch in background (resumable, logged). **NOTE: user interrupted before this ran —
needs explicit go-ahead.**
```bash
cd /workspace/Paper3
nohup python scripts/precompute_latents_fast.py \
  --csv data/nih/Data_Entry_2017.csv \
  --image-dir data/nih/images \
  --vae-ckpt ckpts/vae_step0025000.pt \
  --out-dir /workspace/Paper3/data/latents \
  --batch-size 16 --num-workers 16 --write-workers 16 \
  --split train \
  > runs/precompute_train.log 2>&1 &
```
- Expected: **~65 min** if MooseFS behaves; longer if the ~9.5-min stalls recur.
- Monitor: `tail -f runs/precompute_train.log` and watch the img/s in the final
  per-split line. Verify count when done: `ls data/latents/train/*.pt | wc -l`
  should be **70,607**.

### 2. Compute the scale factor (after train completes)
```bash
python scripts/compute_scale_factor.py \
  --latent-dir data/latents/train \
  --out        data/latents/scale_factor.pt
```

### 3. Optional sanity check (recommended, per bf16 concern)
Decode a couple of cached fp32 latents through the **fp32** decoder and eyeball
them — confirms the encoder path is healthy and the black pixels were purely a
bf16-decode issue.

### 4. Then: LDM training
Point `train_ldm.py --latent-cache data/latents` (consumes `train/`, `val/`,
`scale_factor.pt`). VAE no longer needs to be on the GPU during LDM training.

---

## Lessons / guardrails
- **Always write caches to an absolute path under `/workspace`** (persistent),
  never a relative/ephemeral path. Run long jobs under `nohup` so a disconnect
  doesn't kill them.
- The precompute script is now resumable — safe to restart after any interruption.
- Regenerating on-pod (no transfer) beats re-uploading 18 GB unless local upstream
  is very fast; the raw NIH images already live on the persistent volume.
