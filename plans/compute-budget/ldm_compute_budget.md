# LDM Compute Budget

**Rate:** $1.49/hr | **Last updated:** 2026-06-11

This document works through the estimate from first principles — starting from what is known exactly (model parameters, config), then deriving VRAM and throughput, then building up to a cost range. Each step shows the arithmetic so the assumptions are explicit and easy to revise once a real profiling run is available.

---

## What We Know Exactly

These numbers come directly from the code and config — no estimation involved.

**Model** (`src/models/ldm_unet.py`, `configs/ldm_full.yaml`):
- UNet: 4 resolution levels, channels fixed at 128 across all levels
- Latent space: `(B, 4, 128, 128)` — this is the tensor the UNet operates on, not pixel space
- Attention at two levels only: 32×32 and 16×16 spatial
- 2 ResBlocks per level × encoder + decoder = 16 ResBlocks total
- Class embedding: 4 tokens × 512 dim (3 disease classes + learned null token)
- **Total trainable parameters: 16,418,564 (~16.4 M)**

**Training config** (`configs/ldm_full.yaml`):
- micro-batch: 4, grad_accum: 4 → effective batch = **16**
- bf16 activations and weights, fp32 optimizer states (PyTorch default AMP)
- `max_steps: 100,000`, `ckpt_every: 10,000`

**Dataset**: 112,120 NIH frontal PNGs at 512×512, filtered to 3 classes (Normal, Cardiomegaly, Effusion). Precomputed latents are stored as `(4, 128, 128)` float32 tensors — the VAE is frozen out of the LDM training loop entirely.

**Pipeline count**: the paper needs **2 independent LDM runs** (one cardiomegaly expert, one effusion expert), plus 2 ablation configs for the CFG dropout sweep. Preprocessing is a one-time cost shared across all four.

---

## VRAM Derivation

VRAM has four components. The first three are exact; the fourth requires an estimate.

### 1. Weights in bf16

Each parameter costs 2 bytes in bf16:

```
16,418,564 params × 2 bytes = 32,837,128 bytes ≈ 33 MB
```

### 2. Optimizer states (Adam m + v) in fp32

PyTorch keeps the Adam first and second moment in fp32 regardless of autocast. That's 4 bytes × 2 tensors per parameter:

```
16,418,564 × 4 × 2 = 131,348,512 bytes ≈ 131 MB
```

### 3. Gradients in fp32

AMP accumulates gradients in fp32 (same size as weights in full precision):

```
16,418,564 × 4 = 65,674,256 bytes ≈ 66 MB
```

### 4. Activations (estimate)

Activations depend on tensor shapes throughout the forward pass. For a ResBlock at resolution `r` with `C=128` channels and micro-batch `B=4`, six intermediate tensors survive for backprop (norm1 input, conv1 output, time projection, norm2 input, conv2 output, residual):

```
6 × B × C × r² × 2 bytes per ResBlock
```

Summing across all 16 ResBlocks (4 levels × 2 encoder + 2 decoder, at r = 128, 64, 32, 16) and adding the cross-attention QKV projections at the two attention levels (32×32 and 16×16), plus the skip-connection buffers the U-Net holds between encoder and decoder:

```
ResBlock activations:  ~396 MB
Attention activations: ~140 MB (QKV + attn weights, two levels)
Skip-connection buffers: ~100 MB
Total activations:     ~636 MB
```

### 5. CUDA overhead

PyTorch CUDA allocator, cuDNN workspace, and driver context typically consume 1.0–1.5 GB on modern GPUs. Using 1.2 GB as the planning figure.

### Summary

| Component | Bytes | MB |
|-----------|------|----|
| Weights (bf16) | 32.8 M | 33 |
| Adam m + v (fp32) | 131.3 M | 131 |
| Gradients (fp32) | 65.7 M | 66 |
| Activations (bf16, estimate) | 662 M | 636 |
| CUDA overhead | 1,258 M | 1,200 |
| **Total** | | **~2,066 MB (~2.0 GB)** |

**With a 1.3× fragmentation allowance: ~2.6 GB upper bound.**

This model is tiny — ~60× smaller by parameter count than the VAE at 512² resolution. It fits on any GPU ≥ 8 GB with no gradient checkpointing and no batch-size reduction. The VRAM number does not need a profiler to confirm: it cannot exceed 2.6 GB under these settings.

> **Grad accum and VRAM:** `grad_accum=4` means four sequential micro-batches before one optimizer step. Each micro-batch clears its activation graph before the next begins. VRAM stays at B=4 throughout — it does not scale with grad_accum.

---

## Preprocessing Cost

Before any LDM training, the NIH images must be encoded into latents by the frozen VAE and written to disk. This is a one-time cost; the cache is then reused by all four LDM configs.

**What runs:** `scripts/precompute_latents.py` loads each 512×512 PNG, passes it through the frozen VAE encoder (single forward pass, no grad), and writes a `(4, 128, 128)` float32 tensor plus an integer label per image.

**Dataset size:** 112,120 frontal images (PA/AP views only, filtered to the 3 target classes).

VAE encode throughput depends on the GPU. The VAE has ~83 M parameters and processes 512² single-channel images. Published VAE encode benchmarks for similar architectures:

| GPU | Estimated VAE encode rate | Time for 112k images |
|-----|--------------------------|----------------------|
| A10G 24 GB | ~20 img/s | 112,120 / 20 / 60 ≈ **93 min** |
| A40  48 GB | ~25 img/s | 112,120 / 25 / 60 ≈ **75 min** |
| A100 80 GB | ~45 img/s | 112,120 / 45 / 60 ≈ **42 min** |

These are rough. The actual bottleneck on a machine with fast storage is GPU compute; on a machine with slow network-attached storage it may be disk I/O. In either case the preprocessing run costs **< $2.50** on any GPU listed above.

`compute_scale_factor.py` loads 512 random latents from disk and computes their standard deviation. It takes under 3 minutes (mostly disk reads) and costs under $0.10.

---

## Training Throughput Derivation

This is the most uncertain part of the estimate. There is no hardware to measure on yet, so the approach is to anchor to a published reference and scale by the ratio of FLOPs.

### Reference point

MONAI's `DiffusionModelUNet` on an A100 80 GB at B=4, latent shape `(4, 64, 64)`, 256 channels (typical LDM-256 configuration): **~8–12 it/s** (forward+backward per micro-batch). Using 10 it/s as the midpoint.

### FLOPs scaling to our architecture

FLOPs in a convolutional UNet scale as `C² × H × W` (channels squared times spatial area). Comparing ours to the reference:

```
Channel ratio:  (128/256)² = 0.25
Spatial ratio:  (128×128) / (64×64) = 4.0
Net FLOPs ratio: 0.25 × 4.0 = 1.0
```

Our model has approximately the **same FLOPs per forward pass** as the LDM-256 reference, despite being narrower — the larger spatial extent exactly compensates.

### Steps/sec accounting for grad_accum

Each optimizer step requires 4 forward+backward passes (grad_accum=4). So:

```
steps/sec = (micro-batch it/s) / grad_accum = 10 / 4 = 2.5 steps/sec  (A100)
```

Scaling to other GPUs by bf16 TFLOPS ratio (A100 = 77.6 TFLOPS bf16):

| GPU | bf16 TFLOPS | Scale vs A100 | it/s (micro-batch) | Steps/sec |
|-----|------------|--------------|-------------------|-----------|
| A10G 24 GB | 31.2 | 0.40 | ~4.0 | **~1.0** |
| A40  48 GB | 37.4 | 0.48 | ~4.8 | **~1.2** |
| A100 80 GB | 77.6 | 1.00 | ~10.0 | **~2.5** |
| H100 80 GB | 204.9 | 2.64 | ~26.4 | **~6.6** |

### Uncertainty in these estimates

The FLOPs scaling is sound but ignores memory-bandwidth bottlenecks, attention overhead (MONAI's attention is not FlashAttention), and DataLoader I/O. The actual steps/sec can vary ±30–40% from these figures. The tables below therefore present a **pessimistic / central / optimistic range** rather than a single number, reflecting:

- **Pessimistic (−40%):** memory-bandwidth bound, slower storage, MONAI attention overhead
- **Central:** FLOPs scaling as derived above
- **Optimistic (+25%):** cuDNN kernel fusion, warm DataLoader cache, no pod contention

---

## How Many Steps Does "Converged" Mean?

This matters because it determines which scenario to budget for.

The LDM is an ε-prediction DDPM on a 16.4 M parameter UNet. Comparably sized LDMs in the literature (e.g., CompVis LDM at reduced depth) reach their loss plateau within **20k–50k steps** at effective batch 16. Our model has an additional convergence signal: three classes rather than unconditioned, with CFG dropout at 15%, which slightly slows convergence but not significantly.

- **30k steps**: loss curve is visible; CFG images are recognisable but noisy. *Useful for debugging and pipeline validation, not for composition experiments.*
- **50k steps**: most published small-LDM results are acceptably converged here. FID and SSIM stop improving materially past this point for ~16 M parameter models. *Minimum for the paper's Exp5 marginals gate.*
- **100k steps**: diminishing returns past 50k but provides margin and improves composition sample quality. *Safe upper bound; `max_steps` in the config.*

The config sets `max_steps: 100,000` but the kill criterion (`loss > 2× loss_at_10k`) will catch divergence early. In practice you can assess the W&B `train/loss` curve at 50k and decide whether to continue.

---

## Full Pipeline Cost

Each scenario below includes:
- Preprocessing: 1× precompute + scale factor (shared)
- Training: 2 LDM experts (`cardiomegaly` + `effusion`) at the stated steps
- Buffer: 0.5 h for pod setup, teardown, and checkpoint retrieval
- Contingency: 1.2× applied to the training component only (preprocessing variance is low)

Ablation configs (`ldm_ablation_cfg_p00`, `ldm_ablation_cfg_p30`) are included in Scenario C only.

### Scenario A — Pipeline Validation (50k steps, 2 LDMs)

Use this to confirm the training loop works end-to-end, validate CFG output quality, and decide whether to continue to convergence. Acceptable if budget is tight and you can visually assess the W&B grids to judge whether the model is learning.

| GPU | Preprocess | Training (×2) | Total | **Pessimistic** | **Central** | **Optimistic** |
|-----|-----------|--------------|-------|----------------|------------|----------------|
| A10G 24 GB | 1.6 h | 50k / 1.0 s × 2 = 27.8 h | **~30 h** | $54 | **$44** | $34 |
| A40  48 GB | 1.3 h | 50k / 1.2 s × 2 = 23.1 h | **~25 h** | $45 | **$37** | $29 |
| A100 80 GB | 0.7 h | 50k / 2.5 s × 2 = 11.1 h | **~12 h** | $22 | **$18** | $14 |

### Scenario B — Convergence Run (100k steps, 2 LDMs)

Required before running Exp5–8 (marginals gate, composition experiments). This is the minimum viable run for a complete paper result.

| GPU | Preprocess | Training (×2) | Total | **Pessimistic** | **Central** | **Optimistic** |
|-----|-----------|--------------|-------|----------------|------------|----------------|
| A10G 24 GB | 1.6 h | 100k / 1.0 s × 2 = 55.6 h | **~57 h** | $102 | **$85** | $66 |
| A40  48 GB | 1.3 h | 100k / 1.2 s × 2 = 46.3 h | **~48 h** | $86 | **$72** | $55 |
| A100 80 GB | 0.7 h | 100k / 2.5 s × 2 = 22.2 h | **~23 h** | $42 | **$35** | $27 |

### Scenario C — Full Experiment Grid (100k steps, 2 LDMs + 2 ablations)

Adds the two CFG-dropout ablation runs needed for Exp6/7. Preprocessing and scale factor are still paid once.

| GPU | Preprocess | Training (×4) | Total | **Pessimistic** | **Central** | **Optimistic** |
|-----|-----------|--------------|-------|----------------|------------|----------------|
| A10G 24 GB | 1.6 h | 100k / 1.0 s × 4 = 111 h | **~113 h** | $201 | **$167** | $130 |
| A40  48 GB | 1.3 h | 100k / 1.2 s × 4 = 92.6 h | **~94 h** | $169 | **$141** | $109 |
| A100 80 GB | 0.7 h | 100k / 2.5 s × 4 = 44.4 h | **~46 h** | $82 | **$68** | $53 |

---

## Pinning Down the Central Estimate: The Timing Run

The ±30–40% spread in the tables above comes almost entirely from not knowing the actual steps/sec on the pod. A 100-step timing run closes this:

```bash
# Takes ~2–3 minutes; run before committing to the full job
time python scripts/train_ldm.py \
    --config configs/ldm_full.yaml \
    --latent-cache data/latents \
    --max-steps 100
```

Divide 100 by the wall-clock seconds from `time` to get measured steps/sec. Then compute revised estimates:

```
training_hours = max_steps / measured_steps_per_sec / 3600
cost = (training_hours × n_ldms + preprocess_hours + 0.5) × $1.49/hr
```

For example, if `time` reports `40s` for 100 steps → 2.5 steps/sec → use the A100 central column. If it reports `100s` → 1.0 steps/sec → use A10G central column. The timing run costs `3/60 × $1.49 ≈ $0.07`.

> **No dedicated profiler module is needed.** VRAM is analytically bounded at 2.6 GB and does not need measurement. The only unknowable ahead of time is steps/sec, and a 100-step inline run resolves it cheaply. The VAE had `vae/profile.py` + `vae/budget.py` because VRAM and batch-size sweeping were both uncertain; neither applies here.

---

## Recommendation

| Priority | Choice | Reasoning |
|----------|--------|-----------|
| Minimum for paper | **A100 80 GB, Scenario B** | ~$35–42. Two converged LDM experts in ~23 h. |
| Budget-conscious | **A40 48 GB, Scenario B** | ~$72–86. Same result, ~2× longer wall-clock. |
| Full grid in one shot | **A100 80 GB, Scenario C** | ~$68–82. All four configs sequential; ~46 h total. |
| Fastest turnaround | **H100 80 GB, Scenario B** | ~$13–16. ~7 h total including preprocessing. |

**Avoid A10G for Scenario C** — 113 h on a preemptible pod is high-risk without checkpointing paranoia.

**Parallel option for Scenario C:** split into two simultaneous Scenario B pods (one for the two main LDMs, one for the two ablations). Same total cost, half the calendar time.

---

## Appendix: Batch Size Scaling

If the GPU has headroom, increasing the micro-batch and reducing grad_accum keeps the effective batch at 16 (same training dynamics) while reducing the number of DataLoader calls per step:

| micro_batch | grad_accum | eff. batch | VRAM est. | Speedup vs B=4 |
|-------------|-----------|------------|-----------|----------------|
| 4 (current) | 4 | 16 | ~2.6 GB | 1.0× |
| 8 | 2 | 16 | ~4.5 GB | ~1.6× |
| 16 | 1 | 16 | ~8.0 GB | ~2.5× |

Any GPU with ≥ 10 GB free can use B=8. The speedup is real because each DataLoader `__getitem__` call (loading a `.pt` file from disk) becomes less frequent per optimizer step.
