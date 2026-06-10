# 🔬 Profiling Protocol — VAE & Single-Disease LDM

`feeds: plans/compute-budget/plans/01-workload-sizing.md` · `host: local RTX 4070 12GB → extrapolate` · `pinned: 2026-06-09`

> **Why this exists.** GPU selection is driven by peak VRAM; cost is driven by
> throughput × steps. We cannot afford to discover either by renting and watching.
> This protocol measures both content-independent numbers **locally at batch=1**,
> fits a linear VRAM-vs-batch model, and transfers throughput to the rented tiers
> by a compute-ratio rule — so the pricing survey gets a VRAM target and the cost
> memo gets GPU-hours, **before** the first dollar of rental.

---

## 0 · What we are measuring (and what we are NOT)

Profiling captures only the **content-independent** numbers — they do not depend on
real X-rays, so they can be measured on **noise data** locally:

- **Peak VRAM (GB)** — `torch.cuda.max_memory_allocated()` at steady state.
- **Throughput** — VAE `img/s`, LDM `it/s` (and `img/s = it/s × batch`), median over K steady steps.

We are **not** measuring recon quality, FID, or convergence here — those need real
data and execute downstream under `runpod-execution`. **Do not** let a profiling run
masquerade as a training result.

---

## 1 · Frozen architecture assumptions (state them, or the numbers are unattributable)

The numbers only mean something against a pinned architecture. Record any deviation.

**VAE** (AutoencoderKL-style, label-blind):
- in = 1ch, out = 1ch, latent = **4×128×128 (f=4 at 512²)**; no encoder→decoder skips.
- channel schedule `(128, 256, 512)`, GroupNorm + SiLU, KL bottleneck.
- **Dominant VRAM lever → bottleneck attention.** At f=4 the bottleneck is **128×128 = 16 384 tokens** — full self-attention there is ~10–100× SD's 64×64 attention and is the single biggest VRAM risk. Profile **two variants**: `attn=none` (pure conv) and `attn=windowed/low-res`. Record which.
- 768²→f=6 and 1024²→f=8 reach the **same** 4×128×128 latent by adding encoder/decoder resolution stages — the extra **high-resolution** early conv maps are why VAE VRAM scales with input pixels.

**LDM** (conditional UNet, CFG dropout):
- operates **only** in the 4×128×128 latent → **resolution-invariant** (one config for all of 512/768/1024).
- base channels `128`, mults `(1,2,4,4)` over latent `128→64→32→16`.
- **Dominant VRAM lever → attention resolution.** Attention at the 128×128 / 64×64 latent levels is the cost. Profile `attn@{32,16}` vs `attn@{64,32,16}`. Record which.
- ⚠️ The plan-tree's old `≤128ch, batch≤8, fits 4070 12GB` was written for the **stale 32×32×4 latent** (256× fewer latent pixels than 4×128×128). **Discard it as a sizing input** — re-measure.

Both: **bf16** autocast, **AdamW** (8 bytes/param optimizer state + master), gradient-checkpointing **OFF** for the baseline measurement (measure its effect separately — it is a VRAM lever, see §4).

---

## 2 · The measurement procedure (per config, on the 4070)

For each config (VAE×{512,768,1024}×{attn variants}; LDM×{attn variants}):

1. **Build the module, feed noise** of the correct shape. VAE input `(B,1,R,R)`; LDM input latent `(B,4,128,128)` + a random condition id + a sampled timestep. No dataset needed.
2. **Batch sweep** `B ∈ {1,2,4,8,16,…}` until OOM or 11 GB usable is hit. For each surviving B run a **full train step** (forward + loss + backward + optimizer.step).
3. **Peak VRAM:** `torch.cuda.reset_peak_memory_stats()` before, `torch.cuda.max_memory_allocated()` after — record the steady-state peak (discard step 0, allocator warm-up).
4. **Throughput:** after ≥10 warm-up steps, time K≥50 steps, report **median** `it/s` and `img/s = it/s × B`. Tail a `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1` log alongside — this is the **profiling log** that backs ≥1 VRAM number so the table isn't pure analytics.
5. **If batch=1 itself OOMs locally** (likely for 1024² VAE and the attn@64 LDM on 12 GB): switch to the **fallback ladder** — (a) gradient checkpointing on, (b) the smaller attn variant, (c) if still no fit, **mark the cell `analytic-only`** and fill it from the §3 model rather than measurement. Never silently drop a config — log that it was estimated, not measured.

---

## 3 · The VRAM model (fit locally, extrapolate to the tiers)

Peak VRAM is **device-independent** for a fixed (config, batch) — a tensor is the same
size on a 4070 or an A100. So the local sweep transfers directly. Fit, per config:

```
VRAM(B) ≈ V_fixed + s · B
   V_fixed = params + optimizer state + CUDA context   (batch-independent)
   s       = activation memory per sample              (slope of the sweep)
```

Two surviving batch points give the line; ≥3 lets you check linearity (it should be
near-linear until allocator fragmentation). Then for any target tier, **max usable
batch** = `floor((VRAM_tier·0.9 − V_fixed) / s)` — the 0.9 leaves allocator headroom.
This is how a 12 GB host sizes an 80 GB card it never touches.

## 3b · The throughput transfer (the one estimate that needs a discount)

Throughput is **not** device-independent. Transfer by compute ratio with an efficiency
discount (memory-bound ops and attention don't scale with peak FLOPS):

```
it/s(tier) ≈ it/s(4070) × (BF16_TFLOPS_tier / BF16_TFLOPS_4070) × η
   η ≈ 0.5–0.7   (lower for attention-heavy LDM, higher for conv-heavy VAE)
```

Report the transferred throughput as a **range** (η 0.5→0.7), not a point — and flag
it `ESTIMATE` until §5 confirms it. 4070 dense bf16 ≈ 29 TFLOPS; A100 ≈ 312; H100 ≈ 990;
L40S ≈ 91; A6000 ≈ 39 (use the tier's real spec sheet, not these round numbers).

---

## 4 · Levers to record (each changes the table, so each is a row, not a footnote)

| Lever | Effect | Why we record it |
|---|---|---|
| **Gradient checkpointing on/off** | ~30–40% less activation VRAM, ~20–30% slower | the knob that makes a marginal config fit |
| **Batch size** | linear VRAM, throughput plateaus | sets max-batch per tier (§3) |
| **Attention resolution** (LDM) / bottleneck attention (VAE) | dominant VRAM term | the single biggest swing — see §1 |
| **bf16 vs fp32** | ~2× VRAM, ~2× throughput | confirm bf16 is stable for the VAE KL term first |
| **grad-accum** | trades steps for VRAM at fixed effective batch | lets a small card hit the target effective batch |

---

## 5 · Confirm-on-target (the one cheap check before the full run)

The user chose *extrapolate locally* — so the **only** on-target measurement is a
sanity confirm at provisioning time (under `runpod-execution`), **not** a paid
profiling session now: on the chosen tier, run **one** 50-step micro-run at the
planned batch and check measured peak VRAM is within ~15% of the §3 prediction and
`it/s` within the §3b range. If it diverges, re-fit before committing the full
GPU-hours. This catches a bad η or an allocator surprise for the price of 50 steps.

---

## 6 · Kill-criteria & disk guard (carry into every real run)

- **Kill:** loss NaN/Inf, VRAM climbing across steps (leak) → not steady state, `it/s` collapsing (thermal/throttle or swap).
- **Pre-flight disk guard** (the `/home-mscluster` lesson, applies to RunPod volume too): abort *before* writing checkpoints if the target volume is ≥90% full — a full disk silently kills checkpointing and you lose the run's tail.

```bash
CKPT_DIR=${CKPT_DIR:-/workspace/ckpts}
USED=$(df --output=pcent "$CKPT_DIR" | tail -1 | tr -dc '0-9')
[ "${USED:-100}" -ge 90 ] && { echo "ABORT: $CKPT_DIR ${USED}% full — checkpointing will fail." >&2; exit 1; }
```
