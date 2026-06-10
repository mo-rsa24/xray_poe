# 📊 VAE Profiling Notes (local)

**Scope:** `plans/vae/` · plan 05/08 · measured 2026-06-09
**Hardware:** NVIDIA RTX 4070 Laptop, **8 GB VRAM** (WSL2, torch 2.5.1+cu124)
**Model:** kl-f4 AutoencoderKL (MONAI), **49,104,321 params (49.10M)** — encoder 22.34M, decoder 26.76M

These are the *content-independent* numbers (peak VRAM, throughput) the compute-budget
scope needs. Peak VRAM and throughput depend only on shape/dtype/batch/architecture/optimizer,
so noise tensors are valid measurands (plan 08).

## Peak VRAM — forward+backward, bf16 autocast, AdamW

| res | batch | grad-ckpt | attention | peak VRAM | notes |
|----|------|----------|-----------|-----------|-------|
| 512² | 1 | off | mid (contract) | **10.60 GB** | exceeds 8 GB → ran via WSL2 host-RAM spillover |
| 512² | 1 | **on** | mid (contract) | **7.40 GB** | fits 8 GB; grad-ckpt is the local enabler |
| 512² | 2 | off | mid (contract) | **21.01 GB** | ⇒ ~10.5 GB/sample marginal |
| 512² | 1 | off | **off** (lever) | 6.31 GB | mid-attention at 128² costs ~4.3 GB/sample |
| 512² | 1 | on | off (lever) | 3.88 GB | |
| 256² | 4 | off | mid | 8.68 GB | |
| 256² | 2 | on | mid | **3.03 GB** | clean-fit local config |

## Throughput

The 8 GB card **cannot give a valid 512² img/s**: the contract config sits at/above the
VRAM ceiling and thrashes in shared-memory spillover (measured 0.2 img/s @ 512² b1 +ckpt —
an artifact, not a rate). A clean local rate is only obtainable well under 8 GB:

| res | batch | grad-ckpt | img/s | valid? |
|----|------|----------|-------|--------|
| 256² | 2 | on | ~4.0 | ✅ clean (3.03 GB) |
| 512² | 1 | on | 0.2 | ❌ thrashing (7.8 GB, at ceiling) |

**⇒ the target-resolution (512²) throughput must be measured on the rented GPU.** This is the
scope's hand-off boundary regardless (MASTER_PLAN: real-data train + numbers under
`compute-budget/runpod-execution`).

## Training findings (plan 07/08) — load-bearing for the real train

- **Noise run (plan 08) PASSES — end-to-end, no OOM at the real input shape.** 512² b1,
  bf16, grad-ckpt, sampled path, 25 steps: completed without OOM, **peak VRAM 7.80 GB**,
  0.1 img/s (spillover artifact at the 8 GB ceiling — not a usable rate; target-res
  throughput deferred to the rented GPU). Confirms the loop runs end-to-end on noise tensors
  shaped exactly like real inputs.
- **Overfit-sanity PASSES.** Fixed 2-image batch, deterministic recon, fp32, lr 1e-4 →
  recon **0.918 → 0.0047** in 400 steps; `figures/vae_overfit.png` shows near-identical
  in/out. Confirms the loop/loss/data path are wired and the codec has the capacity to
  reconstruct (an autoencoder-capacity check, by design).
- **σ-drift under the contract's 1e-6 KL.** In small-data / short-schedule regimes the
  *sampled* latent's `z_sigma` drifts large (KL → 10⁵–10⁶) because the 1e-6 KL weight
  applies almost no pressure, so `z = μ + σ·ε` becomes noise-dominated and the **sampled**
  path will not overfit. Consequences adopted here:
  - **Reconstruction is evaluated on the posterior mean** `decode(μ)` (`VAE.reconstruct`) —
    canonical VAE practice (the mode), used by the overfit figure, recon metrics, and the
    ceiling check. The sampled path is the *training* signal only.
  - **The overfit-sanity gate runs deterministic + fp32**; bf16 autocast + MS-SSIM is
    numerically fragile at small scale (it diverged). The noise run / real train use bf16
    (throughput regime) where the loss need only *run*, not converge.
  - **For the real-data train (runpod-execution): monitor `z_sigma` / KL.** With 1e-6 KL on
    a large diverse corpus this is usually fine (it's the SD/LDM value), but if σ drifts,
    a brief KL warm-up or a slightly higher KL weight is the lever — recon is still read off μ.

## Implications for compute-budget (load-bearing)

1. **512² training is steeply VRAM-limited: ~10.6 GB/sample** (7.4 with grad-checkpointing).
   A 24 GB card → batch ~2; 40 GB → ~3–5; 80 GB → ~7–10. Batch size, and thus throughput per
   GPU, is set by this. The mid-block self-attention at the **128² latent (16,384 tokens, O(N²))**
   is ~40 % of the footprint — a cost the architecture-decision note treated as cheap ("lowest
   res") but which is large at f=4 (SD's VAE attends at 64², 16× fewer tokens).
2. **Grad-checkpointing (`use_checkpoint=True`, already wired) cuts ~30 %** of peak VRAM for a
   modest throughput cost — likely the default on the rented GPU to lift batch size.
3. Reproduce on RunPod: `python -m vae.profile --res 512 --precision bf16 --sweep` to get peak
   VRAM + the max batch that fits the chosen card, and a non-thrashing img/s.

*(Raw log: `logs/vae_profile.log`.)*
