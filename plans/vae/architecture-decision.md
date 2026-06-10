# 🏛️ VAE Architecture Decision

**Scope:** `plans/vae/` · **Status:** decided · **Date:** 2026-06-09
**Codec:** label-blind AutoencoderKL · 512² grayscale chest X-ray → **4×128×128 latent (f=4)**

This note fixes the architecture *before* implementation so the codec contract,
profiling shapes, and cost numbers downstream are stable. Produced via `/augment`
— mining the AutoencoderKL / LDM literature as a parts bin (provenance at the end).

---

## 1. Decision in one line

A **kl-f4-class AutoencoderKL** (ResNet blocks, GroupNorm+SiLU, mid-block self-attention
only, KL bottleneck, **no encoder→decoder skips**), 1-channel in/out, **z=4** latent,
trained with **`0.84·MS-SSIM + 0.16·L1 + 1e-6·KL`** — **no adversarial term, no RGB-LPIPS**.

The load-bearing rationale: at f=4 the latent holds 128²×4 = 65,536 values vs the input's
512²×1 = 262,144 — a **~4× compression**, an order of magnitude lighter than SD kl-f8's ~48×.
This is deliberate: a trivially-high reconstruction ceiling, a cheaper VAE, and a latent
with the spatial capacity PoE composition needs. Much of the SD/LDM machinery (perceptual +
adversarial losses, 16-channel latents) exists to fight *high* compression; at 4× we don't
have that problem, and importing those parts would cost more than it buys.

---

## 2. Configuration table (the contract)

| Field | Value | Source / note |
|---|---|---|
| Family | AutoencoderKL (KL-regularized) | LDM kl-f4 |
| Input | `(B, 1, 512, 512)` grayscale | domain |
| Latent | `(B, 4, 128, 128)` | f=4, z=4 |
| Downsample factor `f` | 4 | 2 downsamples |
| Base channels `ch` | 128 | LDM kl-f4 |
| `ch_mult` | `[1, 2, 4]` → 128, 256, 512 | LDM kl-f4 |
| Resolution stages | 512 → 256 → 128 | 3 levels, 2 downsamples |
| `num_res_blocks` per stage | 2 | LDM kl-f4 |
| Norm + activation | GroupNorm (32 groups) + SiLU | LDM/SD standard |
| Attention | **mid-block (lowest res, 128²) only** | `attn_resolutions=[]` |
| `z_channels` / `embed_dim` | **4** | SD kl-f8 (donor) |
| `double_z` | true (encoder emits μ,logσ² → 8 ch) | KL bottleneck |
| Enc→dec skips | **none** | bottleneck contract |
| Output nonlinearity | **none** (raw); data normalized to [−1,1] | SD/LDM convention |
| KL weight | **1e-6** | LDM/SD verbatim |
| Reconstruction loss | **0.84·MS-SSIM + 0.16·L1** | Zhao et al. 2017 |
| Adversarial loss | **none** | rejected (§5) |
| Param estimate | **~55–60M** (exact in plan-02) | kl-f4-class, z=4, 1-ch |

---

## 3. Channel & stage schedule

```
Encoder
  in_conv:        (B,1,512,512)  → (B,128,512,512)
  stage 0  ch=128 (×2 resblock), downsample → 256²
  stage 1  ch=256 (×2 resblock), downsample → 128²
  stage 2  ch=512 (×2 resblock)              128²   (no further downsample)
  mid:     resblock → self-attention → resblock      (attention here only)
  out:     GroupNorm+SiLU → conv → (B, 8, 128, 128)   (μ, logσ²; double_z)
  sample:  z ~ N(μ, σ²)                      (B, 4, 128, 128)

Decoder  (mirror; NO skip connections from encoder)
  in_conv: (B,4,128,128) → (B,512,128,128)
  mid:     resblock → self-attention → resblock
  stage 2  ch=512 (×3 resblock)              128²
  stage 1  ch=256 (×3 resblock), upsample  → 256²
  stage 0  ch=128 (×3 resblock), upsample  → 512²
  out:     GroupNorm+SiLU → conv → (B, 1, 512, 512)   (raw, no tanh)
```

*(Decoder uses `num_res_blocks+1` blocks per stage — LDM convention; minor asymmetry, not skips.)*

**Bottleneck contract.** Information flows encoder → `z (4×128×128)` → decoder and through
nothing else. No skip tensors cross the bottleneck. The latent is the *only* channel, so the
downstream LDM has the full appearance distribution to model — the precondition for the whole
pipeline.

---

## 4. Documented levers (committed default + recorded escape hatch)

These are **off by default**. Recorded so a future run can reach for them with eyes open —
each carries a trade-off, neither is a silent "improvement."

| Lever | Default | Trigger to engage | Trade-off |
|---|---|---|---|
| **z=8** (vs z=4) | z=4 | recon ceiling disappoints in plan-07/ceiling check | buys recon headroom (cheap at f=4); **costs latent diffusability** — wider bottleneck injects high-frequency content the LDM+PoE must model (*Improving the Diffusability of Autoencoders*, arXiv 2502.14831). Since the pipeline is PoE-separability-limited, not recon-limited, expect this to be unnecessary. |
| **RadImageNet-LPIPS** perceptual term | off (MS-SSIM+L1 only) | recon shows texture flatness | buys domain-correct perceptual gradients (used in BS-LDM, X-ray2CTPA); **costs** an external-weights dependency + reproducibility friction. Use the *RadImageNet* backbone, never RGB VGG/AlexNet. |

---

## 5. Rejected alternatives

**f=8 (and the SD/SDXL 48× regime).** Rejected. f=8 forces ~48× compression → the latent loses
fine detail, which is exactly why SD3/FLUX had to widen latents to 16 channels and why SD leans
on perceptual+adversarial losses to mask the loss. f=4 buys a near-trivial recon ceiling and
keeps the latent spatially separable for PoE. Cost accepted: a larger latent the LDM must model
(owned by the LDM/compute-budget scopes).

**Encoder→decoder skip connections (U-Net-style autoencoder).** Rejected outright. Skips let
the decoder bypass the latent, so reconstruction quality stops certifying that information lives
*in z*. The LDM models z; if z isn't the sole channel, the LDM is modeling an incomplete
representation. The latent must be a true information bottleneck — non-negotiable.

**DiT / transformer-token autoencoder.** Rejected for this codec. A patch-token latent breaks the
image-like spatial grid that (a) the convolutional LDM's spatial bias exploits and (b) PoE relies
on for spatial separability of findings (cardiomegaly central, effusion basal). Convolutional
AutoencoderKL keeps the latent a 128² spatial map. DiT may reappear *as the LDM denoiser* — that's
a separate scope decision, not this codec.

**PatchGAN / adversarial reconstruction loss.** Rejected — and deliberately **not** kept as an
optional lever. Adversarial gradients push the decoder toward *plausible-looking* anatomy that
need not correspond to the true input (hallucination; arXiv 2508.14118). The entire pipeline rests
on a *faithful* recon ceiling and faithful disease appearance for PoE — an adversarial VAE would
improve apparent sharpness while silently corrupting the quantity we measure. It exists in SD to
hide f=8 information loss; at f=4 we have little to hide. Leaving it as a lever would invite a
future run to switch it on and break the ceiling, so it is rejected by status, not just by default.

**Off-the-shelf RGB-LPIPS (VGG/AlexNet).** Rejected as a *training* loss. Its perceptual prior is
natural-image color/texture; on grayscale CXR it is out-of-domain. If perceptual gradients are
needed, the RadImageNet-backed variant is the lever (§4). (Plain LPIPS may still serve as an
*eval* metric — that's the recon-gate's call, plan-04, not this loss.)

**`tanh` output.** Rejected-optional. SD/LDM decoders emit raw values and normalize the *data* to
[−1,1]; a `tanh` saturates decoder gradients near the range edges for no benefit at f=4.

---

## 6. Provenance trail (what was borrowed, from where, confidence)

- **kl-f4 skeleton** — `ch=128`, `ch_mult=[1,2,4]`, `num_res_blocks=2`, `attn_resolutions=[]`,
  `kl_weight=1e-6`. Source: CompVis LDM `models/first_stage_models/kl-f4/config.yaml`. *High confidence — read verbatim from the released config.*
- **z_channels=4 + double_z** — Source: Stable Diffusion `autoencoder_kl_32x32x4.yaml` (kl-f8, z=4). *High confidence — verbatim.*
- **MS-SSIM + L1 mix (α≈0.84)** — Zhao, Gallo, Frosio, Kautz, *"Loss Functions for Image Restoration with Neural Networks"*, 2017. *High confidence on the result; α validated on natural images — re-check the mix ratio on CXR in plan-04.*
- **"channels matter only at high compression"** — *Unified Latents* (arXiv 2602.17270); *Improving the Diffusability of Autoencoders* (arXiv 2502.14831). *Medium confidence — informs the z=4-vs-8 call; verify before relying for a publication claim.*
- **Medical reference class** — MedVAE (arXiv 2502.14753); RadImageNet-LPIPS use in BS-LDM (arXiv 2412.15670) and X-ray2CTPA. *Medium confidence — establishes that a medical-tuned VAE, not the RGB SD VAE, is the right precedent.*
- **Implementation chassis (for plan-04)** — HF `diffusers.AutoencoderKL` or MONAI GenerativeModels `AutoencoderKL` (1-channel native). *To be selected in plan-03/04; both expose the config above.*

## 7. Open / deferred

- **Input resolution (512 / 768 / 1024)** — f=4 is fixed; the *input* res is cost-gated and owned
  by `plans/compute-budget/`. This note assumes 512²; if compute-budget selects 768²/1024² the
  latent scales to 192²/256² at the same f=4 and the schedule is unchanged.
- **Exact parameter count** — to be printed in plan-02 (architecture sanity checks). ~55–60M is the
  working figure for cost estimates until then.
- **α re-check + perceptual-lever decision** — plan-04 (implementation + recon metric).
