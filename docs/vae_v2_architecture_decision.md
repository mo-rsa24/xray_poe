# VAE Architecture Decision — f=4 Keep vs KL-f8 Adopt

**Date:** 2026-06-29  
**Author:** Molefe Molefe  
**Status:** ✅ COMMITTED — f=4 retained; KL-f8 rejected

---

## Decision

**f=4 retained.** KL-f8 (SD-style f=8) rejected.

The downstream CTR gate requires sub-pixel-accurate cardiac-border reconstruction
that f=8 cannot reliably deliver. The compute saving (~4–8× cheaper LDM attention)
is insufficient to justify the spatial-resolution loss at the border.

---

## 1. Pixel-margin calculation (512² input)

### Setup

Input image: 512 × 512 pixels.  
Thorax width at 512²: **~410 px** (inner rib cage spans ≈80% of image width on a well-centred PA film).  
Cardiac border edge width (the silhouette edge PSPNet must segment cleanly): **~20 px** at 512².

CTR distribution measured from the NIH frozen separable-both test set
(`data/splits/pixel_verifiable_separable_test.csv`, N=378):

| Percentile | CTR   |
|-----------|-------|
| p10       | 0.477 |
| p50       | 0.532 |
| p90       | 0.605 |

### Heart-width range

Heart width (pixels at 512²) = CTR × thorax_width = CTR × 410:

| Percentile | CTR   | Heart width (512²) | f=4 latent (128²) | f=8 latent (64²) |
|-----------|-------|--------------------|-------------------|------------------|
| p10       | 0.477 | ≈ 196 px           | **49 px**         | **24 px**        |
| p50       | 0.532 | ≈ 218 px           | **55 px**         | **27 px**        |
| p90       | 0.605 | ≈ 248 px           | **62 px**         | **31 px**        |

At both factors the *bulk* of the heart is represented by many latent pixels.
The discriminating constraint is the **border edge**, not the bulk.

### Border-edge margin

A 20 px cardiac silhouette edge at 512² maps to:

| Factor | Latent size | Border edge (latent px) | ≥ 3 px minimum? |
|--------|-------------|-------------------------|-----------------|
| **f=4**    | 128 × 128   | **5.0 px**              | ✅ Yes (1.67×)  |
| **f=8**    | 64 × 64     | **2.5 px**              | ❌ No (0.83×)   |

The 3-pixel minimum is the minimum convolution footprint needed for a transposed-conv
decoder to reconstruct a smooth gradient across the border without aliasing. Below 3 px,
a single erroneous latent activation spans the entire visible border — the decoder
cannot distinguish a clean border from a local intensity artefact.

### Why this matters for CTR

The CTR instrument (`scripts/ctr_overlay.py`, PSPNet) measures the
*widest cardiac diameter* as a ratio of thorax width. A 1-pixel error in the
reconstructed border at latent resolution propagates to:

- **f=4:** 1 latent px → 4 image px → ΔCTR ≈ 4/410 ≈ **0.010** (at our abstain margin)
- **f=8:** 1 latent px → 8 image px → ΔCTR ≈ 8/410 ≈ **0.020** (at our abstain margin)

The Exp-02 codec ceiling established that |ΔCTR| ≤ 0.020 is the acceptable ceiling
(NIH VAE median |ΔCTR| = 0.011; VinDr gate median = 0.009). At f=8, a *single*
erroneous border latent pixel already saturates the ceiling. With only 2.5 border
latent pixels in total, a 40% error rate (1 of 2.5 pixels wrong) would push the
codec beyond the gate threshold with no headroom. At f=4 with 5.0 border pixels,
the equivalent failure rate is 20% — consistent with the 0.009–0.011 observed medians.

### Reconciliation with plan expectation

The plan (`01-architecture-decision.md`) expected heart widths 220–300 px,
corresponding to CTR 0.43–0.60 at a 512 px thorax. Our measurement uses
thorax = 410 px (80% of 512 px), not 512 px (which would imply the thorax fills
the entire image, an overestimate). The resulting heart widths (196–248 px) are
narrower than the plan's estimate but the conclusion is unchanged: f=8 border latents
(2.5 px) are below the 3-pixel minimum; f=4 (5.0 px) are safely above it.

---

## 2. LDM attention cost ratio

Attention scales as O(L²) where L is the sequence length (spatial tokens).

| Factor | Latent size | Tokens L | Attention ratio vs f=8 |
|--------|-------------|----------|------------------------|
| f=4    | 128 × 128   | 16 384   | 16× (naive)            |
| f=8    | 64 × 64     | 4 096    | 1× (reference)         |

In practice with the chunked flash-attention used by our UNet, the runtime ratio
is approximately **4–8×** (not 16×). The measured baseline for the existing
arm1_cleancardio f=4 LDM is **5.25 h / 100k steps on hippo** (single A100-equiv).
At f=8 this would be roughly **0.7–1.3 h / 100k steps**.

**Verdict**: f=4 LDM costs ~5.25 h/100k on hippo; f=8 would cost ~0.7–1.3 h/100k
(estimated 4–8× cheaper). This saving is insufficient to justify the spatial-resolution
loss given our CTR gate — we need ≤0.020 ΔCTR and f=8 leaves no headroom.

---

## 3. Pretrained-decoder argument — and why it doesn't apply here

The Stable Diffusion KL-f8 decoder is powerful and pretrained on large-scale natural
images. RoentGen (arXiv:2211.12737) exploits this by keeping the SD VAE frozen and
fine-tuning only the U-Net, achieving strong perceptual quality (FID, IS) on chest
X-rays without retraining the decoder.

**Why this doesn't apply to our task**: RoentGen's downstream evaluation is
perceptual — FID and visual quality, judged at the whole-image level. Our primary
downstream gate requires |ΔCTR| < 0.020 on **sub-pixel cardiac-border localisation**.
This is a spatial numeric gate, not a perceptual gate. The pretrained f=8 decoder
is a benefit for tasks judged perceptually; it is not a benefit for tasks judged
by sub-pixel cardiac-width ratios. Our Gate-2 (codec ceiling) requires the latter,
and the pixel-margin calculation above shows f=8 fails that requirement.

Additionally, the pretrained f=8 decoder was trained on RGB natural images, not
single-channel grayscale CXRs. Any transfer benefit is further diluted by the domain
mismatch.

---

## 4. Conclusion

**f=4 confirmed. f=8 rejected.**

| Factor | Border lat-px | Below 3-px min? | 1-px err → ΔCTR | Gate-2 headroom | LDM cost on hippo |
|--------|--------------|-----------------|-----------------|-----------------|-------------------|
| f=4    | **5.0**      | No ✅            | 0.010           | **2.1×** ✓     | 5.25 h/100k       |
| f=8    | **2.5**      | Yes ❌           | 0.020           | **1.0×** ❌    | ~1 h/100k         |

f=8 fails the border-resolution minimum at 2.5 latent pixels, and a single pixel
error at f=8 already saturates the 0.020 ΔCTR ceiling with no recovery headroom.
The 4–8× compute saving is real but secondary: given hippo capacity, training time
is not the binding constraint.

This decision is **final for the VAE-v2 scope**: all downstream leaves (diet
preparation, retrain, re-encode, LDM retrain) operate in the f=4 latent space
(4 × 128 × 128). The existing `vae_step0025000.pt` will be used as the **warm-start
initialisation** for VAE-v2 fine-tuning to preserve the spatial priors already
learned from 112k NIH films.

---

## 5. LPIPS backbone constraint (gates leaf 03)

RGB AlexNet/VGG LPIPS is **out-of-domain** for single-channel grayscale CXR and must
not be used. The sanctioned perceptual lever is **RadImageNet-LPIPS** (a torchvision
ResNet backbone pre-trained on medical imaging). The VAE training config controls this
via `VAEConfig.perceptual_weight` (currently 0.0, unwired in `vae/losses.py`).

**Leaf 03 must wire RadImageNet-LPIPS** (`perceptual_weight > 0`), not
`import lpips; lpips.LPIPS(net='alex')`. Using the wrong backbone silently degrades
the perceptual signal without raising an error — the decoder will learn to fool an
out-of-domain feature extractor rather than sharpening anatomically relevant structure.
