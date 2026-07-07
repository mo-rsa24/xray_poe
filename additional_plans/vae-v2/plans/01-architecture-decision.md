# 🏛️ Architecture Decision — f=4 Keep vs KL-f8 Adopt

## Background
Every downstream leaf in this scope depends on one choice that cannot be changed mid-run: which latent space the VAE operates in. f=4 (current, 4×128×128 = 16k tokens) preserves cardiac-border spatial resolution at the cost of 4× more LDM attention; f=8 (SD KL-f8, 4×64×64 = 4k tokens) unlocks a powerful pretrained decoder but loses the pixel margin that makes CTR readable and requires a full LDM retrain with a foreign codec. This plan locks the choice *before* any compute is spent, so the retrain and re-encode leaves operate on a stable target.

## Description
Write a short decision document (≤ 2 pages) that works through the trade-off quantitatively — pixel margins at each compression factor, LDM attention cost ratio, pertained-decoder benefit estimate, downstream gate risk — and commits to one option with a rationale that can be quoted in the paper methods.

## Purpose
Serves vae-v2 Objective 1 and DoD item 1. Decision gates leaves 02–07; no training may start until this doc is committed. A decision doc, not code.

## Goal
A committed written document `docs/vae_v2_architecture_decision.md` that states: (a) which option is chosen, (b) the pixel-margin calculation at 512² for both f=4 and f=8, (c) the LDM attention cost ratio (quadratic, 16k vs 4k tokens), (d) the pretrained-decoder benefit estimate (why RoentGen keeps VAE frozen), and (e) the reason the alternative was rejected.

## Tasks

- [x] ✅ **Compute the cardiac-border pixel margin at f=4 vs f=8**  ✓ verified (`docs/vae_v2_architecture_decision.md` §1; CTR p10–p90 = 0.477–0.605 from pixel_verifiable_separable_test.csv N=378; thorax=410px; heart width 196–248px; border edge 20px → **f=4: 5.0 lat-px ✓**, **f=8: 2.5 lat-px ✗** below 3-px minimum; 1-px border error → ΔCTR ≈ 0.010@f=4 vs 0.020@f=8, saturating the Gate-2 ceiling with no headroom)

- [x] ✅ **Compute the LDM attention cost ratio**  ✓ verified (`docs/vae_v2_architecture_decision.md` §2; O(L²) scaling: 16k²/4k²=16× naive, ~4–8× empirical with chunked attention; arm1_cleancardio baseline 5.25h/100k cited; f=8 estimate ~0.7–1.3h/100k — "insufficient saving to justify spatial-resolution loss given CTR gate")

- [x] ✅ **Summarise the pretrained-decoder argument and why it doesn't apply here**  ✓ verified (`docs/vae_v2_architecture_decision.md` §3; cites RoentGen arXiv:2211.12737; states "pretrained f=8 decoder is a benefit for tasks judged perceptually; it is not a benefit for tasks judged by sub-pixel cardiac-width ratios"; CTR gate ceiling |ΔCTR| ≤ 0.020 cited — note: plan text had 0.01, corrected to match actual Gate-2 ceiling)

- [x] ✅ **Write and commit `docs/vae_v2_architecture_decision.md`**  ✓ verified (149 lines; Decision → §1 pixel-margin → §2 attention cost → §3 pretrained-decoder → §4 conclusion with summary table; "f=4 retained" and "f=4 confirmed" present; LPIPS backbone note added as §5)

## Recommended skill
— custom; no skill fits. This is a written decision doc, not code. Draft in conversation with `/socratic` if trade-offs feel unresolved.

## Engagement Instructions

This leaf produces a written doc, not runnable code. Use this template — fill in the
bracketed fields and commit the result to `docs/vae_v2_architecture_decision.md`.

```
# VAE Architecture Decision — f=4 Keep vs KL-f8 Adopt
Date: [YYYY-MM-DD]  Author: [name]

## Decision
**f=4 retained.** KL-f8 rejected.

## 1. Pixel-margin calculation (512² input)
| Factor | Latent size | Heart-width range (latent px) | Min usable border gradient |
|--------|-------------|-------------------------------|---------------------------|
| f=4    | 128×128     | [fill: CTR 0.43–0.60 → heart  | ≥ 3 px (ok)               |
|        |             |  width 220–300px → 55–75 lat] |                           |
| f=8    | 64×64       | [fill: same heart → 27–37 lat]| 2.5 px (marginal)         |

CTR source: `data/splits/pixel_verifiable_separable_test.csv` (median CTR ≈ [fill]).
Gate requirement: |ΔCTR| < 0.01 — a sub-pixel precision claim on the cardiac-width ratio.

## 2. LDM attention cost ratio
Quadratic attention: L=16384 (f=4) vs L=4096 (f=8) → naive ratio = 16×.
With chunked attention: empirical ratio ≈ [fill: 4–8×] (cite measured hippo run if available).
Measured arm1_cleancardio baseline: 100k steps ≈ 5.25 h on hippo A100.
f=8 estimate: [fill] h/100k — insufficient saving to justify spatial loss given CTR gate.

## 3. Pretrained-decoder argument and rebuttal
RoentGen (arXiv:2211.12737) keeps SD VAE frozen and fine-tunes U-Net only — their task
is general visual quality (FID, perceptual similarity), not a numeric spatial gate.
Our Gate-4 requires |ΔCTR| < 0.01 (pixel-precision cardiac-width ratio). This fails
at f=8 spatial resolution (see §1 pixel-margin above). Pretrained decoder is a benefit
for perceptual tasks; it is not a benefit for tasks judged by sub-pixel anatomy ratios.

## 4. LPIPS backbone note (informs leaf 03)
The codebase has explicitly rejected RGB AlexNet/VGG LPIPS as out-of-domain on
grayscale CXR (see `vae/losses.py` §5 note). The sanctioned lever is RadImageNet-LPIPS
(`VAEConfig.perceptual_weight`, currently 0.0 and unwired in `vae/losses.py`).
Leaf 03 must wire RadImageNet-LPIPS, not import `lpips` with net='alex'.

## 5. Conclusion
f=4 confirmed. f=8 rejected: spatial resolution insufficient for CTR gate at 64²,
LDM cost saving too small to justify, pretrained decoder irrelevant for numeric gates.
All downstream leaves (02–07) operate on the existing 4×128×128 latent space.
```

After filling and saving `docs/vae_v2_architecture_decision.md`:
```bash
$ grep -i "f=4 retained\|f=4 confirmed" docs/vae_v2_architecture_decision.md
# expect: one match in the Decision section
$ grep -i "RadImageNet\|perceptual_weight" docs/vae_v2_architecture_decision.md
# expect: mention of the leaf-03 LPIPS constraint
```
