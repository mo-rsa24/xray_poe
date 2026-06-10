# ✅ VAE Overfit Sanity Run (first pass)

## Description
Drive the VAE to reconstruct a tiny fixed batch (8 synthetic/noise images) to
near-zero error within a few hundred steps — the first confirmation that the loop,
loss, and data path are wired correctly.

## Purpose
Catches a broken loop in minutes instead of after GPU-hours — the cheapest gate,
and it must pass before any longer run. This confirms the **code**, not the data
(see [07] vs [08]: overfit proves correctness, the noise run proves it scales and
measures usage).

## Goal
Reconstruction error on a fixed 8-image batch driven to near zero; the input-vs-output
figure is near-identical.

## Tasks
- [x] ✅ Fix a tiny batch (8 tensors); disable shuffling and augmentation
- [x] ✅ Train to overfit; log recon loss → near zero; save the in/out figure

## Recommended skill
custom; no skill fits — runs the implemented loop.

## Engagement Instructions
```
$ python -m vae.train --overfit --batch 8 --steps 500
# expect: recon loss → near 0; figures/vae_overfit.png shows near-identical in/out
```
