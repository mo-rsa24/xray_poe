# 🔬 LDM Overfit Sanity

## Description
Before full training, overfit the conditional LDM to a handful of single-disease latents
and confirm it regenerates them.

## Purpose
Proves the diffusion training loop, conditioning, and sampler are correct before any GPU-hours.

## Goal
The LDM memorizes a few latents and regenerates them recognizably.

## Tasks
- [ ] ⚠️ Fix a handful of single-disease latents; train to overfit
- [ ] ⚠️ Sample with their conditions; confirm regeneration

## Engagement Instructions
```
$ python -m ldm.train --overfit --n 8 --steps 1000
# expect: sampled latents decode to the memorized images; figures/ldm_overfit.png
```
