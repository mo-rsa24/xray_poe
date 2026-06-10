# 15 · LDM Overfit Sanity

## Reference while you do it
- 📄 Plan: plans/single-disease-ldm/plans/01-overfit-sanity.md

## Section context (paste into the Todoist section)
**Description:** Overfit the conditional LDM to a handful of single-disease latents and confirm regeneration, before full training.
**Objective:** Prove the diffusion loop, conditioning, and sampler before GPU-hours.
**Goal:** The LDM memorizes a few latents and regenerates them recognizably.
**Verify (whole leaf):** `python -m ldm.train --overfit --n 8 --steps 1000` → sampled latents decode to the memorized images; figures/ldm_overfit.png.

## Tasks (one at a time)
- [ ] Fix a handful of single-disease latents; train to overfit
- [ ] Sample with their conditions; confirm regeneration
