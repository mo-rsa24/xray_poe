# 16 · Latent Prep & Condition Set

## Reference while you do it
- 📄 Plan: plans/single-disease-ldm/plans/02-latent-prep-conditions.md

## Section context (paste into the Todoist section)
**Description:** Encode single-disease images to latents (frozen VAE); define the condition set; verify the both-disease hold-out is empty.
**Objective:** Provide clean per-condition training data covering every disease to be composed.
**Goal:** A latent cache keyed by condition, both-disease hold-out verified empty.
**Verify (whole leaf):** `python -m ldm.prepare_latents --vae ckpts/vae_*.pt --out data/latents/` → per-condition counts; assert both-disease count == 0.

## Tasks (one at a time)
- [ ] Encode single-disease images to latents (frozen VAE); cache by condition
- [ ] Define the condition set; VERIFY zero both-disease images present (the hold-out)
