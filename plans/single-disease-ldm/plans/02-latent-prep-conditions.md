# 🧬 Latent Prep & Condition Set

## Description
Encode single-disease images to latents with the frozen VAE and define the conditioning
set (normal, cardiomegaly, effusion, plus the control pair's diseases).

## Purpose
Clean per-condition latents are the LDM's training data; the condition set must cover every
disease we will later compose.

## Goal
A latent cache keyed by condition, with the both-disease hold-out verified empty.

## Tasks
- [ ] ⚠️ Encode single-disease images to latents (frozen VAE); cache by condition
- [ ] ⚠️ Define the condition set; VERIFY zero both-disease images present (the hold-out)

## Engagement Instructions
```
$ python -m ldm.prepare_latents --vae ckpts/vae_*.pt --out data/latents/
# expect: per-condition latent counts; assert both-disease count == 0
```
