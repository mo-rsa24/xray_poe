# 🎲 Train Conditional LDM (CFG dropout)

## Description
Train one conditional LDM on single-disease latents with classifier-free-guidance label
dropout, so a single model supports both the `∅` and `normal` anchors at inference.

## Purpose
CFG dropout gives the unconditional null `ε(z, ∅)`; together with the `normal` condition,
one model powers both H-anchoring anchors with no retraining.

## Goal
A trained LDM checkpoint with CFG dropout, config recorded, both-disease hold-out intact.

## Tasks
- [ ] ⚠️ Train conditional LDM (UNet ≤128ch, batch ≤8 + grad-accum, bf16) with random label dropout
- [ ] ⚠️ Checkpoint to `ckpts/`; record config

## Engagement Instructions
```
$ python -m ldm.train --config configs/ldm.yaml
# expect: checkpoint ckpts/ldm_*.pt; loss decreasing; peak VRAM < 12GB
```
