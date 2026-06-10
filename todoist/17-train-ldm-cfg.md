# 17 · Train Conditional LDM (CFG dropout)

## Reference while you do it
- 📄 Plan: plans/single-disease-ldm/plans/03-train-ldm-cfg.md

## Section context (paste into the Todoist section)
**Description:** Train one conditional LDM on single-disease latents with CFG label dropout, so one model supports both the `∅` and `normal` anchors at inference.
**Objective:** Get the unconditional null `ε(z,∅)` for free so H-anchoring needs no retraining.
**Goal:** A trained LDM checkpoint with CFG dropout, config recorded, both-disease hold-out intact.
**Verify (whole leaf):** `python -m ldm.train --config configs/ldm.yaml` → ckpts/ldm_*.pt; loss decreasing; peak VRAM < 12GB.

## Tasks (one at a time)
- [ ] Train conditional LDM (UNet ≤128ch, batch ≤8 + grad-accum, bf16) with random label dropout
- [ ] Checkpoint to `ckpts/`; record config
