# 🎭 Heart-Mask Conditioning (conv_in + droppable mask)

## Background
Scope 1 tested whether clean data alone fixes cardio. If it nulled, the conditioning is the suspect. This plan adds a spatial heart-mask to the *same* UNet — the highest-leverage change — keeping PoE intact (mask is a CFG-droppable concat input inside `c_cardio`, never a separate stage).

## Description
Widen the UNet `conv_in` to concat a heart-mask channel at latent res; build the PSPNet→mask pipeline (scaled to a sampled CTR distribution); make the mask CFG-droppable (zeroed channels) mirroring the existing null-token drop.

## Purpose
Serves Objectives 1–2 and DoD items 1–2 — the conditioning interface, gated on Scope 1 failing.

## Goal
A training-ready UNet that accepts `[4+1, 128, 128]` (latent + heart-mask), drops the mask with prob p, and warm-starts equivalently to the Scope-1 checkpoint.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks
- [ ] ⚠️ **GATE: confirm Scope 1 nulled**
  - `⊳ decide:` read `additional_plans/vindr-transfer/results/vindr_transfer/eval/transfer_decision.md`; proceed ONLY if recovered=false (null/inconclusive). If Scope 1 recovered, STOP — this scope is moot.
  - `✓ verify:` a recorded go decision citing the Scope-1 yield.
- [ ] ⚠️ **Build the PSPNet heart-mask pipeline (sampled-CTR)**
  - `▶ prompt:` `Write scripts/build_heart_masks.py: run PSPNet (scripts/ctr_overlay.py seg) on each training film → a binary heart mask at latent res (128²); scale each toward a CTR sampled from the real-cardio CTR distribution (NOT a fixed 0.53 template — preserve diversity). Cache to data/latents/masks/<id>.pt.`
  - `✓ verify:` mask cache count == train count; sampled CTRs span the real distribution, not a spike.
- [ ] ⚠️ **Widen conv_in + add CFG-droppable mask channel**
  - `▶ prompt:` `In scripts/train_ldm.py widen the UNet conv_in from 4→5 input channels (latent + heart-mask), zero-initializing the new channel weights so a warm-started model is initially equivalent to Scope-1. Concat the mask to z_noisy. Extend the existing cfg dropout (train_ldm.py:~1068) to ALSO zero the mask channel with the same prob — the spatial null. Mask lives inside c_cardio only.`
  - `✓ verify:` a 1-step run consumes [B,5,128,128]; with mask zeroed + warm-start, loss ≈ the Scope-1 model's (equivalence holds).

## Recommended skill
— custom; core training-loop code (`scripts/train_ldm.py`).

## Engagement Instructions
```bash
$ ls data/latents/masks/*.pt | wc -l   # == train film count
$ $PY -c "import torch,glob; print(torch.load(sorted(glob.glob('data/latents/masks/*.pt'))[0]).shape)"  # (1,128,128)
$ $PY scripts/train_ldm.py --config configs/ldm_maskcond_smoke.yaml --max-steps 1   # consumes [B,5,128,128], no shape error
# expect: mask cache present; conv_in accepts 5ch; warm-start equivalence at step 0
```
