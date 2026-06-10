# 🏛️ SOTA VAE Architecture Selection

## Description
Research and decide the VAE architecture that compresses 512² grayscale X-rays to
a 4×128×128 latent (f=4). Target the AutoencoderKL family: ResNet blocks
(GroupNorm + SiLU), self-attention at the lowest resolution only, a KL bottleneck,
1-channel in/out, optional `tanh` output to [-1,1]. **No encoder→decoder skip
connections** — the latent must be a true information bottleneck or the LDM has
nothing to model.

## Purpose
Fix the architecture before implementation so the codec contract, the profiling
shapes, and the cost numbers are all stable. f=4 keeps per-cell compression light
→ cleaner ceiling, cheaper VAE, and preserved spatial separability for PoE.

## Goal
A written architecture-decision note: block design, channel schedule, downsample
stages, attention placement, latent shape (4×128×128), activation, output
nonlinearity, KL weight, and a parameter-count estimate — with the rejected
alternatives (f=8, encoder→decoder skips, DiT) and the reason each was rejected.

## Tasks
- [x] ✅ Prompt/research a SOTA grayscale VAE for 512²→4×128×128 (AutoencoderKL family); collect reference configs  ✓ verified (architecture-decision.md §6 provenance)
- [x] ✅ Decide the channel schedule + downsample stages (f=4: 512→256→128 spatial) and place attention at low-res only  ✓ verified (§3)
- [x] ✅ Confirm the latent is a pure bottleneck — NO encoder→decoder skips; pick a small KL weight  ✓ verified (§2)
- [x] ✅ Fix activations (GroupNorm + SiLU) and the output (optional `tanh`)  ✓ verified (§2)
- [x] ✅ Write the architecture-decision note with rationale + rejected alternatives  ✓ verified (architecture-decision.md; Todoist VAE 1 completed 2026-06-09)

## Recommended skill
▶ `/augment` ✅ — mine the AutoencoderKL/LDM literature for borrowable config; alt: `/pressure-test` the choice against the literature.

## Engagement Instructions
```
$ cat plans/vae/architecture-decision.md
# expect: a config table (channels, stages, attention, latent 4×128×128, activation,
#   KL weight) + param-count estimate + a "rejected alternatives" section
```
