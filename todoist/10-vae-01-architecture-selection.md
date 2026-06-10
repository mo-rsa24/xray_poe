# VAE 1 · SOTA Architecture Selection

## Reference while you do it
- 📄 Plan: plans/vae/plans/01-architecture-selection.md

## Section context (paste into the Todoist subtask)
**Description:** Research/prompt and decide the VAE architecture for 512² grayscale → 4×128×128 (f=4): AutoencoderKL family — ResNet blocks (GroupNorm+SiLU), attention at low-res only, KL bottleneck, 1-ch in/out, optional tanh out. NO encoder→decoder skips.
**Objective:** Fix the architecture before implementation so the codec contract, profiling shapes, and cost numbers are stable.
**Goal:** A written architecture-decision note (blocks, channels, stages, attention, latent shape, activation, KL weight, param estimate) + rejected alternatives (f=8, enc→dec skips, DiT).
**Verify (whole leaf):** `cat plans/vae/architecture-decision.md` → config table + param estimate + rejected-alternatives section.
**▶ Recommended prompt:** `/augment` ✅ — mine the AutoencoderKL/LDM literature; alt: `/pressure-test`.

## Tasks (one at a time)
- [ ] Prompt/research a SOTA grayscale VAE for 512²→4×128×128; collect reference configs
- [ ] Decide channel schedule + downsample stages (f=4); attention at low-res only
- [ ] Confirm pure bottleneck — NO enc→dec skips; pick small KL weight
- [ ] Fix GroupNorm+SiLU + optional tanh output
- [ ] Write the decision note with rationale + rejected alternatives
