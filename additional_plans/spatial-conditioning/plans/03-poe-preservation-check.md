# ⚖️ PoE Preservation Check

## Description
Verify the mask-conditioned model still composes as a symmetric product of experts: effusion-alone and unconditional sampling work with mask/haze zeroed, and the cardio∧effusion score-sum matches the Scope-1 composition path.

## Purpose
Serves Objective 1 and DoD item 4 — the guardrail that the conditioning didn't quietly break PoE.

## Goal
A recorded check: effusion-alone + unconditional sample correctly with zeroed mask channels; PoE(cardio,effusion) = ε(∅)+Σ(ε(cᵢ)−ε(∅)) numerically intact.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks
- [ ] ⚠️ **Effusion-alone + unconditional sample with mask zeroed**
  - `▶ prompt:` `Sample class=effusion and the unconditional/null with the heart-mask channel zeroed (effusion doesn't carry a heart mask). Confirm both produce sane CXRs — the spatial-null path works.`
  - `✓ verify:` effusion-alone renders effusion; unconditional renders a plausible CXR; neither needs a mask.
- [ ] ⚠️ **Score-sum equality vs Scope-1 composition**
  - `▶ prompt:` `For a fixed seed/latent/timestep, compute ε_poe = ε(∅)+w(ε(c_cardio)−ε(∅))+w(ε(c_effusion)−ε(∅)) on the mask-conditioned model (mask zeroed in the cardio pass for a control, supplied for the test) and confirm the composition machinery is the same score-sum as Scope-1's cfg_compose — no cascade introduced.`
  - `✓ verify:` composition is the documented score-sum; mask supplied only strengthens the cardio score, doesn't change the operator.

## Recommended skill
— custom; `scripts/gate3_yield.py` / `cfg_compose` path.

## Engagement Instructions
```bash
$ grep -iE "effusion|uncond|score.?sum|poe|pass" results/spatial_cond/poe_check.md
# expect: effusion-alone + unconditional OK with mask zeroed; score-sum operator unchanged from Scope-1
```
