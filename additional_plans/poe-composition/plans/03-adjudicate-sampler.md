# ⚖️ Adjudicate — operator vs sampler (corrected sampler on the frozen CXR PoE)

## Description
Separate "the additive operator is wrong" from "the ancestral sampler is wrong" by running a corrected
(Metropolis/ULA) sampler on the frozen CXR PoE and comparing co-presence yield to the ancestral baseline. Model:
**frozen v4-60k** (`ckpts/vae_v2_ldm_v4/model_ema_step0060000`), no retrain. Read ASYMMETRICALLY: ε-prediction
admits only the *unadjusted* sampler cleanly; a Metropolis-adjusted version needs an approximate energy
`f_θ = −½‖s_θ‖²`, so its null is softer than its positive. Runs REGARDLESS of O1 (it's paper-critical — it's the
answer to "did you try MCMC?").

## Purpose
Serves O3 and DoD-2 (resolves the pre-registered fork). Together with the toy (plan 02) it is the paper's
adjudication of the operator/sampler loci; it also feeds the O1 bottleneck verdict's "sampler" locus.

## Goal
Corrected-sampler co-presence yield on CXR reported vs the ancestral baseline, with chain diagnostics
(acceptance rate, autocorrelation), resolving the fork to Abstract-N (support-limited: correction doesn't help)
or Abstract-M (sampler-fixable: correction recovers co-presence).

## Tasks
- [ ] ⚠️ **Implement the unadjusted annealed-Langevin (ULA) corrector** on the composed score — k≈2–5 corrector
  steps per denoise step on the frozen model. Measure yield with `scripts/gate3_yield.py`.
  - [ ] ⚠️ Remedy: if ULA destabilizes (samples blow up / go off-manifold), reduce step size + k and report the
    stable setting; a diverging chain is a diagnostic, not a pass.
- [ ] ⚠️ **Add the approximate-energy Metropolis correction** — `f = −½‖s‖²` + the line-integral MH acceptance
  rule (arXiv 2307.14012); run BOTH corrected samplers and compare. Agreement (both recover the toy, both behave
  the same on CXR) makes the CXR verdict robust to sampler choice.
  - [ ] ⚠️ Remedy: if the two corrected samplers DISAGREE on CXR, report both + their chain diagnostics and mark
    the verdict provisional (do not pick a winner silently); escalate to the parked HMC/energy-distillation only
    if this ambiguity blocks the paper.
- [ ] ⚠️ **Resolve the pre-registered fork** — apply the reading rule fixed in plan 02: correction recovers
  co-presence on CXR → Abstract-M; correction fails → Abstract-N (failure survives correct sampling → localizes
  to target support, locus 4).

## Engagement Instructions
```bash
# corrected samplers on frozen v4-60k PoE; yield re-measured with the shared instrument:
python scripts/gate3_yield.py --ckpt ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors \
  --vae ckpts/vae_v2/vae_v2_step0075000.pt --scale-factor data/latents_v2/joint_perch/scale_factor.pt \
  --n 2000 --out results/poe_composition/sampler_corrected/   # + the corrector hook
```
Verify: corrected-sampler yield reported vs the ancestral 2.6% baseline with acceptance-rate + autocorrelation;
both corrected samplers compared; the fork is resolved to N or M with the reading rule cited. The toy positive
control (plan 02) passes for the SAME corrected sampler (that's what makes the CXR cell legible).

## Recommended skill
— custom; sampler implementation on the frozen model + `scripts/gate3_yield.py` for the yield read.
