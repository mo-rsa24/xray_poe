# 🔬 Diagnose — localize the co-presence bottleneck (codec / data / operator / sampler / support)

## Description
Four cheap, independent probes run in a FIXED ORDER so each result narrows blame before the next is paid for.
Answers: is the co-presence null the **codec** (entangled latent), the **data/ecology** (the comorbid region is
rare or unreachable), the **operator** (additive CFG), the **sampler** (ancestral), or **target support** (the
conjunction isn't learnable from the data)? Model: probes run on the vae-v2 / v4-60k stack (`data/latents_v2`,
`ckpts/vae_v2*`); the base-rate is model-agnostic (real labels). This is the GATE for the whole scope.

## Purpose
Serves O1 and DoD-1. O4/O5 (improve/utility) only fire if these probes say there is something to combine; O2/O3
read differently depending on which locus is live; the parked VAE re-disentangle retrain fires **only** if the
separability probe demands it.

## Goal
A one-page bottleneck verdict attributing the null to exactly one locus (or naming the residual), each probe
reporting its number with a significance/uncertainty band.

## Tasks
- [ ] ⚠️ **Base-rate: real cardio∧effusion co-occurrence in NIH+VinDr labels** (one query) — makes locus-4
  (target support) a measured fact, not borrowed theory.
  - [ ] ⚠️ Remedy: if the uncertain/negative-label policy is ambiguous, report both strict and lenient rates and
    state which the downstream reading uses.
- [ ] ⚠️ **Coverage / density on the real comorbid manifold** — prdc k-NN balls around real comorbid films in an
  xrv embedding (NEVER pixels); report coverage (fraction of real-comorbid neighborhoods reached by ≥1 generated
  comorbid) + density (do generated composites pile into one corner).
  - [ ] ⚠️ Remedy: if k-NN is unstable across feature space, report coverage at two embeddings (xrv + z) and flag
    disagreement rather than trusting one.
- [ ] ⚠️ **Cardio-alone ceiling probe** — sample cardio-alone (single condition), measure heart size with
  PSPNet-CTR. Disambiguates "bad combiner" (renders alone, not composed) vs "empty ingredient" (under-renders
  even alone). Non-negotiable, informs how every operator result is read.
  - [ ] ⚠️ Remedy: reconcile against the known v4-60k cardio-alone CTR (≈0.48–0.54 range from prior runs); if the
    fresh number disagrees, re-run at n≥200 before trusting.
- [ ] ⚠️ **Separability / CAV probe (pixels → z → h-space)** — linear CAV probe (cardio-only vs no-finding;
  effusion-only vs no-finding) at three stages; report per-stage accuracy, CAV-cosine (entanglement geometry),
  CAV weight-mass per latent channel (do they pile into the known dead channels), and a **permutation null** so
  "not separable" is a rejected hypothesis. Re-point the existing h-space probe (`train_ldm.py:550`, currently
  downstream/label-conditioned — wrong stage) to `z` and pixels.
  - [ ] ⚠️ Remedy: if logistic-regression CAVs are unstable across random-negative draws, switch to a
    difference-of-means (h-space) direction estimator and report both.
- [ ] ⚠️ **Write the bottleneck verdict** — attribute to one locus (or residual) with each probe's number + band;
  set the O4/O5 gates and the VAE-retrain park condition.

## Engagement Instructions
```bash
# base-rate (model-agnostic): co-occurrence over the label CSVs + Wilson band
python - <<'PY'  # pandas over data/splits/*.csv → cardio∧effusion count / N, Wilson CI
PY
# coverage/density: prdc on xrv embeddings of real-comorbid vs generated composites (never pixels)
# ceiling: cardio-alone sample (scripts/gate3_yield.py-style single-condition) + PSPNet-CTR
# separability: re-point the train_ldm.py:550 probe to z + pixels; ≥100 label-shuffle permutation null → p
```
Verify: a verdict file names exactly one locus with all four numbers + bands; the permutation-null p is reported;
the O4/O5 gate decision (proceed / park) is stated.

## Recommended skill
— custom; project scripts (`gate3_yield.py`, the re-pointed `train_ldm.py:550` probe).
   alt: `/data-distributions` for the base-rate label-frequency view.
