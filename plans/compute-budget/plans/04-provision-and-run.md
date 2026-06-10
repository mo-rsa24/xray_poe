# 🚀 Provision & Run on RunPod

## Background
Gated on `03`'s go decision. This is the execution that turns the costed plan
into trained checkpoints — large enough to become its own sub-scope.

## Description
Once the cost memo greenlights a config, provision it on RunPod and run the
trains end-to-end: environment, corpus transfer, VAE then LDM training,
checkpoint retrieval, teardown.

## Purpose
Converts the budget decision into the actual shared-VAE and LDM checkpoints the
rest of Paper3 depends on. Serves the broadened Objective 7 / DoD 6.

## Goal
Provisioned pod at the chosen config; corpus on the volume; VAE and LDM trains
run to their gates; checkpoints retrieved to `ckpts/`; pod torn down to stop billing.

## Tasks
- [ ] ⚠️ Provision the chosen RunPod config and run the VAE + LDM trains end-to-end  → decomposed: see `plans/runpod-execution/MASTER_PLAN.md`

## Engagement Instructions
```
$ ls ckpts/   # expect vae-*.ckpt and ldm-*.ckpt retrieved from the pod
# RunPod console shows the pod terminated (no ongoing charge)
```

## Recommended skill
— custom; no skill fits (infra/ops execution); will be decomposed into its own sub-scope.
