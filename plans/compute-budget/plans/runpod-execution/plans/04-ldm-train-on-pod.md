# 🎲 Train the Single-Disease LDM on the Pod

## Background
Runs after `03-vae-train-on-pod` — needs the frozen VAE checkpoint to encode
latents. The LDM code, CFG-dropout training loop, latent-prep, and FID eval were
written and unit-tested locally (the `single-disease-ldm` scope) and proven on the
overfit gate. This plan runs the real-data LDM train on the pod.

## Description
On the pod, encode the single-disease images to latents with the frozen VAE, run
the LDM overfit-sanity gate on a handful of latents, then launch the full
conditional LDM train (classifier-free-guidance label dropout), checkpointing to
`ckpts/`. Confirm the both-disease hold-out stays empty throughout.

## Purpose
The single-disease LDM is the model the composition experiments build on; it must
be trained on real latents from the shared VAE. Gating on overfit before the long
train is the same cheap correctness check applied to the diffusion loop. Serves
Objective 4 and Definition-of-Done #4.

## Goal
A trained single-disease LDM checkpoint plus logs in `ckpts/` on the volume,
produced after the overfit-sanity gate passed, config recorded, both-disease
hold-out verified empty.

## Tasks
- [ ] ⚠️ Encode single-disease images to latents with the frozen VAE (`ckpts/vae_*.pt`); cache by condition on the volume
- [ ] ⚠️ Run the LDM overfit-sanity gate on a handful of latents; confirm they regenerate before proceeding
- [ ] ⚠️ VERIFY the both-disease hold-out is empty in the training latents (count == 0)
- [ ] ⚠️ Launch the full conditional LDM train with CFG label dropout (`bash scripts/train_ldm.sh`); checkpoint to `ckpts/`; record config
- [ ] ⚠️ Compute single-disease FID vs real; save a sample grid into the run log

## Recommended skill
— custom; no skill fits (runs the locally-built `ldm` loop on rented hardware).
   — alt: `/analyze-run` to read the training curve / FID once logging.

## Engagement Instructions
```
# DO THIS — on the pod: latents, gate, then the real train
$ python -m ldm.prepare_latents --vae ckpts/vae_*.pt --out /workspace/data/latents/
# expect: per-condition latent counts printed; assert both-disease count == 0
$ bash scripts/train_ldm.sh --overfit          # gate: a few latents
# expect: sampled latents decode to the memorized images  → gate PASSED
$ bash scripts/train_ldm.sh --config configs/ldm.yaml     # full real-data CFG train

# GET THAT — checkpoint + logs on the volume, FID recorded
$ ls -t ckpts/ldm_*.pt | head -1               # expect: a saved LDM checkpoint
$ python -m ldm.evaluate --ckpt ckpts/ldm_*.pt --fid   # expect: single-disease FID reported; both null modes callable
$ tail runs/ldm_train.log                      # expect: loss decreased; peak VRAM < provisioned GPU
```
