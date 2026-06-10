# 🗜️ Train the Shared VAE on the Pod

## Background
Runs after `02-corpus-transfer`. The VAE code, training loop, and eval/ceiling
metrics were written and unit-tested locally (the `vae` scope) and proven to run
end-to-end on noise data before renting. This plan runs the *real-data* train on
the pod — the first time the VAE sees the actual corpus.

## Description
On the pod, run the VAE overfit-sanity gate on a tiny real batch first; only once
it passes, launch the full real-data VAE train, checkpointing to `ckpts/` on the
volume and logging recon loss + the SSIM/LPIPS ceiling-check.

## Purpose
The shared VAE is the codec the whole pipeline depends on — the LDM trains in its
latent space. Running the overfit gate before the long train is the cheap
correctness check that the code is wired right on real data before spending
GPU-hours. Serves Objective 3 and Definition-of-Done #3.

## Goal
A trained shared-VAE checkpoint plus logs in `ckpts/` on the volume, produced
after the overfit-sanity gate passed on real data, with the recon/ceiling metric
recorded.

## Tasks
- [ ] ⚠️ Run the VAE overfit-sanity gate on a small real-data batch; confirm recon loss → near zero before proceeding
- [ ] ⚠️ Launch the full real-data VAE train (`bash scripts/train_vae.sh`); checkpoint periodically to `ckpts/` on the volume
- [ ] ⚠️ Watch the run: recon loss decreasing, no OOM/crash, peak VRAM within the provisioned GPU
- [ ] ⚠️ Run the recon/ceiling-check on a held-out real batch; record SSIM/LPIPS into the run log

## Recommended skill
— custom; no skill fits (runs the locally-built `vae` loop on rented hardware).
   — alt: `/analyze-run` to read the W&B/training curve once the run is logging.

## Engagement Instructions
```
# DO THIS — on the pod, gate first, then the real train
$ bash scripts/train_vae.sh --overfit          # gate: tiny real batch
# expect: recon loss → near 0; figures/vae_overfit_real.png near-identical in/out  → gate PASSED
$ bash scripts/train_vae.sh --config configs/vae.yaml   # full real-data train

# GET THAT — checkpoint + logs on the volume, ceiling recorded
$ ls -t ckpts/vae_*.pt | head -1               # expect: a saved VAE checkpoint
$ python -m vae.eval --ckpt ckpts/vae_*.pt --ceiling   # expect: SSIM/LPIPS reported on held-out real data
$ tail runs/vae_train.log                      # expect: recon loss decreased; peak VRAM < provisioned GPU
```
