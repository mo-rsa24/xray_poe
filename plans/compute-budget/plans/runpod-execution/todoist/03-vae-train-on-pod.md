# 03 · Train the shared VAE on the pod

[⌂ Index](00-INDEX.md) · [← prev 02](02-corpus-transfer.md) · [next → 04](04-ldm-train-on-pod.md)

## Reference while you do it
- 📄 Plan: plans/compute-budget/plans/runpod-execution/plans/03-vae-train-on-pod.md

## Section context (paste into the Todoist section)
**Description:** On the pod, run the VAE overfit-sanity gate on a tiny real batch first; only once it passes, launch the full real-data VAE train, checkpointing to `ckpts/` on the volume and logging recon loss + the SSIM/LPIPS ceiling-check.
**Objective:** Produce the shared VAE codec the whole pipeline depends on (the LDM trains in its latent space), gating on overfit before spending GPU-hours.
**Goal:** A trained shared-VAE checkpoint plus logs in `ckpts/` on the volume, produced after the overfit-sanity gate passed on real data, with the recon/ceiling metric recorded.
**Verify (whole leaf):** `bash scripts/train_vae.sh --overfit` → recon loss → near 0, near-identical in/out (gate PASSED); `ls -t ckpts/vae_*.pt | head -1` → a saved checkpoint; `python -m vae.eval --ckpt ckpts/vae_*.pt --ceiling` → SSIM/LPIPS reported; `tail runs/vae_train.log` → loss decreased, peak VRAM < provisioned GPU.
**▶ Recommended prompt:** — custom; no skill fits (runs the locally-built `vae` loop on rented hardware). alt: `/analyze-run` to read the W&B/training curve.

## Tasks (one at a time)
- [ ] Run the VAE overfit-sanity gate on a small real-data batch; confirm recon loss → near zero before proceeding
- [ ] Launch the full real-data VAE train (`bash scripts/train_vae.sh`); checkpoint periodically to `ckpts/` on the volume
- [ ] Watch the run: recon loss decreasing, no OOM/crash, peak VRAM within the provisioned GPU
- [ ] Run the recon/ceiling-check on a held-out real batch; record SSIM/LPIPS into the run log
