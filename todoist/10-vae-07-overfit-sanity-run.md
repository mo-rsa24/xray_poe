# VAE 7 · Overfit Sanity Run (first pass)

## Reference while you do it
- 📄 Plan: plans/vae/plans/07-overfit-sanity-run.md

## Section context (paste into the Todoist subtask)
**Description:** Drive the VAE to reconstruct a fixed 8-image (synthetic/noise) batch to near-zero error within a few hundred steps — first confirmation the loop, loss, and data path are wired correctly.
**Objective:** Catch a broken loop in minutes; cheapest gate; must pass before any longer run. Confirms the code, not the data.
**Goal:** Recon error on a fixed 8-image batch driven to near zero; in/out figure near-identical.
**Verify (whole leaf):** `python -m vae.train --overfit --batch 8 --steps 500` → recon loss ~0; figures/vae_overfit.png near-identical in/out.

## Tasks (one at a time)
- [ ] Fix a tiny batch (8 tensors); disable shuffle + augmentation
- [ ] Train to overfit; log recon loss → near zero; save in/out figure
