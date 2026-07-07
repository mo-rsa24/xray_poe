# Preprocess + Photometric Norm + Border Filter

**Description:** Standardize VinDr to NIH prep (incl. MONOCHROME1→2 normalization), confirm PA-only, border-visible filter, encode to latents.
**Objective:** a clean, projection-tagged, border-visible latent cache for Stage-1 (DoD 2).
**Goal:** data/latents/vindr/ of border-visible films, per-class N reported.
**Verify:** latent count = kept count; shape (4,128,128).
**▶ Recommended prompt:** /visualize-data-samples /home/molefe/data/vindr512

## Tasks
- Standardize images to the NIH preprocessing  [+ MONOCHROME1→2 normalization]
- Confirm PA-only + tag projection
- Apply the border-visible filter + map labels→classes (class 2 = 09's pick)
- Encode the kept set to latents through our VAE
