# 04 · Train the single-disease LDM on the pod

[⌂ Index](00-INDEX.md) · [← prev 03](03-vae-train-on-pod.md) · [next → 05](05-retrieve-teardown-reconcile.md)

## Reference while you do it
- 📄 Plan: plans/compute-budget/plans/runpod-execution/plans/04-ldm-train-on-pod.md

## Section context (paste into the Todoist section)
**Description:** On the pod, encode single-disease images to latents with the frozen VAE, run the LDM overfit-sanity gate on a handful of latents, then launch the full conditional LDM train (CFG label dropout), checkpointing to `ckpts/`. Confirm the both-disease hold-out stays empty.
**Objective:** Produce the single-disease LDM the composition experiments build on, trained on real latents from the shared VAE, gating on overfit before the long train.
**Goal:** A trained single-disease LDM checkpoint plus logs in `ckpts/` on the volume, produced after the overfit gate passed, config recorded, both-disease hold-out verified empty.
**Verify (whole leaf):** `python -m ldm.prepare_latents --vae ckpts/vae_*.pt --out /workspace/data/latents/` → per-condition counts, both-disease count == 0; `bash scripts/train_ldm.sh --overfit` → memorized latents regenerate (gate PASSED); `ls -t ckpts/ldm_*.pt | head -1` → a saved checkpoint; `python -m ldm.evaluate --ckpt ckpts/ldm_*.pt --fid` → FID reported, both null modes callable.
**▶ Recommended prompt:** — custom; no skill fits (runs the locally-built `ldm` loop on rented hardware). alt: `/analyze-run` to read the training curve / FID.

## Tasks (one at a time)
- [ ] Encode single-disease images to latents with the frozen VAE (`ckpts/vae_*.pt`); cache by condition on the volume
- [ ] Run the LDM overfit-sanity gate on a handful of latents; confirm they regenerate before proceeding
- [ ] VERIFY the both-disease hold-out is empty in the training latents (count == 0)
- [ ] Launch the full conditional LDM train with CFG label dropout (`bash scripts/train_ldm.sh`); checkpoint to `ckpts/`; record config
- [ ] Compute single-disease FID vs real; save a sample grid into the run log
