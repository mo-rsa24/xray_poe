# 04 · Shared stretch-512 preprocessing pipeline

[⌂ Index](00-INDEX.md) · [← prev 03](03-four-group-partition.md) · [next → 05](05-integrity-manifest.md)

## Reference while you do it
- 📄 Plan: plans/data-foundation/plans/06-shared-preprocessing.md

## Section context (paste into the Todoist section)
**Description:** One preprocessing function both datasets pass through — load grayscale → per-image min–max → stretch-resize to 512² → [-1,1] — so the model can't read dataset identity off intensity or aspect ratio. Stretch (not crop) because NIH ships pre-squished PNGs and cropping would cut basal effusion / apical pneumothorax.
**Objective:** A validated, shared loader is the on-the-fly path every downstream scope (VAE curation, LDM latent prep) reads through.
**Goal:** `data/preprocess.py` with one `preprocess(path, target=512)` function, validated on a sample panel from BOTH datasets — ranges match [-1,1], shape (channels,512,512), no aspect/inversion surprises.
**Verify (whole leaf):** `python -m data.preprocess --selftest --nih data/nih/images --vindr data/vindr/images --panel out/preprocess_panel.png`; each output `(1,512,512)`, min≈-1, max≈+1, float32 — identical convention regardless of source; NIH & VinDr tiles indistinguishable in framing.
**▶ Recommended prompt:** `/preprocess-vision data/nih data/vindr` ✅ — vision preprocessing scaffold (resize / normalize / sanity panel). alt: `/visualize-data-samples` for the contact sheet; the stretch-vs-crop + per-image min–max policy is custom.

## Tasks (one at a time)
- [ ] Implement `preprocess(path, target=512, channels=1)`: grayscale load → per-image min–max → stretch-resize to 512² → [-1,1]; `channels=3` replicate path for a pretrained-VAE warm-start
- [ ] Run it on a mixed sample panel (NIH + VinDr); save a contact sheet to eyeball aspect/intensity consistency
- [ ] Assert output invariants: dtype float32, shape (channels,512,512), range ⊆ [-1,1], finite; matching stat distribution across both sources
- [ ] Document it as the canonical on-the-fly loader for downstream scopes (one import, both datasets)
