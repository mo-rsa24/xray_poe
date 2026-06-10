# 🔁 Shared Stretch-512 Preprocessing Pipeline

## Background
NIH (train) and VinDr (eval) arrive as differently-shaped PNGs — NIH pre-squished
1024×1024 8-bit, VinDr aspect-preserving from 16-bit DICOM. A single shared
preprocessing function must turn both into identical-convention 512×512 tensors,
so the model cannot read dataset identity off intensity or aspect ratio (a real
risk for a *generative* model, which will happily encode any systematic
cross-dataset difference).

## Description
One function both datasets pass through, producing model-ready tensors.

1. Pipeline: load grayscale → per-image min–max to [0,1] → stretch-resize to
   512×512 (bilinear) → scale to [-1,1] → single channel (3-channel replicate only
   when warm-starting a pretrained VAE).
2. **Stretch, not crop** — deliberate: NIH only ships pre-squished square PNGs
   (forces stretch), and cropping would cut basal effusion / apical pneumothorax.
   CXR aspect ratios are near-1:1, so the distortion is mild.
3. **Per-image min–max** equalizes NIH's 8-bit vs VinDr's windowed intensities,
   removing a dataset-identity cue.
4. 512 is pinned by the adopted 4×128×128 latent at f=4 (see the latent-shape
   decision); the pipeline targets 512 but takes `target` as a parameter so
   768/1024 stay reachable if that decision reopens.

## Purpose
A validated, shared preprocessing path is the on-the-fly loader every downstream
scope reads through (VAE curation, LDM latent prep). Serves Objective 2,
Definition-of-Done #2, and the loader half of #6.

## Goal
A `data/preprocess.py` exposing one `preprocess(path, target=512, channels=1)`
function, validated on a sample panel drawn from BOTH datasets — intensity ranges
match ([-1,1]), output shape (channels,512,512), no aspect/inversion surprises —
with the panel saved for eyeballing.

## Tasks
- [ ] ⚠️ Implement `preprocess(path, target=512, channels=1)`: grayscale load → per-image min–max → stretch-resize to 512² → [-1,1]; `channels=3` replicate path for a pretrained-VAE warm-start
- [ ] ⚠️ Run it on a mixed sample panel (NIH + VinDr); save a contact sheet to eyeball aspect/intensity consistency
- [ ] ⚠️ Assert output invariants: dtype float32, shape (channels,512,512), range ⊆ [-1,1], finite; matching stat distribution across both sources
- [ ] ⚠️ Document it as the canonical on-the-fly loader for downstream scopes (one import, both datasets)

## Recommended skill
▶ `/preprocess-vision data/nih data/vindr` ✅ — vision preprocessing scaffold (resize / normalize / sanity panel).
   alt: `/visualize-data-samples` for the eyeball contact sheet; the stretch-vs-crop + per-image min–max policy is custom.

## Engagement Instructions
```
# DO THIS — build the shared loader + the mixed-source validation panel
$ python -m data.preprocess --selftest --nih data/nih/images --vindr data/vindr/images --panel out/preprocess_panel.png

# GET THAT — invariants hold across BOTH datasets
$ python -c "
import glob
from data.preprocess import preprocess
paths = ['data/nih/images/00000001_000.png', sorted(glob.glob('data/vindr/images/*.png'))[0]]
for p in paths:
    t = preprocess(p)
    print(p, tuple(t.shape), round(float(t.min()),3), round(float(t.max()),3), t.dtype)
"
# expect each: shape (1,512,512), min≈-1.0, max≈+1.0, float32 — identical convention regardless of source
$ open out/preprocess_panel.png   # NIH and VinDr tiles indistinguishable in framing / intensity convention
```
