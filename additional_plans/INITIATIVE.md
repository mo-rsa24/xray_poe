# Track B — VinDr-Mix Cardio Improvement Initiative

> Parent index for the parallel improvement scopes. Created 2026-07-02 when the plan tree
> split into two tracks (see top-level `MASTER_PLAN.md` → "Scope status").

## What this is
One combined effort to **mix VinDr deeper and lift arm1's cardio results** beyond the frozen
`gtr6kqpk` checkpoint. It is a **separate initiative, parallel to and NOT blocking the arm1 paper.**
The arm1 paper (Track A) ships on the frozen `ckpts_arm1_cleancardio/model_step0100000` regardless of
whatever this track produces.

## Why it exists
Arm1 (Track A) already mixed a first pass of VinDr cardio (NIH 854 basal-clean + VinDr-kept ≈ 1,725,
from scratch) and **plateaued at the diagnostic line** — cardio CTR 0.499 vs real 0.535, PoE Gate-3
yield 0.125. The arm1 verdict named the two levers that would push past the plateau, and they are
exactly the two scopes below:
1. **Unique-example scarcity** → deeper external VinDr mix.
2. **Basal-haze occlusion + coarse class-embed entanglement** → a sharper codec (and, if needed,
   spatial conditioning).

Full arm1 verdict: `../results/section4/analysis_gtr6kqpk_cardio_retrain.md`.

## Member scopes
- **`vae-v2/`** — the live vehicle. Perceptual (LPIPS + PatchGAN) fine-tune of the VAE on a combined
  NIH+VinDr border-visible diet, then retrain the LDM from scratch in the new latent space. Targets
  synth-cardio sensitivity ≥ 0.80 and PoE Gate-3 yield ≥ 0.30.
- **`vindr-transfer/`** — the VinDr **data-prep** feeder (acquire, projection-filter, roundtrip-gate,
  morbidity-criteria). Its original two-stage *transfer* thesis is retired; its 6,671 border-visible
  films now feed the vae-v2 diet. Opacity-instrument + composition-operators sub-scopes survive as
  standalone, model-agnostic eval tooling.
- **`spatial-conditioning/`** — gated follow-up: heart-mask + haze-map conditioning, run only if the
  vae-v2 joint-from-scratch LDM still misses the Gate-3 yield bar.

## Boundary with Track A
- Track A = **analysis only** on the frozen `gtr6kqpk` cardio expert; leads with the mechanistic null.
- Track B = **new training** allowed; chases the utility win Track A explicitly does not require.
- No Track A deliverable waits on a Track B result.
