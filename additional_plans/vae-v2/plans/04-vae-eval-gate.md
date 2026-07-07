# 🚦 VAE-v2 Eval Gate — Go / No-Go Decision

## Background
Before re-encoding 70k NIH latents and retraining the LDM, we need to confirm the fine-tuned VAE-v2 actually improved perceptual quality without regressing the spatial fidelity gate. The current `vae_step0025000.pt` already passes |ΔCTR| < 0.01 (measured at 0.009); the risk is that LPIPS + discriminator pressure sharpens texture at the cost of spatial accuracy. This plan runs the three-gate evaluation on the held-out border-visible set and records an explicit go/no-go.

## Description
Run a batch eval script on the best vae-v2 checkpoint (step 75k or the best mid-run checkpoint per W&B LPIPS curve) against the held-out border-visible val split. Compute: median |ΔCTR|, lower-zone SSIM, full-image LPIPS, and cardiac-border crop LPIPS. Compare to baseline (`vae_step0025000.pt`) on the same set. Record verdict. If any gate fails, return to leaf 03 with an adjusted recipe (see contingency below).

## Purpose
Serves vae-v2 Objective 4 and DoD item 4. Hard gate: leaf 05 (re-encode) must not start until this verdict is `GO`. Also blocks leaf 06 (LDM retrain) and leaf 08 (downstream eval).

## Goal
A written verdict file `results/vae_v2/eval_gate/verdict.md` with per-gate pass/fail, comparison to `vae_step0025000.pt` baseline, overall GO/NO-GO, and — if NO-GO — the specific recipe adjustment for the leaf 03 retry.

## ⚠️ GATE GAP — added Gate 5 (2026-07-01)

All four original gates measure **reconstruction** quality (encode→decode→compare), which
uses the posterior **mean μ** — and those genuinely PASS (LPIPS 0.022, |ΔCTR| 0.003). But
none checked whether the VAE's **latent is usable for LDM training**. It is not, when
*sampled*: VAE-v2 is posterior-collapsed (σ ≈ 0.999 ≈ prior, μ_std ≈ 0.187 → **signal
fraction 3.4%**, vs V1 21%). This gap let two LDM retrains (mzd3jwy9, gtsw5bq7) burn on
sampled latents before the collapse surfaced. **Gate 5 below closes it** — any future
VAE fine-tune must clear it before spending a re-encode. The GO verdict stands for
reconstruction, but the LDM must train on **μ** (`encode_moments`), not sampled latents
(plan 05 CORRECTION). See memory `vindr-latent-scale-bug`.

## Tasks

- [x] ✅ **Write `scripts/eval_vae_v2.py` — per-gate eval script**  ✓ verified (`scripts/eval_vae_v2.py` 230 lines; smoke-test on 1 film passed; CSV columns confirmed: patient_id, source, label, checkpoint, ctr_orig, ctr_recon, delta_ctr, lower_zone_ssim, lpips_full, lpips_cardiac_crop; 1-film preview: all 4 gates pass on vae_v2_step0075000)
  - **Objective:** a single-run script that loads a checkpoint, runs the held-out val set through the codec, and computes all four gates.
  - **Inputs:** checkpoint path (arg), val manifest (10% hold-out from `data/vae_v2_diet/train_manifest.csv`), original NIH images.
  - **Gates to compute:**
    1. `delta_ctr_median`: encode→decode each film; run `ctr_overlay.py` on original + reconstruction; |ΔCTR| = |CTR_orig − CTR_recon|; report median. **Target: < 0.01 (regression hold).**
    2. `lower_zone_ssim`: compute SSIM on the bottom third (rows 340–512 of 512px image) of each original vs reconstruction; report median. **Target: ≥ 0.88.**
    3. `lpips_full`: LPIPS (AlexNet) on full 512×512 original vs recon; report median. **Target: < 0.10 (primary improvement gate).**
    4. `lpips_cardiac_border`: LPIPS on a 256×256 crop centred on the cardiac border (use heart mask centroid from PSPNet); report median. **Target: < 0.08.**
  - **Baseline comparison:** run same eval on `vae_step0025000.pt`; write delta columns.
  - **Saved artifact:** `scripts/eval_vae_v2.py`, `results/vae_v2/eval_gate/metrics.csv` (per-film + aggregate).
  - **Verify:** script runs on one film without error; output CSV has four gate columns + source + patient_id.

- [x] ✅ **Run eval on best vae-v2 checkpoint + baseline**  ✓ verified — eval ran on n=854 held-out films (`results/vae_v2/eval_gate/verdict.md`, run date 2026-06-30): all 4 gates pass with large margins (delta_ctr 0.0027, ssim 0.9757, lpips_full 0.0222, lpips_cardiac 0.0199) vs baseline `vae_step0025000.pt`.
  - **Objective:** populate the metrics CSV for verdict.
  - **Method:**
    ```bash
    python scripts/eval_vae_v2.py \
      --checkpoint ckpts/vae_v2/vae_v2_step0075000.pt \
      --baseline ckpts/vae_step0025000.pt \
      --manifest data/vae_v2_diet/val_manifest.csv \
      --output results/vae_v2/eval_gate/
    ```
  - **Expected output:** printed table of 4 gates × 2 checkpoints + Δ column.
  - **Verify:** `results/vae_v2/eval_gate/metrics.csv` exists and is non-empty; all four gate columns populated.

- [x] ✅ **Write `results/vae_v2/eval_gate/verdict.md` and record GO/NO-GO**  ✓ verified (`results/vae_v2/eval_gate/verdict.md` written; n=854 films; all 4 gates pass: delta_ctr=0.0027✓ ssim=0.9757✓ lpips_full=0.0222✓ lpips_cardiac=0.0199✓; **Overall: GO**; checkpoint `ckpts/vae_v2/vae_v2_step0075000.pt` cleared for leaf 05)
  - **Objective:** the explicit gate decision.
  - **Format:**
    ```
    ## VAE-v2 Eval Gate — Verdict

    | Gate               | Target  | Baseline | vae-v2  | Δ      | Pass? |
    |--------------------|---------|----------|---------|--------|-------|
    | delta_ctr_median   | < 0.01  | 0.009    | X.XXX   | +X.XXX | ✓/✗  |
    | lower_zone_ssim    | ≥ 0.88  | X.XXX    | X.XXX   | +X.XXX | ✓/✗  |
    | lpips_full         | < 0.10  | X.XXX    | X.XXX   | -X.XXX | ✓/✗  |
    | lpips_cardiac_crop | < 0.08  | X.XXX    | X.XXX   | -X.XXX | ✓/✗  |

    **Overall: GO / NO-GO**

    KL spike in first 2k steps: YES (max X.Xe-X) / NO
    ```
  - **If GO:** state "Proceed to leaf 05 (re-encode). Checkpoint: `ckpts/vae_v2/vae_v2_step0075000.pt`."
  - **If NO-GO:** state which gate(s) failed and the recipe adjustment:
    - `lpips_full` still > 0.10 → extend training 20k more steps, re-eval.
    - `delta_ctr_median` regressed > 0.01 → reduce `λ_perc` to 0.05, reduce disc weight to 0.5, re-run from step 50k.
    - `lower_zone_ssim` < 0.88 → add explicit L2 term on lower-zone crop (λ=0.5), re-run from step 50k.
  - **Saved artifact:** `results/vae_v2/eval_gate/verdict.md`
  - **Verify:** `grep -E "GO|NO-GO" results/vae_v2/eval_gate/verdict.md` returns exactly one line.

- [x] ✅ **Gate 5 — latent LDM-usability (posterior signal fraction) — ADDED 2026-07-01**  ✓ resolved — signal fraction VAE-v2 **3.4–3.6%** vs V1 21% = posterior collapsed → decision: LDM trains on **μ** (`encode_moments`), not sampled latents (leaf 05 re-encodes as μ). Recorded in `results/vae_v2/eval_gate/verdict.md` (Gate 5 section) + DECISION_TIMELINE [G5b] + memory `vindr-latent-scale-bug`.
  - **Objective:** confirm the VAE's latent carries enough signal for an LDM to denoise from
    N(0,1), not just enough for μ-reconstruction. This is the gate whose absence cost two runs.
  - **Method:** on a sample of held-out films, read posterior params via `vae.encode_moments`
    and compute the aggregate **signal fraction** `sf = μ_std² / (μ_std² + σ_mean²)`:
    ```bash
    micromamba run -n jaxstack python3 - <<'PY'
    import torch, glob, numpy as np
    from PIL import Image; from torchvision import transforms
    import sys; sys.path.insert(0,'.')
    from vae.model import VAE; from vae.config import VAEConfig
    R=transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
    vae=VAE(VAEConfig(input_resolution=512,use_checkpoint=False)).cuda().eval()
    ck=torch.load('ckpts/vae_v2/vae_v2_step0075000.pt',map_location='cuda',weights_only=False)
    vae.load_state_dict(ck.get('ema',ck.get('model',ck)),strict=False)
    mu_s=[];sg=[]
    for f in glob.glob('data/nih/images/*.png')[:32]:
        mu,sigma=vae.encode_moments(R(Image.open(f).convert('L')).unsqueeze(0).cuda())
        mu_s.append(mu.std().item()); sg.append(sigma.mean().item())
    mu_std=np.mean(mu_s); sig=np.mean(sg)
    print(f'mu_std={mu_std:.3f} sigma={sig:.3f} signal_fraction={mu_std**2/(mu_std**2+sig**2):.1%}')
    PY
    ```
  - **Target: signal fraction > ~15%.** VAE-v2 measured **3.6% (FAIL for sampled)** → resolution
    is μ-encoding (plan 05). V1 VAE was 21% (PASS). If a future fine-tune clears >15%, sampled
    latents are fine; if not, the LDM must use μ or the fine-tune must add KL/free-bits pressure.
  - **Per-channel note (measured 2026-07-01):** the collapse is channel-concentrated — signal
    fraction ch1 ≈12%, ch0/2/3 = 0.1–1.4% (VAE-v2 uses ~1 of 4 latent channels). Consequence:
    per-channel normalization is NOT the fix (it would amplify the near-empty channels); global
    scaling on μ is correct. The principled deep fix for a channel-collapsed VAE is a re-fine-tune
    with KL/free-bits (or a lower KL weight) so information spreads across all 4 channels.
  - **Saved artifact:** append a `## Gate 5 — latent signal fraction` section to `verdict.md`.
  - **Verify:** signal fraction recorded in `verdict.md` with the chosen encode-mode (μ vs sampled) justified by it.

## Recommended skill
— custom; project-specific evaluation. No existing skill fits.

## Engagement Instructions

```bash
# ── env ───────────────────────────────────────────────────────────────────────
# On hippo: micromamba activate jaxstack

# ── Gate 1: |ΔCTR| regression — use vae_recon_pilot.py + ctr_overlay.py ─────
# vae_recon_pilot.py is the existing recon eval script (matches Exp02 patterns).
# ctr_overlay.py single-film mode: --film <path> --out-overlay <png>
# For batch |ΔCTR| on val set, use vae_recon_pilot.py with --manifest:
$ python scripts/vae_recon_pilot.py \
    --vae-ckpt ckpts/vae_v2/vae_v2_step0075000.pt \
    --manifest data/vae_v2_diet/val_manifest.csv \
    --out-csv results/vae_v2/eval_gate/recon_metrics.csv \
    --localized
# compare against baseline (vae_step0025000.pt baseline = 0.009 median |ΔCTR|):
$ python3 -c "
import pandas as pd
m = pd.read_csv('results/vae_v2/eval_gate/recon_metrics.csv')
print('delta_ctr_median (v2):', m['delta_ctr'].abs().median())
print('target: < 0.01 (regression hold; baseline was 0.009)')
"

# ── Gate 2: VinDr round-trip gate (re-run with VAE-v2 for cross-check) ───────
# vindr_vae_gate.py already exists; pass new checkpoint to compare
$ python scripts/vindr_vae_gate.py \
    --vae-ckpt ckpts/vae_v2/vae_v2_step0075000.pt \
    --scale-factor data/latents/scale_factor.pt \
    --out-csv results/vae_v2/eval_gate/vindr_roundtrip_v2.csv \
    --out-pairs results/vae_v2/eval_gate/vindr_pairs_v2
$ python3 -c "
import pandas as pd
m = pd.read_csv('results/vae_v2/eval_gate/vindr_roundtrip_v2.csv')
print('VinDr |ΔCTR| median (v2):', m['delta_ctr'].abs().median())
print('lower_zone_ssim median:', m['lower_zone_ssim'].median(), '  target: ≥ 0.88')
"

# ── Gate 3: LPIPS — write eval_vae_v2.py (new script, use recon_pilot as pattern)
# metrics/eval_checkpoint.py is an in-memory module (not CLI) — use it as a library
# to score decoded tensors. For held-out set LPIPS, write scripts/eval_vae_v2.py:
#   1. Load VAE-v2; encode+decode val films
#   2. Run LPIPS (RadImageNet backbone) on full 512×512 and cardiac-crop
#   3. Compute lower-zone SSIM (rows 340:512 of 512px)
#   4. Write results/vae_v2/eval_gate/metrics.csv
$ python scripts/eval_vae_v2.py \
    --vae-ckpt ckpts/vae_v2/vae_v2_step0075000.pt \
    --baseline ckpts/vae_step0025000.pt \
    --manifest data/vae_v2_diet/val_manifest.csv \
    --out results/vae_v2/eval_gate/metrics.csv
$ python3 -c "
import pandas as pd
m = pd.read_csv('results/vae_v2/eval_gate/metrics.csv')
for col in ['lpips_full_v2','lpips_cardiac_v2','lower_zone_ssim_v2']:
    print(col+':', m[col].median())
print('LPIPS target: < 0.10 (full), < 0.08 (cardiac)')
print('SSIM target: ≥ 0.88')
"

# ── Write and check verdict ───────────────────────────────────────────────────
# Fill in results/vae_v2/eval_gate/verdict.md using the table format in the tasks above
$ grep -E "Overall: GO|Overall: NO-GO" results/vae_v2/eval_gate/verdict.md
# expect: exactly one line
```
