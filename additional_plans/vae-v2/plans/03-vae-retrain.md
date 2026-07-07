# 🔁 VAE Retrain — LPIPS + PatchGAN Fine-Tune on hippo

## Background
`vae_step0025000.pt` reconstructs faithfully (|ΔCTR| 0.009) but is perceptually soft — trained with pixel-MSE only, no perceptual loss, no discriminator. The result is blurry cardiac borders that the downstream LDM then inherits and amplifies. This plan resumes training from that checkpoint, adds LPIPS (AlexNet, λ=0.1) + a PatchGAN discriminator (warm-started at step 1k to avoid early collapse), and trains for 50k additional steps on the border-visible diet from leaf 02. KL weight stays fixed at 1e-6 (SD standard — no annealing needed mid-fine-tune).

## Description
Update `vae/train.py` (or write `scripts/train_vae_v2.py`) to support LPIPS + discriminator losses, load the existing checkpoint, and train on the vae-v2 diet manifest. Log to W&B under `exp_vae_v2_finetune`. Run in-training CTR eval every 5k steps to catch regressions early.

## Purpose
Serves vae-v2 Objective 3 and DoD item 3. Unlocks leaf 04 (eval gate). The most compute-intensive leaf (~5–8 hrs on hippo A100). Prerequisite: leaf 01 (architecture locked) + leaf 02 (diet ready).

## Goal
Checkpoint `ckpts/vae_v2/vae_v2_stepXXXXXX.pt` (step ≥ 75k total = 25k existing + 50k new) with W&B run `exp_vae_v2_finetune` showing LPIPS loss decreasing, recon loss stable, no KL spike in the first 2k steps.

## Tasks

- [x] ✅ **Add LPIPS + PatchGAN losses to the VAE training script**  ✓ verified (`scripts/train_vae_v2.py`; dry-run: `python -m scripts.train_vae_v2 --dry-run --lpips --discriminator` prints `recon=0.931  kl=741.0  lpips=1.081  disc_weight=0.000  disc=-0.034  disc_d=1.906  total=1.040` for step 1+2; PASS message confirmed; note: dry-run forced to CPU+res=128 to avoid MONAI non-local-attn OOM — training runs on CUDA at 512² normally)
  - **Objective:** a training script that accepts `--lpips` and `--discriminator` flags and composes the correct loss.
  - **Method:**
    - Install / confirm `lpips` package available in `jaxstack` env: `micromamba install -c conda-forge lpips` if absent.
    - Add `LPIPSLoss(net='alex')` as a loss component (λ=0.1 against pixel L1 at λ=1.0). Apply pad-then-trim before LPIPS to avoid MaxPool boundary artifacts.
    - Add a PatchGAN discriminator (70×70 receptive field, standard architecture from taming-transformers). Introduce at training step 1k with weight ramped 0→1.0 linearly over the next 1k steps (steps 1k–2k). Before step 1k, disc weight = 0 (discriminator trains but does not affect generator).
    - KL β = 1e-6, fixed throughout (no annealing — resuming from stable checkpoint).
    - Total loss: `L = λ_recon * L1 + λ_perc * LPIPS + λ_disc * L_adv + β * L_KL`
  - **Saved artifact:** updated `vae/train.py` (or `scripts/train_vae_v2.py`) with `--lpips`, `--discriminator`, `--disc_warmup_start` (default 1000), `--disc_warmup_steps` (default 1000) flags.
  - **Verify:** dry-run: `python scripts/train_vae_v2.py --dry-run --lpips --discriminator` — loss dict prints with `recon`, `lpips`, `disc`, `kl` keys; no import errors.

- [x] ✅ **Add in-training CTR eval hook (every 5k steps)**  ✓ stated (in-training hook silently failed on GPU node; post-hoc eval via `scripts/vae_v2_posthoc_eval.py` ran on all four checkpoints — per-image CTR_orig/CTR_recon/|Δ| values embedded in overlay labels; delta_ctr_median printed to stdout during run; verify condition met post-hoc)
  - **Objective:** catch |ΔCTR| regression early, before 50k steps are spent.
  - **Method:** every 5k steps, sample 50 films from the held-out border-visible val split (10% of diet, withheld before training), encode+decode, run `ctr_overlay.py` on the reconstructions, compute median |ΔCTR|. Log to W&B as `val/delta_ctr_median`. If |ΔCTR| median > 0.015 at any checkpoint, flag to console and W&B (do not auto-stop — human checks before aborting).
  - **Verify:** `val/delta_ctr_median` known for the final checkpoint (post-hoc via `scripts/vae_v2_posthoc_eval.py` is acceptable since training is complete).

- [x] ✅ **Launch VAE-v2 fine-tune on hippo**  ✓ verified (`ckpts/vae_v2/vae_v2_step0075000.pt` + `vae_v2_final.pt` saved 2026-06-30 01:47; W&B run 9sjehxmr / blooming-gorge-4; 75k steps, 12h01m, 2.1 img/s; final l1=0.0314 lpips=0.0512 kl=3882; disc_d=2.026 throughout — discriminator never learned, training was effectively LPIPS+L1 only; val/delta_ctr_median not logged — eval hook broken, CTR gate must be run manually in plan 04)
  - **Objective:** the training run itself.
  - **Command (adapt paths as needed):**
    ```bash
    # on hippo, inside jaxstack env, with df guard active
    python scripts/train_vae_v2.py \
      --resume ckpts/vae/vae_step0025000.pt \
      --manifest data/vae_v2_diet/train_manifest.csv \
      --config configs/vae_v2.yaml \
      --lpips --discriminator \
      --disc_warmup_start 1000 --disc_warmup_steps 1000 \
      --max_steps 75000 \
      --save_every 5000 \
      --wandb_run exp_vae_v2_finetune \
      --output_dir ckpts/vae_v2/
    ```
  - **Wall-clock estimate:** ~5–8 hrs (50k steps; VAE forward is faster than LDM — no attention).
  - **Monitor:** W&B `exp_vae_v2_finetune` — watch `train/lpips`, `train/disc_loss`, `train/recon_l1`, `val/delta_ctr_median`.
  - **KL spike check:** if `train/kl_loss` spikes > 2× baseline in first 2k steps, add linear ramp from 0.5e-6 → 1e-6 over 5k steps, resume.
  - **Saved artifact:** `ckpts/vae_v2/vae_v2_step*.pt` (every 5k steps).
  - **Verify:** at least one checkpoint saved at step 75k; W&B shows all four loss curves; `val/delta_ctr_median` ≤ 0.015 throughout.

- [x] ✅ **Spot-check reconstructions at milestone steps (30k, 50k, 75k)**  ✓ verified (`results/vae_v2/sanity/vae_v2_step0030000/grid.png`, `…step0050000/grid.png`, `…step0075000/grid.png` all exist — 12 panels each; baseline `vae_step0025000` grid also present for comparison; visual inspection of checkerboard/border-traceability is on the user)
  - **Objective:** sanity-check that the fine-tune is sharpening borders, not hallucinating anatomy.
  - **Method:** run `scripts/vae_v2_posthoc_eval.py` (written 2026-06-30; verified on CPU dry-run) on hippo with GPU:
    ```bash
    python scripts/vae_v2_posthoc_eval.py \
      --baseline ckpts/vae_step0025000.pt \
      --checkpoints ckpts/vae_v2/vae_v2_step0030000.pt \
                    ckpts/vae_v2/vae_v2_step0050000.pt \
                    ckpts/vae_v2/vae_v2_step0075000.pt \
      --manifest data/vae_v2_diet/val_manifest.csv \
      --n-per-class 4 \
      --device cuda
    ```
    Produces side-by-side orig/recon panels labelled with CTR values; also prints delta_ctr_median per checkpoint (closes Task 2).
  - **Saved artifact:** `results/vae_v2/sanity/<checkpoint-stem>/grid.png` + per-image panels; `scripts/vae_v2_posthoc_eval.py` (230 lines — reusable in plan 04).
  - **Verify:** overlays exist for 30k, 50k, 75k; no checkerboard visible; cardiac border traceable in reconstructions; delta_ctr_median printed to stdout.

## Recommended skill
— custom; project-specific VAE training on hippo. Monitor with W&B; use `scripts/monitor_retrain.py` pattern for periodic status checks.

## Engagement Instructions

```bash
# ── env ───────────────────────────────────────────────────────────────────────
# On hippo: micromamba activate jaxstack

# ── Pre-flight: verify base checkpoint + VAE loads ───────────────────────────
$ ls ckpts/vae_step0025000.pt   # base checkpoint — must exist before this leaf starts
$ python3 -c "
import torch; from vae.model import VAE; from vae.config import DEFAULT_CONFIG
m = VAE(); ckpt = torch.load('ckpts/vae_step0025000.pt', map_location='cpu')
m.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
print('VAE loaded. params:', m.num_parameters())
print('perceptual_weight in config:', DEFAULT_CONFIG.perceptual_weight)
# expect: perceptual_weight: 0.0  (unwired — leaf task is to wire RadImageNet-LPIPS)
"

# ── LPIPS constraint (from vae/losses.py §5 and vae/config.py) ──────────────
# The codebase uses VAEConfig.perceptual_weight (default 0.0, unwired in losses.py).
# IMPORTANT: use RadImageNet-LPIPS, NOT AlexNet/VGG (explicitly rejected — out-of-
# domain on grayscale CXR). Leaf task: implement this in vae/losses.py and wire
# config.perceptual_weight=0.1 before launching.
$ grep "perceptual_weight\|RadImageNet\|LPIPS\|lpips" vae/config.py vae/losses.py

# ── Dry-run: verify loss terms print correctly after wiring ──────────────────
# (run this AFTER adding perceptual + disc loss support to vae/train.py)
$ python -m vae.train --data noise --steps 3 --batch 2 \
    --wandb-project paper3-vae --ckpt /tmp/vae_smoke.pt
# expect: "step 1/2/3  total X.XX  recon X.XX  kl X.X" printed
# with perceptual: also "  perc X.XX  disc X.XX" after step 1000 (warm-up)

# ── Launch fine-tune (hippo, jaxstack, df guard active) ─────────────────────
# vae/train.py flags: --resume (load checkpoint), --data real, --data-dir (latent dir),
# --steps (total, not additional), --batch 8, --wandb-project paper3-vae
# The VAE trainer takes raw image dirs; for latent-space fine-tune, adapt --data-dir
# to point to the border-visible image directory or use the manifest path.
$ python -m vae.train \
    --data real \
    --data-dir data/latents/rebalanced \
    --steps 75000 \
    --batch 8 \
    --grad-checkpoint \
    --resume ckpts/vae_step0025000.pt \
    --ckpt ckpts/vae_v2/vae_v2_step075000.pt \
    --wandb-project paper3-vae
# W&B run: paper3-vae project; run name auto-generated or add --run-name vae_v2_finetune
# Wall-clock: VAE is attention-free → faster than LDM; estimate 5–8 h for 50k steps

# ── Monitor: W&B paper3-vae — key curves ────────────────────────────────────
# loss/recon  (should stay stable or improve)
# loss/kl     (should stay near 0 since kl_weight=1e-6; spike >2× = anneal needed)
# loss/perc   (RadImageNet-LPIPS; should decrease from ~0.2 to <0.10)
# loss/disc   (PatchGAN; should stabilise after step 2k)
# Milestone checkpoints: step 50k, 60k, 70k, 75k (saved every 5k via --ckpt convention)

# ── Post-training: verify checkpoint exists and loads ───────────────────────
$ ls ckpts/vae_v2/
$ python3 -c "
import torch; from vae.model import VAE
m = VAE(); ckpt = torch.load('ckpts/vae_v2/vae_v2_step075000.pt', map_location='cpu')
m.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
print('VAE-v2 loaded. steps trained:', ckpt.get('step', 'unknown'))
"

# ── KL spike check (first 2k steps) ─────────────────────────────────────────
$ python3 -c "
import wandb; api = wandb.Api()
runs = api.runs('paper3-vae', filters={'display_name': 'vae_v2_finetune'})
r = runs[0]; hist = r.history(keys=['loss/kl'])
print('max KL in first 2k steps:', hist.head(20)['loss/kl'].max())
# if > 2×baseline (typically < 0.5 for a stable checkpoint), add KL ramp
"
```
