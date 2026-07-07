# 🩹 Collapse-Prevention + Consistency — LDM Recipe v2 (EMA + min-SNR + class rebalance)

## Background
The μ-latent retrain `exp_vae_v2_ldm_mu` (W&B 6wj35ips) VALIDATED the μ-encode fix — cardio
CTR climbed 0.445→0.508 and the model generates structured chest X-rays (vs the two prior
noise runs). But it exposed two new failures:
1. **Effusion collapse** — `presence/effusion` fell 0.378→0.003 by 25k and stayed pinned near
   zero. Effusion CTR held (~0.44) so the images are real X-rays, but the *blunting feature*
   stopped rendering. Traced to the cardio-heavy `class_mult` (0.40) starving the effusion
   conditional as cardio sharpened.
2. **Per-seed inconsistency** — `presence/cardio` swinging 0.00–0.54; some seeds render clean
   cardio, others anatomically distorted ("trying to become X-rays").

## Description
Apply the sanctioned improvement-ladder levers (memory `ldm-cardio-undertrained-rebalance-lever`:
longer→**EMA**→CFG/sampler→**min-SNR**→capacity) in ONE clean retrain, AND first rule out the
VAE as the cause of effusion collapse (a plausible channel-collapse mechanism exists — see Task 1).

## Purpose
Extends vae-v2 Objective 6/7 (a usable LDM whose single-disease renders are clear+consistent so
PoE Gate-3 yield can recover). Prerequisite: plan 05 (μ cache) ✅, plan 06 (μ retrain validated).
Blocks plan 08 downstream eval on quality grounds.

## Goal
An LDM (`exp_vae_v2_ldm_mu_v2`, `ckpts/vae_v2_ldm_v2/`) whose `presence/effusion` HOLDS through
30k+ (no collapse) and whose per-seed cardio rendering is consistent (EMA), with the VAE
explicitly exonerated (or indicted) as the effusion bottleneck first.

## Tasks

- [ ] ⚠️ **Task 1 — VAE exoneration: disease-feature round-trip (DEFERRED diagnostic — NOT a hard gate)**
  - **STATUS (2026-07-01):** deliberately deferred — v2/v3 launched WITHOUT this (cheap-test-first
    call). Still worth running: if v3 plateaus or effusion regresses, this settles whether the VAE
    under-encodes the effusion feature before any VAE re-fine-tune (Task 5).
  - **Objective:** settle whether the VAE preserves the effusion blunting feature, or whether
    the channel-collapse (VAE uses ~1 of 4 latent channels) under-encodes the subtle costophrenic
    signal — which would make effusion collapse a VAE problem, not an LDM one.
  - **Method:** take ~50 REAL effusion films (border-visible, held-out) + ~50 real cardio films;
    encode→decode via μ (`vae.encode_moments` mean → `vae.decode`); run the instrument on BOTH
    original and reconstruction:
    - effusion: `metrics.eval_checkpoint.score_decoded_array` → `cp_left/cp_right.blunted`
      (blunting presence). Compare blunting-present rate: real vs μ-recon.
    - cardio: CTR on real vs μ-recon (already ~ΔCTR 0.003 in plan 04, re-confirm on this subset).
  - **Saved artifact:** `results/vae_v2/vae_roundtrip/effusion_blunting_roundtrip.csv` + a short
    verdict note.
  - **Verify / decision:**
    - blunting SURVIVES recon (recon blunting-rate ≥ ~0.8× real) → **VAE exonerated**, effusion
      collapse is LDM-side → proceed to Task 3 (v2 retrain).
    - blunting LOST in recon (recon-rate ≪ real) → **VAE is the bottleneck** → v2 will not fix it;
      escalate to Task 5 (VAE re-fine-tune) instead.

- [x] ✅ **Task 2 — Implement the recipe (code + config)**  ✓ verified 2026-07-01
  - Weight-EMA (`_WeightEMA`, decay 0.9999) + min-SNR-γ (`_min_snr_weight`, γ=5) added to
    `scripts/train_ldm.py`, config-gated (`use_ema`/`use_min_snr`, default off → old configs
    unchanged). EMA weights swapped in for all eval panels + saved as `model_ema_step*.safetensors`
    (raw weights keep training/resume). Both helpers unit-tested on CPU (EMA lag+swap+restore OK;
    min-SNR down-weights high-SNR steps OK). `configs/ldm_vae_v2_v2.yaml`: class_mult
    0.28/0.34/0.38 (effusion plurality), ckpt_dir ckpts/vae_v2_ldm_v2, run exp_vae_v2_ldm_mu_v2.

- [x] ✅ **Task 3 — Launch retrain(s)**  ✓ verified (v2 `umqnrs0o` + v3 `yrcj8qgl` launched 2026-07-01; ckpts/vae_v2_ldm_v2 & _v3 exist; Task-1 gate bypassed by the cheap-test-first call. Per-version status now tracked in 06's version timeline.)
  - **Command (template):**
    ```bash
    cd ~/PhD/Paper3 && mkdir -p logs
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup micromamba run -n jaxstack python scripts/train_ldm.py \
      --config          configs/ldm_vae_v2_v2.yaml \
      --vae-decode-ckpt ckpts/vae_v2/vae_v2_step0075000.pt \
      --run-name        exp_vae_v2_ldm_mu_v3 \
      > logs/train_vae_v2_ldm_mu_v3.log 2>&1 &
    ```
    (config `ldm_vae_v2_v2.yaml` now carries the v3 values: class_mult 0.20/0.40/0.40,
    ckpt_dir ckpts/vae_v2_ldm_v3, run exp_vae_v2_ldm_mu_v3. v2 = umqnrs0o, v3 = yrcj8qgl.)
  - **Verify:** `model_ema_step*.safetensors` appear alongside `model_step*`; W&B run live.

- [x] ✅ **Task 4 — Monitor: effusion holds + cardio consistency**  ✓ resolved via the v1→v4 iteration (per-version detail in 06's version timeline). OUTCOME: v4 (per-channel whitening) renders all three classes coherently — collapse/washed-out fixed; effusion holds. Cardio UNDER-RENDERS geometrically (CTR 0.482) — carried to plan 07-ldm-enhance, not this plan. min-SNR ruled out; EMA-warmup + rebalance kept.
  - **Success criteria:** `presence/effusion` HOLDS (v1 collapsed to 0.003); `ctr_med/cardio` stays
    ~0.50; both diseases render together; then `yield/poe_gate3` recovers.
  - **Method:** pull eval scalars via `wandb.Api()` at 5k/10k/…; training-analyst deeper sweep.
  - **Iteration log (2026-07-01):**
    - **v2 — `umqnrs0o` (class_mult 0.28/0.34/0.38):** effusion FIXED (presence 0.263 @20k raw, no
      collapse) but cardio REGRESSED (CTR 0.508→0.385, presence 0.263→0.024, 77% cardio→effusion
      leak). A see-saw — the rebalance over-favoured effusion. Also exposed an EMA-warmup artifact:
      flat 0.9999 decay made the eval panels look like noise until ~50k (raw val/loss was identical
      to v1 the whole time — the run was fine, the monitoring wasn't). Two code fixes followed:
      (a) EMA decay WARMUP `min(decay,(step+1)/(step+10))`, (b) class_mult → 0.20/0.40/0.40.
    - **v3 — `yrcj8qgl` (class_mult 0.20/0.40/0.40 + EMA-warmup fix) — BEST RUN, IN PROGRESS:**
      first run with BOTH conditionals alive. @5k/10k: ctr_med/cardio 0.488/0.496 (real ref 0.535),
      presence/effusion 0.315/0.159 (healthy, not the 0.659 floor, not collapsed), border 0.47→0.56,
      yield/poe_gate3 0.156/0.125, leak/cardio→effusion falling 0.469→0.375, no catalog failure modes.
      Recognizable CXR anatomy by 5k (EMA-warmup fix → panels meaningful early). WATCH at 20k:
      (1) does presence/effusion stabilise (5k→10k dip 0.315→0.159 looks like tightening, not the
      v1 slide to 0)? (2) does ctr_med/cardio hold ≥0.5?
  - **NOTE — decode fix staged (would be v4):** a separate SCALE-FACTOR DECODE BUG was found +
    FIXED after v3 launched (so it is NOT in v3): the in-training dashboard decoded generated latents
    WITHOUT dividing by scale_factor — dormant for v1 (sf≈1), 5× over-scale for v2 μ (sf 5.034). Fixed
    in `_eval_gen_decode`/`_log_quality_panels`/`_log_cfg_ablation`/`_log_compose_grid`/`monitor.decode_grid`
    + `eval_ckpt_offline.py`/`monitor_retrain.py` (`generate.py` already divided). v3 is unaffected in
    practice because (i) early samples are under-dispersed (z0 std ~0.33) so buggy-decode ≈ correct,
    and (ii) final synthetic data goes through `generate.py` which divides correctly. **If v3 fails, OR
    its panels over-saturate as it converges (z0 std → 1.0 makes buggy-decode 5× over), relaunch as
    v4 WITH the decode fix.** Judge converged-checkpoint quality via the fixed `eval_ckpt_offline.py`.

- [ ] ⚠️ **Task 5 — FALLBACK: VAE re-fine-tune (only if Task 1 indicts the VAE, or v2/v3 quality plateaus)**
  - **Trigger:** Task 1 shows blunting lost in recon, OR v2 effusion still collapses despite the
    rebalance+min-SNR.
  - **Method:** re-fine-tune VAE-v2 with stronger KL / free-bits so signal spreads across all 4
    channels (undo the collapse), re-run plan 04 Gate-5 (signal fraction >15%), re-encode (plan 05),
    retrain. This is the ~6–10 h detour deliberately deferred until evidence forces it.

## Recommended skill
— custom; same train_ldm.py + W&B monitoring pattern as plan 06.

## Engagement Instructions

```bash
# ── env: micromamba activate jaxstack (hippo) ─────────────────────────────────

# ── Task 1: VAE disease-feature round-trip (the certainty test) ──────────────
# Reuses metrics.eval_checkpoint.score_decoded_array (the PSPNet CP/blunting instrument).
# Encode real effusion films to μ (encode_moments mean), decode, compare blunting rate.
# Real effusion source: data/nih/images_effusion_only/ (or the border-visible manifest).
# GPU must be free (VAE encode+decode). ~2 min for 50+50 films.

# ── Task 3: launch v2 (only after Task 1 exonerates the VAE) ──────────────────
# Watch first 4 min for step-100 loss lines; then eval at 5k.

# ── Task 4: pulse-check scalars ──────────────────────────────────────────────
# micromamba run -n jaxstack python3 -c "import wandb; r=wandb.Api().run('prime_lab/paper3-ldm/<id>'); ..."
# KEY metric: presence/effusion must NOT decay toward 0 as cardio improves.
```
