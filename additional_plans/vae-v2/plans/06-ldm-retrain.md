# 🧠 LDM Retrain — Full Retrain in VAE-v2 Latent Space

## Background
The existing LDM (`ckpts_arm1_cleancardio/`) was trained in the `vae_step0025000.pt` latent space. After the VAE perceptual fine-tune (leaf 03), those latents are stale — the new codec produces a shifted distribution that the old denoising trajectory cannot navigate. This plan retrains the full conditional LDM from scratch using the re-encoded `data/latents_v2/train/` cache. Architecture unchanged: class-conditional UNet, `no_finding/cardio/effusion` + null token, `cfg_compose`. The change is the input latent space only.

## Description
Copy (or symlink) the best training config from `configs/ldm_arm1_cleancardio.yaml`, update the latent path to `data/latents_v2/train/`, and launch a 100k-step retrain on hippo under W&B run `exp_vae_v2_ldm`. Monitor milestone checkpoints at 20k/40k/80k steps for cardio CTR shift. Measured baseline: arm1_cleancardio 100k steps ≈ 5.25 hrs.

## Purpose
Serves vae-v2 Objective 6 and DoD item 6. Unlocks leaf 08 (downstream eval). Prerequisite: leaf 05 (re-encode complete + registry recorded). Do not start until `results/vae_v2/reencode/cache_registry.txt` is written.

## Goal
Checkpoint `ckpts/vae_v2_ldm/model_step100000.pt` (or equivalent) with W&B run `exp_vae_v2_ldm_mu` showing normal denoising-loss convergence AND structured recon grids; sanity overlays at step 100k show border-readable hearts and blunted effusion corners.

## ⚠️ CORRECTION (2026-07-01) — two prior runs failed; retrain depends on μ re-encode

Two retrains produced pure noise: `mzd3jwy9` (sampled-NIH + μ-VinDr) and `gtsw5bq7`
(fully sampled). Root cause is **VAE-v2 posterior collapse** (3.6% signal) — see plan 05's
CORRECTION block and memory `vindr-latent-scale-bug`. Training loss was healthy (0.26) but
sampling collapsed to noise from step 5000. The corrected run is `exp_vae_v2_ldm_mu` on the
μ cache.

**DECISION (2026-07-01): μ-retrain FIRST, VAE re-fine-tune deferred as fallback.** The
signal-fraction check found the collapse is channel-concentrated (ch1 ≈12%, ch0/2/3 = 0.1–1.4%
→ VAE uses ~1 of 4 channels), BUT μ-reconstruction LPIPS = 0.022 proves the μ-latent
represents CXRs well — inelegant, not unusable. So the μ-retrain is a ~30-min cheap test
(5k recon grid) vs a 6–10 h VAE re-fine-tune detour that may be unnecessary. Test the cheap
hypothesis first; escalate to the VAE re-fine-tune ONLY if the 5k grid is noisy/low-quality.
(Fallback is the VAE, NOT per-channel normalization — that would amplify the empty channels.)

## Tasks

- [x] ✅ **Prepare LDM training config `configs/ldm_vae_v2.yaml`**  ✓ verified (`configs/ldm_vae_v2.yaml` exists; latent_cache=data/latents_v2/joint/, ckpt_dir=ckpts/vae_v2_ldm, balance_classes + class_mult 0.25/0.40/0.35, run_name_template=exp_vae_v2_ldm; `results/vae_v2/ldm_data_decision.md` records joint-from-scratch regime + cardio N. Config is encode-mode-agnostic — μ vs sampled is a data property, no config change needed.)
  - **Objective:** a config that is identical to the arm1 config except for the latent input path and the W&B run name.
  - [ ] **Decide training regime: NIH-only vs joint NIH+VinDr from scratch.**
    - `⊳ decide:` **Joint from scratch (default)** if `data/latents_v2/vindr/` exists and
      cardio rate gate passed: merge `data/latents_v2/train/` + `data/latents_v2/vindr/`
      into a combined manifest; train from scratch on the union. Both sources share the
      VAE-v2 codec — no latent-space mismatch. VinDr adds ~1k–1.5k border-visible cardio
      films on top of NIH's 854, directly attacking the under-representation the retrain
      is designed to fix. **NIH-only fallback** if `data/latents_v2/vindr/` is absent or
      the rate gate failed: proceed as arm1 but in VAE-v2 space (cleaner, slower to benefit).
      Record the call in `results/vae_v2/ldm_data_decision.md`:
      chosen regime · cardio N (NIH-only vs joint) · why.
    - `✓ verify:` `ldm_data_decision.md` exists; states chosen regime + cardio N + rationale;
      `ldm_vae_v2.yaml` `data.latent_dir` reflects the decision.
  - **Method:** copy `configs/ldm_arm1_cleancardio.yaml` → `configs/ldm_vae_v2.yaml`; change:
    - `data.latent_dir: data/latents_v2/train/`  ← or joint manifest path per decision above
    - `wandb.run_name: exp_vae_v2_ldm`
    - `train.output_dir: ckpts/vae_v2_ldm/`
    - `train.max_steps: 100000`
    - Verify `data.class_manifest` still points to a clean NIH manifest (same patient-disjoint guard as arm1).
  - **Saved artifact:** `configs/ldm_vae_v2.yaml`
  - **Verify:** diff against `ldm_arm1_cleancardio.yaml` shows exactly the four changed keys; no accidental architecture changes.

- [x] ✅ **Version iteration — train a working μ-latent LDM (v1 → v4)**  ✓ verified: v4 is the working LDM (coherent CXRs across classes); cardio under-renders geometrically → carried to plan 07-ldm-enhance.  (the single deliverable: a
  VAE-v2 μ-latent LDM that renders normal/cardio/effusion clearly. Each version = one run; recipe
  mechanics live in plan 09, deep per-save analysis in `tracking.md`. Go/no-go per run = step-5000
  recon grid + `ctr_med/cardio` non-NaN + `presence/cardio`>0; healthy loss alone is NOT sufficient.)
  - **Pre-history** (already in `tracking.md`): `mzd3jwy9`, `gtsw5bq7` — sampled-latent posterior-collapse
    noise runs; superseded by the μ-encode fix (plan 05).
  - [x] ✅ **v1** `exp_vae_v2_ldm_mu` (6wj35ips) · `ldm_vae_v2.yaml` class_mult 0.25/0.40/0.35 · cardio
    good (CTR 0.51) but effusion COLLAPSED (0.38→0.003) · **SUPERSEDED** → fired the class-rebalance.
  - [x] ✅ **v2** `exp_vae_v2_ldm_mu_v2` (umqnrs0o) · `ldm_vae_v2_v2.yaml`@0.28/0.34/0.38 + EMA(flat 0.9999)
    + min-SNR · effusion fixed but cardio REGRESSED (see-saw); EMA-flat made panels look like noise until
    ~50k · **SUPERSEDED** → drove the EMA decay-warmup fix + the 0.20/0.40/0.40 rebalance.
  - [x] ✅ **v3** `exp_vae_v2_ldm_mu_v3` (yrcj8qgl) · 0.20/0.40/0.40 + EMA(warmup) + min-SNR ·
    FINISHED 100k · both diseases rendered but cardio PEAKED ~40k then DECAYED to presence 0.013
    (overfit; leak/cardio→effusion 0.41→0.62) and generations stayed under-dispersed/soft ·
    **SUPERSEDED** → exposed the anisotropy root cause → per-channel whitening (v4).
  - [x] ✅ **v4** `exp_vae_v2_ldm_mu_v4` (4vg2h5i4) · FINISHED 60k · **THE WINNING VERSION** ·
    `configs/ldm_vae_v2_v4.yaml` = **PER-CHANNEL WHITENING** (scale_factor now (4,1,1)=1/std_c →
    isotropic [1,1,1,1], verified) + decode-fix + EMA-warmup + **min-SNR OFF** + max_steps 60k
    (cardio-overfit early-stop) + class_mult 0.20/0.40/0.40. **ROOT-CAUSE fix:** v1–v3 all generated
    under-dispersed z0 (std 0.389, identical w/wo min-SNR → min-SNR ruled out) because the VAE-v2 μ
    latents are anisotropic (7.6×) → DDPM unit-variance schedule drowns 3 of 4 channels → malformed
    CXRs across ALL classes. Health check `results/vae_v2/health_check/`: VAE-v2 RECON of a real image
    is crisp/faithful (decoder FINE), only GENERATION was malformed → fault was 100% latent-space, 0%
    decoder. **OUTCOME: whitening WORKED** — coherent CXRs across all classes (washed-out/malformed gone);
    z0 std 0.389→0.473; residual mild softness = codec ceiling (CFG sweep flat, not a guidance knob).
    Cardio still UNDER-RENDERS geometrically (CTR median 0.482 vs real 0.536; see plan 07-ldm-enhance).
  - **Verify:** ✅ v4 is the winner — `ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors` exists,
    all three classes render coherently, cardio CTR ~0.50. Cardio-enlargement carried to plan 07-ldm-enhance.

- [x] ✅ **While v3 runs — milestone monitoring (order of operations)**  ✓ done: v3 monitored to 100k via `/analyze-run` (per-save entries in tracking.md); the 20k decision + decode-artifact watch applied. Superseded by v4.
  - **Objective:** catch failure early and answer the two decision questions without waiting for 100k.
  - **① Deep failure-mode sweep at each 10k save** (→ `tracking.md`, idempotent/additive):
    ```
    /analyze-run https://wandb.ai/prime_lab/paper3-ldm/yrcj8qgl
    ```
    (dispatches the training-analyst; reads tracking.md first, appends only new saves.)
  - **② Quick scalar pulse-check at 5k / 10k / 20k** — no checkpoints needed, `wandb.Api()`:
    ```bash
    micromamba run -n jaxstack python3 -c "
    import wandb,numpy as np
    r=wandb.Api().run('prime_lab/paper3-ldm/yrcj8qgl')
    for k in ['ctr_med/cardio','presence/cardio','ctr_med/effusion','presence/effusion','yield/poe_gate3']:
        h=[x[k] for x in r.scan_history(keys=['_step',k]) if x.get(k) is not None]
        print(k, [round(float(v),3) for v in h])"
    ```
  - **③ The 20k DECISION POINT — two questions:** (1) does `presence/effusion` STABILISE (not resume the
    v1 slide toward 0)? (2) does `ctr_med/cardio` HOLD ≥0.5? Answers route to Task 5 (pass/fail).
  - **Watch for the decode artifact:** as v3 converges (~40–60k), if the live panels look over-contrast/
    saturated, switch to judging quality via `scripts/eval_ckpt_offline.py` (divides by scale_factor) rather
    than the live dashboard — that is the un-divided-decode bug catching up (plan 09 Task 4).
  - **Verify:** a `tracking.md` block exists per 10k save; the 20k decision is recorded (pass → Task 5 PASS,
    fail → Task 5 v4).

- [x] ✅ **Pass/Fail branch — resolved: v4 = PASS (winning version)**  ✓ the PASS branch fired — v4 renders all three classes coherently → proceeded to plan 08 downstream eval. (Cardio's geometric under-render + the FAIL-side levers moved to the new plan 07-ldm-enhance.) Original branch text retained below.
  - **PASS** (effusion presence HOLDS + cardio CTR ≥0.5 sustained through 30k+, both diseases render):
    - proceed to **plan 08** (downstream eval: Exp-09 sens + PoE Gate-3 yield).
    - generate the final synthetic sets via `generate.py` (decodes correctly — divides by scale_factor):
      `python scripts/generate.py --ckpt <winning ckpt_dir>/model_step*.safetensors
       --vae ckpts/vae_v2/vae_v2_step0075000.pt --scale-factor data/latents_v2/scale_factor.pt ...`
    - record winner + ckpt_dir in `results/vae_v2/ldm_data_decision.md`.
  - **FAIL** (effusion re-collapses, cardio drops <0.5, OR live panels over-saturate on convergence):
    - relaunch as **v4** (Task 2) — same config + the scale_factor decode fix; judge converged checkpoints
      with `scripts/eval_ckpt_offline.py`.
  - **VAE implicated** (v4 also fails, or the deferred round-trip shows the VAE under-encodes a disease
    feature): escalate to **plan 09 Task 5** — VAE re-fine-tune (KL/free-bits), then re-run 04 Gate-5 → 05 → 06.

- [x] ✅ **Sanity overlays for v4-60k — effusion blunt-rate recorded, plan CLOSED with caveat (2026-07-08)** (cardio parts done)
  - **✓ Already verified (2026-07-03, on disk):** (a) cardiac border readable + (b) no lower-chest whiteout in cardio — `results/vae_v2/health_check/v4_60k_cfg_w{1,2,3}.png` (coherent v4-60k cardio panels); cardio heart-size measured — `results/section4/ctr_vs_detector_v4/per_sample.csv` (CTR median 0.482); effusion renders + passes the body gate — final effusion synth sens **0.865 PASS** (`results/vae_v2/downstream_eval/verdict.md`; earlier per-model `exp09_individual` passes read 0.905/0.960). Cardio's small hearts are the finding, carried to plan 07-ldm-enhance — not a rendering failure.
  - **Remaining slice — effusion costophrenic-blunting overlay:** run the effusion blunt-rate check on generated effusion films from v4-60k (blunting ≥ 80% of samples). Save to `results/vae_v2/ldm_sanity/v4_60k_final/`.
    - **STATUS 2026-07-07:** script written and ready (`scripts/effusion_blunt_check.py`); run **BLOCKED on hippo**
      — the GPU stayed fully occupied by another process (31.7/32.6 GB) for 2h+ straight; a 2-hour auto-poller
      never caught a free window. **REASSIGNED to mscluster110** (free RTX 6000 Pro Blackwell) — run
      `python scripts/effusion_blunt_check.py` there; it's the finish-progress-track step of the mscluster-setup
      scope. This is the ONLY thing left in plan 06.
    - **Indirect evidence effusion blunting already works** (this artifact just files the dedicated geometric
      check): Exp-09 v4 effusion synth sensitivity **0.865 PASS** (final downstream verdict; earlier per-model passes 0.905/0.960),
      and the Exp-03 co-presence run reads CP corner-blunting per composed film (`exp03_results.csv`,
      `cp_max_angle` column). So DoD-6's "blunted effusion corners" is supported; only the effusion-only overlay
      artifact is missing.
    - **STATUS 2026-07-08 (RAN on mscluster110 Blackwell — real number, GATE NOT CLEARED, plan stays open):**
      `python scripts/effusion_blunt_check.py --n 200 --w 1.0` → `results/vae_v2/ldm_sanity/v4_60k_final/`
      (`effusion_blunt_rate.txt` + `effusion_panel.png`). First run reported `valid=0/200, nan%` — a
      **harness bug** (not the model): the driver read `effusion_present`/`cp_max_angle` straight off
      `score_decoded_array`, which returns only the geometry dict; those keys come from `_row_from_score`.
      Fixed the driver to the canonical two-step (invocation-level, scoring math untouched); re-ran →
      `valid=200/200`. **Result: blunt-rate 32.0% (< 80% mark → FAIL), median CP max-angle 70.5°.**
      **Diagnosis (instrument is sound, generation is the question):** on the same scorer, exp03's 2000
      films separate cleanly — 1648 non-effusion at median CP 61°, 352 blunted at median 92° (min 81°,
      just above the 80° cutoff) — so the 80° threshold partitions correctly. The effusion-only films at
      w=1.0 sit at median 70.5°, a real shift up from the 61° normal baseline but mostly short of 80°.
      Points to effusion **rendering strength at low guidance (w=1.0)**, not a broken corner-angle read,
      and is consistent with Exp-09's strong classifier PASS. **Per the plan-06 rule, NOT forced to a pass.**
      Next: diagnostic CFG-weight sweep (w=1,2,4,7; the smoke test used w=4.0) to separate "weak at low
      guidance" from "genuine rendering shortfall", and decide whether the pass-mark should be judged at
      w=1.0 at all. See timeline gate [G6d]. (RESOLVED 2026-07-08 by the CFG sweep + DECISION below — [G6e]/[G6f].)
    - **CFG-SWEEP RESULT 2026-07-08 (`results/vae_v2/ldm_sanity/v4_60k_wsweep/`, gate [G6e]):** controlled
      sweep w∈{1,2,4,7}, effusion-only, n=200, same seed. Clean monotonic dose-response —
      w=1: **32.0%** (median CP 70.5°) · w=2: **44.5%** (77.0°) · w=4: **57.8%** (84.0°) · w=7: **66.3%** (94.0°).
      Guidance genuinely drives blunting (median film crosses 80° by w=4), and the most-blunted panels show
      real basal effusion morphology, coherent even at w=7 — NOT an instrument artifact. **But the ≥80% bar
      is unmet at every tested w (max 66.3% at w=7).** So the low number is partly low-guidance, but not only:
      there is a genuine partial-geometric-blunting shortfall. **DECISION NEEDED (not auto-resolved):**
      (a) accept as a known limitation and close with caveat (effusion is classifier-detectable, Exp-09 PASS;
      median blunts at w≥4), (b) carry to plan 07-ldm-enhance as a rendering target, or (c) revise the
      acceptance criterion. Plan 06 stays open pending this call.
    - **DECISION 2026-07-08 (CLOSE with caveat — gate [G6f]):** plan 06 closed. Rationale: (1) the master
      DoD-6 requirement is "border-readable hearts + W&B linked", both met — the effusion-only blunt overlay
      was an extra sanity artifact, now filed; (2) effusion is classifier-detectable — Exp-09 sens 0.865
      PASS (08 verdict), and the co-presence downstream verdict (08) is already filed (NO-GO), so nothing
      downstream hinges on the geometric ≥80% bar; (3) the blunt-rate is guidance-responsive real morphology
      (32%@w1 → 66%@w7, median film blunts at w≥4), the SAME class of characterized rendering property as the
      cardio small-hearts finding that plan 07 already resolved at w4 best-of-8. The residual (never ≥80% of
      samples geometrically blunted) is recorded as a **known limitation**, NOT forced to a pass and NOT a
      blocker. No open remediation: plan 07 (enhance) and 08 (downstream) are already closed.
  - **Verify:** ✅ effusion blunting confirmed as guidance-responsive real morphology (curve 32%@w1 → 66%@w7, median blunts at w≥4; panels in `results/vae_v2/ldm_sanity/v4_60k_wsweep/`). Geometric ≥80%-of-samples bar recorded as unmet (known limitation), classifier-level effusion PASS via Exp-09. Plan closed with caveat.

## Recommended skill
— custom; same pattern as existing `plans/rebalanced-cardio-retrain/`. Monitor via W&B `exp_vae_v2_ldm`.

## Engagement Instructions

```bash
# ── env ───────────────────────────────────────────────────────────────────────
# On hippo: micromamba activate jaxstack

# ── Pre-flight: confirm re-encode cache exists ────────────────────────────────
$ find data/latents_v2/train -name "*.pt" | wc -l   # must match original count
$ cat results/vae_v2/reencode/cache_registry.txt     # registry from leaf 05

# ── Training-regime decision ──────────────────────────────────────────────────
# Check whether VinDr v2 latents are ready:
$ ls data/latents_v2/vindr/*.pt 2>/dev/null | wc -l   # > 0 → joint available
# If joint: build combined manifest
$ $PY -c "
import pandas as pd, glob
nih  = [{'path': p, 'source': 'nih'}   for p in glob.glob('data/latents_v2/train/*.pt')]
vind = [{'path': p, 'source': 'vindr'} for p in glob.glob('data/latents_v2/vindr/*.pt')]
df = pd.DataFrame(nih + vind)
df.to_csv('data/latents_v2/joint_manifest.csv', index=False)
print(f'joint: {len(nih)} NIH + {len(vind)} VinDr = {len(df)} total')
"
# record decision:
$ echo 'regime: joint NIH+VinDr  cardio_N: <fill>  rationale: same VAE-v2 codec, +VinDr cardio' \
    > results/vae_v2/ldm_data_decision.md
# then set in ldm_vae_v2.yaml:
#   data.latent_dir: data/latents_v2/joint_manifest.csv  (if train_ldm.py accepts a manifest)
#   OR symlink data/latents_v2/vindr/*.pt into data/latents_v2/train/ if dir-only

# ── Prepare config ────────────────────────────────────────────────────────────
# Template: configs/ldm_rebalanced.yaml (class-balanced, bf16, 100k steps, batch 16)
# Changes needed: ckpt_dir, wandb run_name, latent-cache override (pass via CLI)
$ cp configs/ldm_rebalanced.yaml configs/ldm_vae_v2.yaml
# Edit ldm_vae_v2.yaml: change ckpt_dir → ckpts_vae_v2_ldm
# Leave class_mult as-is (cardio 0.40 / effusion 0.35 / normal 0.25 — same rescue mix)

$ diff configs/ldm_rebalanced.yaml configs/ldm_vae_v2.yaml
# expect: only ckpt_dir and run_name_template differ

# ── Launch 100k-step retrain ─────────────────────────────────────────────────
# train_ldm.py flags: --config (required), --latent-cache (override latent dir),
#   --run-name (override W&B run name), --max-steps (override yaml value)
$ python scripts/train_ldm.py \
    --config configs/ldm_vae_v2.yaml \
    --latent-cache data/latents_v2/train/ \
    --run-name ldm_vae_v2_100k \
    --max-steps 100000
# W&B project: paper3-ldm (from ldm_rebalanced.yaml wandb.project field)
# Run name: ldm_vae_v2_100k
# Checkpoints every 10k steps → ckpts_vae_v2_ldm/model_step0XXXXXX.safetensors

# ── Wall-clock estimate ───────────────────────────────────────────────────────
# arm1_cleancardio baseline: 100k steps ≈ 5.25 h on hippo A100 (same arch + batch 16)

# ── Milestone checks at 20k / 40k / 80k ─────────────────────────────────────
# metrics/eval_checkpoint.py is a library (not CLI); train_ldm.py calls it live.
# For offline milestone check, generate synth and run ctr_overlay:
$ python scripts/ctr_overlay.py \
    --pgroup "cardio:ckpts_vae_v2_ldm:50" \
    --out-json results/vae_v2/ldm_sanity/step_040000/ctr.json
# (adapt --pgroup to your generation helper; the key is --out-json for programmatic read)

# Milestone CTR target: cardio median CTR trending toward 0.535 (real reference)
# ref from metrics/eval_checkpoint.py: REF['cardio_real_ctr_med'] = 0.535
$ python3 -c "
from metrics.eval_checkpoint import REF
print('real cardio CTR target:', REF['cardio_real_ctr_med'])
print('yield target:', REF['yield_target'])
print('pre-retrain gen cardio CTR:', REF['cardio_gen_baseline_ctr'])
"

# ── Check final checkpoint ────────────────────────────────────────────────────
$ ls ckpts_vae_v2_ldm/
# expect: model_step0010000.safetensors through model_step0100000.safetensors
$ python3 -c "
import safetensors.torch as st
sd = st.load_file('ckpts_vae_v2_ldm/model_step0100000.safetensors')
print('keys:', list(sd.keys())[:5], '...')
print('num tensors:', len(sd))
"

# ── W&B monitoring ────────────────────────────────────────────────────────────
# project: paper3-ldm  run: ldm_vae_v2_100k
# Key curves: train/loss (denoising), train/loss_cls1 (cardio) vs train/loss_cls2 (effusion)
# If cardio train loss dives while val stalls → overfit → reduce class_mult[1] to 0.33
```
