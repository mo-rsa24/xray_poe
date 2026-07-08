# VAE-v2 — Perceptual Fine-Tune for Cardiac Border Sharpness

> This scope runs **after** rebalanced-cardio-retrain and external-transfer tracks
> complete. It is not a dependency of them; it is a parallel improvement lever.
> Source-of-truth for all numbers: the augmented brief (2026-06-28 session).

## Mission
Fine-tune `vae_step0025000.pt` (f=4, 4×128×128 latent, ~25k steps, pixel-MSE only)
with LPIPS perceptual loss (AlexNet, λ=0.1) + PatchGAN discriminator on a combined
NIH+VinDr border-visible diet (~5.5k films), sharpening cardiac border rendering
without regressing |ΔCTR|; then re-encode all NIH latents under the frozen VAE-v2 and
retrain the LDM from scratch in the new latent space, targeting synth-cardio classifier
sensitivity ≥ 0.80 and PoE Gate-3 co-presence yield ≥ 0.30.

## Objectives
1. Lock the architecture (f=4 vs KL-f8) before any compute is spent on training.
2. Build and audit the combined border-visible diet; confirm patient-disjointness from
   the NIH frozen test.
3. Add perceptual pressure (LPIPS + PatchGAN) to the existing VAE checkpoint and train
   to sharpness convergence.
4. Gate the VAE-v2 on perceptual quality while holding the spatial fidelity regression.
5. Re-encode the full NIH training set under the frozen VAE-v2 codec.
6. Retrain the LDM from scratch in the VAE-v2 latent space.
7. Verify the downstream hypothesis: sharper codec → synth-cardio body-fidelity + PoE
   yield recovery.

## Goals
1. **Architecture decided** — f=4 confirmed; f8 rejected with trade-off log (spatial
   resolution at 128² vs 64², LDM attention cost at 16k vs 4k tokens, retrain
   requirement, cardiac-border pixel margin). Decision doc committed.
2. **Diet assembled** — NIH border=Y (N reported, ~4k) + VinDr border=Y (N reported,
   ~1.47k) latents audited and symlink-selected; patient-disjoint from NIH 235
   reserved + frozen test verified; flip + random-crop ±5% augmentation configured
   (~44k effective passes/epoch).
3. **VAE-v2 trained** — fine-tune resumed from `vae_step0025000.pt`; LPIPS λ=0.1 +
   PatchGAN introduced at step 1k (weight ramped 0→1.0 over 1k), KL fixed at 1e-6;
   ≥50k additional steps (total ~75k); convergence verified; checkpoint saved to
   `ckpts/vae_v2/`; W&B run linked.
4. **VAE-v2 gate passed** — on held-out border-visible films: LPIPS < 0.10 (primary),
   lower-zone SSIM ≥ 0.88, |ΔCTR| median < 0.01 (regression hold), cardiac-border
   LPIPS < 0.08; go/no-go recorded in plan 04.
5. **NIH latents re-encoded** — ~70k NIH training films encoded at VAE-v2 frozen
   weights (~1.5 hrs at 19.4 img/s); shape 4×128×128 + finite + N matches manifest
   verified; new cache path documented.
6. **LDM retrained** — 100k steps in VAE-v2 latent space; W&B-logged; sanity overlays
   generated; estimated wall-clock ~5.5 hrs (hippo A100, measured at 100k/5.25hr).
7. **Downstream gates** — synth-cardio Exp-09 classifier sensitivity ≥ 0.80 AND PoE
   Gate-3 yield ≥ 0.30 on NIH frozen separable set; verdict filed against
   pre-registration table; winner checkpoint released to Exp 5–7 or null recorded.

## Expected Outcome
A sharpened VAE-v2 checkpoint (LPIPS < 0.10) that holds spatial fidelity (|ΔCTR|
< 0.01) and drives a downstream LDM whose synth-cardio body-fidelity and PoE yield
clear the headline gates. If gates pass, VAE-v2 + its LDM supersede
`vae_step0025000.pt` and `ckpts_arm1_cleancardio/` for all Exp 5–7 reporting. If
gates fail, the null is recorded and the current codec stands — a clean, logged result
either way. Total estimated pipeline: ~1.5 GPU days (VAE fine-tune ~5–8 hrs +
re-encode ~1.5 hrs + LDM retrain ~5.5 hrs + eval headroom).

## Definition of Done
1. ✅ Architecture decision doc committed — f=4 confirmed, f8 rejected with pixel-margin
   and attention-cost calculation; no ambiguity about which checkpoint family this
   scope operates on.
2. ✅ Combined diet built — NIH border=Y + VinDr border=Y image paths confirmed; N per
   split reported (N_nih=10,409 N_vindr=6,671 N_total=17,080); patient-disjoint check
   logged (intersection=0); augmentation config committed (configs/vae_v2.yaml).
3. ✅ VAE-v2 fine-tune complete — LPIPS + PatchGAN training log shows loss decreasing;
   checkpoint `ckpts/vae_v2/vae_v2_final.pt` saved; W&B run URL recorded (9sjehxmr / blooming-gorge-4).
4. ✅ VAE-v2 gate evaluated — per-gate verdict (pass/fail/margin) in plan 04; overall
   go/no-go signed off; KL spike check for first 2k steps logged.
   → GO on `vae_v2_step0075000.pt` (n=854, all 4 recon gates pass; KL spike NO); Gate 5 posterior
   collapse (3.4% signal) → LDM trains on μ. `results/vae_v2/eval_gate/verdict.md`, timeline [G5a]/[G5b].
5. ✅ NIH+VinDr re-encode complete as μ (encode_moments) — sampled cache was invalidated by
   VAE-v2 posterior collapse (3.6% signal); redone μ (NIH 70,613 + val 3,715, VinDr 6,671,
   std ~0.20, sf 5.034); integrity + signal-fraction logged; V1 cache preserved.
6. ✅ LDM retrain complete — 100k steps from scratch; sanity overlays show border-readable
   hearts; W&B run linked. v4-60k `4vg2h5i4` is the winning LDM. Effusion blunt-rate sanity
   filed 2026-07-08 (guidance-responsive: 32%@w1→66%@w7, median blunts at w≥4; classifier-PASS
   Exp-09 0.865); geometric ≥80% bar recorded as a known limitation, not a blocker — [G6f].
7. ✅ Downstream gates evaluated — Exp-09 sens + Gate-3 yield reported vs targets;
   outcome filed (recovered / null / inconclusive); pre-registration table row updated.
   → Exp-09 recovered (cardio 0.875 via 07 fix, effusion 0.865); Exp-03 co-presence NULL (2.6% « 30%,
   robust to guidance) = mechanistic finding; outcome filed in results/vae_v2/downstream_eval/verdict.md.

## Sub-Scopes
(none)

## Plans
Dependency order: `01 → 02 → 03 → 04 → 05 → 06 → 09 → 07 → 08`
(09-collapse-prevention ran alongside 06's v1→v4 iteration; 07-ldm-enhance diagnostics run before the final 08-downstream re-eval)
- ✅ 01-architecture-decision.md — f=4 vs KL-f8 one-time decision; locks the codec
  family before any training (Goal 1)
- ✅ 02-diet-preparation.md — NIH+VinDr border=Y combined diet; latent audit +
  patient-disjoint check; augmentation config (Goal 2)
- ✅ 03-vae-retrain.md — LPIPS + PatchGAN fine-tune from `vae_step0025000.pt`;
  hippo training run; W&B monitoring (Goal 3)
- ✅ 04-vae-eval-gate.md — |ΔCTR|, SSIM, LPIPS on held-out set; go/no-go decision. GO (n=854, all 4
  gates pass with margin) + Gate 5 posterior-collapse resolved (μ-encode); timeline [G5a]/[G5b] (Goal 4)
- ✅ 05-ldm-reencode.md — re-encode NIH + VinDr latents as μ (encode_moments) under frozen
  VAE-v2; sampled cache was invalidated by posterior collapse, redone μ (Goal 5)
- ✅ 06-ldm-retrain.md — full LDM retrain in VAE-v2 μ-latent space (Goal 6). v1→v4 iteration DONE;
  **v4-60k `4vg2h5i4` per-channel-whitening is the winning LDM** (all classes render, validated in Acts 6–9,
  used by every downstream result). Timeline Act 6. Effusion blunt-rate overlay RAN on mscluster110 Blackwell
  (2026-07-08): guidance-responsive real morphology, 32%@w1→66%@w7, median film blunts at w≥4; geometric
  ≥80% bar recorded as a known limitation (not forced, not a blocker). Closed with caveat — [G6e]/[G6f]
- ✅ 07-ldm-enhance.md — cardio "bigger hearts" utility push on v4-60k: diagnostics A/B/C DONE
  (data is good, model under-draws, big hearts reachable) → free inference-only fix, no retrain;
  target MET — synth-cardio detector sens 0.875 > real 0.806 via w4 best-of-8 (/augment retired,
  retrain rungs parked, 10k premium pile parked — not needed); before→after filed in 08 verdict (Goal 7)
- ✅ 08-downstream-eval.md — synth-cardio sens + PoE yield on NIH frozen set. DONE — verdict NO-GO on
  co-presence: Exp-09 PASS (cardio 0.875 via 07 fix, effusion 0.865), Exp-03 co-presence 2.6% « 30% robust to
  w_cardio → mechanistic null; verdict + pre-reg row (table2_augmentation.md) + exploratory augment-probe
  (null, figure filed) all complete (Goal 7)
- ✅ 09-collapse-prevention.md — recipe (EMA + min-SNR + class rebalance) to stop effusion
  collapse & fix per-seed consistency; resolved via the v1→v4 iteration (Goal 6/7)
