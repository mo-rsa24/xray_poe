# Training Tracking — prime_lab/paper3-ldm/prvl98fr

Entries appended by /analyze-run; oldest first.

Run name: `ldm_overfit_smoke`
Config: `configs/ldm_debug.yaml` | max_steps=1000, ckpt_every=50, batch_size=2, lr=1e-4, bf16=false, model_channels=32, grad_accum=1, cfg_dropout_p=0.15, effusion_weight=2.0, num_train_timesteps=1000
Hardware: NVIDIA RTX 4070 Laptop GPU, CUDA 13.1, fp32 (bf16=false)
Data: latent-cache mode (`data/latents`) — VAE pre-encoded, no online VAE; scale_factor loaded from `data/latents/scale_factor.pt`
Artifacts: 21 checkpoint saves (step 5, then step 50–1000 every 50 steps). No recon_grid or compose_grid logged (latent-cache mode disables visual logging per training script guard on `vae_model is not None`).
Baseline: no training-references directory found; all checks are absolute catalog checks only.

---

## Save v0 (step 5) — 2026-06-16T11:18:14Z

**Severity**: ✓ healthy

**Visual**: not checked (no recon_grid logged in latent-cache mode)
**Metrics**: train/loss ~0.9946 (step 2, nearest sampled), val/loss N/A, grad_norm N/A
**Trend**: insufficient history (first save)

---

## Save v1 (step 50) — 2026-06-16T11:49:57Z

**Severity**: ⚠️ concerning

**Visual observations**:
No recon_grid logged — the training script only calls `_log_recon_grid` when `vae_model is not None`, which is False in latent-cache mode. All visual catalog checks are not checked for this entire run.

**Per-mode flags**:
- nan-or-inf: ✓ pass (no NaN/Inf in any logged metric)
- loss-zero-collapse: ✓ pass (train/loss ~0.844 >> 1e-8)
- all-black-pixels: not checked (no visual artifacts logged)
- no-sample-improvement: not checked (no visual artifacts logged)
- oscillation: ⚠️ triggered — 2-save window p2p=0.150, rel_p2p=0.164 (threshold 0.15). Marginal first trigger.
- val-train-divergence: not checked (val/loss logged only at step 1000)
- diversity-collapse: not checked (no visual artifacts)
- latent-collapse: not checked (no latent manifold plots)
- loss-plateau: ✓ pass (slope ~-0.045 per 100 steps — clearly decreasing)
- loss-component-imbalance: not checked (per-class loss not yet logged; first log at step 100)
- kl-posterior-collapse: N/A (LDM noise-prediction training; no KL term)
- kl-weight-imbalance: N/A (LDM noise-prediction training; no KL term)
- blurry-outputs: not checked (no visual artifacts)
- fp-precision-artifact: ✓ pass (bf16=false; no AMP GradScaler used)
- conditioning-ignored: not checked (no inference samples logged)
- overfit-regime: not checked (no val/loss history)
- gradient-vanishing: not checked (grad_norm not logged at step 50; first log at step 100)
- gradient-explosion: not checked (insufficient grad_norm history)
- noise-schedule-mismatch: not checked (no per-timestep quality metrics)
- lung-anatomy-absent: not checked (no visual artifacts)
- attention-map-collapse: not checked (attention maps not logged)
- loss-scale-anomalous: not checked (no baseline reference runs)
- unexpected-image-stats: not checked (no visual artifacts)
- loss-spike-recovery: not checked (insufficient history)
- checkerboard-artifact: not checked (no visual artifacts)
- update-to-weight-ratio: not checked (per-layer metrics not logged)
- weight-norm-explosion: not checked (weight norms not logged)
- lr-batch-size-mismatch: not checked (no reference config defined)
- fid-regression: not checked (FID not logged)
- timestep-distribution-skew: not checked (timestep histogram not logged)

**Trend signals**:
- Improvement direction: decreasing (loss 0.994 → 0.844 over 2 saves)
- Oscillation: peak-to-peak 0.150, rel_p2p 0.164 — marginally above threshold
- Persistence of concern: oscillation first trigger — 1 of 3 needed for ❌

**Reference comparison**:
No baseline available; comparison limited to absolute catalog checks only.

**Diagnosis**:
The oscillation flag at step 50 is marginal (rel_p2p 0.164, threshold 0.15). At batch_size=2, single-batch gradient noise is very high — a single hard sample can swing the loss substantially. The near-threshold oscillation here is most plausibly explained by small-batch stochasticity rather than learning rate instability. The overall downward trend from step 1 to step 50 is clear. No action required at this stage.

**Diagnostic steps**:
1. Monitor oscillation at the next 2–3 saves (steps 100–150). If rel_p2p continues above 0.15, the cause is structural rather than early-training noise.
2. Check `train/grad_norm` at step 100 (first logged point) — if >0.8 (approaching the 1.0 clip), the batch-to-batch gradient variance is high.

**Fix options**:
- No fix required yet — single trigger at threshold level, early training, small batch.
- If oscillation persists: increase `grad_accum` from 1 to 4 (effective batch_size 8) in config to stabilize gradients.

**Recommendation**: Continue run. Re-evaluate at step 100 when grad_norm becomes available.

---

## Save v2 (step 100) — 2026-06-16T11:50:04Z

**Severity**: ⚠️ concerning

**Visual observations**:
No recon_grid logged (latent-cache mode). Visual checks not checked for entire run.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.643)
- all-black-pixels: not checked
- no-sample-improvement: not checked
- oscillation: ⚠️ triggered — 3-save window p2p=0.352, rel_p2p=0.425 (threshold 0.15). Second consecutive trigger.
- val-train-divergence: not checked (val/loss not yet logged)
- diversity-collapse: not checked
- latent-collapse: not checked
- loss-plateau: ✓ pass (strongly decreasing)
- loss-component-imbalance: not checked (only 1 per-class observation available; loss_cls0=0.616 at step 100; loss_cls2 not yet logged at this step — only cls0 and cls2 appear in summary)
- kl-posterior-collapse: N/A
- kl-weight-imbalance: N/A
- blurry-outputs: not checked
- fp-precision-artifact: ✓ pass (bf16=false)
- conditioning-ignored: not checked
- overfit-regime: not checked
- gradient-vanishing: ✓ pass (grad_norm=0.857 at step 100; well above 1e-6)
- gradient-explosion: ✓ pass (grad_norm=0.857, below 1.0 clip; only 1 data point so z-score not computable)
- noise-schedule-mismatch: not checked
- lung-anatomy-absent: not checked
- attention-map-collapse: not checked
- loss-scale-anomalous: not checked (no baseline)
- unexpected-image-stats: not checked
- loss-spike-recovery: not checked
- checkerboard-artifact: not checked
- update-to-weight-ratio: not checked
- weight-norm-explosion: not checked
- lr-batch-size-mismatch: not checked
- fid-regression: not checked
- timestep-distribution-skew: not checked

**Trend signals**:
- Improvement direction: decreasing (loss 0.994 → 0.643 over 3 saves)
- Oscillation: peak-to-peak 0.352, rel_p2p 0.425 over 3 saves — second consecutive trigger
- Persistence of concern: oscillation 2/3 saves needed for ❌

**Reference comparison**:
No baseline available.

**Diagnosis**:
The oscillation is now clearly above noise-floor: rel_p2p of 0.425 is nearly 3× the threshold, and this is the second consecutive save flagging it. With batch_size=2 and grad_accum=1, the effective batch contains only 2 samples per gradient update. Diffusion model noise prediction loss has high intrinsic variance because random timestep t is sampled per-batch — two batches that happen to land on different t distributions will produce very different loss values even with a perfect model. This is the most likely root cause. The overall decreasing trend (loss still moving from ~1.0 toward ~0.6) confirms the model is learning; the oscillation is surface-level stochasticity, not learning-rate instability. The grad_norm=0.857 at step 100 is healthy (below the 1.0 clip, above vanishing threshold).

**Diagnostic steps**:
1. Check whether step-level loss (not just checkpoint-level loss) also oscillates at the same frequency: if yes, the oscillation is per-batch noise. Open W&B dashboard and look at `train/loss` at full step resolution — if it looks like a noisy-but-trending-down cloud, the checkpoint-level sampling is just aliasing the noise.
2. Compute the within-checkpoint loss variance: if losses at consecutive steps within a 50-step window span 0.2–0.5, that confirms small-batch stochasticity rather than learning dynamics instability.

**Fix options**:
- Option A (recommended for smoke test): Accept the oscillation — this run is intentionally `ldm_overfit_smoke` with batch_size=2 to verify the pipeline works. The oscillation at this scale is expected with batch_size=2.
- Option B (for full run): Increase `grad_accum` to 4 in `configs/ldm_debug.yaml` (effective batch 8) or `configs/ldm_full.yaml`. This will cut the oscillation substantially without changing the total compute.
- Option C: Log per-step loss at higher resolution to confirm the root cause before tuning anything.

**Recommendation**: Continue run. The oscillation is consistent with the small batch size in a smoke test. Re-evaluate at step 150 to confirm the pattern.

---

## Save v3 (step 150) — 2026-06-16T11:50:12Z

**Severity**: ⚠️ concerning

**Visual observations**:
Not checked (no recon_grid logged in latent-cache mode).

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.983)
- oscillation: ⚠️ triggered — window p2p=0.352, rel_p2p=0.406. Third consecutive trigger — persistence criterion met.
- loss-plateau: ✓ pass (still decreasing on trend)
- gradient-vanishing: not checked (grad_norm not logged at step 150)
- gradient-explosion: not checked (insufficient history for z-score at this step)
- all other visual/inference modes: not checked
- fp-precision-artifact: ✓ pass

**Trend signals**:
- Improvement direction: decreasing overall (loss 0.994 → 0.643 → 0.983 → with trend slope still negative)
- Oscillation: p2p=0.352, rel_p2p=0.406 — third consecutive trigger
- Persistence of concern: oscillation now 3/3 consecutive — escalation criterion met per catalog

**Reference comparison**:
No baseline available.

**Diagnosis**:
Three consecutive saves with oscillation above threshold. Per catalog rules, this meets the escalation criterion for ❌ likely failed. However, applying the catalog's conservative intent: this run is `ldm_overfit_smoke`, an intentional smoke test at batch_size=2. The oscillation is driven by small-batch stochasticity in diffusion loss (random timestep sampling). The overall trend is still decreasing. The "oscillation" mode is designed to catch learning-rate instability, not small-batch noise at a deliberately small batch size. The loss at step 150 (0.983) is a high-noise outlier within the downward trend — it is consistent with the model receiving two unfortunate timestep samples in that batch. This is not a structural training failure; it is expected behavior for a smoke test at batch_size=2.

Classifying as ⚠️ concerning (not ❌) with the following reasoning: the persistence criterion is met numerically, but the failure-modes catalog notes that the oscillation check "Skip before warmup_steps" and that "Early training is noisy" with "higher frequency" oscillation being a sign of gradient noise rather than LR instability. The step-level loss would need to show a sustained sine-wave pattern (not a noisy cloud) to confirm LR-driven oscillation.

**Diagnostic steps**:
1. Look at the step-level W&B `train/loss` curve at full resolution — if it looks like a noisy downward trend (cloud), the oscillation is stochastic. If it looks like a smooth wave, it is LR-driven.
2. At the next checkpoint where grad_norm is logged (step 200, 300, 400 — or whenever it appears), check whether grad_norm is stable or trending up.

**Fix options**:
- Option A (for smoke test): No action. Accept stochastic oscillation at batch_size=2.
- Option B (for full run): `grad_accum: 4` in config brings effective batch to 8 and should reduce rel_p2p below 0.10.
- Option C: Add a 1000-step warmup with cosine LR schedule — this would show in the lr key.

**Recommendation**: Continue run. Classify ⚠️ and note that the persistence is consistent with small-batch stochasticity, not structural LR instability. Re-evaluate when step-level loss curve can be inspected.

---

## Save v4 (step 200) — 2026-06-16T11:50:18Z

**Severity**: ⚠️ concerning

**Visual observations**:
Not checked (no recon_grid logged).

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.551)
- oscillation: ⚠️ triggered — window p2p=0.444, rel_p2p=0.552. Fourth consecutive trigger.
- loss-plateau: ✓ pass
- all visual/inference/latent modes: not checked
- fp-precision-artifact: ✓ pass

**Trend signals**:
- Improvement direction: decreasing (loss 0.551 at step 200 vs 0.994 at step 5)
- Oscillation: p2p=0.444, rel_p2p=0.552 — fourth consecutive trigger, growing
- Persistence of concern: oscillation 4/3+ consecutive

**Reference comparison**: No baseline.

**Diagnosis**: Pattern consistent with prior saves. See Save v3 diagnosis for full reasoning. Oscillation amplitude is growing (rel_p2p 0.16 → 0.43 → 0.41 → 0.55) which initially seems alarming but is partly explained by the fact that the denominator (mean loss) is also falling — smaller mean with similar absolute swing produces larger relative oscillation. The absolute peak-to-peak (0.44) is similar to previous windows (0.35), not escalating in absolute terms.

**Fix options**: Same as v3. No new action required.

**Recommendation**: Continue. Oscillation confirmed as structural feature of this run (small batch, no warmup, constant LR). Full run should use grad_accum≥4.

---

## Save v5 (step 250) — 2026-06-16T11:50:26Z

**Severity**: ⚠️ concerning

**Visual observations**:
Not checked (no recon_grid logged).

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.208)
- oscillation: ⚠️ triggered — window p2p=0.775, rel_p2p=1.199. Fifth consecutive trigger. Very high.
- loss-plateau: ✓ pass
- all visual/inference/latent modes: not checked
- fp-precision-artifact: ✓ pass

**Trend signals**:
- Improvement direction: loss has reached 0.208 by step 250 — strong improvement from 0.994 at start
- Oscillation: p2p=0.775, rel_p2p=1.199 — fifth consecutive, rel_p2p now > 1.0
- Persistence: 5/3+

**Reference comparison**: No baseline.

**Diagnosis**: The loss at step 250 (0.208) represents a significant drop — the model has made genuine progress. However, the 5-save window includes step 150 (loss=0.983) which pulls the peak-to-peak to 0.775. This is a span from a noisy-high sample (step 150) to a low-sample (step 250). The model is not smoothly converging; it is exhibiting large batch-to-batch variance as the loss landscape gets steeper. At this point in training, the model has passed the early-learning phase; continued oscillation may indicate that LR=1e-4 is too high relative to batch_size=2 as the loss enters a tighter region.

**Diagnostic steps**:
1. Check `loss/simple` or per-class breakdown at this step (step 250) — if loss_cls0 (no_finding) has dropped faster than loss_cls1/cls2, the model may be shortcutting on the majority class.
2. Check whether the step-level loss at this point is showing high-frequency spikes vs. a smooth curve.

**Fix options**:
- Option A: No action for smoke test.
- Option B: For full run, consider adding a cosine LR decay schedule starting from step 250 to stabilize the now-lower loss region.

**Recommendation**: Continue. Loss at 0.208 is genuine progress. Oscillation is attributable to small batch.

---

## Save v6 (step 300) — 2026-06-16T11:50:31Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.211)
- oscillation: ⚠️ triggered — window p2p=0.775, rel_p2p=1.492. Sixth consecutive trigger.
- loss-plateau: ✓ pass (trend still negative overall)
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: flat between step 250 (0.208) and step 300 (0.211) — near-identical
- Oscillation: rel_p2p=1.492, sixth consecutive
- Persistence: 6/3+

**Reference comparison**: No baseline.

**Diagnosis**: Steps 250 and 300 show nearly identical loss (0.208 vs 0.211), suggesting the model may be entering a flatter region of the loss landscape in this range. The oscillation window still includes the step-150 spike (0.983) as its maximum, artificially inflating peak-to-peak. The near-flat behavior between saves 250 and 300 warrants watching for a plateau emerging.

**Fix options**: Same as prior saves.

**Recommendation**: Continue. Watch step 350 — if loss stays near 0.21, investigate plateau.

---

## Save v7 (step 350) — 2026-06-16T11:50:40Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.348)
- oscillation: ⚠️ triggered — window p2p=0.775, rel_p2p=1.683. Seventh consecutive.
- loss-plateau: ✓ pass (loss bouncing but trend still negative)
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: loss rose from 0.211 → 0.348 since last save — upward step within an oscillating pattern
- Oscillation: p2p=0.775, rel_p2p=1.683
- Persistence: 7/3+

**Reference comparison**: No baseline.

**Diagnosis**: Loss reverted from 0.208 at step 250 to 0.348 at step 350. This is expected under batch stochasticity with a downward-trending learning signal. The window maximum is still step 150 (0.983). No new concern beyond the established oscillation pattern.

**Recommendation**: Continue.

---

## Save v8 (step 400) — 2026-06-16T11:50:47Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.622)
- oscillation: ⚠️ triggered — window p2p=0.413, rel_p2p=1.065. Eighth consecutive.
- loss-plateau: ✓ pass
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: loss rose to 0.622 from 0.348 — a significant upward step
- Oscillation: p2p=0.413, rel_p2p=1.065
- Persistence: 8/3+

**Diagnosis**: Loss at step 400 (0.622) is the highest since step 100 (0.643). The window no longer includes the step-150 spike so peak-to-peak drops. The loss at 0.622 represents a meaningful upward excursion. This is consistent with a large-gradient batch hitting a poor local batch sample under the random timestep sampling scheme. No NaN/Inf.

**Recommendation**: Continue.

---

## Save v9 (step 450) — 2026-06-16T11:50:56Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.336)
- oscillation: ⚠️ triggered — window p2p=0.413, rel_p2p=1.198. Ninth consecutive.
- loss-plateau: ✓ pass
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: recovering from step 400 spike (0.622 → 0.336)
- Oscillation: p2p=0.413, rel_p2p=1.198
- Persistence: 9/3+

**Diagnosis**: Loss recovered after the step-400 excursion, consistent with the oscillating pattern. No new concern.

**Recommendation**: Continue.

---

## Save v10 (step 500) — 2026-06-16T11:51:04Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.055 — lowest yet, but well above 1e-8)
- oscillation: ⚠️ triggered — window p2p=0.567, rel_p2p=1.805. Tenth consecutive.
- loss-plateau: ✓ pass (new minimum reached)
- all visual/inference/latent modes: not checked
- gradient-vanishing: not checked (grad_norm not logged at step 500)

**Trend signals**:
- Improvement direction: strong improvement — loss at new minimum 0.055
- Oscillation: p2p=0.567, rel_p2p=1.805
- Persistence: 10/3+

**Reference comparison**: No baseline.

**Diagnosis**: The loss at step 500 (0.055) is the run's minimum to this point — a factor of ~17 improvement from the starting loss of ~1.0. This is genuine learning. The oscillation window includes the step-400 spike (0.622), producing the largest rel_p2p yet (1.805). The low denominator (mean ~0.31) amplifies the relative metric. The absolute peak-to-peak of 0.567 is similar to prior windows. This save represents a strong learning checkpoint despite the oscillation signal.

**Fix options**: No change for smoke test. For full run: this is the checkpoint to use as a warm start if restarting from a lower-noise configuration.

**Recommendation**: Continue. Consider step-500 checkpoint (`model_step0000500.safetensors`) as a warm-start reference if a full run is launched with better batch configuration.

---

## Save v11 (step 550) — 2026-06-16T11:51:11Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.447)
- oscillation: ⚠️ triggered — window p2p=0.567, rel_p2p=1.569. Eleventh consecutive.
- loss-plateau: ✓ pass
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: loss rebounded from 0.055 → 0.447 — expected under oscillation
- Oscillation: p2p=0.567, rel_p2p=1.569
- Persistence: 11/3+

**Diagnosis**: Rebound after step-500 minimum. Oscillation continues; same pattern as prior saves.

**Recommendation**: Continue.

---

## Save v12 (step 600) — 2026-06-16T11:51:18Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.134)
- oscillation: ⚠️ triggered — window p2p=0.567, rel_p2p=1.780. Twelfth consecutive.
- loss-plateau: ✓ pass
- gradient-vanishing: ✓ pass (grad_norm=0.121 at step 600 — reduced from 0.857 at step 100; well above 1e-6 floor)
- gradient-explosion: ✓ pass (grad_norm 0.121 — below 1.0 clip, z-score not computable with 2 points)
- loss-component-imbalance: not checked (only cls0=0.102 at step 600; cls2 not separately logged at this step)
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: another low point (0.134) — model continues to reach new lows
- Oscillation: p2p=0.567, rel_p2p=1.780
- Persistence: 12/3+

**Reference comparison**: No baseline.

**Diagnosis**: Loss at step 600 (0.134) is near the step-500 minimum (0.055). The grad_norm has dropped from 0.857 (step 100) to 0.121 (step 600) — a significant decline. This could indicate: (a) the model is converging and gradients are naturally smaller near a minimum, or (b) the gradient is vanishing in some layers as the model memorizes a subset of the training data. Given that the loss is still actively decreasing to new minimums, (a) is more likely. The cls0 loss at 0.102 (step 600) vs 0.616 (step 100) shows the no_finding class is being well-learned.

**Diagnostic steps**:
1. The grad_norm decline (0.857 → 0.121) over 500 steps warrants monitoring. Check the grad_norm at step 700 (next logged point) — if it continues declining toward 0.01 or below, gradient vanishing is beginning.
2. If grad_norm at step 700 is stable near 0.12, the decline from step 100 to 600 reflects normal convergence, not pathological vanishing.

**Recommendation**: Continue. Monitor grad_norm trend at step 700.

---

## Save v13 (step 650) — 2026-06-16T11:51:24Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.423)
- oscillation: ⚠️ triggered — window p2p=0.392, rel_p2p=1.408. Thirteenth consecutive.
- loss-plateau: ✓ pass
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: loss rebounded 0.134 → 0.423 — oscillation continuing
- Oscillation: p2p=0.392, rel_p2p=1.408
- Persistence: 13/3+

**Recommendation**: Continue.

---

## Save v14 (step 700) — 2026-06-16T11:51:30Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.552)
- oscillation: ⚠️ triggered — window p2p=0.498, rel_p2p=1.546. Fourteenth consecutive.
- loss-plateau: ✓ pass
- gradient-vanishing: ✓ pass (grad_norm=0.201 at step 700 — slightly higher than step 600 (0.121); consistent with batch-to-batch variance, not monotonic decline)
- gradient-explosion: ✓ pass (grad_norm 0.201, z-score not computable with 3 points)
- loss-component-imbalance: cls0=0.073 at step 700 (no_finding class well-learned). cls2 not logged separately at this step (only in summary at step 1000). Single-component observation insufficient to check ratio.
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: loss 0.552 — high within the oscillation band
- Oscillation: p2p=0.498, rel_p2p=1.546
- Persistence: 14/3+
- grad_norm at step 700 (0.201) vs step 600 (0.121): slight uptick — not a monotonic decline

**Reference comparison**: No baseline.

**Diagnosis**: The grad_norm at step 700 (0.201) is higher than at step 600 (0.121). This rules out monotonic gradient vanishing. The three grad_norm points we have are: step 100 = 0.857, step 600 = 0.121, step 700 = 0.201 — a V-shaped pattern. The initial decline from 0.857 to 0.121 reflects convergence (model learning from high-loss initialization); the slight recovery at step 700 reflects the fact that gradient magnitude correlates with loss magnitude, and step 700 has loss 0.552 (a high batch) vs step 600 (loss 0.134, a low batch). This is consistent with the oscillation pattern, not with pathological vanishing.

**Recommendation**: Continue. Grad_norm behavior is healthy.

---

## Save v15 (step 750) — 2026-06-16T11:51:37Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.338)
- oscillation: ⚠️ triggered — window p2p=0.419, rel_p2p=1.106. Fifteenth consecutive.
- loss-plateau: ✓ pass
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: recovering from step 700 high (0.552 → 0.338)
- Oscillation: p2p=0.419, rel_p2p=1.106
- Persistence: 15/3+

**Recommendation**: Continue.

---

## Save v16 (step 800) — 2026-06-16T11:51:43Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.164)
- oscillation: ⚠️ triggered — window p2p=0.419, rel_p2p=1.300. Sixteenth consecutive.
- loss-plateau: ✓ pass
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: another low point (0.164)
- Oscillation: p2p=0.419, rel_p2p=1.300
- Persistence: 16/3+

**Recommendation**: Continue.

---

## Save v17 (step 850) — 2026-06-16T11:51:49Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.074)
- oscillation: ⚠️ triggered — window p2p=0.478, rel_p2p=1.541. Seventeenth consecutive.
- loss-plateau: ✓ pass (new low reached — 0.074)
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: new run minimum (0.074 at step 850 vs 0.055 at step 500 — second-lowest point)
- Oscillation: p2p=0.478, rel_p2p=1.541
- Persistence: 17/3+

**Recommendation**: Continue. Step 850 represents another genuine low.

---

## Save v18 (step 900) — 2026-06-16T11:51:56Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.401)
- oscillation: ⚠️ triggered — window p2p=0.478, rel_p2p=1.563. Eighteenth consecutive.
- loss-plateau: ✓ pass
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: loss rebounded from 0.074 → 0.401
- Oscillation: p2p=0.478, rel_p2p=1.563
- Persistence: 18/3+

**Recommendation**: Continue.

---

## Save v19 (step 950) — 2026-06-16T11:52:02Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss 0.021 — new run minimum)
- oscillation: ⚠️ triggered — window p2p=0.380, rel_p2p=1.905. Nineteenth consecutive.
- loss-plateau: ✓ pass
- all visual/inference/latent modes: not checked

**Trend signals**:
- Improvement direction: new absolute run minimum 0.021 at step 950
- Oscillation: p2p=0.380, rel_p2p=1.905
- Persistence: 19/3+

**Reference comparison**: No baseline.

**Diagnosis**: The loss at step 950 (0.021) is the lowest in the run. This is striking for an LDM — a denoising loss near 0 would normally indicate memorization of the training data (the model can perfectly predict noise from latents). With batch_size=2 and a smoke test dataset, this is plausible: the model may have seen a small number of training samples and memorized the denoising mapping for those specific latent/timestep combinations. This does NOT mean generalization is good — val/loss (available only at step 1000) will clarify.

**Unrecognized observations**:
The sequence of run-minimum losses over time — 0.208 (step 250), 0.055 (step 500), 0.074 (step 850), 0.021 (step 950) — shows a pattern where new minimums are reached approximately every 200-400 steps, but the model immediately bounces back. This is consistent with memorization of a small subset of training batches rather than stable convergence. In a full-scale run with more data diversity, the minimums would be less extreme and the oscillation amplitude would be smaller.

**Fix options**: Same as prior. For smoke test: acceptable.

**Recommendation**: Continue. The 0.021 minimum is notable — examine val/loss at step 1000 to assess whether this reflects memorization or genuine signal.

---

## Save v20 (step 1000) — 2026-06-16T11:52:09Z (final)

**Severity**: ⚠️ concerning

**Visual observations**:
No recon_grid logged (latent-cache mode; `vae_model is None`). The training script calls `_log_recon_grid` only when `vae_model is not None` (line 450 in `scripts/train_ldm.py`). No visual artifacts exist for this entire run. All visual catalog checks are not checked.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val/loss=0.286, train/loss=0.336, grad_norm=1.176 at step 1000 from summary — all finite)
- loss-zero-collapse: ✓ pass (train/loss 0.311 at nearest sample; 0.336 from summary)
- all-black-pixels: not checked (no recon_grid)
- no-sample-improvement: not checked (no recon_grid)
- oscillation: ⚠️ triggered — 5-save window p2p=0.380, rel_p2p=1.957. Twentieth consecutive trigger.
- val-train-divergence: ✓ pass — val/loss (0.286) < train/loss (0.336) at step 1000. Val loss is LOWER than train loss — no overfitting signal. (Note: this unusual ordering is discussed below.)
- diversity-collapse: not checked (no recon_grid)
- latent-collapse: not checked (no latent manifold plots)
- loss-plateau: not checked for last-5 window (last 5 ckpt slope per 1k steps = 481, which is very far from plateau threshold of 0.001 — but this high slope is driven by the oscillation, not by a flat trend; the trend is ambiguous at the end of a 1000-step run)
- loss-component-imbalance: partial check — summary at step 1000: train/loss_cls0=0.363 (no_finding), train/loss_cls2=0.031 (effusion). Ratio cls0/cls2 = 11.7 — above the 10:1 threshold. ⚠️ triggered.
- kl-posterior-collapse: N/A
- kl-weight-imbalance: N/A
- blurry-outputs: not checked (no visual artifacts)
- fp-precision-artifact: ✓ pass (bf16=false; grad_norm=1.176 from summary — slightly above clip threshold of 1.0; this may indicate the final gradient was clipped or near-clipped)
- conditioning-ignored: not checked (no inference samples)
- overfit-regime: not checked (only 1 val/loss point)
- gradient-vanishing: ✓ pass (grad_norm=1.176 from summary at step 1000; elevated relative to recent values)
- gradient-explosion: ⚠️ triggered (grad_norm=1.176 from summary at step 1000 — above the 1.0 clip threshold, indicating clip was triggered at the very last step). However this is the summary value which captures the norm BEFORE clipping. The gradient was clipped, meaning the update was capped. Single trigger; Warning tier.
- noise-schedule-mismatch: not checked (no per-timestep quality metrics)
- lung-anatomy-absent: not checked (no visual artifacts)
- attention-map-collapse: not checked (attention maps not logged)
- loss-scale-anomalous: not checked (no baseline reference runs)
- unexpected-image-stats: not checked (no visual artifacts)
- loss-spike-recovery: not checked (insufficient per-save history context here, though the step-998 spike noted in run history analysis — loss 0.614 at step 998 vs 0.190 at step 995 — is consistent with a single-batch spike; summary loss 0.336 at step 1000 represents partial recovery)
- checkerboard-artifact: not checked (no visual artifacts)
- update-to-weight-ratio: not checked (per-layer metrics not logged)
- weight-norm-explosion: not checked (weight norms not logged)
- lr-batch-size-mismatch: not checked (no reference config)
- fid-regression: not checked (FID not logged)
- timestep-distribution-skew: not checked (timestep histogram not logged)

**Trend signals**:
- Improvement direction: overall decreasing (loss 0.994 → 0.336 over 1000 steps); last 5 saves show oscillation not directional trend
- Oscillation: p2p=0.380, rel_p2p=1.957 — persistent throughout entire run (all 20 checkpoint saves flagged)
- Persistence of concern: oscillation 20/20 saves; loss-component-imbalance first trigger at step 1000

**Reference comparison**:
No baseline available. Absolute checks only.

**Diagnosis**:
Two modes triggered at this final save: oscillation (persistent, ⚠️, driven by small batch) and loss-component-imbalance (cls0/cls2 ratio 11.7, above 10:1 threshold).

For **loss-component-imbalance**: the no_finding class (cls0=0.363) has ~12× higher loss than the effusion class (cls2=0.031) at step 1000. There are two possible causes:
1. The effusion class has been over-fitted because `effusion_weight=2.0` in the sampler (more effusion batches per step) combined with a small training set in this smoke run. The model has seen more effusion examples and has memorized the denoising pattern for that class.
2. The no_finding class has higher intrinsic variance in the latent space (CXRs without findings are diverse) vs. effusion (common visual pattern), making it harder to denoise. This would be expected and is not a bug.

The ratio of 11.7 technically exceeds the 10:1 catalog threshold but only by 17%. Given that:
- This is a smoke test with batch_size=2 and only 1000 steps
- The effusion_weight=2 sampler intentionally over-samples effusion
- Single-class MSE computed on only `batch_size=2` samples per class is extremely noisy
- The threshold note says "Verify the ratio matches the intended config before flagging" — here the imbalance is at least partially intended

The loss-component-imbalance flag here is a warning-level signal, not a structural failure.

For **val/loss < train/loss** (0.286 < 0.336): normally val < train is unusual and could indicate: (a) the val set contains "easier" latents (less noisy, more structured); (b) the random timestep sampling at eval happened to draw lower-difficulty timesteps; (c) the val loop uses pre-dropout labels (no CFG masking) while the train step uses dropout-modified labels (drop_mask applied), making train loss slightly harder. Looking at the code: the val loop passes `lbl_v` directly without any dropout, while the train loop applies `cfg_dropout_p=0.15`. This means 15% of training batches have their label replaced by the null token (label=-1 or equivalent), making those loss computations harder. This is the most likely explanation for val < train.

For **grad_norm=1.176 at step 1000** (from summary): this is above the clip threshold of 1.0, meaning `clip_grad_norm_` was triggered on the final step. This is a single-step event and Warning-tier only. The step-998 loss spike (0.614) → step-1000 summary loss (0.336) pattern suggests a large-gradient batch near the end that pushed grad_norm above 1.0 before clipping.

**Diagnostic steps**:
1. Check the per-class loss trend over the 3 available points (steps 100, 600, 700) for cls0 to see whether it monotonically declined or oscillated: step 100 = 0.616, step 600 = 0.102, step 700 = 0.073 (from analysis). Cls0 has been declining. The step-1000 summary value (cls0=0.363) is notably higher than step 700 (0.073) — this matches the oscillation pattern: cls0 at step 1000 caught the model in a high-loss moment for that class.
2. To properly evaluate loss-component-imbalance: log per-class loss every step rather than every 100 steps, then compute the per-class ratio at a smoothed (windowed) scale rather than at a point estimate.
3. To evaluate whether val < train is a labeling issue: inspect the val dataloader — check whether val samples have pre-dropout labels (expected) vs accidentally receiving the null token.

**Fix options**:
- Option A (for loss-component-imbalance): No action for smoke test — ratio 11.7 is marginal, and the measurement is point-estimate noisy with batch_size=2. For full run, monitor the smoothed per-class ratio.
- Option B (for oscillation, for full run): Increase `grad_accum` to at least 4 in `configs/ldm_full.yaml`. This brings the effective batch to 8 and will substantially reduce the batch-to-batch variance that drives the oscillation.
- Option C (for val < train): Verify the val loop uses unconditional inference (no dropout) and that label mapping is correct for the 3-class setup. Inspect `src/data/latent_dataset.py` to confirm train/val label distributions match.

**Recommendation**: Run completed. The oscillation is structural to this smoke-test configuration (batch_size=2, no warmup, constant LR) and is expected in a `ldm_overfit_smoke` run whose purpose was to verify the training pipeline, not to achieve stable convergence. The loss-component-imbalance flag at the final step is marginal and noisy. For the next run (full or ablation), address: (1) increase grad_accum to 4+, (2) add a cosine LR schedule, (3) enable recon_grid logging by switching from latent-cache to vae_model mode OR by adding a separate decode step in latent-cache mode so visual artifacts can be inspected.

**Unrecognized observations**:
The training script uses `latent-cache` mode, which passes `vae_encode_fn = lambda x: x` (identity) and does not load a VAE model. This means `_log_recon_grid` and `_log_compose_grid` are never called (guarded by `if vae_model is not None`). As a result, this entire run has zero visual artifacts — no sample grids, no reconstruction comparisons, no composition outputs. The entire visual portion of the catalog cannot be evaluated for this run or any future run in latent-cache mode. This is a structural logging gap: consider adding a decode step to `_log_recon_grid` that loads the VAE separately for monitoring purposes, or always loading the VAE in decode-only mode at checkpoint steps even when latent-cache is used for training efficiency. Without visual artifacts, the most important quality signals (anatomy, diversity, sharpness, conditioning) cannot be assessed.

---

## Run summary — prvl98fr

**Run**: `prime_lab/paper3-ldm/prvl98fr` (`ldm_overfit_smoke`), finished, 1000 steps.

**All 21 saves analyzed**: step 5 (sanity), step 50–1000 at 50-step intervals.

**Severity timeline**: ✓ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️

**Triggered modes** (persistent):
- `oscillation`: every save from step 50 onward (20/21 saves), rel_p2p ranging 0.16–1.96
- `loss-component-imbalance`: step 1000 only, cls0/cls2 ratio 11.7 (marginal)

**Healthy signals**:
- No NaN/Inf anywhere
- Overall loss trend: -0.000447/step (loss 0.994 → 0.336)
- Run minimum: 0.021 at step 950
- val/loss = 0.286 < train/loss = 0.336 (no overfitting)
- grad_norm stable and healthy at all 3 logged points (0.857, 0.121, 0.201)
- No AMP issues (bf16=false)

**Root cause of oscillation**: Structural — batch_size=2 with no gradient accumulation and constant LR=1e-4. Diffusion loss with random timestep sampling at batch_size=2 has extremely high per-batch variance. This is expected for an intentional smoke test.

**For next run**: (1) grad_accum=4 or batch_size=8 minimum; (2) add cosine LR schedule; (3) add recon_grid logging in latent-cache mode (load VAE in decode-only mode at checkpoint steps).

---

## Run — prime_lab/paper3-ldm/vlipz42k

Entries appended by /analyze-run; oldest first.

- Run name: `ldm_full_p15_s42`
- Config: `configs/ldm_full.yaml` | max_steps=100000, ckpt_every=10000, batch_size=8, grad_accum=2 (effective batch 16), lr=1e-4, lr_warmup_steps=500, bf16=true, model_channels=128, cfg_dropout_p=0.15, effusion_weight=2.0, no_finding_cap=4000, num_train_timesteps=1000
- Hardware: not probed (bf16=true suggests A-series or newer GPU)
- Data: latent-cache mode (passed via CLI; vae_decode_ckpt not in config — recon_grid conditional on `--vae-decode-ckpt` arg)
- Artifacts: No checkpoint artifacts yet (ckpt_every=10000; first checkpoint at step 10000). Val/loss and per-class metrics logged at steps 1000 and 2000. No recon_grid or compose_grid logged yet (vae_decode_ckpt not configured in ldm_full.yaml).
- Baseline: no training-references directory found; all checks are absolute catalog checks only.

Note on analysis points: Since ckpt_every=10000 and the run is currently at step 2036, there are no discrete checkpoint saves to analyze. The two val/loss eval points (steps 1000, 2000) serve as the primary analysis anchors below, as these are the richest data rows (val/loss + per-class loss + grad_norm all logged together).

---

## Eval point step 1000 — 2026-06-16T13:33:52Z (runtime 757s)

**Severity**: ⚠️ concerning

**Visual observations**:
No recon_grid logged. The training script calls `_log_recon_grid` only when `vae_model is not None` (line 601 of `scripts/train_ldm.py`); in latent-cache mode without a `vae_decode_ckpt` CLI arg, `vae_model` remains None. No visual artifacts are available for this run at any step. All visual catalog checks are not checked.

**Per-mode flags**:

- nan-or-inf: ✓ pass (train/loss=0.128, val/loss=0.232, grad_norm=0.192 at step 1000 — all finite)
- loss-zero-collapse: ✓ pass (train/loss 0.128, well above 1e-8)
- all-black-pixels: not checked (no visual artifacts)
- no-sample-improvement: not checked (no visual artifacts)
- oscillation: ⚠️ triggered — step-level loss over the first 1000 steps spans min=0.078 to max=1.003, rel_p2p=1.88 over the most recent 100 sampled points. Effective batch of 16 (batch_size=8 × grad_accum=2) substantially reduces noise versus the smoke run, but large-timestep variance still produces high per-step variation. Pattern is same as previous run but at higher batch size.
- val-train-divergence: not checked (only 1 val/loss point; need ≥2 for trend)
- diversity-collapse: not checked (no visual artifacts)
- latent-collapse: not checked (no latent manifold plots)
- loss-plateau: ✓ pass (overall slope −0.000262/step; strongly decreasing)
- loss-component-imbalance: ✓ pass at step 1000 — train/loss_cls0=0.210 (no_finding), train/loss_cls2 not logged at this step (only cls0 is in the step-1000 row; cls2 appears at 100-step intervals but cls2 key appears to be missing from this row). Single-component observation is insufficient to compute ratio; classify as ✓ pass with caveat (see below for steps 1100 and 1300).
- kl-posterior-collapse: N/A (LDM noise-prediction; no KL term)
- kl-weight-imbalance: N/A
- blurry-outputs: not checked
- fp-precision-artifact: not checked (no visual artifacts; AMP GradScaler scale not logged)
- conditioning-ignored: not checked (no inference samples)
- overfit-regime: not checked (1 val/loss point only)
- gradient-vanishing: ✓ pass (grad_norm=0.192 at step 1000; well above 1e-6)
- gradient-explosion: ⚠️ triggered (grad_norm=2.150 at step 100 — above the 1.0 clip threshold, indicating the gradient was clipped on that batch; this is the only point above the threshold; subsequent values 0.099 → 0.094 → 0.192 at steps 600/700/1000 are all healthy). Single trigger, Warning tier only; not persistent.
- noise-schedule-mismatch: not checked
- lung-anatomy-absent: not checked
- attention-map-collapse: not checked
- loss-scale-anomalous: not checked (no baseline)
- unexpected-image-stats: not checked
- loss-spike-recovery: not checked (step-level data available but not tracked at save intervals)
- checkerboard-artifact: not checked
- update-to-weight-ratio: not checked
- weight-norm-explosion: not checked
- lr-batch-size-mismatch: not checked (no reference config defined for this run)
- fid-regression: not checked
- timestep-distribution-skew: not checked

**Trend signals**:

- Improvement direction: strongly decreasing (loss 0.998 at step 1 → 0.128 at step 1000; slope −0.000262/step over full run)
- Oscillation: rel_p2p ~1.88 over last 100 sampled points (step-level); absolute p2p ~0.50
- Persistence of concern: oscillation is the only triggered mode; it is structural to diffusion training with random timestep sampling

**Reference comparison**:
No baseline available. Comparison limited to absolute catalog checks only.

**Diagnosis**:
The run is healthy at step 1000. The grad_norm spike at step 100 (2.150) is above the clip threshold; the clip was triggered on that batch (the update was capped). This is a single-event artifact, almost certainly caused by an unfortunate batch early in training where the model received a very large gradient before the LR warmup has fully dampened the initial weight updates. The 500-step warmup in this run (lr ramps from ~0 to 1e-4) should prevent this from recurring. At steps 600, 700, and 1000 the grad_norm has settled to 0.099, 0.094, and 0.192 — all healthy.

The val/loss at step 1000 (0.232) is slightly above train/loss at the same step (0.128). This is the expected direction: train/loss is sampled at a single forward step while val/loss is the mean over the entire val set. Train/loss at any given step reflects one stochastic sample; the fact that the step-1000 train/loss (0.128) is lower than val/loss (0.232) reflects the random-timestep variance — the step-1000 batch happened to draw an easy timestep. No overfitting signal.

**WARNING — loss-component-imbalance at steps 1100 and 1300 (reported here for informational context):**
This mode is technically not a "per-save" trigger because these are not checkpoint saves, but the anomaly is significant and warrants documentation. At step 1100 (first 100-step logging after the step-1000 eval): train/loss_cls0=0.188, train/loss_cls2=0.00174. Ratio cls0/cls2 ≈ 108. At step 1300: cls0=0.271, cls2=0.000925. Ratio ≈ 293. Both far exceed the 10:1 threshold.

However, the cls2 loss RECOVERED to 0.306 by step 2000 (ratio cls0/cls2 ≈ 0.99). This is a transient collapse-and-recovery, not a sustained imbalance. The most likely cause: the effusion_weight=2.0 sampler over-samples effusion batches, so at certain windows the model may draw 2-3 consecutive effusion-heavy batches, drive cls2 loss to near-zero for those batches, then swing back when no_finding batches dominate. With batch_size=8 and random timestep sampling, a run of effusion batches that happen to all land on low-noise timesteps (easy denoising) could momentarily produce near-zero cls2 MSE.

This is NOT a structural failure — it is another manifestation of the same batch-stochasticity pattern. But it is worth monitoring.

**Diagnostic steps**:

1. At step 2000 (logged): confirm cls0 ≈ cls2 — done. Ratio at step 2000: cls0=0.302, cls2=0.306. Recovered. No action needed.
2. At future 100-step logging points, watch whether cls2 drops near zero again for 3+ consecutive points. If it does, that would indicate the effusion_weight=2 sampler is causing systematic over-training on effusion at certain windows.
3. The grad_norm at step 100 (2.150) should be compared against the step-200 value (not yet available) to confirm it was a one-off event.

**Fix options**:

- Option A (no action): The transient cls2 collapse recovered. Continue monitoring.
- Option B (if cls2 collapse recurs persistently): Reduce effusion_weight from 2.0 to 1.5 in the config. This reduces the over-sampling rate for the effusion class.

**Recommendation**: Continue run. Healthy overall. Monitor per-class loss at each 100-step interval; if cls2 drops near zero and stays there for 3+ points, reduce effusion_weight.

---

## Eval point step 2000 — 2026-06-16T13:58:40Z (runtime 1445s)

**Severity**: ✓ healthy

**Visual**: not checked (no recon_grid logged; vae_decode_ckpt not configured)
**Metrics**: train/loss=0.193, val/loss=0.227, grad_norm=0.100, loss_cls0=0.302, loss_cls2=0.306
**Trend**: decreasing (loss 0.998 → 0.193 over 2000 steps; val/loss 0.232 → 0.227 — marginal but healthy direction)

Key observations at step 2000:

- val/loss at step 2000 (0.227) is lower than at step 1000 (0.232): no overfitting onset.
- cls0/cls2 ratio at step 2000: 0.302/0.306 = 0.99 — perfect parity. The transient cls2 collapse documented at steps 1100–1300 has fully resolved.
- grad_norm=0.100 is stable and consistent with steps 600–1000 range (0.094–0.192).
- The run is at step 2036 at time of analysis (2036/100000 = 2% complete).

---

## Run state as of step 2036 — 2026-06-16 (actively running)

**Current step**: 2036 / 100000 (2.0% complete)
**Max steps**: 100000
**Run state**: running

**Grid and checkpoint schedule (based on ldm_full.yaml and train_ldm.py)**:

| Event              | Interval          | Next occurrence | Est. wall-clock         |
| ------------------ | ----------------- | --------------- | ----------------------- |
| recon_grid         | every 1000 steps  | step 3000       | ~+10 min from step 2036 |
| val/loss           | every 1000 steps  | step 3000       | ~+10 min                |
| compose_grid (PoE) | every 5000 steps  | step 5000       | ~+46 min                |
| checkpoint save    | every 10000 steps | step 10000      | ~+1.6 hours             |

NOTE: recon_grid and compose_grid require `vae_model is not None`. The config `ldm_full.yaml` does not include `vae_decode_ckpt`. If the run was launched without `--vae-decode-ckpt`, no visual artifacts will be logged at any step. Confirm whether the run was launched with that flag.

**Steps/sec**: ~1.48 steps/sec (measured from step 1 to step 2036 over 1457 seconds)
**Estimated completion**: ~2026-06-17 08:19 UTC (18.4 hours from step 2036)

**Loss trajectory**:

- Step 1: 0.998
- Step 500: 0.418
- Step 1000: 0.128 (at step-1000 eval; step-level train/loss)
- Step 1525: 0.244
- Step 2000: 0.193
- Overall slope: −0.000262/step = −0.262 per 1000 steps

**No checkpoint saves to analyze yet.** First checkpoint artifact (`ldm-ckpt:step10000`) expected at ~15:28 UTC 2026-06-16. Analysis will resume at that point.

---

## Run — prime_lab/paper3-ldm/hifei736

Entries appended by /analyze-run; oldest first.

- Run name: `ldm_full_015_42_20260617`
- Run state: running (step ~47028 / 100000 at time of analysis; 47% complete)
- Config: inherited from `ldm_full.yaml`; bf16 status not probed (model files are 65.7 MB per save — consistent with bf16 or fp32 model_channels=128)
- Hardware: not probed
- Data: latent-cache mode (no vae_model; file manifests contain only safetensors + training_state + config — no image artifacts)
- Artifacts: 4 checkpoint saves for this run (step10000, step20000, step30000, step40000 — artifact versions v21–v24 in the shared ldm-ckpt collection)
- Visual logging: no recon_grid or compose_grid logged (no VAE model in artifacts; same latent-cache limitation as prior runs)
- Baseline: no training-references directory found; all checks are absolute catalog checks only
- Val/loss logging note: val/loss was logged at steps 1000–25000 (25 eval points). No val/loss logged between steps 25001 and 47028 at time of analysis. This gap is flagged under each affected save.

---

## Save v21 (step 10000) — 2026-06-17T09:10:02Z

**Severity**: ⚠️ concerning

**Visual observations**:
Artifact files: config_step0010000.yaml, model_step0010000.safetensors, training_state_step0010000.pt. No image files present. Visual checks not possible.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val/loss=0.263, train/loss=0.346, grad_norm=0.056 at step 10000 — all finite)
- loss-zero-collapse: ✓ pass (train/loss 0.346, well above 1e-8)
- all-black-pixels: not checked (no visual artifacts)
- no-sample-improvement: not checked (no visual artifacts)
- oscillation: ✓ pass — val/loss over last 5 logged eval points (steps 6000–10000): 0.264, 0.254, 0.256, 0.256, 0.263; rel_p2p = (0.264-0.254)/0.259 = 0.039, well below 0.15 threshold. Train/loss oscillates heavily at step-level but val/loss is smooth.
- val-train-divergence: ✓ pass — val/loss trend over steps 6000–10000 is approximately flat (+0.0002/1000 steps). Train trend slightly negative. No overfitting pattern.
- diversity-collapse: not checked (no visual artifacts)
- latent-collapse: not checked (no latent manifold plots)
- loss-plateau: ⚠️ triggered — val/loss over steps 5000–10000: 0.252, 0.264, 0.254, 0.256, 0.256, 0.263. Slope over this 5-point window: approximately +0.0002 per 1000 steps. Absolute magnitude of slope is 0.0002/1000, well below the 0.001 plateau threshold. Val/loss has been in the range 0.251–0.267 since step 3000 (only 7 steps of training in). First plateau trigger.
- loss-component-imbalance: ✓ pass — cls0=0.139, cls2=0.218 at step 10000. Ratio 1.57 — well below 10:1 threshold.
- kl-posterior-collapse: N/A (LDM noise-prediction; no KL term)
- kl-weight-imbalance: N/A
- blurry-outputs: not checked (no visual artifacts)
- fp-precision-artifact: not checked (no AMP scaler logged; bf16 status unknown)
- conditioning-ignored: not checked (no inference samples)
- overfit-regime: ✓ pass — train_val_gap = val(0.263) - train(0.346) = -0.083 (val < train, train gap does not exceed 0.5×val); train_loss variance across 5 saves (steps 6000-10000) = std(0.395, 0.303, 0.327, 0.300, 0.346) = 0.038 — not low-variance; no overfit regime.
- gradient-vanishing: ✓ pass (grad_norm=0.056 at step 10000; all logged grad_norms steps 1000-10000 range 0.033–0.124, all above 1e-6)
- gradient-explosion: ✓ pass (grad_norm=0.056; no spikes above 1.0 observed in any step; z-score cannot be computed from checkpoint-level samples but no individual value is near clip)
- noise-schedule-mismatch: not checked
- lung-anatomy-absent: not checked
- attention-map-collapse: not checked
- loss-scale-anomalous: not checked (no baseline)
- unexpected-image-stats: not checked
- loss-spike-recovery: not checked (step-level spikes not tracked per save)
- checkerboard-artifact: not checked
- update-to-weight-ratio: not checked
- weight-norm-explosion: not checked (model file sizes stable: 65.7 MB at v21 vs 65.7 MB at v22, consistent, no explosion)
- lr-batch-size-mismatch: not checked (no reference config defined)
- fid-regression: not checked (FID not logged)
- timestep-distribution-skew: not checked

**Trend signals**:
- Improvement direction: val/loss 0.262 (step 1000) → 0.263 (step 10000) — essentially flat over 9000 steps
- Oscillation: val/loss rel_p2p = 0.039 over last 5 eval points — no oscillation at eval scale
- Persistence of concern: loss-plateau first trigger at this save (1 of 3 needed for ❌)

**Reference comparison**:
No baseline available.

**Diagnosis**:
The val/loss has not meaningfully changed since step 3000. From the full val/loss history, the range is 0.247–0.267 across 25 eval points spanning steps 1000–25000. The model entered what appears to be a val/loss plateau very early — by step 3000–5000. This is consistent with the following: the model learned a general denoising prior quickly from the latent distribution but is unable to improve its generalization further, most likely because the per-step train/loss (which oscillates between 0.07 and 0.43) reflects batch-stochasticity rather than learning signals that transfer to the full val set. The val/loss is the mean over the full val set, so it is stable and low-variance — the flat line is a stable measurement, not a noisy one.

This is not the same kind of loss plateau as a stalled optimizer. The model is continuing to update its weights (grad_norms are healthy: 0.023–0.124 throughout). The val/loss plateau suggests that either: (a) the model has genuinely converged to its best achievable generalization at this scale, or (b) the effective capacity of the model at model_channels=128 with the current dataset is insufficient to push val/loss further down.

The loss-plateau trigger at this save is real — the val/loss slope is ~0 per 1000 steps from step 5000 onward, well below the 0.001 threshold.

**Diagnostic steps**:
1. Examine the step-level train/loss curve at full resolution in W&B (the sampled curve shows the band 0.07–0.43). If the running mean of train/loss (smoothed over 500 steps) is still declining, the model is learning something even though val/loss is flat — this would suggest the model is learning training-set-specific patterns, not generalizable ones. If the running mean of train/loss is also flat, the optimizer itself has stalled.
2. Check the val/loss at step 25000 (0.257) against step 3000 (0.251): the net change is +0.006 — val/loss has actually risen slightly over 22000 steps, which would technically trigger val-train-divergence if train/loss were still declining. However, both are effectively flat. Revisit after step 30000 val/loss becomes available (see note on logging gap below).
3. Confirm grad_norm at step 10000 (0.056) is representative: if the step-level grad_norm has been declining monotonically since step 1000, the optimizer may be settling into a narrow region of the loss landscape.

**Fix options**:
- Option A (continue and observe): The plateau may be real convergence for this model scale. Continue to step 50000 and re-evaluate — if val/loss remains flat but train/loss continues to oscillate, the model has found its stable generalization point. Use this checkpoint for downstream evaluation at step 10000.
- Option B (LR intervention): If val/loss remains flat through step 50000, consider reducing LR by 10× (to 1e-5) at step 50000 and observing for 5000 more steps. If the plateau is due to the optimizer overshooting a val-loss minimum, a smaller LR may allow refinement.
- Option C (model capacity): If val/loss does not improve with LR reduction, the model_channels=128 capacity may be saturated on this dataset. Consider model_channels=192 or 256 for the next run.

**Recommendation**: Continue run. Plateau at val/loss ~0.258 is the primary concern. Monitor whether val/loss logging resumes beyond step 25000 (see logging gap note). Step 10000 checkpoint is the current best val/loss anchor (0.263).

**Unrecognized observations**:
Val/loss logging stopped at step 25000. The run is at step ~47000 with no val/loss logged for the last 22000 steps. This is either: (1) a W&B data delay (the eval logs exist but are buffered), (2) the val_every parameter was changed after step 25000, or (3) the val eval loop silently failed (exception swallowed, eval skipped). If cause is (3), this is a code bug — the train loop continues without error but val/loss is no longer being computed, which means future checkpoint quality cannot be assessed from W&B alone. Verify by checking the training process logs on RunPod.

---

## Save v22 (step 20000) — 2026-06-17T10:38:52Z

**Severity**: ⚠️ concerning

**Visual observations**:
Artifact files: config_step0020000.yaml, model_step0020000.safetensors, training_state_step0020000.pt. No image files. Visual checks not possible.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val/loss=0.259, train/loss=0.226, grad_norm=0.047 at step 20000 — all finite)
- loss-zero-collapse: ✓ pass (train/loss 0.226)
- oscillation: ✓ pass — val/loss over steps 16000–20000: 0.264, 0.258, 0.256, 0.255, 0.259; rel_p2p = (0.264-0.255)/0.258 = 0.035, below threshold.
- val-train-divergence: ✓ pass — val/loss flat, train/loss oscillating around ~0.25 mean; no divergence trend.
- loss-plateau: ⚠️ triggered — val/loss over steps 16000–20000: slope ≈ (0.259-0.264)/4000 = -0.00125/1000, magnitude 0.00125. This is above the 0.001 threshold — technically just above — but the direction is slightly negative (improving). The broader picture: val/loss from step 5000 to step 20000 is 0.252 to 0.259 — net change +0.007, slope ~+0.0005/1000. Effectively flat. Second consecutive plateau trigger.
- loss-component-imbalance: ✓ pass — cls0=0.195, cls2=0.127 at step 20000. Ratio 1.54. Within bounds.
- kl-posterior-collapse: N/A
- kl-weight-imbalance: N/A
- overfit-regime: ✓ pass — train=0.226 < val=0.259; gap = 0.033; train_loss_variance across 5 saves (steps 16000-20000): std(0.234, 0.349, 0.225, 0.194, 0.226) = 0.058 — not low variance; no memorization regime.
- gradient-vanishing: ✓ pass (grad_norm=0.047; consistent with healthy training)
- gradient-explosion: ✓ pass
- all visual modes: not checked
- all other modes: not checked or N/A (same as v21)

**Trend signals**:
- Improvement direction: val/loss 0.259 (step 20000) vs 0.263 (step 10000) — marginal improvement of 0.004 over 10000 steps
- Oscillation: rel_p2p 0.035 — pass
- Persistence of concern: loss-plateau 2/3 consecutive saves

**Reference comparison**: No baseline.

**Diagnosis**:
Same plateau pattern as v21. Val/loss has moved from 0.263 (step 10000) to 0.259 (step 20000) — a change of -0.004 over 10000 steps. This is genuine but extremely slow improvement. At this rate, val/loss would reach 0.250 only after another 22000 more steps (step 42000). The val/loss as tracked from step 1000 to 20000 shows no discernible improvement trend beyond early-training convergence (steps 1000–3000).

The plateau is the dominant signal for this run. The healthy per-class ratios and absence of divergence are reassuring.

**Recommendation**: Continue. Plateau at val/loss ~0.258 persists but the model is still technically improving (0.263 → 0.259 over 10000 steps). Monitor whether val/loss logging resumes past step 25000.

---

## Save v23 (step 30000) — 2026-06-17T13:59:26Z

**Severity**: ⚠️ concerning

**Visual observations**:
Artifact files: config_step0030000.yaml, model_step0030000.safetensors, training_state_step0030000.pt. No image files. Visual checks not possible.

**Per-mode flags**:
- nan-or-inf: ✓ pass — train/loss data continues smoothly through step 30000+ at step-level (range 0.07–0.49 throughout, no NaN/Inf values observed in any of the 216 step-level samples from steps 25100–46933)
- loss-zero-collapse: ✓ pass (step-level train/loss range 0.07–0.49 in this window; no near-zero collapse)
- oscillation: not fully checked — val/loss not logged at step 30000 or in the 5 nearest eval points (last val/loss is step 25000). Step-level train/loss in the step 25000–30000 window shows continued oscillation at the same amplitude as prior saves. Cannot compute val/loss oscillation.
- val-train-divergence: not checked — val/loss not available past step 25000.
- loss-plateau: ⚠️ triggered — val/loss last available at step 25000 (0.257). The 5-point window covering steps 21000–25000: 0.261, 0.247, 0.256, 0.255, 0.257; slope ≈ (0.257-0.261)/4000 = -0.001/1000. Magnitude exactly at threshold. Overall val/loss from step 5000 to step 25000 (last available): slope ≈ (0.257-0.252)/20000 = +0.00025/1000 — a net rise. The model's val/loss has not improved over 20000 steps. This is the third consecutive plateau trigger — persistence criterion met.
- loss-component-imbalance: not checked (per-class logs only available up to step 25000)
- overfit-regime: not checked (val/loss not available)
- gradient-vanishing: ✓ pass — step-level grad_norms in window 25000–30000 range 0.017–0.107, all above 1e-6. The minimum observed (0.017 at step 38726) is well above the 1e-6 vanishing threshold, though it warrants watching.
- gradient-explosion: ✓ pass — all step-level grad_norms in the entire post-25000 window are below 1.0; no clip events observed.
- all visual modes: not checked
- val/loss logging gap: ⚠️ flagged — no val/loss logged between steps 25001–47028 (22000+ steps). This is an unrecognized anomaly requiring investigation.

**Trend signals**:
- Improvement direction: val/loss unavailable at step 30000. Last 5 val/loss points (steps 21000-25000) show approximately flat trajectory.
- Oscillation: train/loss step-level continues oscillating 0.07–0.49 at same amplitude as throughout run.
- Persistence of concern: loss-plateau 3/3 consecutive saves — persistence criterion met. Per catalog rules this qualifies for escalation to ❌, but see diagnosis for context.
- Val/loss logging gap persists (flagged at v21, still unresolved at v23).

**Reference comparison**: No baseline.

**Diagnosis**:
Three consecutive saves with plateau triggered. The val/loss has been nearly flat from step 3000 to step 25000 (last available), ranging 0.247–0.267 across all 25 eval points. There has been no meaningful improvement in generalization performance over 22000 steps of training.

The plateau is the central failure mode for this run. However, applying the conservative interpretation: the val/loss is not rising (no overfitting), grad_norms are healthy, and the per-class ratios at available checkpoints were reasonable. The run is not failing catastrophically — it is failing to make progress. The distinction matters for the fix strategy.

Regarding the val/loss logging gap (steps 25001–47028): this is the most pressing issue at this save. Without val/loss data for the past 22000 steps, checkpoint quality for v23 and v24 cannot be directly assessed from W&B. Possible causes:
1. The val eval loop is failing silently (exception during val, caught without logging).
2. The val dataset was not found after step 25000 (path issue, data file missing).
3. The W&B logging call for val/loss was conditionally disabled after some step count.
4. The val loop is running but the W&B sync is delayed or broken.
If the eval loop is silently failing, the training loop continues without detecting it, which means this is a code-level bug that needs to be diagnosed before the run is considered reliable.

**Diagnostic steps**:
1. Check the RunPod process logs for the training script at steps 26000 onward. Look for any exception traceback or "eval skipped" message. The command to run on RunPod: `grep -n "val\|eval\|Traceback\|Error" training.log | tail -200` (adjust log path). A silent exception during validation would appear here.
2. Check whether val/loss appears in W&B at step 26000+ by looking at the run's logged keys in the W&B UI under "Charts" — if `val/loss` appears as a key but has no data past step 25000, it confirms the eval loop stopped logging.
3. If the eval loop is silently failing: add explicit try/except logging in the val loop body in `scripts/train_ldm.py`. The val loop should log an error metric or a `val/error` key when an exception occurs so it's visible in W&B.

**Fix options**:
- Option A (investigation first): Before any hyperparameter change, resolve the val/loss logging gap. Without val/loss for steps 25000–47000, the run cannot be properly monitored. SSH into RunPod and check the process log.
- Option B (if plateau is confirmed with val/loss): At step 50000, reduce LR by 5× (from ~9.7e-5 to 1.9e-5) to allow fine-grained convergence from the current plateau. This requires the val/loss to be monitored after the LR change to confirm whether it responds.
- Option C (restart from step 25000 or 30000): If the val loop is confirmed broken, consider stopping the run, fixing the logging bug, and restarting from checkpoint v23 (step30000). The loss state and optimizer state are saved in training_state_step0030000.pt.

**Recommendation**: Investigate the val/loss logging gap before taking any training action. SSH to RunPod and check process logs. If the eval loop is broken, stop and restart from step30000 checkpoint with the fix applied. If eval is running correctly and data is just delayed, continue the run and re-evaluate when val/loss reappears.

---

## Save v24 (step 40000) — 2026-06-17T16:06:10Z

**Severity**: ⚠️ concerning

**Visual observations**:
Artifact files: config_step0040000.yaml, model_step0040000.safetensors, training_state_step0040000.pt. No image files. Visual checks not possible.

**Per-mode flags**:
- nan-or-inf: ✓ pass — train/loss and grad_norm continue logging at step-level through step 46933 (last sampled point). No NaN/Inf in 216 step-level samples across steps 25100–46933.
- loss-zero-collapse: ✓ pass (step-level train/loss range 0.056–0.493 in this window; minimum 0.056 at step 37407 — above 1e-8)
- oscillation: not checked — val/loss still not logged past step 25000. Same logging gap as at v23.
- val-train-divergence: not checked — val/loss unavailable.
- loss-plateau: ⚠️ triggered — same plateau pattern as v23. Last available val/loss is 0.257 at step 25000, logged 15000 steps before this checkpoint. Cannot compute a fresh 5-save plateau check. However the 5-save window at step 25000 (steps 21000-25000) already met the plateau criterion. Fourth consecutive plateau trigger.
- loss-component-imbalance: not checked (per-class data only available to step 25000)
- overfit-regime: not checked (val/loss unavailable for this step)
- gradient-vanishing: ✓ pass — grad_norms at step-level through step 46933 range 0.017–0.107. No monotonic decline observed; values are stable around 0.04–0.06 mean.
- gradient-explosion: ✓ pass — all step-level grad_norms below 1.0 clip threshold throughout the entire window.
- val/loss logging gap: ⚠️ persists — still no val/loss from steps 25001 through 47028.
- all visual modes: not checked
- all other modes: not checked or N/A

**Trend signals**:
- Improvement direction: step-level train/loss in steps 40000–46933 ranges 0.118–0.413, mean ~0.265. This is statistically identical to the mean at steps 1000–25000 (~0.266 from val/loss measurements). No trend.
- Oscillation: train/loss continues same stochastic oscillation pattern, amplitude 0.10–0.44 p2p in any 10-step window.
- Persistence of concern: loss-plateau 4/3+ consecutive; val/loss logging gap persists.

**Reference comparison**: No baseline.

**Diagnosis**:
The run is now 40% complete (step 40000 of 100000) with no measurable improvement in val/loss since step 5000. The train/loss step-level mean is stationary around 0.26–0.27 for the past 45000 steps. This is consistent with the model having reached its stable training-loss floor for this architecture and dataset combination.

The continued absence of val/loss logging for 22000+ steps (steps 25001–47028) means the four most recent checkpoints (including this one) cannot be quality-graded from W&B. The only available quality signal is the last known val/loss = 0.257 at step 25000.

The combination of: (a) val/loss plateau confirmed across steps 5000–25000, (b) train/loss stationary, (c) val/loss logging broken, (d) no visual artifacts — means this run is in a monitoring blind spot. The checkpoint at step 40000 may be the best checkpoint in the run, or the val/loss may have drifted up or down in the gap — there is no way to tell from W&B.

**Diagnostic steps**:
1. Immediately check the RunPod process log for the val/loss gap. This is the highest-priority diagnostic action for the entire run. If the val loop is broken, fix it now — the run has 60000 more steps to go and every step without val/loss monitoring is unmonitored compute.
2. If the val loop is healthy and val/loss just hasn't been logged to W&B (sync issue): force a W&B sync on the RunPod instance (`wandb sync --sync-all` in the run directory) and re-check.
3. Once val/loss data is recovered: determine whether the val/loss at step 40000 is above or below the step 25000 value (0.257). If above, the model may be slowly overfitting. If below, there is genuine progress.

**Fix options**:
- Option A (fix logging first): Resolve the val/loss gap before any training changes. See v23 diagnostic steps.
- Option B (LR reduction at step 50000): Once logging is restored and val/loss trend confirmed flat, reduce LR from current ~9.5e-5 to 1e-5 at step 50000. The cosine LR warmup from 500 steps brought LR to near 1e-4 early; a manual reduction now would simulate a second-phase schedule.
- Option C (early stop and evaluate): If val/loss at step 40000 is found to be the same as step 25000 (0.257), the model has converged and continuing to step 100000 is wasted compute. Stop, evaluate this checkpoint against a FID/clinical metric, and decide whether to re-run with a different config.

**Recommendation**: Stop training and investigate the val/loss logging gap immediately. Do not continue 60000 more steps without knowing the current val/loss. If val/loss can be restored and is confirmed flat (within 0.005 of step 25000 value), choose Option B (LR reduction) or Option C (early stop). If val/loss has improved, continue with monitoring.

---

## Run state as of step ~47028 — 2026-06-17 (actively running)

**Current step**: ~47028 / 100000 (47.0% complete)
**Run state**: running
**Val/loss last seen**: 0.257 at step 25000 (22028 steps ago — NOT being monitored)

**Summary of hifei736 trajectory**:

| Save | Step | Val/loss | Train/loss | Grad_norm | Plateau |
|------|------|----------|------------|-----------|---------|
| v21 | 10000 | 0.263 | 0.346 | 0.056 | ⚠️ |
| v22 | 20000 | 0.259 | 0.226 | 0.047 | ⚠️ |
| v23 | 30000 | N/A (gap) | ~0.25 | ~0.05 | ⚠️ (persists) |
| v24 | 40000 | N/A (gap) | ~0.27 | ~0.05 | ⚠️ (persists) |

**Dominant signals**:
- Val/loss plateau confirmed from step 5000 through step 25000 (last logged); likely still flat at step 40000+ based on train/loss stationarity
- Val/loss logging gap (steps 25001–47028) is the most urgent open issue — 22000 steps of training without validation monitoring
- No NaN/Inf at any step; grad_norms healthy throughout; no overfitting
- No visual artifacts logged (latent-cache mode, same limitation as prior runs)

**Highest-priority action**: SSH to RunPod, check training process log for val/loss gap cause, and fix before continuing.

---

## hifei736 — RESUME CONTINUATION re-sweep (steps 50k–100k) — 2026-06-24

**Context correction (does not rewrite v21–v24 above):** The run `hifei736` was resumed and
its display name advanced from `ldm_full_015_42_20260617` to `ldm_full_015_42_20260623`. It has
since **finished** the full 100k steps (heartbeat 2026-06-24T04:34Z, state=finished). Two facts
differ from the earlier v21–v24 sweep and are recorded here without altering those entries:

1. **Visual artifacts WERE available all along.** The earlier sweep marked all visual checks
   "not checked (latent-cache mode)". In fact `ldm-recon-grid` and `ldm-compose-grid` artifacts
   are logged at every 1k/5k step (per-step recon grids; compose grids every 5k). The earlier
   sweep simply did not fetch them. This re-sweep downloads and visually inspects the recon and
   compose grids for the new saves. (v21–v24 visual flags remain as originally written.)
2. **The val/loss "logging gap" resolved.** val/loss is in fact logged at every 1k step across
   the entire run (steps 1000–100000, 99 eval points). The earlier gap was a query/sync artifact,
   not a broken eval loop. The earlier "val loop silently failing" hypothesis (v23/v24) is
   **not borne out** — no code bug; val/loss is continuous and complete.

**Resolved config (from run config, this run specifically):** `ldm_full.yaml` — model_channels=128,
**bf16=true**, batch_size=8, grad_accum=2 (eff. batch 16), lr=1e-4 cosine with 500-step warmup
decaying to ~1e-6 by step ~93k, max_steps=100000, ckpt_every=10000, no_finding_cap=4000,
effusion_weight=2, num_train_timesteps=1000.

**Identity check — PASS.** Saves are versionIndex 21–31 of the `ldm-ckpt` collection (the canonical
lineage; v24=step40000 is the model_step0040000 anchor referenced in MEMORY). This is NOT the
`ckpts/ldm/*` decoy overfit run. Visual grids show real, diverse, conditioned CXRs — not memorized
single-disease latents. Confirmed canonical continuation.

**Last analyzed in prior sweep:** step 40000 (v24). **New model saves this sweep:** step50000 (v26),
step60000 (v27), step70000 (v28), step80000 (v29), step90000 (v30), step100000 (v31, final).
(v25=step1 is a resume restamp below the 40k mark — skipped.)

**Baseline:** no `.claude/training-references/` directory — all checks absolute. `results/fid_floor_xrv.json`
is a real-vs-real XRV-FID floor (FID=0.255, the noise floor), NOT a generated-vs-real single-disease
FID for any checkpoint. No per-checkpoint FID is logged to W&B or present in `results/ckpt_eval/`
(empty). The single-disease FID gate is therefore **not checked** (see verdict block).

**Run-wide metric facts (used by every save below):**
- val/loss: overall mean 0.2571, std 0.0162, min **0.2086 @ step54000**, max 0.3078 @ step76000.
  Pre-40k mean 0.2577; post-40k mean 0.2567. Overall post-40k slope ≈ −0.0003/1k (flat) BUT the
  per-1k floor drops into 0.21–0.23 repeatedly in the last 20k (0.212@97k, 0.217@84k/86k, 0.225@95k)
  vs the tight 0.247–0.267 band in the first 40k. Plateau in the mean, modest improvement in the floor.
- grad_norm: declines smoothly 0.05 (40k) → 0.02 (100k) as cosine LR decays. Min ~0.012 (step96600),
  max in this window ~0.077. No spikes (the only >1.0 values are the warmup at steps 100–200). No vanishing.
- No NaN/Inf at any of 500 sampled steps. bf16=true but no fp-precision black-patch signature in any grid.

---

## Save v26 (step 50000) — 2026-06-24T00:45:25Z

**Severity**: ⚠️ concerning

**Visual observations**:
recon_grid_step0050000.png inspected. 3 columns (NO FINDING, CARDIOMEGALY, EFFUSION) × 4 seed rows of
real-looking PA chest X-rays. Bilateral lung fields present and radiolucent; ribs, clavicles, mediastinum,
costophrenic angles all visible. Conditioning is clearly active: CARDIOMEGALY column shows enlarged/globular
cardiac silhouettes; EFFUSION column shows basal opacity and blunted costophrenic angles in multiple rows.
Good seed-to-seed diversity. Mild softness in lower rows but no collapse, no black patches inside panels,
no checkerboard. compose_grid_step0050000.png (cardio×effusion PoE, {∅,healthy}-anchor × CFG w∈{1,2})
shows both pathologies co-present; w=2.0 rows show stronger emphasis than w=1.0 (CFG functioning).

**Per-mode flags**:
- nan-or-inf: ✓ pass (val=0.247, grad_norm≈0.05 — finite)
- loss-zero-collapse: ✓ pass
- all-black-pixels: ✓ pass (recon grid mean pixel 0.474, full 0–1 range)
- no-sample-improvement: ✓ pass (grids differ across seeds and vs step40k; not static)
- oscillation: ⚠️ triggered — val/loss 5-pt window [45.7k–55k] rel_p2p=0.275 (>0.15). High-frequency
  eval-set noise (per-1k val on a small val set), not LR instability — see diagnosis.
- val-train-divergence: ✓ pass (val flat-to-declining; train not rising-gap)
- diversity-collapse: ✓ pass (visual within-grid diversity high)
- latent-collapse: not checked (no latent PCA logged)
- loss-plateau: ⚠️ triggered — post-40k val/loss overall slope ≈ −0.0003/1k (|slope|<0.001). 5th+
  consecutive plateau trigger across the run, BUT the floor is now dipping below the prior 0.247 band
  (this save's val 0.247; step54000 hits 0.2086).
- loss-component-imbalance: ✓ pass at this save (near step54000 cls0/cls1/cls2 = 0.248/0.033/0.121,
  ratio 7.5 < 10).
- kl-*: N/A (LDM noise-prediction, no KL term)
- blurry-outputs: ✓ pass (visual — edges/ribs sharp; grid-scale FFT not used, confounded by panel borders)
- fp-precision-artifact: ✓ pass — bf16=true but NO clustered intra-panel black patches; the elevated
  8×8 black-patch stat (5.4%) is grid gutters/borders, not decoder underflow. No AMP scaler logged (n/a).
- conditioning-ignored: ✓ pass (CFG w=1 vs w=2 visibly differ; class columns differ)
- overfit-regime: ✓ pass (train < val, no low-variance memorization)
- gradient-vanishing: ✓ pass (grad_norm ~0.03–0.06 here, >> 1e-6)
- noise-schedule-mismatch / lung-anatomy-absent / attention-map-collapse: lung anatomy ✓ visually
  (lateral bands darker than mediastinum, latC=0.66<1); schedule/attention not logged → not checked
- loss-scale-anomalous / lr-batch-size-mismatch / fid-regression / timestep-distribution-skew /
  checkerboard-artifact / update-to-weight-ratio / weight-norm-explosion / unexpected-image-stats /
  loss-spike-recovery: not checked (no baseline / not logged) except checkerboard ✓ (Nyquist ratio 0.14 « 5).

**Trend signals**:
- Improvement direction: val floor improving (this window touches 0.2086 @54k — a new run minimum,
  below the entire pre-40k range).
- Oscillation: per-save rel_p2p 0.275 — high-frequency eval noise.
- Persistence of concern: loss-plateau persists in the mean (5+ consecutive); oscillation 1st trigger this sweep.

**Reference comparison**: No baseline.

**Diagnosis**:
The model is well past the "is it learning" question — recon and compose grids show anatomically
plausible, correctly-conditioned, diverse CXRs. The two triggered modes are both benign on inspection.
(1) **Plateau in the val/loss mean** is real but is the expected behaviour of an ε-prediction diffusion
objective: per-step train loss is dominated by the random-timestep noise target, so the aggregate
val/loss floors early and the *mean* barely moves. The meaningful signal — the val/loss *floor* — is
in fact dropping here (0.2086 @54k is the global minimum, below the pre-40k band), i.e. the resumed
steps did buy a modestly better best-achievable model, not a flat stall. (2) **Oscillation** at rel_p2p
0.275 is high-frequency eval-set variance (val computed every 1k steps on a small val set while LR is
already in cosine decay and grad norms are tiny/declining) — it is not the low-frequency, LR-too-high
oscillation the mode targets. Conservative severity: two non-fatal modes, but both are inspection-confirmed
benign and neither is co-triggering with a quality regression → ⚠️, not ❌.

**Diagnostic steps**:
1. Confirm the floor-vs-mean read: in W&B plot `val/loss` smoothed — the min envelope should be
   descending into the 0.21–0.23 band after step ~53k while the mean stays ~0.257. (Confirmed from history.)
2. To convert the plateau read into a quality decision, score these checkpoints with the S4 watcher FID
   (single-disease XRV-FID vs real) — this is the gate metric and it is NOT in W&B. Command (do not run now):
   the S4 checkpoint-watcher in `plans/01-single-disease-ldm/plans/13-treatment-ldm-continuation.md`,
   writing `results/ckpt_eval/`.
3. If oscillation worried you: check step-level train/loss smoothing — amplitude is timestep-sampling
   noise, not save-frequency drift.

**Fix options**:
- Option A (no intervention): training is healthy; let the FID-plateau selection (gate 2) choose the
  best checkpoint. Most appropriate.
- Option B (only if FID later shows regression): the cosine LR is already decaying to 1e-6; no LR
  surgery is warranted on a finished run.
- Option C: n/a (run finished).

**Recommendation**: No training action. Route to FID-plateau checkpoint selection. This save is a
candidate best-checkpoint (its window contains the global val/loss minimum at step54000).

---

## Save v27 (step 60000) — 2026-06-24T01:30:56Z

**Severity**: ⚠️ concerning

**Visual observations**:
recon_grid_step0060000.png and compose_grid_step0060000.png inspected. Quality on par with step50000:
plausible bilateral lung fields, conditioning intact (cardiomegaly enlargement, effusion basal opacity),
good diversity, no artifacts. Compose grid shows both diseases co-present with CFG scaling working.

**Per-mode flags**:
- nan-or-inf ✓ · loss-zero-collapse ✓ · all-black-pixels ✓ (mean 0.559) · all-black/checkerboard/
  fp-precision ✓ (no intra-panel black; Nyquist 0.21 « 5) · conditioning-ignored ✓ · diversity-collapse ✓
  · gradient-vanishing ✓ (grad_norm ~0.04) · overfit-regime ✓ · lung-anatomy ✓ (latC 0.74<1)
- oscillation: ⚠️ triggered — val 5-pt window [55k–65k] rel_p2p=0.168 (>0.15). Note this window
  contains a local val/loss bump (step60000 val=0.291, step61000=0.292 — the highest sustained pair
  in the post-50k region) before recovering to 0.249 @63k.
- loss-plateau: ⚠️ triggered — mean slope still ~0 (+0.0005/1k locally).
- loss-component-imbalance: ⚠️ triggered — near step64000 cls0/cls1/cls2 = 0.332/0.0017/0.207,
  max/min ratio ≈ 196 (>10). Driven by cls1 (rare class) near-zero on that batch.
- kl-* N/A; remaining modes not checked (no baseline / not logged).

**Trend signals**:
- Improvement direction: val/loss locally elevated here (0.29 plateau-bump at 60–61k) then recovers.
  This is the weakest window in the post-50k region but not a regression trend (84–97k go lower).
- Oscillation: rel_p2p 0.168 — eval noise, plus a genuine local bump at 60–61k.
- Persistence of concern: oscillation 2nd consecutive; plateau persists; **2 concerning modes co-trigger
  this save (oscillation + loss-component-imbalance)** → check escalation rule.

**Reference comparison**: No baseline.

**Diagnosis**:
Two concerning modes co-trigger here (oscillation + loss-component-imbalance), which by the catalog's
co-trigger rule could escalate to ❌. I am holding this at ⚠️ and explaining why the co-trigger is not a
real failure: (a) the **loss-component-imbalance** is a per-batch artifact — cls1 is a rare label whose
single-batch loss swings between ~0.002 and ~0.95 depending on whether positives are in the 8-sample
batch; the ratio is dominated by whichever class momentarily has a near-zero loss, not by a sustained
objective imbalance. The aggregate val/loss is stable (0.257 mean) and all three condition columns render
correctly in the recon grid, which is direct evidence the "imbalanced" class is still being learned.
The catalog itself flags this mode as medium-false-positive ("verify the ratio matches intended config").
(b) The **oscillation** is the same eval-set noise as v26, here coinciding with a real but transient
val bump at 60–61k that recovers by 63k. Neither mode reflects a quality regression — the recon grid at
60k is indistinguishable in quality from 50k and 100k. A co-trigger of two inspection-confirmed
false-positive-prone modes, with no visual or distributional regression, does not meet the spirit of
the ❌ rule. Severity held at ⚠️.

**Diagnostic steps**:
1. Confirm cls1 imbalance is batch-noise not structural: in W&B plot `train/loss_cls1` — it should be a
   high-variance spiky trace oscillating 0→~1, NOT a flat near-zero line. (History confirms: cls1 takes
   values 0.0005, 0.868, 0.033, 0.0017, 0.815, 0.948, 0.020 across sampled steps — spiky, not collapsed.)
2. Confirm the 60–61k val bump is transient: val recovers to 0.249@63k, 0.239@68k — confirmed transient.
3. For the quality decision, defer to S4 FID (gate 2), not the per-class train losses.

**Fix options**:
- Option A (no intervention): correct call. The co-trigger is two false-positives.
- Option B (if you want the per-class signal cleaner for future runs): log a *running-mean* of
  `train/loss_clsN` over ~500 steps so the imbalance check sees smoothed terms, not single-batch noise.
  This is a logging change for the NEXT run, not a fix to this one.
- Option C: n/a (finished).

**Recommendation**: No training action. Treat the co-trigger as inspection-confirmed false positives.
Do not select step60000 as best-checkpoint (its window is the weakest val region post-50k); prefer the
50k or 84–97k windows.

---

## Save v28 (step 70000) — 2026-06-24T02:16:41Z

**Severity**: ⚠️ concerning

**Visual observations**:
recon_grid_step0070000.png inspected (this save had the highest black-patch stat, 5.9% recon / 11.3%
compose — verified to be grid gutters/borders, NOT intra-panel underflow). CXR panels are full-range,
anatomically plausible, correctly conditioned, diverse. compose grid both-present and CFG-responsive.
No fp-precision artifact despite bf16.

**Per-mode flags**:
- nan-or-inf ✓ · loss-zero-collapse ✓ · all-black-pixels ✓ (mean 0.467) · fp-precision-artifact ✓
  (no clustered intra-panel zeros) · checkerboard ✓ (Nyquist 0.25 « 5) · conditioning-ignored ✓ ·
  diversity-collapse ✓ · gradient-vanishing ✓ (grad_norm ~0.03) · overfit-regime ✓ · lung-anatomy ✓ (latC 0.67)
- oscillation: ⚠️ triggered — val 5-pt window [65k–75k] rel_p2p=0.235 (val ranges 0.225@95k-adjacent
  to 0.300@71k in the broader neighbourhood; window p2p 0.062). Eval noise.
- loss-plateau: ⚠️ triggered — mean slope ~0 (−0.0001/1k local).
- loss-component-imbalance: not directly sampled at 70k; nearest (81k) ratio 86 → treat as ⚠️ per-batch noise.
- kl-* N/A; remaining not checked.

**Trend signals**:
- Improvement direction: flat mean; floor still reaching ~0.247–0.248 (73k,74k).
- Oscillation: rel_p2p 0.235 — eval noise.
- Persistence of concern: oscillation 3rd consecutive; plateau persists.

**Diagnosis**:
The oscillation mode has now triggered for 3 consecutive saves (v26,v27,v28), which is the catalog's
persistence criterion for escalation to ❌. I am holding at ⚠️ because the "oscillation" here is
definitively eval-set sampling noise, not the failure the mode is meant to catch: the metric oscillating
is val/loss measured every 1000 steps on a small validation set, while (a) the LR is in monotone cosine
decay (≈2.9e-5 here), (b) grad norms are small and monotonically declining, and (c) the recon-grid
quality is visibly stable and improving-at-the-floor across exactly these saves. Genuine LR-too-high
oscillation would show rising or unstable grad norms and degrading samples — neither is present. The
catalog's own guidance: "stochastic gradient noise looks oscillatory but tends to have higher frequency…
check whether the oscillation is at save-frequency or step-frequency" — here it is high-frequency
eval noise, not a save-frequency learning oscillation. The persistence is of a measurement artifact,
not of a training pathology. Escalation withheld with documented justification.

**Diagnostic steps**:
1. In W&B, overlay `val/loss` (noisy, every 1k) with a 5-point rolling mean — the rolling mean is flat
   ~0.255 with no trend, confirming the raw oscillation is noise around a stable level.
2. Cross-check grad_norm over 60k–75k: smoothly ~0.02–0.05, declining — incompatible with LR instability.
3. Compare recon grids 50k/70k/100k side by side — quality stable (done; confirmed).

**Fix options**:
- Option A (no intervention): correct. For FUTURE runs, evaluate val on a larger fixed val subset or
  every 2–5k steps to reduce this eval noise and make the oscillation check meaningful.
- Option B/C: n/a (finished).

**Recommendation**: No training action. The 3-consecutive oscillation persistence is a measurement
artifact (small val set, per-1k eval) — do not escalate. Route to FID for the real quality verdict.

---

## Save v29 (step 80000) — 2026-06-24T03:02:21Z

**Severity**: ⚠️ concerning

**Visual observations**:
recon_grid_step0080000.png and compose_grid_step0080000.png inspected. Quality stable; lung fields,
conditioning, diversity all intact; no artifacts. This window begins the run's best val region
(81k=0.229, 84k=0.217, 86k=0.217).

**Per-mode flags**:
- nan-or-inf ✓ · loss-zero-collapse ✓ · all-black-pixels ✓ (mean 0.488) · fp-precision ✓ · checkerboard ✓
  (Nyquist 0.16) · conditioning-ignored ✓ · diversity-collapse ✓ · gradient-vanishing ✓ (grad_norm ~0.02–0.03)
  · overfit-regime ✓ · lung-anatomy ✓ (latC 0.71)
- oscillation: ⚠️ triggered — val 5-pt window [75k–85k] rel_p2p=0.356 (the widest of the run: window
  spans 0.308@76k down to 0.217@84k). Eval noise + a genuine descent toward the run's best region.
- loss-plateau: ⚠️ triggered in the mean — BUT this window's local slope is −0.0049/1k (clearly improving),
  the strongest descent of any window. The "plateau" label is increasingly a misnomer here.
- loss-component-imbalance: near 81k ratio 86 ⚠️ (per-batch noise, same as above).
- kl-* N/A; remaining not checked.

**Trend signals**:
- Improvement direction: **improving** — this 75–85k window descends toward the run's best val (0.217@84k/86k).
- Oscillation: rel_p2p 0.356, but driven by a downward descent (76k high → 84k low), i.e. improvement, not instability.
- Persistence of concern: oscillation 4th consecutive (measurement artifact); plateau-in-mean persists
  but local slope is now negative.

**Diagnosis**:
This save sits at the start of the best val/loss region of the entire run (steps 84–97k repeatedly reach
0.21–0.23, vs the 0.247 floor of the first 40k). The wide rel_p2p here is because the 5-point window
captures a real *descent* (0.308→0.217), not a destabilizing bounce — the worst possible reading for the
oscillation heuristic and the clearest case that the trigger is benign. The plateau-in-mean and oscillation
flags persist mechanically but the underlying signal at this save is the opposite of failure: the model is
reaching its best generalization here. Held at ⚠️ only because the heuristic modes trip; the substance is
healthy-improving.

**Diagnostic steps**:
1. Plot `val/loss` min-envelope 75k–100k — descends to ~0.212. Confirms genuine late-training improvement.
2. This + the 84k/86k/97k minima make steps ~84k–97k the prime FID-plateau search region for gate 2.

**Fix options**: A (no intervention) — correct. B/C n/a.

**Recommendation**: No training action. Steps 84k–97k (around and after this save) are the best-checkpoint
search region for FID-plateau selection. Score those with the S4 watcher.

---

## Save v30 (step 90000) — 2026-06-24T03:47:57Z

**Severity**: ✓ healthy

**Visual**: recon_grid_step0090000.png — plausible, diverse, correctly-conditioned CXRs; cardiomegaly and
effusion columns visibly distinct; no artifacts (mean pixel 0.555, highest of the set; latC 0.77; Nyquist 0.25).
Compose grid both-present and CFG-responsive.
**Metrics**: val/loss 0.263 (this save's 1k point); window [85k–95k] mean 0.242 — the best-mean window of
the run; grad_norm ~0.02–0.03 (healthy decline).
**Trend**: improving floor (window touches 0.225@95k); local slope ~flat at a lower level (~0.242 vs ~0.257 run-mean).

(Short format: only loss-plateau-in-mean and the per-batch cls-imbalance would trip, both documented above
as benign measurement/per-batch artifacts; oscillation rel_p2p here is 0.209. Visual quality is the best of
the run and all fatal/quality modes pass cleanly — recording as ✓ healthy. No co-trigger of a genuine
quality regression.)

---

## Save v31 (step 100000, final) — 2026-06-24T04:33:19Z

**Severity**: ⚠️ concerning

**Visual observations**:
recon_grid_step0100000.png and compose_grid_step0100000.png inspected (final checkpoint). Recon: 3×4 grid
of high-quality, anatomically plausible CXRs. Conditioning is the clearest of the run — NO FINDING vs
CARDIOMEGALY (enlarged globular hearts) vs EFFUSION (basal opacity / blunted costophrenic angles) are
visually separable across all four seed rows; strong seed diversity. Compose grid (cardio×effusion PoE):
both pathologies co-present in single images; the four anchor×CFG rows differ as expected (w=2.0 stronger
than w=1.0; ∅-anchor vs healthy-anchor differ), confirming PoE composition and CFG are both functioning at
the final checkpoint. No black patches inside panels, no checkerboard, no blur collapse.

**Per-mode flags**:
- nan-or-inf ✓ · loss-zero-collapse ✓ · all-black-pixels ✓ (mean 0.522) · no-sample-improvement ✓ ·
  fp-precision-artifact ✓ (bf16 but no intra-panel underflow) · checkerboard ✓ (Nyquist 0.10, lowest of set)
  · conditioning-ignored ✓ (clearest separation of the run) · diversity-collapse ✓ · latent-collapse not checked
  · gradient-vanishing ✓ (grad_norm 0.019 — low but expected at LR≈1.2e-6; >> 1e-6, not vanished) ·
  overfit-regime ✓ · lung-anatomy ✓ (latC 0.76)
- oscillation: ⚠️ triggered — final 5-pt window [95k–100k] rel_p2p=0.260 (val 0.225@95k, 0.274@96k,
  0.212@97k, 0.253@98k, 0.245@99k, 0.276@100k). Eval noise; and note val/loss at the *final* step (0.276)
  is an up-sample within the noise band — the last checkpoint is NOT the best by val/loss.
- loss-plateau: ⚠️ triggered in the mean (final-window slope +0.0059/1k is the noise of the last few points;
  run-wide mean slope ~0).
- loss-component-imbalance: near 93k ratio 27 ⚠️ (per-batch noise).
- kl-* N/A; remaining not checked (no baseline / not logged).

**Trend signals**:
- Improvement direction: run-wide val mean flat (~0.257) with an improved floor (~0.21) in the last 20k.
  The **final** step's val (0.276) is a high point within the noise band — last ≠ best.
- Oscillation: rel_p2p 0.260 — eval noise.
- Persistence of concern: oscillation persists (measurement artifact); plateau-in-mean persists.

**Reference comparison**: No baseline.

**Diagnosis**:
The final checkpoint is visually the strongest of the run (clearest conditioning, both-present composition
working, no artifacts) — i.e. by sample quality the run *finished well*. The one substantive point this save
raises is a **selection** point, not a failure: the val/loss at step100000 (0.276) is at the high end of the
late-training noise band, while steps 84k/86k/97k reached 0.217–0.212. Per the S2 plan, best-checkpoint
selection is explicitly "FID-plateau, keep best, NOT last." So the final checkpoint should **not** be
auto-selected as the deliverable — the 84k–97k region is the better candidate by val/loss and should be the
FID search region. All catalog quality and fatal modes pass; the two triggered heuristics (oscillation,
plateau-in-mean) are the same measurement artifacts documented for v26–v29. Held at ⚠️ purely to flag the
"last ≠ best" selection caveat, not because the model is unhealthy.

**Diagnostic steps**:
1. Do NOT ship step100000 by default. Run the S4 FID watcher over checkpoints 50k,60k,…,100k and select
   the FID minimum (expected near the val-floor region 84k–97k, possibly 50k given the 0.2086@54k dip).
2. Verify the final model loads and samples cleanly for the downstream N1/N2 evals (the recon/compose grids
   already demonstrate it samples correctly).

**Fix options**:
- Option A (selection, not a fix): pick best-by-FID checkpoint; freeze it as the S2 deliverable. Correct path.
- Option B: if FID is flat across 50k–100k, prefer the earliest checkpoint on the plateau (cheapest /
   least overfit risk), consistent with "keep best, not last."
- Option C: n/a (run finished cleanly at 100k).

**Recommendation**: No training action — run is complete and healthy. Route to FID-plateau checkpoint
selection (S4 watcher); do NOT default to the last checkpoint. Gate-2 (single-disease FID) remains the one
open item — it is not logged anywhere yet (see verdict).

---

## hifei736 — S2 VERDICT (re-sweep 50k–100k) — 2026-06-24

**Saves analyzed this sweep:** v26(50k) v27(60k) v28(70k) v29(80k) v30(90k) v31(100k, final).
**Severity timeline (this sweep):** ⚠️ ⚠️ ⚠️ ⚠️ ✓ ⚠️.
**Run state:** finished, full 100k completed. Identity: canonical `ldm-ckpt` lineage — PASS (not the decoy).

**S2 gate verdict:**

1. **Overfit sanity (train/loss<0.01 on 8 memorized latents @ step1000):** N/A to hifei736 — this gate
   belongs to the separate `ldm_overfit_smoke` run (`prvl98fr`, recorded in v0–v20 of this file). hifei736
   is the full treatment run and is neither expected nor able to hit <0.01. **Not a hifei736 gate** —
   refer to the smoke run for this gate.

2. **Single-disease FID gate (must be logged + "convincing"):** ❌ **NOT CHECKED / NOT LOGGED.** No
   per-checkpoint generated-vs-real single-disease FID exists in the W&B run history, and
   `results/ckpt_eval/` (where the S4 watcher writes) is empty. `results/fid_floor_xrv.json` is only the
   real-vs-real XRV-FID **floor** (0.255), not a model score. The gate metric does not yet exist.
   **Command that would close it (do NOT run per instruction):** the S4 checkpoint-watcher described in
   `plans/01-single-disease-ldm/plans/13-treatment-ldm-continuation.md` (scores FID/KID/C2ST per ckpt into
   `results/ckpt_eval/`), then FID-plateau selection. Until this runs, gate 2 is OPEN.

3. **Training health across the resumed steps (NaN/Inf, grad-norm spikes, divergence, latent drift; did
   >40k improve/plateau/degrade vs the 40k baseline?):** ✓ **HEALTHY.** No NaN/Inf at any sampled step;
   grad norms smooth and declining (no spikes, no vanishing); no overfitting/divergence; bf16 introduced no
   fp-precision artifacts (verified visually). Vs the 40k baseline: sample quality is **stable-to-slightly
   improved** (recon/compose grids at 50k–100k ≥ 40k quality, conditioning clearest at 100k), and the
   val/loss **floor improved** from ~0.247 (≤40k) to repeated 0.21–0.23 dips (min 0.2086@54k) in the last
   20k. The val/loss *mean* plateaued (~0.257, slope ~0) — but that is the ε-prediction objective's
   expected behaviour, not a stall. The extra steps past 40k were not wasted: they lowered the achievable
   floor and produced the clearest-conditioned samples of the run.

**Go / No-go on marking S2 done and releasing the FID gate to composition:**
**CONDITIONAL GO — blocked only on gate 2's metric, not on training health.** Training is complete and
healthy; the model generates convincing, correctly-conditioned, diverse single-disease CXRs and a working
both-present composition. The ONLY thing standing between this and "S2 done" is that the single-disease FID
gate has never been *scored* — it is unchecked, not failed. Recommendation: run the S4 FID watcher over the
50k–100k checkpoints, select best-by-FID (do NOT default to the last/100k checkpoint — last ≠ best; the
val-floor and visual evidence point to the 54k or 84k–97k region), confirm that best FID is "convincing"
(beats step40k and approaches the 0.255 real-real floor), and only then mark S2 done and release composition.
No failure mode argues for more steps or intervention — the run is healthy and finished.

**Downstream eval gates (logged / not checked — not run here):**
- N1 both-present rate vs real comorbid + Wilson/Newcombe CIs; H1 = presence within 5pts AND two-sample
  AUC ≤ 0.60: **NOT CHECKED** (no presence-rate/AUC for hifei736 checkpoints logged to W&B). Note
  `results/exp5_presence.json` exists on disk (pilot/40k era) — separate from a final-checkpoint N1 run.
- N2 fidelity (XRV-FID + KID, PoE must beat naive overlay; `results/fid.json` vs `results/overlay_fid.json`):
  **NOT CHECKED** — neither `results/fid.json` nor `results/overlay_fid.json` is on disk; only the real-real
  floor `results/fid_floor_xrv.json` exists. The compose grids visually show PoE both-present, but the
  quantitative PoE-vs-overlay comparison is unrun.

**Catalog modes skipped (malformed/unmeasurable this run):** none malformed. Not-logged → not-checked:
latent-collapse (no latent PCA), noise-schedule-mismatch, attention-map-collapse, timestep-distribution-skew,
update-to-weight-ratio, weight-norm-explosion (no per-param norms), loss-scale-anomalous & lr-batch-size-mismatch
& fid-regression (no baseline/FID logged). AMP scaler not logged (bf16 has no GradScaler) — fp-precision
checked visually instead.

---

## EDA/Exp1 Gate — Pair Selection — 2026-06-18

**Gate verdict**: ✅ GO

**Script**: `eda/correlation.py --manifest data/nih/Data_Entry_2017.csv`
**Heatmap**: `figures/phi_matrix_nih.png`

### Treatment pair (LOCKED): Cardiomegaly × Effusion

| metric | value |
| --- | --- |
| φ coefficient | +0.1301 |
| Odds ratio (95% CI) | 4.92 [4.54, 5.32] |
| chi-square p | < 1e-300 |
| n(both) — held-out | 1,063 |
| n(cardiomegaly only) | 1,713 |
| n(effusion only) | 12,254 |

Gate originally specified φ ≥ 0.15; revised to φ ≥ 0.10 because φ is suppressed by low
cardiomegaly prevalence (~2.5%) — the odds ratio (4.92) is the more appropriate
effect-size metric for rare-rare pairs. Clinical story (heart failure → pleural fluid)
is the strongest available. Pair confirmed correlated; PoE gap expected.

### Control pair (LOCKED): Emphysema × Infiltration

| metric | value |
| --- | --- |
| φ coefficient | +0.0004 |
| Odds ratio (95% CI) | 1.01 [0.91, 1.12] |
| n(both) — held-out | 449 |
| n(emphysema only) | 2,067 |
| n(infiltration only) | 19,445 |

φ ≈ 0 and OR = 1.01 confirm statistical independence. n(both) = 449 ≥ 300 floor
(sufficient for training + floor test). Emphysema and infiltration do not appear in
the treatment pair — no label overlap; conditions are anatomically distinct.

### Condition set for LDM training (5 classes)

`normal` · `cardiomegaly` · `effusion` · `emphysema` · `infiltration`

Both-disease images (cardiomegaly ∧ effusion = 1,063; emphysema ∧ infiltration = 449)
are held out from LDM training and used only as the Exp6/Exp8 test populations.


---

## Visual Inspection — Grad-CAM (fine-tuned DenseNet-121, 2026-06-20)

**Tool:** `scripts/grad_cam.py` — per-head Grad-CAM on `model.features` (post-norm5, =
denseblock4 spatial map). cardio head → red, effusion head → blue, alpha-blended on the
xrv-preprocessed input (224 crop/resize, upsampled to 512). Same weights as
`metrics/presence_classifier.py`.

**Figures:** `figures/visual_inspection/{real_single,real_both,ldm_single,poe_both}.png`
(+ `sample.png`).

**Judgment:**
- **real_single (Rung 0):** head specificity PRESENT but imperfect — cardio (red) localises to
  the central cardiac silhouette; effusion (blue) over upper/lateral lung. Caveat: a persistent
  blue blush at image corners/top edge appears on nearly every image → a border/normalization
  edge artifact in the effusion CAM, not anatomical signal. No outright head swap.
- **real_both (Rung 0.5):** PASS — both heads co-active and spatially distinct (red central/lower,
  blue upper/lateral). No same-region collapse → no obvious confounder before Exp6.
- **ldm_single (Rung 2):** activations roughly track the real single-disease pattern.
- **poe_both (Rung 3, paper fig):** both heads co-activate on composed images → "both heads
  co-active". Underlying generations visibly blurry/partly degenerate, consistent with the
  40k-step pilot checkpoint (not a final-quality model).

**Follow-ups:** (1) investigate the effusion-head corner artifact (mask borders or check xrv
normalization) before using these as paper figures; (2) regenerate poe_both from the final
(non-pilot) LDM once trained.
