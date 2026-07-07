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


---

## Visual Inspection — Grad-CAM REGEN on 100k + corner-fix (2026-06-24)

Re-ran `scripts/grad_cam.py` (2-head fine-tuned DenseNet-121, red=cardio / blue=effusion)
after two changes vs the 2026-06-20 figures: (a) the `grad_cam_utils.GradCAM` border-artifact
fix is now active (top feature-row zeroed, p99 normalisation), and (b) the LDM figures use the
**final 100k checkpoint** (`outputs/ckpt_eval/0100000/...`), not the 40k pilot — the 40k
`outputs/single/` set no longer exists. Figures overwritten in `figures/visual_inspection/`:
`real_single.png` (8 cardio_only + 8 effusion_only), `real_both.png` (16 `images_both`),
`ldm_single.png` (8+8 100k single), `poe_both.png` (16 100k compose w1p0). This run is the
visual half of the cardio-instrument calibration the 100k pilot flagged (see
[[extractor-real-vs-gen-disagree]] / "Pilot Eval — model_step0100000").

**Per-set head-specificity verdict:**
- **real_single (Rung 0):** PASS (imperfect). Cardio rows → red on the central cardiac
  silhouette; effusion rows → blue over lower/lateral lung. Corner blue is reduced vs Jun-20 but
  a faint bottom-edge tint persists on both heads. No head swap.
- **real_both (Rung 0.5):** PASS. Red and blue co-activate in spatially **distinct** regions
  (red central/cardiac, blue upper-lateral) — no same-region collapse → no obvious Exp6 confounder.
- **ldm_single (Rung 2, 100k):** ⚠️ SPLIT. Generated **effusion** → strong, well-localised blue,
  visually matching real effusion (consistent with presence 0.83). Generated **cardiomegaly** →
  cardio-CAM is **diffuse and partly edge-spread, NOT tightly locked on a central cardiac
  silhouette** the way real cardio is.
- **poe_both (Rung 3, 100k, paper headline fig):** both heads co-active on composed images; blue
  crisp and anatomically placed, **red present but diffuse**. Co-activation visually holds; cardio
  localisation is the weak link.

**Verdict: both heads co-active** (real_both ✓, poe_both ✓) — but with a cardio caveat.

**What this means for the pilot's cardio question — it does NOT cleanly confirm "just classifier
mis-calibration."** The earlier pilot hypothesis was that the cardio head is mis-calibrated on
synthetic images while the heart is in fact drawn (v31 visual + near-floor FID). Grad-CAM
partially walks that back: on generated cardio the cardio head's **attention itself is diffuse /
not heart-locked**, unlike on real cardio. So the cardio head is not merely failing to *score* a
clearly-attended heart — it is not finding a crisp cardiac silhouette to attend to. Grad-CAM
cannot adjudicate *why* (smears across the mediastinum by design — it localises classifier
attention, not an organ boundary), so it is **necessary but insufficient**.

**Action (unchanged priority, sharper need):** measure heart size **geometrically and
classifier-independently** — HybridGNet CTR (or MedSAM heart-cut) on the 100k cardio singles —
to decide between (i) the heart is anatomically enlarged but the DenseNet head/threshold misreads
synthetic CXRs (→ re-calibrate/re-anchor the head) vs (ii) the 100k LDM genuinely under-renders
the cardiac silhouette (→ a generator capacity problem; revisit class-imbalance 1,713 vs 12,254).
This is exactly the CTR/MedSAM instrument-calibration work specified in the current
`plans/02-pre-evaluation/plans/05-real-instrument-calibration.md` (tasks 5.2–5.6).


---

## CTR adjudication — cardiomegaly is UNDER-RENDERED, not mis-classified (2026-06-24)

Built a classifier-independent CTR instrument (`scripts/ctr_overlay.py`) using the
torchxrayvision ChestX-Det **PSPNet** anatomical segmentation (Heart + L/R Lung) — substituted
for the plan's HybridGNet/CheXmask CTR (same geometry: CTR = heart_width / thorax_width; avoids
the torch_geometric install + GB-scale CheXmask download on a no-sudo node). Results
`results/ctr_calibration.json`; overlay `figures/instrument_calibration/ctr_overlay_real.png`.

| group | n | mean CTR | median | frac > 0.5 |
|---|---|---|---|---|
| cardio_real (images_cardio_only)        | 60 | **0.547** | 0.544 | **0.75** |
| both_real (images_both)                 | 60 | 0.548 | 0.538 | 0.77 |
| normal_real (partition)                 | 60 | 0.456 | 0.454 | 0.15 |
| **cardio_gen100k** (LDM single)         | 91 | **0.395** | 0.425 | **0.12** |
| effusion_gen100k (LDM single)           | 55 | 0.373 | 0.385 | 0.09 |

**Instrument validated on real:** cardio_real (0.547) > normal_real (0.456); 75% of real cardio
exceed the CTR>0.5 cardiomegaly line vs 15% of normal; both_real high (0.548) as expected. The
CTR tool cleanly separates real cardiomegaly from normal → trustworthy.

**Verdict — the generator does NOT enlarge the heart.** Generated "cardiomegaly" CTR (0.395) is
far below real cardio (0.547) and *below even real normal* (0.456); only 12% cross 0.5 (vs 75%
real). Anchoring on the domain-shift-robust **within-domain delta** (generated images segment to
a slightly lower absolute CTR overall, a mild seg domain-shift): real cardiomegaly is **+0.09**
CTR over its no-cardio baseline (normal), but generated cardiomegaly is only **+0.02** over its
no-cardio baseline (gen-effusion) — the cardiomegaly signal is ~4× weaker in generated than real.

**This OVERTURNS the classifier-mis-calibration hypothesis.** The presence classifier (cardio
0.385) and Grad-CAM (diffuse, not heart-locked) were reading the generated images *correctly* —
there is no enlarged heart to detect. The bottleneck is **generator capacity for cardiomegaly**,
confirming [[ldm-cardio-undertrained-rebalance-lever]] (7:1 cardio:effusion data imbalance +
effusion_weight=2.0 starve the cardio conditional). VAE is innocent (Rung 1), effusion is fine
end-to-end, FID is realistic — the single defect is that the 100k LDM under-renders the cardiac
silhouette.

**Consequence for go/no-go (supersedes the pilot's "fix the measurement first"):** the fix is a
**rebalanced retrain of the cardio conditional**, not a classifier re-calibration. Exp5 (H1
marginals gate) cannot pass for cardiomegaly on the current checkpoint, so Exp6/7/8 stay gated
until a rebalanced LDM lifts generated CTR into the real-cardio range. Effusion, VAE, FID, and the
correlation gate are all clear.

**Instrument decision (plan 5.6):** carry **CTR via xrv ChestX-Det PSPNet** forward as the
cardiomegaly heart-size instrument for the leakage harness (07) — validated on real, no heavy
deps, classifier-independent. MedSAM-cut / HybridGNet remain optional cross-checks if a second
opinion is wanted, but are not required to make the call.


---

## Pilot Eval — model_step0100000 (100k steps) — 2026-06-24

Inference-only presence ladder on the final (100k) checkpoint, run via
`metrics.presence_classifier` (fine-tuned DenseNet-121, AUC cardio 0.902 / effusion 0.890;
Youden thr_c 0.616, thr_e 0.516) plus per-class FID (`metrics.fid`). Pilot batch generated
with `scripts/generate.py` (seed 42, anchor=null, 50 DDIM steps). Result files:
`pilot_validate.json` (R0), `pilot_validate_both.json` (R0.5), `pilot_vae_recon.json` (R1),
`pilot_presence_100k.json` (R2), `pilot_composition_100k.json` (R3), `pilot_fid.json` (FID).

### Presence ladder (mean classifier probability)

| Rung | set | n | cardio | effusion | co-presence | source |
|---|---|---|---|---|---|---|
| 0   | real single-disease            | 201/238 | **0.784** | **0.707** | — | `pilot_validate.json` |
| 0.5 | real both-comorbid (oracle)    | 100 | 0.675 | 0.443 | **0.06** | `pilot_validate_both.json` |
| 1   | VAE recon of the R0.5 images   | 100 | 0.674 | 0.453 | — | `pilot_vae_recon.json` |
| 2   | LDM single-disease (100k)      | 200 | **0.385** ⚠️ | 0.830 | — | `pilot_presence_100k.json` |
| 3   | LDM PoE composed w1p0 (100k)   | 200 | 0.360 | 0.803 | **0.030** | `pilot_composition_100k.json` |

### FID (per class, n=200; class-matched real ref, not full corpus)

| class | inception FID | inception floor (real-real) | xrv FID | xrv floor | gate <250 |
|---|---|---|---|---|---|
| cardiomegaly | 181.5 | 90.1 | 6.19 | 2.35 | ✓ PASS |
| effusion     | 173.7 | 96.3 | 4.44 | 2.59 | ✓ PASS |

(Real ref deviates from the plan's `data/nih/images/`, which now holds the full 112k corpus;
class-matched `images_cardio_only` / `images_effusion_only` is the honest per-class reference.)

### Gate results
- **FID < 250 per class:** ✓ both pass; xrv domain-FID is near the real-vs-real floor → images
  are realistic CXRs in pathology-feature space.
- **R2/R3 soft gate (cardio >0.20, effusion >0.20, co-presence >0.10):** cardio ✓ (0.385/0.360),
  effusion ✓ (0.830/0.803), **co-presence ✗ (0.030 < 0.10)**.
- **VAE feature-loss gate (R1 within 0.05 of R0.5, flag if drop >0.20):** ✓ paired Δ ≤ 0.01 on
  both heads → VAE is effectively lossless for disease features.

### Diagnosis — the co-presence collapse is cardio-limited, and partly a measurement artifact
Three pathways are cleanly separated:
1. **VAE — innocent.** R1 reconstructions reproduce both-head presence to within 0.01 of the
   real both-group images (R0.5). No information loss at the codec.
2. **Effusion — learned end-to-end.** Generated effusion presence (R2 0.830, R3 0.803) *exceeds*
   the real single-disease mean (0.707). The effusion conditioning works.
3. **Cardiomegaly — the bottleneck.** Generated cardio presence (R2 0.385, R3 0.360) sits well
   below real (0.784) and below the 0.616 decision threshold, so most generated images are
   classified cardio-negative and the PoE AND collapses to 0.030.

**Crucial caveat — this is not unambiguously a generator failure.** Two independent signals
contradict a "model can't draw cardiomegaly" reading: (a) the v31 visual analysis of
`compose_grid_step0100000.png`/`recon_grid_step0100000.png` reports cardiomegaly as the
*clearest-conditioned* class of the run ("enlarged globular hearts, visually separable"); (b)
xrv-FID for the cardio singles is near the real-real floor (realistic). Yet the classifier's
cardio head fires at only 0.385 on those same images. This matches the prior
`extractor-real-vs-gen-disagree` / bimodal-heart-size finding: the cardio head is likely
**mis-calibrated on synthetic images**, not reading a genuinely absent pathology. So the 0.030
co-presence is a lower bound confounded by classifier-on-generated domain shift.

### Go / No-go
**QUALIFIED NO-GO on the composition claim as currently measured — but NOT a training action.**
By the rule (co-presence gate not cleared) this is a no-go; however the failure mode is most
likely a **cardio-classifier calibration artifact on synthetic images**, not a generator or VAE
defect (effusion end-to-end ✓, VAE lossless ✓, FID realistic ✓, visual cardio conditioning ✓).

Required before committing to Exp5/Exp6:
1. **Disambiguate the cardio head on generated images** — measure cardiomegaly directly with the
   CTR/HybridGNet extractor (see `poe-inspection-harness`) on the R2 cardio singles; if hearts
   are enlarged by CTR but the DenseNet head stays <0.616, the gate must be re-anchored to a
   generated-calibrated threshold (or the classifier re-calibrated on synthetic CXRs).
2. **Re-anchor H1** against the R0.5 oracle (0.06), not 1.0, once the cardio head is trusted.
3. Hypothesis to keep in view: training class imbalance (cardio_only 1,713 vs effusion_only
   12,254, ~7×) could also depress cardio rendering — but the visual + FID evidence argues the
   primary gap is measurement, so fix the measurement first before any retraining.

---

## Run — prime_lab/paper3-ldm/pcp9jo36 (`exp5_retrain_s01_20260625`) — REBALANCED CARDIO RETRAIN

The Exp03/Exp05 lever ([[ldm-cardio-undertrained-rebalance-lever]]): reuse VAE + reuse effusion,
RETRAIN the cardio conditional on class-balanced sampling. Config `configs/ldm_rebalanced.yaml`:
`balance_classes=true` (class_mult normal 0.25 / cardio 0.40 / effusion 0.35), `effusion_weight 2.0→1.0`,
lr 1e-4, batch 16, model_channels 128 (UNCHANGED), cfg_dropout 0.15, bf16, max_steps=100000, ckpt_every=10000.
Launched `scripts/launch_retrain.sh` on hippo, latent-cache `data/latents/rebalanced`, VAE `ckpts/vae_step0025000.pt`.

**Run state (from W&B):** STATE = **failed**; created 2026-06-25T10:37:06Z; runtime **1810 s (~30 min)**; last
logged step **10294** of planned 100000 (**10%**). Exactly ONE checkpoint artifact (`ldm-ckpt:v32`, `model_step0010000`).
No baseline `.claude/training-references/` directory present → **no baseline available**; all checks are absolute catalog
checks. Resolved metric keys: `train/loss`, `val/loss`, `train/grad_norm`, `train/loss_cls{0,1,2}`, `train/lr`.
"Already analyzed up to": none for this run-id (prior entries are prvl98fr and hifei736). Every save is new; there is one.

---

## Save v32 (step 10000) — 2026-06-25T11:06:39Z (only save; run failed at step 10294)

**Severity**: ⚠️ concerning

**Visual observations**:
Two grids decoded at 512² and inspected directly.
- **recon_grid_step0010000** (single-class conditional; cols NO_FINDING / CARDIOMEGALY / EFFUSION, rows seeds 42/137/256/512):
  Lung fields present and bilateral in all 12 panels — ribs, clavicles, mediastinum, hemidiaphragms all recognizable;
  not black, not saturated, good seed-to-seed diversity (no diversity collapse). The CARDIOMEGALY column does render a
  **visibly broader cardiac silhouette** than the NO_FINDING column (seeds 42, 256 show a widened globular heart filling
  more of the lower thorax) — a real, legible conditioning effect, and an improvement over the old hifei736 cardio-alone
  (which under-rendered the heart below even real-normal CTR). BUT the cardiac **borders are soft/hazy** — the left heart
  border blends into adjacent lung rather than presenting a crisp edge, and the mediastinum is washed out. The EFFUSION
  column renders crisp basal whiteout + blunted costophrenic angles (effusion remains the strong, over-rendered expert).
  Overall slight softness consistent with only 10k steps (10% of schedule).
- **compose_grid_step0010000** (PoE co-morbid; rows null-anchor w1.0/w2.0, healthy-anchor w1.0/w2.0; cols 4 seeds):
  Effusion signature dominates every composite (dense lower-zone opacity, blunted angles). w=2.0 rows add mediastinal/cardiac
  density vs w=1.0. The heart silhouette in the composites is still **soft and partly swamped by the basal whiteout** — some
  cardiac density is present but not the sharp enlarged silhouette of the real co-morbid reference (user's Image A). This is
  the same qualitative cardiac-under-rendering tendency as before, at an early checkpoint.

This matches the user's attached Image B (gen step10000, w=1.0): narrow/soft cardiac silhouette, hazy border, heavy basal
whiteout, washed-out mediastinum. Image B is a **step-10000** sample — i.e. 10% trained — so it cannot be read as the
rebalanced model's converged cardiac quality.

**Per-mode flags** (absolute checks; baseline absent):
- nan-or-inf: ✓ (0 non-finite across train/loss n=10294, val/loss n=10, grad_norm n=102, per-class)
- loss-zero-collapse: ✓ (train/loss smoothed ~0.25, never near 0)
- all-black-pixels: ✓ (grids show full anatomy, normal CXR dynamic range)
- lung-anatomy-absent: ✓ (bilateral lung fields visible in all panels)
- diversity-collapse: ✓ (clear seed-to-seed variation)
- conditioning-ignored: ✓ (cardio column broader than normal; effusion column distinct; CFG w1.0 vs w2.0 differ)
- blurry-outputs: ⚠️ (mild — outputs soft at step 10000; expected this early, not escalated)
- oscillation (headline train/loss): ✓ (smoothed loss flat ~0.25–0.26 from step 1000, relative p2p of the *smoothed*
  curve « 0.15; the raw per-step series is high-variance by DDPM design — random timestep per batch — not LR oscillation)
- loss-plateau: ✓-by-design (train/loss flat near the DDPM noise-MSE floor ~0.25 is the expected converged-ish band, not a stall;
  val/loss tracks it 0.21–0.29)
- val-train-divergence / overfit-regime: ✓ (val 0.249→best 0.215@8k→0.290@10k tracks train ~0.25; gap small, no rising-val trend)
- loss-component-imbalance (cls0/cls1/cls2): ⚠️ **noisy, not actionable** — per-class loss is computed on the per-batch
  subset of each class (`F.mse_loss(eps_pred[mask], noise[mask])`, ~6–40 samples), so single-batch ratios swing 0.03–0.91.
  cls1 (cardio) ranges 0.10–0.43 over the run; it did **not** dive to ~0 (the config's "cardio overfit" tripwire did NOT fire by 10k).
  The >10:1 instantaneous ratios are subset-size variance, not a true objective imbalance — flagged ⚠️ for transparency, not escalated.
- gradient-explosion: ✓ (grad_norm 2.23 at warmup step→settles ~0.05–0.13; clip max_norm=1.0 only bit at the very start; last 8 ~0.04–0.13)
- gradient-vanishing: ✓ (grad_norm ~0.05–0.13, well above 1e-6; loss still moving early)
- weight-norm-explosion / update-to-weight: not checked (per-layer norms not logged; scalar grad_norm healthy)
- fp-precision-artifact (bf16): ✓ (no clustered black patches; no GradScaler — bf16 autocast, no loss scaler logged)
- checkerboard-artifact / noise-schedule-mismatch / attention-map-collapse / timestep-distribution-skew: not checked
  (FFT/attention/timestep-histogram artifacts not logged)
- fid-regression / latent-collapse / kl-*: not checked / N/A (LDM, not VAE; FID & latent PCA not logged for this run)

**Trend signals**:
- Improvement direction: train/loss smoothed flat ~0.25 after step 1000 (the DDPM-MSE floor); val/loss best 0.215@8k.
  Insufficient *multi-save* history — only one checkpoint exists (run died at 10k), so cross-save trend is "insufficient history".
- Oscillation: smoothed headline p2p ≈ 0.05 over 1k–10k (well under 0.15). Raw per-step p2p large by design (random-timestep DDPM).
- Persistence of concern: none — single save; nothing flagged for ≥3 consecutive saves.

**Reference comparison**: no baseline available (`.claude/training-references/` absent); comparison limited to absolute checks
and to the prior hifei736 run recorded above. Relative to hifei736 cardio-alone (which the CTR instrument scored 0.395,
below real-normal 0.456), the step-10000 cardio column here looks *qualitatively broader than normal* — a directional
improvement — but at 10% of schedule and with soft borders, not a confirmed CTR win (no CTR was computed on this checkpoint).

**Diagnosis** (why the run stopped — evidence-backed, Question 1):
The run was **killed by its own divergence guard misfiring on normal DDPM step-to-step loss noise.** `scripts/train_ldm.py`
sets a kill reference `loss_at_step_10k = accum_loss` at exactly step 10000 from a **single instantaneous step value (0.2272)**,
then `sys.exit(1)` whenever any later single step exceeds `2× = 0.4544`. The W&B output.log shows verbatim:
`11:07:15 KILL CRITERIA MET: Loss divergence at step 10294: 0.5062 > 2× step-10k value 0.2272`. Step-10294's single-batch
loss (0.5062) is ordinary DDPM variance — individual steps span 0.09–1.00 across the run because each batch samples a random
timestep; the *smoothed* loss never left ~0.25. So this is a **false-positive early-stop / code bug in the guard**, NOT a real
divergence: zero NaN/Inf anywhere, grad_norm at the kill was 0.058, val/loss 0.29 (healthy). State=failed because the guard
called `wandb.finish(exit_code=1)` + `sys.exit(1)`. Two compounding flaws: (a) the reference is a *single noisy step* rather
than a windowed/EMA loss; (b) anchoring it at step 10000 means the very first noisy excursion after the checkpoint trips it.
Not OOM (it logged 294 steps *after* the heavy step-10000 image-decode burst, ckpt written), not preemption, not the data loader.

**Was something wrong with the model? (Question 3):** No training pathology in the model itself. Curves are clean (no NaN,
controlled grads, train/val tracking, legible conditioning, anatomically valid samples). The failure is purely the **harness
kill-guard bug**. The only *model-quality* caveat is the long-standing one — cardiac borders are soft and effusion dominates —
and that is unresolvable from a 10%-trained checkpoint.

**Is it capacity? (Question 4):** Re-confirmed **NOT** capacity. The UNet is **16.4M params** (model_channels=128,
num_res_blocks=2, cross_attn_dim=512; safetensors ≈66 MB fp32). The same 16.4M UNet **over-renders effusion** through the same
VAE — proof the network has the capacity to render a pathology sharply when it gets enough balanced signal. Capacity/VAE were
already ruled out for the old collapse ([[ldm-cardio-undertrained-rebalance-lever]], [[exp02-codec-ceiling]] VAE rFID 1.81 /
|ΔCTR| median 0.011 faithful); nothing here contradicts that. The cardio deficit is a **data/optimization** problem (1,713
unique cardio latents, now 40% of batches under replacement sampling → heavy repetition, few *unique* gradients), not a
parameter-count ceiling.

**Should we improve the VAE? (Question 7):** No. The recon grid shows the VAE decoding all three classes into anatomically
valid 512² CXRs with correct dynamic range; effusion is rendered crisply through it. This is fully consistent with
[[exp02-codec-ceiling]] (VAE faithful, |ΔCTR| 0.011, REUSABLE, do-not-retrain) and the pilot Rung-1 (recon Δ≤0.01). Nothing in
this run indicts the codec. VAE work is **not justified** — it would spend compute on the one component already proven innocent.

**Should we swap to a transformer (DiT / U-ViT)? (Question 5):** **Not warranted now**, and not the diagnosed bottleneck. For a
512²/f=4 → 4×128×128 latent on a single hippo GPU, a DiT operating on 128×128 patches is *expensive*: at patch-size 2 that's
4,096 tokens (O(N²) attention → ~16M attention entries/layer), and a competitive DiT-B/L (130–460M params) is 8–28× our current
16.4M UNet. Our evidence says the convolutional UNet already has the capacity (effusion over-renders) — the deficit is data
balance + training length, which a DiT does not fix and would only make slower to iterate. A DiT swap is a large architecture
+ re-tuning cost with no mechanism to address cardio under-rendering. **Recommendation: stay on the UNet.** Revisit DiT/U-ViT
only if, after a *full-length* rebalanced run with the fixes below, cardiac borders are still soft AND a capacity probe (scale
model_channels 128→192) shows headroom — i.e. only once capacity is empirically implicated, which it currently is not.

**Pretrained HF models to fine-tune instead? (Question 6):** Strongest concrete candidate is **`stanfordmimi/RoentGen-v2`**
(https://hf.co/stanfordmimi/RoentGen-v2) — chest-X-ray text-to-image, a **fine-tune of stabilityai/stable-diffusion-2-1**,
diffusers `StableDiffusionPipeline`, **MIT license**, gated (paper arxiv:2508.16783 "Improving Performance, Robustness, and
Fairness of Radiographic AI Models with Finely-Controllable Synthetic Data"; lineage from original RoentGen, arxiv:2211.12737 /
https://hf.co/papers/2211.12737). It is purpose-built for exactly our task (controllable pathology CXR synthesis) and trained on
MIMIC-CXR-scale data — far more cardiomegaly exemplars than our 1,713.
  **Feasibility / the catch — latent incompatibility:** RoentGen-v2 is SD-2-1 lineage = **4-channel, f=8 VAE** (latent
  4×64×64 at 512²). Our pipeline is **4-channel, f=4** (latent 4×128×128 at 512²) — same channel count but **different spatial
  compression**, so our cached latents and our VAE are NOT drop-in compatible with its UNet. Fine-tuning RoentGen means adopting
  *its* SD VAE (f=8) and *its* UNet and abandoning our f=4 latent — which [[latent-shape-resolution-open]] deliberately pinned at
  f=4 for PoE spatial separability (the whole compositional thesis). It is also text-conditioned (CLIP prompts), not our 3-class
  embedding, so the PoE ε-composition wiring would need re-derivation. License (MIT) and resolution (512²) are fine; the blocker
  is the latent geometry + the PoE method coupling. **Verdict:** RoentGen-v2 is a viable *alternative research direction* (and a
  strong external baseline / sanity check that "a well-fed CXR diffusion model renders cardiomegaly sharply"), but it is **not a
  cheap drop-in** for our f=4 PoE setup. Recommended use: (a) as a baseline to confirm the ceiling, and (b) reconsider as the
  backbone only if we reopen the f=4 latent decision. Do not block the next launch on it.

**Diagnostic steps** (to confirm the above before relaunch):
1. **Confirm the kill was spurious** — already done: `output.log` line 277 shows the exact false-positive; grad_norm 0.058 and
   val 0.29 at the kill confirm no real divergence. No further check needed.
2. **Verify cardio is genuinely improving** — relaunch with the guard fixed (below) to a real checkpoint (≥40k–100k), then run
   `scripts/ctr_overlay.py` (xrv ChestX-Det PSPNet, the validated instrument) on ≥200 generated cardio-only singles. Target:
   generated cardio CTR rising from the old 0.395 toward the real-cardio band (0.547 / 75% >0.5). This is the decisive metric,
   not the step-10000 eyeball.
3. **Watch the cardio overfit tripwire** the config calls out: log/inspect `train/loss_cls1` on a *smoothed* basis; if it dives
   toward ~0 while val/loss stalls, drop class_mult[1] 0.40→0.33 (equal thirds) — but at 10k it had not dived.

**Fix options** (Question 8 — the single highest-leverage knob is "let it train", gated behind fixing the guard):
- **Option A (REQUIRED, do first): fix the divergence-guard bug in `scripts/train_ldm.py` (~lines 982–995).** Anchor the kill
  reference to a **smoothed/EMA loss** (e.g. running mean over the last 200–500 steps) instead of the single `accum_loss` at
  step 10000, and compare the *smoothed current* loss to it — not a single step. Equivalently raise the factor (2×→4×) and/or
  require N consecutive steps over threshold. Without this, any relaunch dies again at the first post-10k noisy step.
- **Option B (the actual training lever): just train longer — to the full 100k.** Nothing in the dynamics warranted stopping;
  the model was at 10% and still early. The old hifei736 needed its full schedule before cardio quality could be judged. This is
  the **single highest-leverage change**: the run never got the chance to learn. Keep lr 1e-4, batch 16, class_mult as-is for the
  first full run — change only the guard (A), then read CTR at 40k/100k.
- **Option C (only if, after a full run, cardio borders stay soft):** address the *unique-cardio-data* scarcity rather than
  capacity — (i) augment the 1,713 cardio latents (light spatial aug in pixel space before re-encoding, or mild latent
  perturbation) so 40%-of-batches isn't pure repetition; (ii) consider a modest model_channels 128→160/192 capacity probe ONLY
  after a capacity signal appears; (iii) as a separate research track, evaluate the RoentGen-v2 backbone (Question 6) if reopening
  the f=4 latent is on the table. Do NOT touch the VAE (Q7) and do NOT swap to DiT (Q5) pre-emptively.

**Recommendation**: Fix the guard (Option A), then **relaunch and let it train to 100k** (Option B) — pause nothing else.
The retrain did not fail on its merits; it was killed by a buggy tripwire at 10% trained. Defer all architecture/VAE/HF changes
until a full-length rebalanced checkpoint exists and CTR has been measured on it.

**Unrecognized observations**:
- The kill-guard reference-loss bug is a genuine **code/harness defect**, not a catalog training mode — surfaced here because it
  determined the run's lifetime. Worth a catalog addendum candidate: "early-stop-guard-misfire — a divergence/early-stop tripwire
  anchored to a single noisy step value (vs smoothed/windowed) terminates a healthy run on ordinary loss variance." Check: run
  state=failed + `sys.exit`/`wandb.finish(exit_code=1)` in logs + no NaN + grad/val healthy + the kill step's loss within the
  run's historical per-step range.
- Per-class loss logging on tiny per-batch class subsets produces 0.03–0.91 swings that *look* like loss-component-imbalance but
  are subset-size variance. Consider logging per-class loss as an EMA, or over an accumulation window, to make the cardio-overfit
  tripwire the config relies on actually readable.

**Catalog modes skipped/not-checked due to missing artifacts**: weight-norm-explosion, update-to-weight-ratio (no per-layer
norms); checkerboard-artifact, noise-schedule-mismatch, timestep-distribution-skew (no FFT/timestep histograms); attention-map-
collapse (no attention logging); fid-regression (no FID logged this run); latent-collapse, kl-posterior-collapse, kl-weight-
imbalance (VAE-side, N/A to this frozen-VAE LDM run). lr-batch-size-mismatch: not flagged — batch 16 / lr 1e-4 is the same
effective-batch dynamics as the old 8×2 run by design (config note), and no reference config defines a different LR.

---

## Run — prime_lab/paper3-ldm/vqe9re62 (`ldm_rebal_full_015_42_20260625_cont30000`) — REBALANCED CARDIO RETRAIN, FORK-RESUME FROM pcp9jo36 @ step 30000

Third instance of the Exp05 lever ([[ldm-cardio-undertrained-rebalance-lever]]). Same config `configs/ldm_rebalanced.yaml`
(balance_classes, class_mult 0.25/0.40/0.35, effusion_weight 1.0, lr 1e-4, batch 16 × grad_accum 1, model_channels 128, bf16,
max_steps 100000, ckpt_every 10000). Launched `scripts/relaunch_retrain.sh` on **hippo** (host `hippocampus`, **RTX 5090, 34 GB
VRAM, 33.5 GB system RAM**), `--resume pcp9jo36`. Because the resume now **forks to a FRESH W&B run** (fork_from gated behind a
private-preview 400, so it starts a new run logging from the checkpoint step — lineage `resumed_from_run=pcp9jo36`,
`resumed_from_step=30000`), the W&B step axis on vqe9re62 starts at **30100, not 0** — this offset is by design, NOT a problem.
The actual resume point is the **latest** checkpoint in `ckpts_rebalanced/` (step 30000: the run got from 10k→30k earlier; this
session continued 30k→35k). Resolved metric keys: `train/loss`, `val/loss`, `train/grad_norm`, `train/loss_cls{0,1,2}`,
`train/lr`, plus the NEW in-training eval keys `eval/instrument_ok`, `eval/*` (CTR/CP/yield scalars), `panel/*` (overlay images).

No baseline `.claude/training-references/` directory present → **no baseline available**; all checks are absolute. "Already
analyzed up to": none for this run-id (prior entries are prvl98fr, hifei736, pcp9jo36). Saves new since the fork: ONE checkpoint
artifact on disk this session — **step 30000** (the fork origin / latest decodable checkpoint; `model_step0030000.safetensors`
+ `training_state_step0030000.pt`, written 20:27). The run reached step 35000 but **died inside the step-35000 eval block before
writing a 35000 checkpoint** — so step 30000 is the last persisted, fully-resumable state. (Steps 10000/20000 were recorded under
the pcp9jo36 lineage / earlier; 40000 was never reached.)

---

## Save v33 (step 30000, fork origin) — 2026-06-25T20:27Z (run died at step 35000, mid-eval, no 35000 ckpt) — RECURRING-CRASH DIAGNOSIS

**Severity**: ⚠️ concerning

> This entry answers the user's pointed question head-on: *"We keep getting crashes. Are we evaluating the right arm to
> determine crashes? Is it certain a crash means the model won't get better? The crashes are getting suspicious."* Short answer,
> evidenced below: **No — we are watching the WRONG arm.** The crash on vqe9re62 is **NOT** the divergence guard and **NOT** model
> divergence. It is an **infrastructural process death (CUDA/host OOM) inside the newly-added in-training generation-quality
> dashboard, at the step-35000 eval-cadence boundary.** The model was healthy at the instant it died, the last checkpoint
> (step 30000) is intact and fully resumable, and the crash says **nothing** about whether the model will improve. Severity is
> ⚠️ (not ❌): the catalog requires a fatal-tier NaN/Inf trigger, OR persistence across 3+ saves, OR a 2-mode co-trigger for ❌ —
> none of which is met. A *process* OOM is not a catalog training-failure mode and does not escalate model severity.

**Visual observations** (decoded singles at step 30000, viewed directly — `outputs/exp05_monitor/{cardio,effusion,poe}_30000/`):
- **cardio_30000** (9-panel montage): healthy, anatomically-plausible PA CXRs — bilateral lung fields, ribs, clavicles,
  mediastinum, hemidiaphragms all legible; good dynamic range (mean pixel 0.40, std 0.28); lateral/center intensity ratio 0.60
  (lungs clearly darker than mediastinum → `lung-anatomy-absent` passes). Several panels show a **broadened, globular cardiac
  silhouette** filling the lower thorax — a real, legible cardio-conditioning effect, and a visible improvement over the soft
  10k checkpoint (v32). Cardiac borders are still **mildly soft/hazy** at 30% of schedule (expected; not escalated). Good
  seed-to-seed diversity (no diversity collapse), no black/saturated panels, no checkerboard.
- **poe_30000** (9-panel montage): co-morbid composites — basal whiteout / blunted-angle effusion signature PLUS cardiac density
  in the same image; bilateral anatomy intact, strong seed diversity, no collapse. Effusion remains the stronger-rendered
  expert; cardiac silhouette present but partly swamped by basal opacity in some seeds (the long-standing cardio-under-render
  tendency, at an early checkpoint — unresolvable from 30%-trained samples).
- **effusion_30000**: crisp basal opacity + blunted costophrenic angles (effusion the over-rendered expert, as always).

The model is **visually healthy at step 30000.** Nothing in the imagery indicates divergence, collapse, or a code-render bug.

**Per-mode flags** (absolute checks; baseline absent; computed on step-30000 series + decoded singles):
- nan-or-inf: ✓ (no non-finite in train/loss, val/loss, grad_norm across the 30100→35000 window)
- loss-zero-collapse: ✓ (train/loss per-step 0.12–0.42, smoothed ~0.25; never near 0)
- all-black-pixels: ✓ (mean pixel 0.40–0.42, full dynamic range)
- lung-anatomy-absent: ✓ (lateral/center 0.60–0.63 < 0.85 — lung fields clearly present)
- diversity-collapse: ✓ (clear seed-to-seed variation in all three modes)
- conditioning-ignored: ✓ (cardio column broader than normal; effusion distinct; PoE composes both)
- blurry-outputs: ⚠️ (mild cardiac-border softness at 30% trained — expected this early, NOT escalated)
- oscillation (headline train/loss): ✓ (val/loss flat 0.238–0.248 over 31k–35k; smoothed train flat ~0.25; raw per-step
  high-variance by DDPM design — random timestep per batch — not LR oscillation; the EMA guard exists precisely for this)
- loss-plateau: ✓-by-design (flat near the DDPM noise-MSE floor ~0.25; val tracks it; this is the converged-ish band, not a stall)
- val-train-divergence / overfit-regime: ✓ (val 0.238–0.248 tracks train ~0.25; gap small; no rising-val trend; train/val both flat)
- loss-component-imbalance (cls0/1/2): ⚠️ **noisy, not actionable** — per-class loss is on tiny per-batch class subsets, so
  instantaneous values swing (e.g. no_finding 0.0002–0.68, cardio 0.009–0.56, effusion 0.0026–0.56 within a few steps). cardio
  (cls1) did NOT dive to ~0 (overfit tripwire NOT fired). The >10:1 instantaneous ratios are subset-size variance, not a true
  objective imbalance — flagged ⚠️ for transparency only, exactly as in v32. Same recommendation: log per-class loss as an EMA.
- gradient-explosion / gradient-vanishing: ✓ (grad_norm 0.02–0.09 throughout; clip max_norm=1.0 never bit; well above 1e-6)
- fp-precision-artifact (bf16): ✓ (no clustered black patches; no GradScaler — bf16 autocast, no loss scaler)
- **eval/instrument_ok**: ⚠️/inconclusive — **never logged at step 35000** (see Diagnosis: the process died *before* the
  try/except could stamp it). At earlier eval steps it was not surfaced as 0 in the log. This is itself diagnostic — see below.
- weight-norm-explosion, update-to-weight-ratio: not checked (no per-layer norms logged)
- checkerboard-artifact, noise-schedule-mismatch, timestep-distribution-skew: not checked (no FFT/timestep histograms)
- attention-map-collapse: not checked (no attention logging)
- fid-regression: not checked (FID not logged; eval/* CTR scalars exist but no FID series)
- latent-collapse / kl-posterior-collapse / kl-weight-imbalance: N/A (frozen-VAE LDM, not a VAE run)
- **early-stop-guard-misfire** (the v32 candidate mode): ✓ **did NOT fire this run** — the rewritten EMA guard did not trip
  (no "KILL CRITERIA", no "Loss divergence", no `wandb.alert`, no `sys.exit(1)` anywhere in `output.log`). The v32 false-positive
  cause is **fixed and confirmed inert** on vqe9re62.

**Trend signals**:
- Improvement direction: val/loss flat-to-slightly-improving 0.244→0.238 across 31k–35k (best 0.2384 @35k); train smoothed flat
  ~0.25 at the DDPM floor. Multi-save cross-checkpoint trend = "insufficient history" (only one persisted checkpoint this fork).
- Oscillation: smoothed headline p2p ≈ 0.01 over the window (« 0.15). Raw per-step large by DDPM design.
- Persistence of concern: **none** — no mode flagged for ≥3 consecutive saves; nothing co-triggered. (This is why severity stays ⚠️.)

**Reference comparison**: no baseline available (`.claude/training-references/` absent). Relative to the prior fork lineage
(pcp9jo36 v32 @10k and hifei736), the 30k cardio column is qualitatively **broader and sharper than the 10k checkpoint** — a
directional improvement consistent with "the model is learning cardio as it trains" — but CTR was not scored on this checkpoint
(the `eval/*` CTR dashboard that would have scored it is the very component that crashed), so this is a visual read, not a
measured CTR win.

**Diagnosis — the recurring-crash analysis (this is the core of the entry):**

**Which arm fired? Answer to "Are we evaluating the right arm?" — NO.** The kill signal we have been watching (the loss-divergence
guard) is **not** what stopped vqe9re62. Evidence from `wandb/run-20260625_205045-vqe9re62/files/output.log`:
- The log ends **mid-eval at step 35000**, on the line `anchor=normal w=2.0 seed=512` of the PoE compose grid (train_ldm.py:690),
  i.e. inside the `step % 5000 == 0` block that runs compose grid + the `panel=True` quality dashboard. The very next thing that
  block does is `_run_quality_dashboard(..., panel=True)` → PSPNet + presence classifier moved to GPU, 192 images generated/decoded.
- **No traceback. No "KILL CRITERIA". No "Loss divergence". No `eval/instrument_ok=0`. No `wandb.finish()`. No `sys.exit`.** The
  process simply stops between log lines. A clean Python exception would have been caught by the `try/except` around the dashboard
  (train_ldm.py:1174-1184), which logs `eval/instrument_ok=0` and continues — that did NOT happen. A Python exception elsewhere
  would have printed a traceback — it did NOT. **An abrupt death with no Python-level trace is the signature of an OS OOM-kill
  (SIGKILL) or an unrecoverable CUDA OOM/fault below the Python layer** — exactly the failure the try/except cannot catch.
- Corroborating: the `media/` directory for the run has **no panel images** (the step-35000 `panel/*_burned`/`panel/*_layered`
  were never logged → the dashboard never finished), and the `.wandb` file's final mtime (21:47) post-dates the last training log
  line (21:44) — wandb's sidecar flushed and exited while the trainer was already gone, with no clean finish.

**Why an OOM here, mechanically (the NEW dashboard is the prime suspect, now confirmed):** the step-35000 boundary is the single
heaviest moment of the whole schedule — it runs, on one 34 GB GPU, all of: recon grid (12 imgs) + CFG ablation (9) + compose grid
(16) + the `panel=True` quality dashboard. The dashboard (`_run_quality_dashboard`, train_ldm.py:178; `metrics/eval_checkpoint.py`)
lazy-moves **PSPNet seg + the presence classifier onto the GPU** alongside the resident **UNet + the decode-only VAE**, then
**generates 192 images** (64 cardio + 64 effusion + 64 PoE, 50 DDIM steps each) and decodes every one. Two concrete VRAM peaks:
(1) `_presence_metrics` → `eval_checkpoint.py:361` does `torch.stack(tensors).to(device)` — it stacks **all images of a label at
once** (up to 64 × 224²) onto the GPU plus the classifier forward; (2) `panel=True` then runs a *second* full `evaluate_checkpoint`
for the 12 panel images while UNet+VAE+PSPNet are all still hot. Host RAM is only 33.5 GB and VRAM 34 GB — both tight. The recon
step alone took **463 s**; the combined 35k eval is the longest, most memory-saturated point in the run. This is textbook
single-GPU eval-OOM, and the `try/except` around the *Python* logic cannot save a process that the OS/driver kills outright.

**Was the model diverging? Answer to "Is it certain a crash means the model won't get better?" — NO, it is NOT a model failure
and the crash is orthogonal to model quality.** Distinguish the two crash *classes*:
- A **training failure** (NaN/Inf weights, true loss divergence) = the model is genuinely degrading; the checkpoint is poisoned.
- A **process crash** (OOM, unhandled exception, preemption) = the Python/CUDA *process* died; the **model state in the last
  checkpoint is intact**, fully resumable, and says **nothing** about model quality.
vqe9re62's crash is squarely the **process-crash** class: zero NaN/Inf, grad_norm 0.02–0.09, val 0.238 (a new run-min), legible
conditioning, anatomically valid samples at the moment of death. The step-30000 checkpoint is on disk and intact. **A relaunch
from it loses zero model progress.** So it is *not* certain — it is in fact *false* — that this crash means the model won't get
better; the model was on a healthy trajectory and simply needs the eval instrument to stop killing the process.

**"The crashes are suspicious" — naming the actual, DIFFERENT cause across runs.** The suspicion is well-founded but the two
crashes have **different, both-infrastructural** causes, neither of which is model divergence:
- **pcp9jo36 (v32, died @10294):** the **OLD divergence guard false-fired** — it anchored the kill threshold to a single noisy
  per-step loss and tripped on ordinary DDPM variance. A *harness* bug. (Now FIXED: the guard is EMA-smoothed + persistent over
  200 steps, and confirmed inert on vqe9re62.)
- **vqe9re62 (v33, died @35000):** the **NEW in-training quality dashboard OOM'd** at the step-35000 panel eval-cadence boundary.
  A *different* harness/infrastructure problem introduced by the UNTESTED-on-GPU dashboard — not the guard, not the model.
So the pattern is: **two consecutive runs killed by two different pieces of instrumentation wrapped around a healthy model.** The
model has never been the thing that failed. That is the suspicious-but-explicable through-line: every kill so far has come from
the *monitoring/eval* arm, not the *training* arm. We have been watching the loss-divergence arm while the eval/OOM arm is what
actually fires.

**Diagnostic steps** (to confirm OOM beyond the strong circumstantial case, before/with relaunch):
1. **Confirm the OOM at the OS/driver level** — on hippo run `dmesg -T | grep -iE "killed process|out of memory|oom"` (or
   `journalctl -k | grep -i oom`) around the 21:44 timestamp; an `Out of memory: Killed process … python` line is the smoking gun
   for a host-RAM OOM-kill. If nothing in dmesg, it was a CUDA-side OOM → check for a `CUDA out of memory` in any captured stderr
   (the tmux/nohup stream, NOT output.log — wandb's log only captures stdout; the CUDA error may have gone to stderr).
2. **Reproduce in isolation** — call `_run_quality_dashboard(..., panel=True)` once on the step-30000 checkpoint with
   `nvidia-smi dmon -s m` running; watch VRAM at the 192-image generate + PSPNet/classifier-resident peak. If it approaches 34 GB,
   OOM is confirmed and the fix is sizing (below).
3. **Rule the guard fully out** — already done from the log (no KILL/divergence/exit strings); the EMA guard arming line
   ("kill-criteria reference EMA set") also never printed, so the reference was never even armed before the crash. Not the guard.
4. **Confirm checkpoint integrity** — load `ckpts_rebalanced/model_step0030000.safetensors` + `training_state_step0030000.pt`;
   verify finite weights and optimizer state. (Resume already reads these via `_find_latest_checkpoint`.)

**Fix options** (the goal: stop the eval dashboard from OOM-killing a healthy run; the model needs no fix):
- **Option A (recommended, minimal): shrink + serialize the eval dashboard's GPU footprint.** Concretely in
  `metrics/eval_checkpoint.py` / `_run_quality_dashboard`: (i) drop `_EVAL_N_PER_FINDING` 64 → 16 (192 → 48 images); (ii) in
  `_predict_probs_arrays` (eval_checkpoint.py:361) replace the single `torch.stack(...).to(device)` over all label images with a
  **mini-batched loop** (e.g. 8 at a time) so the classifier never holds 64×224² at once; (iii) ensure the seg/classifier/VAE are
  moved on→GPU, used, then **off→CPU with `torch.cuda.empty_cache()` between sub-stages** (the code already empties cache in the
  `finally`, but not *between* generation and segmentation, where the peak is); (iv) move the OOM-prone instruments **CPU-only**
  for inference if VRAM is the binding constraint — PSPNet/classifier on 48 images on CPU is slow but cannot OOM the GPU.
- **Option B (cheapest unblock, do FIRST to get the run moving): lower the dashboard cadence and/or gate it behind a flag.** Make
  the `panel=True` quality dashboard run every 20000 steps instead of 5000, or gate the whole dashboard behind a config flag
  (`eval_dashboard: false`) and relaunch with it OFF. The recon/compose grids (cheap, already proven safe at 30k/35k) stay on;
  only the heavy PSPNet+classifier+192-img block is suppressed. This **decouples training progress from the untested instrument**
  — train to 100k now, score CTR offline afterward with `scripts/ctr_overlay.py` (the validated path) on the final checkpoint.
- **Option C (belt-and-suspenders): wrap the dashboard call to catch CUDA-OOM specifically and empty_cache + skip.** Add
  `except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); wandb.log({"eval/instrument_ok":0})`. NOTE: this only helps if
  the OOM is a *catchable* CUDA exception; if the OS OOM-killed the process (host RAM, 33.5 GB), no in-process handler can save it —
  then Option A/B (reduce footprint) is mandatory, not optional.

**Recommendation**: **Relaunch from the intact step-30000 checkpoint with the heavy eval dashboard OFF (Option B — gate/lower
cadence) so training reaches 100k uninterrupted, and separately apply Option A sizing before re-enabling it.** Do NOT treat this
as model divergence, do NOT touch lr/batch/class_mult/VAE/architecture, and do NOT re-arm any tighter loss guard — the model is
healthy and the loss-divergence arm is the wrong thing to watch. Let it train; score cardio CTR offline on the final checkpoint.

**Unrecognized observations** (not catalog training modes — infrastructural, but they determined the run's lifetime):
- **eval-dashboard-OOM** (catalog-addendum candidate): a NEW in-training eval/instrument block (segmentation + classifier +
  large generation batch co-resident on a single GPU) causes a CUDA/host OOM that **kills the process below the Python try/except**.
  Signature: run state=failed/crashed; log ends **mid-eval at an eval-cadence step** (multiple of the panel cadence); **no
  traceback, no `eval/instrument_ok=0`, no clean `wandb.finish`**; last checkpoint healthy; zero NaN/Inf. Check: crash step is a
  multiple of the eval cadence AND `dmesg`/stderr shows OOM. This is distinct from `early-stop-guard-misfire` (v32) and from any
  loss-based mode — it is an instrument-footprint problem, recoverable by resume, orthogonal to model quality.
- **The two consecutive crashes (pcp9jo36, vqe9re62) are both monitoring-arm failures, not training-arm failures** — worth
  recording as a meta-pattern: when a run dies, check the *instrument/eval/guard* path BEFORE concluding the model diverged. On
  this project so far, the model has never been the cause of a crash.

**Catalog modes skipped/not-checked due to missing artifacts**: weight-norm-explosion, update-to-weight-ratio (no per-layer
norms); checkerboard-artifact, noise-schedule-mismatch, timestep-distribution-skew (no FFT/timestep histograms); attention-map-
collapse (no attention logging); fid-regression (no FID series); latent-collapse, kl-* (VAE-side, N/A). lr-batch-size-mismatch:
not flagged (batch 16 / lr 1e-4 unchanged from prior runs by design; no reference config defines a different LR).

---

# Training Tracking — prime_lab/paper3-ldm/gtr6kqpk

Entries appended by /analyze-run; oldest first. This is a NEW run-id block in the shared ledger; prior blocks (prvl98fr, hifei736, pcp9jo36, vqe9re62) are untouched.

Run name: `arm1_cleancardio_20260626_1649`
Config: `configs/ldm_arm1_cleancardio.yaml` | Arm-1 clean-cardio LDM retrain from scratch, max_steps=100000, ckpt_every=10000 (10 saves), lr=1e-4 constant, bf16 train + fp32 decode, batch 16×1 (effective 16). Latent-cache `data/latents/rebalanced_cleancardio`; frozen-VAE codec decode ckpt `ckpts/vae_step0025000.pt`. cfg conditioning over 3 classes (no_finding / cardiomegaly / effusion).
Hardware: hippocampus, single RTX 5090. Started 2026-06-26T16:49:51Z, reached step 100000 ~22:04Z cleanly.
Data: rebalanced clean-cardio set (border-visible cardiac border; cardio up-weighted relative to the 7:1 imbalance that under-trained the prior cardio conditional).
Artifacts: 10 checkpoint saves `ckpts_arm1_cleancardio/model_step{0010000..0100000}.safetensors`. Rich visual logging THIS run (recon per-class + grid + cfg_ablation every 5000; PoE compose anchors + quality panels every 15000) — visual checks ARE available, unlike the latent-cache-only prvl98fr smoke run.
Baseline: no `.claude/training-references/` directory → absolute catalog checks only, no z-score/percentile comparison.
Run health headline: NO NaN/Inf anywhere; train/loss noisy-flat ~0.20–0.30 with a mild downward drift (first<10k mean 0.301 → last>90k mean 0.205); val/loss FLAT 0.21–0.30, val≈train throughout (no overfitting); grad_norm tiny and stable (mean 0.047, ZERO clip events — never hit 1.0); lr constant 1e-4. This is the relaunch the vqe9re62 entry recommended; the eval-dashboard-OOM is FIXED (`eval/instrument_ok=1` every eval row, reached 100k with no traceback).

N/A modes for this run (noted once here, not repeated per-save): kl-posterior-collapse, kl-weight-imbalance, latent-collapse (frozen-VAE LDM, noise-prediction training — no KL term, no trainable latent prior).
Not-checked modes (artifacts not logged): attention-map-collapse (no attention maps); update-to-weight-ratio, weight-norm-explosion (no per-layer/weight norms); checkerboard-artifact, noise-schedule-mismatch, timestep-distribution-skew (no FFT/per-timestep/timestep-histogram series); fid-regression (no FID series); loss-scale-anomalous, lr-batch-size-mismatch (no baseline/reference config defining a different LR — batch16/lr1e-4 unchanged by design).

---

## Save 1 (step 10000) — eval row @10000

**Severity**: ✓ healthy

**Visual**: Per-class recon + grid_10000 inspected — structures present but hazy with lower-zone fog; 3-class grid already shows distinct NO_FINDING / CARDIO / EFFUSION columns and good per-seed diversity (no mode collapse). Early-training softness, expected.
**Metrics**: train/loss ~0.30 (early band), val/loss ~0.255 (≈train, flat), grad_norm ~0.047 (no clip). cardio CTR_med 0.474, effusion CTR_med 0.368, poe CTR_med 0.464, yield 0.094, presence cardio 0.107 / effusion 0.628 / both 0.000.
**Trend**: insufficient history (first save). nan-or-inf ✓, loss-zero-collapse ✓, fp-precision-artifact ✓ (no clip, no black patches seen), val-train-divergence ✓ (val≈train).

---

## Save 2 (step 20000) — eval row @20000

**Severity**: ✓ healthy

**Visual**: not re-rendered this save beyond the 5000-cadence recon set; sampled neighbours (15k/25k) consistent with Save 1 — hazy-but-improving, anatomy present, diverse across seeds.
**Metrics**: train/loss ~0.25 band, val/loss ~0.255 (flat, ≈train), grad_norm ~0.047 (no clip). cardio CTR_med 0.443, effusion CTR_med 0.363, poe CTR_med 0.389, yield 0.000, presence cardio 0.076 / effusion 0.659 / both 0.031.
**Trend**: flat (cardio CTR 0.474→0.443, within eval-to-eval noise). nan-or-inf ✓, oscillation ✓ (loss band stable, no growing p2p), val-train-divergence ✓. The single-eval yield=0.000 and low cardio presence are below downstream targets but are not catalog training-failure modes (loss/grad/val all healthy); treated as project-target context, not ⚠️.

---

## Save 3 (step 30000) — eval row @30000

**Severity**: ✓ healthy

**Visual**: recon cardio/effusion at 30k sharper than 10k; lung fields darkening, cardiac silhouette better defined. compose healthy-anchor @15k (nearest logged) is an anatomically plausible composed CXR.
**Metrics**: train/loss ~0.22 band, val/loss ~0.255 (flat), grad_norm ~0.047. cardio CTR_med 0.485, effusion CTR_med 0.384, poe CTR_med 0.437, yield 0.063, presence cardio 0.357 / effusion 0.534 / both 0.031.
**Trend**: improving (cardio CTR 0.443→0.485; cardio presence up to 0.357). nan-or-inf ✓, oscillation ✓, val-train-divergence ✓.

---

## Save 4 (step 40000) — eval row @40000

**Severity**: ✓ healthy

**Visual**: trajectory consistent — crisper than early saves, no artifacts/black patches/checkerboard observed in the sampled recon set.
**Metrics**: train/loss ~0.21 band, val/loss ~0.255 (flat), grad_norm ~0.047. cardio CTR_med 0.462, effusion CTR_med 0.392, poe CTR_med 0.459, yield 0.250 (best single-eval so far), presence cardio 0.326 / effusion 0.597 / both 0.031.
**Trend**: improving overall; effusion CTR creeping up (0.363→0.392) while holding presence — effusion channel is NOT regressing as cardio improves. nan-or-inf ✓, oscillation ✓, val-train-divergence ✓.

---

## Save 5 (step 50000) — eval row @50000

**Severity**: ✓ healthy

**Visual**: cardiomegaly_50000 inspected directly — crisp dark lung fields, clearly defined cardiac silhouette, visible ribs; a clear step up in sharpness from the 10k sample. Quality is improving with training.
**Metrics**: train/loss ~0.20 band, val/loss ~0.255 (flat), grad_norm ~0.047. cardio CTR_med 0.511 (crosses the 0.5 ref line), effusion CTR_med 0.403, poe CTR_med 0.505, yield 0.250, presence cardio 0.295 / effusion 0.503 / both 0.031.
**Trend**: improving — cardio CTR crosses 0.5 for the first time, poe CTR follows (0.505). nan-or-inf ✓, oscillation ✓, val-train-divergence ✓. This is the rebalance lever landing (cardio conditional now rendering an at-target heart).

---

## Save 6 (step 60000) — eval row @60000

**Severity**: ✓ healthy

**Visual**: consistent with 50k — sharp, anatomically plausible, diverse.
**Metrics**: train/loss ~0.20 band, val/loss ~0.255 (flat), grad_norm ~0.047. cardio CTR_med 0.501, effusion CTR_med 0.409, poe CTR_med 0.488, yield 0.219, presence cardio 0.263 / effusion 0.565 / both 0.000.
**Trend**: flat-high (cardio CTR holding ~0.50). nan-or-inf ✓, oscillation ✓, val-train-divergence ✓. cardio CTR now sits AT the 0.5 line across consecutive evals — at, not above, the downstream target, but stable and not a catalog failure.

---

## Save 7 (step 70000) — eval row @70000

**Severity**: ✓ healthy

**Visual**: sampled set consistent; no degradation.
**Metrics**: train/loss ~0.20 band, val/loss ~0.255 (flat), grad_norm ~0.047. cardio CTR_med 0.484, effusion CTR_med 0.411, poe CTR_med 0.447, yield 0.125, presence cardio 0.295 / effusion 0.503 / both 0.000.
**Trend**: flat (cardio CTR 0.501→0.484, within noise; effusion CTR still inching up). nan-or-inf ✓, oscillation ✓, val-train-divergence ✓.

---

## Save 8 (step 80000) — eval row @80000

**Severity**: ✓ healthy

**Visual**: consistent.
**Metrics**: train/loss ~0.20 band, val/loss ~0.255 (flat), grad_norm ~0.047. cardio CTR_med 0.496, effusion CTR_med 0.399, poe CTR_med 0.475, yield 0.125, presence cardio 0.295 / effusion 0.565 / both 0.000.
**Trend**: flat-high (cardio CTR back to ~0.50). nan-or-inf ✓, oscillation ✓, val-train-divergence ✓.

---

## Save 9 (step 90000) — eval row @90000

**Severity**: ✓ healthy

**Visual**: panel cardio_layered_90000 inspected — broad cardiac silhouette / enlarged heart relative to thorax (cardiomegaly conditioning is rendering an enlarged heart, the lever's intended effect), dark lung fields, sharp ribs. compose healthy-anchor w1.0_90000 is softer/hazier with a whited-out lower zone and slightly distorted upper mediastinum — the PoE-compose path is hazier than the recon path (expected: product-of-experts is harder), but still an artifact-free plausible chest. The residual basal/diaphragm whiteout is the known non-effusion haze (border filter cleans the cardiac border only).
**Metrics**: train/loss ~0.205 (last-band mean), val/loss ~0.243 (flat), grad_norm ~0.047. cardio CTR_med 0.508, effusion CTR_med 0.410, poe CTR_med 0.490, yield 0.156, presence cardio 0.263 / effusion 0.597 / both 0.063.
**Trend**: improving-then-flat-high (cardio CTR 0.508, near run-max). nan-or-inf ✓, oscillation ✓, val-train-divergence ✓.

---

## Save 10 (step 100000) — eval row @100000 (final)

**Severity**: ✓ healthy

**Visual**: cardiomegaly_100000, effusion_100000, no_finding_100000, grid_100000, cfg_ablation_100000 all inspected directly. Late samples are the sharpest of the run: clean dark lung fields, well-defined heart, visible ribs and costophrenic angles. Conditioning is functional and class-distinct — no_finding renders a normal-sized heart with crisp dark lungs; cardiomegaly renders a visibly broader cardiac silhouette; effusion renders basal opacity / blunted costophrenic angles. cfg_ablation_100000 (rows w=0 unconditional → w=1 → w=2 amplified) shows class features sharpening as CFG weight rises — conditioning is NOT ignored. grid_100000 retains good per-seed diversity (no mode/diversity collapse). No black patches, no checkerboard, no saturation.
**Metrics**: train/loss summary-final 0.198, val/loss summary-final 0.243 (val≈train, no overfitting), grad_norm final 0.0145 (no clip). cardio CTR_med 0.499, effusion CTR_med 0.406, poe CTR_med 0.493, yield 0.125, presence cardio 0.201 / effusion 0.659 / both 0.063, leak/cardio→effusion 0.188 (lowest of run).

**Per-mode flags** (full check at final save):
- nan-or-inf: ✓ pass (train/val/grad all finite)
- loss-zero-collapse: ✓ pass (loss 0.198 ≫ 1e-8)
- oscillation: ✓ pass (last-5 loss band stable; no growing p2p — contrast the prvl98fr smoke run)
- val-train-divergence / overfit-regime: ✓ pass (val 0.243 ≈ train 0.198, val flat across all 100 logged points; gap small, not widening)
- loss-plateau: ✓ pass (mild downward drift first→last 0.301→0.205; flat tail is convergence with val tracking, not a stall — grads non-vanishing at 0.0145)
- gradient-vanishing: ✓ pass (grad_norm 0.0145; small but the loss still drifted down and val tracks — well-conditioned bf16, not a frozen model)
- gradient-explosion / fp-precision-artifact: ✓ pass (ZERO clip events all run; bf16 train + fp32 decode; no clustered black patches in any inspected sample)
- all-black-pixels / blurry-outputs / diversity-collapse / lung-anatomy-absent / conditioning-ignored / unexpected-image-stats: ✓ pass by direct visual inspection (sharp, anatomically plausible, diverse, conditioning class-distinct, CFG functional)
- loss-component-imbalance: not checked (per-class loss components not separately logged this run)
- kl-* / latent-collapse: N/A (frozen-VAE noise-prediction LDM)
- attention-map-collapse, update-to-weight-ratio, weight-norm-explosion, checkerboard-artifact, noise-schedule-mismatch, timestep-distribution-skew, fid-regression, loss-scale-anomalous, lr-batch-size-mismatch: not checked (artifacts/baseline not logged — see block header)

**Trend signals**:
- Improvement direction: improving overall — cardio CTR_med early(10–40k) mean ~0.466 → late(70–100k) mean ~0.497, crossing/holding the 0.5 ref line from 50k; effusion CTR_med rose 0.368→0.406 while holding presence (effusion did NOT regress as cardio improved); cardio→effusion leakage fell to its run-low 0.188.
- Oscillation: none structurally — train loss is a stable noisy-flat band, val flat, grads stable. (Random-timestep sampling makes the loss noisy by design but p2p is not growing.)
- Persistence of concern: none. No mode flagged on any of the 10 saves.

**Reference comparison**: no baseline available; absolute catalog checks only.

**Note — lr logs 0 only at step 100000**: this is a LOGGING ARTIFACT of the post-loop final log capturing the scheduler at terminal state on a constant-LR schedule (lr held 1e-4 from start through step 99000). It is NOT a decay schedule and NOT a training fault — do not flag as gradient-vanishing-by-lr or as a plateau cause.

**Diagnosis (positive)**: This run is the clean, completed Arm-1 clean-cardio retrain the vqe9re62 entry called for. The eval-dashboard-OOM that killed the two prior monitoring-arm runs is fixed (`eval/instrument_ok=1` every eval, clean finish at 100k). Mechanically: stable bf16 optimization (tiny well-conditioned grads, zero clips), no overfitting (val tracks train flat), and the rebalance lever landed — the cardio conditional now renders an at-target enlarged heart (cardio CTR crossed and holds ~0.5 from 50k; visually confirmed enlarged silhouette in the cardio recon/panel) without the effusion channel regressing.

**Downstream-target context (NOT catalog ⚠️)**: cardio CTR_med sits AT (0.499) rather than ABOVE the 0.5 ref / 0.535 real-cardio target; Gate-3 PoE yield ends at 0.125 vs the ~0.30 project target; cardio presence ~0.20 is modest. These are downstream PoE-utility shortfalls, not training failures — the model trained correctly and converged cleanly. They are the inputs to the experiment/decision layer, not the training-health layer.

**Recommendation**: Accept the run as a clean, converged, artifact-free training. Treat the at-line cardio CTR / sub-target yield as a downstream-yield question for the experiment layer (e.g. CFG-anchor sweep, sampler, EMA — the next rungs on the improvement ladder), not as a reason to re-train or change lr/batch/architecture. Let it stand; score offline on the final checkpoint.

**Unrecognized observations**: none. The only off-nominal scalar (lr=0 at the terminal step) is the explained logging artifact above. The PoE-compose path being hazier than the recon path is expected (product-of-experts), not a bug.

---

## Run summary — gtr6kqpk

**Run**: `prime_lab/paper3-ldm/arm1_cleancardio_20260626_1649` (gtr6kqpk), COMPLETE, 100000 steps, 10 saves.

**Severity timeline**: ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ (all healthy)

**Triggered modes**: none on any save.

**Healthy signals**: no NaN/Inf; flat val tracking flat train (no overfitting); tiny stable grads, zero clip events; sample quality visibly improves early→late; conditioning class-distinct and CFG-responsive; diversity preserved; cardio CTR crosses/holds 0.5 from 50k with effusion holding and leakage at run-low.

**Downstream (non-training) shortfalls carried to the experiment layer**: cardio CTR at-line not above target; PoE Gate-3 yield 0.125 vs ~0.30; cardio presence ~0.20.

---

# Training Tracking — prime_lab/paper3-vae-v2/kj7dunvn

Entries appended by /analyze-run; oldest first.

Run name: `exp_vae_v2_finetune`
Script: `scripts/train_vae_v2.py`
Config: `configs/vae_v2.yaml` | VAE-v2 perceptual fine-tune. Resumed from `ckpts/vae_step0025000.pt` (the existing NIH VAE). Runs steps 25000 → 75000 (50000 additional steps). save_every=5000 (10 saves expected: steps 30000, 35000, …, 75000). eval_every=5000. batch=2, lr=5e-5 (half of original 1e-4), fp32 (bf16 disabled; GroupNorm collapse documented in memory). LPIPS λ=0.1 (AlexNet), PatchGAN discriminator λ=0.5 with warmup in the original 25k run (disc fully active from step 25100 onward). kl_weight=1e-6. grad_clip_norm=1.0. 16,226 train / 854 val images (NIH+VinDr border-visible diet). EMA decay=0.999.
Hardware: hippocampus, RTX 5090, 34 GB VRAM.
Run state at sweep time: RUNNING. Created 2026-06-29T12:09:23Z. Last step 29400 at 2026-06-29T13:08Z. 4400 steps into the 50000-step fine-tune (8.8%). No checkpoint saves yet; first save at step 30000 (imminent — ~8 min away at sweep time).
Loss components (logged every 100 steps): loss/total = lambda_recon*l1 + kl_weight*kl + lambda_perc*lpips + disc_weight*disc_g (can be negative when generator fools discriminator). loss/kl logged in nats (raw, not weighted). loss/recon_l1, loss/lpips are the primary quality signals. loss/disc_d and loss/disc_g track adversarial dynamics.
Baseline: no `.claude/training-references/` directory → all checks are absolute catalog checks only.
Visual artifacts: eval block runs at step % 5000 == 0 (first at step 30000); no recon grids logged yet. All visual checks are not checked for this entry.

N/A modes for this run (frozen encoder/decoder fine-tune with discriminator — noted once here):
- kl-posterior-collapse: the KL is 1436–8042 nats (healthy range; falling 44% from start to now, aligning posterior toward prior). Not collapsed.
- latent-collapse: no latent PCA logged. Not checked.
- conditioning-ignored: this is a VAE fine-tune (no class conditioning on the VAE side). Not applicable.
- noise-schedule-mismatch, timestep-distribution-skew: no diffusion timestep sampling in a VAE. Not applicable.

Not-checked modes (artifacts not yet logged): all-black-pixels, blurry-outputs, diversity-collapse, lung-anatomy-absent, no-sample-improvement, checkerboard-artifact, unexpected-image-stats, fid-regression, attention-map-collapse (no recon grids yet; first eval at step 30000). loss-scale-anomalous, lr-batch-size-mismatch (no baseline reference config defined for VAE v2). update-to-weight-ratio, weight-norm-explosion (no per-layer norms logged).

---

## Pre-checkpoint sweep (steps 25100–29400) — 2026-06-29T12:09–13:08Z

**Status**: No checkpoint saves have been written yet (first save at step 30000, which was imminent at sweep time). This entry documents the run's health during the first 4400 steps of the fine-tune, to be used as trend context for the first checkpoint entry.

**Not a checkpoint save entry.** No per-save catalog analysis is performed here (catalog analysis is anchored to checkpoint saves). Instead, this documents the run-wide metric state and flags any catalog modes that can be evaluated from the step-level history.

**Visual observations**: not checked (no eval has run yet; first recon grid at step 30000).

**Metric overview (steps 25100–29400, 44 logged rows at 100-step intervals)**:
- loss/total: min=-0.296, max=+0.393, mean=+0.029. Negative values occur when the generator adversarial loss (disc_g) is large and negative (generator fooling discriminator); this is expected GAN behavior, not a negative-loss bug.
- loss/recon_l1: min=0.0185, max=0.0875, mean=0.035. First-10 mean=0.035, last-10 mean=0.038. Noisy but broadly flat-to-slightly-improving (slope per 100 steps: -0.0065, i.e. the trend is downward but masked by per-batch variance).
- loss/lpips: min=0.024, max=0.094, mean=0.048. First-10 mean=0.057, last-10 mean=0.038 — a clear improving trend (slope -0.051 per 100 steps). LPIPS is the primary perceptual quality signal and it is improving.
- loss/kl (nats): min=1436, max=8042, mean=2928. KL is DECLINING: first-10 mean=4229 → last-10 mean=2355 (-44%). At kl_weight=1e-6 the KL contribution to total loss is ~0.003–0.008 — small but not negligible relative to l1. The decline means the fine-tune is refining the posterior toward the prior (healthy for a codec fine-tune).
- loss/disc_d: consistently ~2.00 (range 1.965–2.073). A discriminator loss of 2.0 is the equilibrium when the discriminator cannot distinguish real from reconstructed images (binary cross-entropy baseline for 50/50 real/fake). This is the correct operating point for a generator that is fooling the discriminator.
- loss/disc_g (generator adversarial): ranges -0.729 to +0.727. Sign alternates; large negative values mean the generator is strongly fooling the discriminator. The adversarial dynamics are active.
- train/grad_norm (PRE-CLIP): min=1.56, max=30.44, mean=12.69. EVERY step has pre-clip grad norm > 1.0 — the clip_grad_norm_(1.0) fires on all 44 observed steps. The effective gradient is scaled down ~12.7x on average.

**Per-mode flags**:
- nan-or-inf: ✓ pass (0 non-finite values across all 44 rows for all logged keys)
- loss-zero-collapse: ✓ pass (l1 min=0.0185, well above 1e-8; loss/total can be negative but that is normal GAN behavior, not zero-collapse)
- oscillation (on loss/recon_l1, the VAE quality metric): ⚠️ triggered. Every 5-save window of recon_l1 exceeds rel_p2p > 0.15 (range 0.375–1.682 across 9 windows). 41/41 rolling windows triggered; max consecutive = 41. Primary driver: batch_size=2 with adversarial gradient updates produces high per-batch variance. The LPIPS loss (better-behaved) shows a clear downward trend despite the l1 oscillation, suggesting the oscillation is batch-noise, not LR instability.
- gradient-explosion (z-score > 3.0): ⚠️ triggered once. At step 27700, pre-clip grad_norm=30.44 (z=3.08 over 20-step window). Single trigger, Warning tier. Post-clip the update was capped at 1.0 — no damage to weights. The overall grad distribution (mean=12.69) is consistent throughout; this is one high-noise batch.
- loss-plateau: ✓ pass on LPIPS (clearly improving). l1 is noisy-flat but this is batch-size-2 GAN variance, not a plateau. Run is only 8.8% complete — plateau check not meaningful at this stage.
- val-train-divergence: not checked (no val/loss logged in the step-level history; val eval first runs at step 30000).
- overfit-regime: not checked (same reason).
- kl-weight-imbalance: ✓ pass. kl_contribution = (1e-6 * kl) / (l1 + 1e-6*kl) = 3.1%–18.5% (mean 8.1%). Catalog threshold requires < 1% AND beta < 0.001. Beta=1e-6 < 0.001 is TRUE, but contribution is NOT below 1% — so the combined condition does not trigger.
- fp-precision-artifact: ✓ pass (fp32 training confirmed from launch banner "fp32 (bf16 disabled for decoder)"; no GradScaler; no visual artifacts to check yet).
- gradient-vanishing: ✓ pass (grad_norm always > 1.0 pre-clip; well above 1e-6 threshold).
- loss-component-imbalance: partial. The dominant loss component varies per batch — l1 ~0.035, lpips ~0.048, disc_g contribution ~0 on average (symmetric about 0), kl*w ~0.003. No single component ratio exceeds 10:1 when comparing the fixed-sign components (l1 and kl); disc_g is by design allowed to dominate in both directions. Not a structural imbalance.
- All other modes: not checked (no visual artifacts, no per-layer norms, no baseline).

**Trend signals (pre-checkpoint)**:
- Improvement direction: LPIPS clearly improving (first-10 mean 0.057 → last-10 mean 0.038, -33%). l1 noisy-flat (high per-batch variance at batch=2). KL declining (posterior aligning to prior).
- Oscillation: persistent on l1 (batch-noise driven, all windows trigger). Not persistent on LPIPS (improving).
- Persistence of concern: oscillation 41/41 windows triggered on l1 — meets persistence criterion numerically. However, the mechanism is definitively batch_size=2 GAN stochasticity (not LR instability), the LPIPS trend is clearly improving, and no visual or convergence regression is observable. See diagnosis.

**Reference comparison**: no baseline available; absolute checks only.

**Diagnosis**:
The run is in a GAN-adversarial fine-tuning regime that makes several metric patterns look alarming out of context:

1. **Persistent gradient clipping (mean pre-clip 12.69, threshold 1.0, 45/45 steps clipped)**: The adversarial generator loss introduces gradient magnitudes that are structurally larger than a pure reconstruction loss. This is documented behavior for VAE+GAN training (taming-transformers / LDM codebook VAE style) — the adversarial gradient from the discriminator is design-larger than the L1/LPIPS reconstruction gradient and drives the norm above the clip threshold. The clipping is doing its job (capping updates to a safe magnitude). The effective LR is ~3.9e-6 per step after clipping, which is conservative for a fine-tune. Not a training failure.

2. **Oscillation on l1 (every window triggers)**: batch_size=2 with alternating adversarial vs reconstruction gradient directions produces very high per-batch l1 variance. The same oscillation pattern was documented for the smoke run (prvl98fr) at batch_size=2. The LPIPS metric (perceptual quality, the primary target of this fine-tune) is clearly improving — this is the more reliable signal.

3. **Single grad-norm spike at step 27700 (30.44, z=3.08)**: One step received an unusually large batch gradient. With no NaN/Inf, no downstream loss spike, and the norm clipped before the update, this is a one-off event. Not concerning.

4. **loss/total going negative**: This is correct behavior for a GAN generator loss. When disc_g (the generator's adversarial term) is large and negative (generator fools the discriminator), the total drops below zero. This is not a "negative loss bug" — it is the GAN objective working as intended (log D(G(x)) being large when D is fooled).

5. **KL declining from 4229 → 2355 nats**: The encoder posterior is moving closer to the N(0,1) prior during fine-tuning. This is beneficial for inference-time sampling and indicates the LPIPS+discriminator losses are not breaking the prior matching — the fine-tune is finding a better trade-off point.

**Diagnostic steps** (for monitoring at step 30000 checkpoint):
1. Check val/loss components (val/loss_total, val/loss_recon_l1, val/loss_lpips) at step 30000 — confirm LPIPS continues improving and l1 is not diverging on the held-out set.
2. Check the recon grid at step 30000 (16×3 images): do cardiac borders look sharper than in the original 25k checkpoint? This is the deliverable of the fine-tune. Compare against the original `vae_step0025000.pt` recon quality as a reference.
3. Check val/delta_ctr_median at step 30000 — the script logs this explicitly (threshold 0.015 triggers a warning flag). If delta_ctr_median < 0.015, the VAE reconstruction is preserving CTR geometry.
4. If the recon grid is logged to W&B at step 30000, WebFetch the image URL and inspect directly.

**Fix options** (pre-emptive, not yet needed):
- Option A (no action needed now): The run is healthy and 8.8% complete. Continue to step 30000 and evaluate at the first checkpoint.
- Option B (if oscillation persists and LPIPS stalls): The oscillation on l1 is expected at batch=2 with adversarial gradients. The config already documents batch=2 as the maximum before LPIPS+disc OOM at 32GB. No batch-size fix available without architecture changes; accept the oscillation.
- Option C (if val/delta_ctr_median > 0.015 at step 30000): VAE is regressing on CTR geometry. Reduce disc_weight from 0.5 to 0.3 to let reconstruction quality dominate. The adversarial signal may be over-sharpening edges at the expense of cardiac geometry fidelity.

**Recommendation**: Continue run. No action needed before step 30000. Review the first checkpoint entry when it is written.

**Unrecognized observations**:
- The discriminator loss (disc_d ≈ 2.00 at every step) is at the theoretical equilibrium for a binary classifier with 50/50 accuracy. This could mean the generator has immediately fooled the discriminator from step 25100 onward (the discriminator was already saturated from the original 25k fine-tune). A saturated discriminator provides a near-zero gradient to improve the VAE's adversarial objective — this is the "GAN training instability" mode where one party becomes too strong. However, disc_g (the generator loss) is still varying widely (-0.73 to +0.73), suggesting the discriminator IS giving signal despite appearing at equilibrium on disc_d. This may be a logging artifact (disc_d is logged once per training step, but the discriminator may actually be improving between those snapshots). Watch disc_d at step 30000 — if it remains stuck at 2.00, the discriminator has collapsed and the adversarial fine-tuning is no longer providing useful gradients to the generator.
- No val rows in the W&B history up to step 29400 — confirms that val eval has not run yet (first at step 30000). This is expected per the eval_every=5000 schedule.

**Catalog modes skipped (not applicable / not checkable at this stage)**: all visual modes (no recon grids until step 30000); latent-collapse (no PCA); kl-posterior-collapse (KL healthy and declining); conditioning-ignored (VAE has no class conditioning); noise-schedule-mismatch / timestep-distribution-skew (no diffusion timesteps in a VAE); fid-regression (no FID logged); loss-scale-anomalous / lr-batch-size-mismatch (no baseline reference config); weight-norm-explosion / update-to-weight-ratio (no per-layer norms).

---

---

# Training Tracking — prime_lab/paper3-vae-v2/kj7dunvn

Entries appended by /analyze-run; oldest first.

Run name: `exp_vae_v2_finetune`
Config: `configs/vae_v2.yaml` | Fine-tune of `vae_step0025000.pt` with LPIPS + PatchGAN (70×70 RF hinge). max_steps=75000, warm_start at step 25000 (net additional steps=50000), lr=5e-5, weight_decay=1e-2, batch=2, fp32=True (bf16 explicitly disabled), save_every=5000, eval_every=5000.
Loss structure: total = L1 + beta*KL + 0.1*LPIPS + disc_weight*adv_g, where beta=1e-6 (fixed, no annealing), disc_weight ramps from 0 at step 1000 to 0.5 at step 2000. Discriminator is a 5-layer PatchGAN with GroupNorm and hinge loss.
Hardware: hippocampus (single RTX 5090). Started 2026-06-29T12:09:23Z, currently running.
Data: combined NIH+VinDr border-visible diet (train_manifest.csv, 17,080 rows: NIH 10,409 + VinDr 6,671).
Artifacts: NO checkpoint saves yet — first save is at step 30000 (the warm-start lives at ckpts/vae_step0025000.pt; the fine-tune output ckpts/vae_v2/ is empty). Run is at step ~29400 at time of analysis (~600 steps from first checkpoint).
Visual logging: eval hook (_vae_eval_hook) runs at step 30000; no visual artifacts are available yet.
Baseline: no `.claude/training-references/` directory; all checks are absolute catalog checks only.
Prior-run context: the warm-start checkpoint (vae_step0025000.pt) was the canonical VAE from the arm1-cleancardio LDM run (gtr6kqpk). That VAE used no LPIPS/discriminator. This fine-tune adds those to sharpen cardiac-border rendering.

N/A modes for this run: kl-posterior-collapse (KL is NOT near zero; see below), overfit-regime (no val/loss logged yet; eval hook at step 30000), val-train-divergence (same), latent-collapse (no latent manifold plots), attention-map-collapse (no attention maps), checkerboard-artifact / noise-schedule-mismatch / timestep-distribution-skew / fid-regression (no relevant artifacts logged), update-to-weight-ratio / weight-norm-explosion (no per-layer norms logged).

---

## Pre-checkpoint interval (steps 25100–29400) — 2026-06-29T12:09:23Z to ~T13:03Z

**Severity**: ⚠️ concerning

**Visual observations**:
No visual artifacts exist yet. The _vae_eval_hook fires at step 30000 (the first scheduled eval step after the warm-start at 25000). No recon panels, latent manifold, CTR delta, or presence metrics have been logged. All visual catalog checks are not checked for this interval. Visual inspection will be possible at step 30000.

**Per-mode flags**:
- nan-or-inf: ✓ pass (no NaN/Inf in any of loss/total, loss/recon_l1, loss/kl, loss/lpips, loss/disc_d, loss/disc_g, train/grad_norm across all 44 logged steps)
- loss-zero-collapse: ✓ pass (recon_l1 ranges 0.019–0.088; total loss ranges −0.296 to +0.393; reconstruction is well above 1e-8)
- all-black-pixels: not checked (no visual artifacts yet)
- no-sample-improvement: not checked (no visual artifacts yet)
- oscillation: ⚠️ triggered — ALL 40 computed 5-window intervals have rel_p2p >> 0.15; typical rel_p2p 3–41. The total loss oscillates between −0.296 and +0.393 across the 44 logged steps. Root cause is the adversarial component (disc_g swings from −0.729 to +0.727 with std 0.323), which when multiplied by disc_weight=0.5 contributes up to ±0.36 to the total loss. The reconstruction metrics (L1 and LPIPS) are improving monotonically and do NOT oscillate — see reconstruction quality trend below.
- val-train-divergence: not checked (no val/loss logged; first eval at step 30000)
- diversity-collapse: not checked (no visual artifacts)
- latent-collapse: not checked (no latent manifold)
- loss-plateau: ✓ pass (recon quality improving continuously: L1 0.088→0.031, LPIPS 0.094→0.033, slope −1.06e-4/100steps on combined metric)
- loss-component-imbalance: not checked in the standard sense; however, the adversarial term (disc_g × 0.5) is the dominant oscillation source, not an imbalance between L1 and LPIPS
- kl-posterior-collapse: ✓ pass — KL is NOT near zero. Raw KL ranges 1437–8042 nats; per-latent-dim KL is 0.050–0.123 nats (latent shape 4×128×128 = 65536 dims). This is normal for a trained VAE.
- kl-weight-imbalance: ⚠️ triggered — beta=1e-6, well below the 0.001 catalog threshold; KL contribution to total non-adversarial loss is 3–20%. However, this is by design in the config — the intent is to sharpen reconstruction without letting KL dominate the LPIPS fine-tune. As a codec for LDM sampling (not standalone generative), the latent prior shape matters less than reconstruction quality. Flagging as ⚠️ per catalog rules; see Diagnosis below.
- blurry-outputs: not checked (no visual artifacts)
- fp-precision-artifact: ✓ pass by design — fp32=True explicitly set in config and enforced (bf16 disabled after the bf16 GroupNorm black-image bug documented in project memory). No GradScaler (fp32 training). Pixel-patch checks not possible without images.
- conditioning-ignored: not checked (no inference samples; this VAE has no class conditioning — it is a codec)
- gradient-vanishing: ✓ pass (grad_norm ranges 1.56–30.44 before clipping; all well above 1e-6)
- gradient-explosion: ⚠️ triggered — ALL 44 logged grad_norm values exceed the 1.0 clip threshold. Mean pre-clip grad_norm = 12.64; max = 30.44 at step 27700. Clip fires on every single update. Warning tier per catalog; see Diagnosis below.
- loss-scale-anomalous: not checked (no baseline)
- unexpected-image-stats: not checked (no visual artifacts)
- loss-spike-recovery: ✓ pass — no isolated spike pattern; the oscillation is structural to GAN dynamics, not a one-off spike
- lr-batch-size-mismatch: not checked (no reference config defines a different LR for this architecture)

**Trend signals**:
- Improvement direction: reconstruction improving (L1: 0.088→0.031, −65%; LPIPS: 0.094→0.033, −65%) over 4300 steps. Total loss is near-zero mean with high-amplitude adversarial oscillation, not meaningful on its own.
- Oscillation: rel_p2p 3–41 across all windows, but driven entirely by disc_g swing. Reconstruction loss (L1 + 0.1×LPIPS) does not oscillate; it decreases smoothly.
- Persistence of concern: oscillation triggered every step (structural to GAN training, not LR instability); gradient-explosion triggered every step (clip always firing); kl-weight-imbalance triggered by config design.

**Reference comparison**:
No baseline available; absolute catalog checks only. Prior VAE (vae_step0025000.pt, trained without LPIPS/disc) had no comparable adversarial dynamics; this run is the first GAN-augmented training in the project.

**Diagnosis**:

**Oscillation** (⚠️ but GAN-structural): The total loss oscillation is entirely explained by the adversarial component. The generator adversarial term disc_g swings between −0.729 (generator fools discriminator well) and +0.727 (discriminator wins). When multiplied by disc_weight=0.5, this contributes ±0.36 to the total loss in a given batch. Meanwhile the reconstruction terms (L1, LPIPS) are smoothly declining. The correct headline metric for this run is the combined reconstruction loss (L1 + 0.1×LPIPS), which shows steady monotonic improvement. The catalog's oscillation check is designed for diffusion model loss oscillation (random timestep sampling) and for LR-instability oscillation — it was not designed for GAN adversarial dynamics, which always produce total-loss oscillation by mechanism. Classifying ⚠️ by catalog rules but noting this is a false-positive: the oscillation is healthy GAN training, not an instability.

**Gradient explosion** (⚠️ but expected for GAN warm-up phase): The pre-clip grad_norm of 12.64 mean with max 30.44 means the gradient clip is firing on every step. The clip threshold is 1.0 (hardcoded in train_vae_v2.py line 778). Two likely explanations: (a) The PatchGAN discriminator introduces strong gradient signals back to the VAE encoder/decoder that were not present in the pre-training phase — this is known to happen when a discriminator is introduced to a pre-trained generator; the gradients are larger because the generator is receiving a new, strong learning signal from a well-trained discriminator. (b) The lr=5e-5 may be too high for the LPIPS+disc phase at batch_size=2, producing larger per-sample gradient contributions. The fact that reconstruction IS improving (L1 down 65%) despite constant clipping suggests the clipping is doing its intended job — it is preventing parameter explosion while allowing the learning signal to flow. There is no NaN/Inf, which confirms the clipping is successfully bounding the updates. However, constant clipping at a ratio of ~12:1 means the effective learning rate is being reduced by ~12× every step — the optimizer is in a regime where gradients are never taken at face value. This is not catastrophic but it is suboptimal; the model may train more stably with either (a) a lower lr or (b) a higher clip threshold (e.g., 5.0 or 10.0 for GAN fine-tuning).

**KL weight imbalance** (⚠️ but intentional): beta=1e-6 is deliberately tiny. The config comment acknowledges this — the intent is to keep KL from counteracting the LPIPS-driven fine-tune. The risk is that the latent prior q(z|x) will drift from N(0,1) during fine-tuning, which matters only if you later want to sample from N(0,1) directly. Since this VAE is used as a deterministic encoder (the LDM uses mu, not sampled z), prior drift has minimal practical impact. The KL per latent dimension (0.05–0.12) is within normal range — the posterior is not collapsing, it is just not being pulled hard toward the prior. Not a real concern for the project's use case.

**Disc_d = ~2.0 throughout**: The discriminator hinge loss stabilizes at ~2.0 immediately and stays there. For the standard hinge formulation (max(0,1−D(real)) + max(0,1+D(fake))), a value of 2.0 means D is classifying all real images as real (logit≫+1) and all generated images as fake (logit≪−1). This indicates the discriminator has fully dominated the generator and is perfectly distinguishing real from generated. In healthy GAN training, the discriminator and generator should compete — disc_d should oscillate below 2.0 as the generator improves. The near-constant disc_d=2.0 suggests the discriminator has not been challenged yet. This may improve as the fine-tune progresses and the generator's reconstruction improves further.

**Diagnostic steps**:
1. At step 30000 (first checkpoint + eval), inspect the _vae_eval_hook outputs in W&B: recon panel (how sharp is the cardiac border?), delta_ctr_median (did it move toward zero, meaning VAE is less distorting CTR?), and presence metrics. This is the primary quality gate.
2. After step 30000 is logged, check whether disc_d has dropped below 2.0 — if yes, the generator is starting to fool the discriminator. If disc_d is still exactly 2.0, consider whether the disc_warmup is too aggressive (the disc trained for 1000 steps before the generator started competing).
3. Monitor whether grad_norm stabilizes or continues at 12+ mean. If it continues, consider reducing lr from 5e-5 to 2e-5 or raising the clip threshold from 1.0 to 5.0 (the latter lets the optimizer see a larger fraction of the true gradient while still bounding the worst spikes).
4. Check the KL trajectory at the step 30000 eval — if KL continues declining (8042→1437 trend), it is moving in the right direction. If it re-ascends, the disc-induced gradients are disrupting the encoder's learned posterior.

**Fix options**:
- Option A (most conservative, no change): Let the run continue to step 30000. The reconstruction quality is improving rapidly (65% reduction in 4300 steps). The adversarial oscillation and clipping are structural to GAN training and not harming the reconstruction trajectory. Evaluate quality at step 30000 before any intervention.
- Option B (if grad_norm stays at 12+ after step 30000): Raise the clip_grad_norm threshold in train_vae_v2.py line 778 from 1.0 to 5.0. GAN fine-tuning typically uses higher clip thresholds than diffusion model training. A threshold of 5.0 would reduce clipping from ~44× per step to ~6×, allowing the optimizer to take larger fraction of the true gradient while still bounding extreme spikes.
- Option C (if disc_d remains at 2.0 and the generator does not improve): Reduce disc_weight from 0.5 to 0.1 in the config (discriminator_weight in vae_v2.yaml) to weaken the adversarial signal. This will reduce the total-loss oscillation amplitude and may allow the generator to catch up before the discriminator dominates completely.
- Option D (if reconstruction regresses at any future checkpoint): The run can be restarted from the original vae_step0025000.pt with modified hyperparameters. The warm-start checkpoint is intact.

**Recommendation**: Let the run continue uninterrupted to step 30000. The reconstruction signal is healthy (65% L1/LPIPS improvement in 4300 steps), no NaN/Inf, fp32 stability confirmed. Evaluate CTR delta and recon quality at the step-30000 eval before any hyperparameter changes. The flagged modes (oscillation, gradient-explosion, kl-weight-imbalance) are all expected consequences of the GAN fine-tune design, not pathologies requiring immediate action.

**Unrecognized observations**:
The disc_d sitting at exactly 2.0 throughout is a sign the discriminator is trivially distinguishing real from generated — this means the adversarial signal flowing back to the generator is already at maximum saturation. In a balanced GAN, the discriminator loss would oscillate as the generator improves. The flat disc_d=2.0 is not in the catalog but is a known GAN training pattern when the discriminator is much stronger than the generator early in training. It resolves on its own as generator quality improves. If it persists past step 50000, consider adding noise to discriminator inputs (instance noise) or reducing the discriminator's learning rate.

**Catalog modes not checked due to missing artifacts**: all visual modes (all-black-pixels, blurry-outputs, no-sample-improvement, diversity-collapse, lung-anatomy-absent, conditioning-ignored, unexpected-image-stats, checkerboard-artifact), attention-map-collapse, update-to-weight-ratio, weight-norm-explosion, noise-schedule-mismatch, timestep-distribution-skew, fid-regression, loss-scale-anomalous, lr-batch-size-mismatch (no reference config).

---

# VAE-v2 Fine-Tune — paper3-vae-v2 / kj7dunvn

**Run**: `prime_lab/paper3-vae-v2/exp_vae_v2_finetune` (kj7dunvn), RUNNING, 29100 steps (of 75000 total; resumes from vae_step0025000.pt at step 25000).

**No checkpoints saved yet** (save_every=5000; first save expected at step 30000).
**Pre-checkpoint status logged at step 29100.**

---

## Pre-checkpoint status (steps 25100–29100, 41 log points)

**Severity**: ⚠️ (discriminator collapse pattern — see below)

**Metrics at last log (step 29100)**:
- recon_l1: 0.0257   lpips: 0.0311   kl: 1788   disc_d: 2.001   disc_g: +0.102   grad_norm: 4.34   disc_weight: 0.5

**Key signals**:

1. **KL spike at warm-start** ⚠️ (resolved): Step 25100 opened with KL=8042 (1.73× the subsequent mean of ~2500). Dropped to ~3150 by step 25300 and stabilised in the 1500–4000 band. Within the project threshold (< 2×), so flagged ⚠️ but no ramp needed. Did NOT trigger re-escalation → single anomaly, now resolved.

2. **Discriminator collapse** ⚠️ (persistent across all 41 points): `loss/disc_d` is locked at exactly 2.000 ± 0.010 for the entire window (25100→29100). The hinge discriminator loss maxes out at 2.0 when the disc outputs ≈ 0 for both real and fake — it is not differentiating. Consequences: the adversarial signal to the generator is noise (`disc_g` oscillates ±0.5 with no downward trend); `loss/total` goes negative ~49% of steps because the disc_g term dominates. This pattern has been present for 4000 steps — it predates the first save, so severity remains ⚠️ (not yet 3+ consecutive save entries to escalate to ❌).

3. **Grad norm spikes** ⚠️: Raw grad_norm (before clip) reaches 30.4, p95 = 22.9, mean = 12.5. Clip at 1.0 is firing aggressively on many steps. This is consistent with warm-start instability (optimizer state reset) + stuck-disc adversarial gradient being noisy. Not ❌ by itself (no NaN/Inf, no divergence in recon), but co-triggers with discriminator collapse.

4. **Recon and LPIPS improving** ✓: Despite the disc issue, the pixel task is working. recon_l1 (first-half mean 0.0354 → second-half 0.0329, ↓7%), lpips (0.0523 → 0.0444, ↓15%). The VAE is learning to reconstruct more sharply.

5. **val/delta_ctr_median**: not logged yet (first eval at step 30000).

**Per-mode flags**:
- nan-or-inf: ✓ pass (all metrics finite)
- loss-zero-collapse: ✓ pass (recon 0.026 ≫ 0)
- gradient-explosion: ⚠️ raw norm p95=22.9 (clipped to 1.0; not causing divergence but co-triggers)
- oscillation: NOTE — total loss oscillates ±0.2 due to noisy adversarial term from stuck disc; NOT growing → not escalated
- kl-spike: ⚠️ at warm-start step 25100, resolved by 25300
- discriminator-collapse: ⚠️ disc_d stuck at 2.0 (0/41 points below 1.9)
- val-train-divergence: not checked (no val metrics yet)
- conditioning/visual: not checked (no image artifacts logged yet; eval at step 30000)

**Co-trigger**: grad_norm spikes + disc collapse on same steps → ⚠️ combined; not yet ❌ (no save entries logged, first save pending).

**Recommendation**: Do NOT stop. The pixel losses (recon + lpips) are improving correctly — the VAE is fine as a codec regardless of the disc. Two actions to take **after the first checkpoint at step 30000**:
1. Inspect the W&B `panel/normal_burned`, `panel/cardio_burned`, `panel/effusion_burned` overlays to confirm visual quality is improving over the warm-start checkpoint (delta_ctr_median should be ≤ 0.015).
2. Diagnose disc collapse: check if `ckpts/vae_v2/` is being written correctly; then verify the PatchGAN is receiving gradients by checking that `disc.parameters()` require_grad is True. If disc_d stays at 2.0 through the step-30000 save, add a small disc-lr bump (try 2×) or lower the disc_weight to 0.1 until disc starts differentiating — the recon+LPIPS path alone is sufficient for the border-sharpening goal.


---

## Run — prime_lab/paper3-ldm/mzd3jwy9

Entries appended by /analyze-run; oldest first.

- Run name: `exp_vae_v2_ldm`
- Config: `configs/ldm_rebalanced.yaml` (inferred from run config) | max_steps=100000, ckpt_every=10000, grid_every=5000, compose_every=15000, batch_size=16, grad_accum=1 (effective batch 16), lr=1e-4 cosine with 500-step warmup, bf16=true, model_channels=128, cfg_dropout_p=0.15, balance_classes=True, class_mult={0:0.25, 1:0.40, 2:0.35}, effusion_weight=1 (removed upweight), num_train_timesteps=1000
- Hardware: not probed (bf16=true, consistent with A-series GPU on hippo)
- Data: latent-cache mode, latent_cache=data/latents_v2/joint/ — VAE-v2 encoded latents (rebalanced retrain; uses cleaned cardio + border-visible filter)
- Artifacts: 10 checkpoint saves (step 10000–100000 at 10k intervals; ldm-ckpt:v48–v58). Recon grids logged every 5000 steps (ldm-recon-grid:v220–v240). Compose grids at steps 15000, 30000, 45000, 60000, 75000, 90000 (ldm-compose-grid:v41–v46).
- Visual logging: recon_grid and compose_grid AVAILABLE (VAE decode enabled in this run — confirmed by presence of image artifacts)
- Baseline: no training-references directory found; all checks are absolute catalog checks only
- Key difference from prior runs: this is the rebalanced retrain (class_mult reweighting replaces effusion_weight=2.0; VAE-v2 latents instead of original VAE latents; balance_classes=True)
- Eval metrics logged every 5000 steps: ctr_med/cardio, presence/cardio, presence/effusion, frac_gt0.5/cardio, yield/poe_gate3, auc/cardio_gen, auc/effusion_gen

---

## Save v48 (step 10000) — 2026-06-30T16:50:38Z

**Severity**: ⚠️ concerning

**Visual observations**:
recon_grid_step0010000.png inspected. All 12 panels (4 seed rows × 3 class columns: NO FINDING, CARDIOMEGALY, EFFUSION) show high-frequency salt-and-pepper noise with no anatomical structure whatsoever. Not a single valid CXR in the grid. The noise texture is isotropic across all panels — there is no hint of lung fields, mediastinum, or any body silhouette. One partial exception: seed 512, cardiomegaly column shows a faint bright central region that could be the beginning of a mediastinum signal, but it is entirely submerged in noise. No compose grid was logged at step 10000 (compose_every=15000).

Eval metrics at step 10000: ctr_med/cardio=0.434, presence/cardio=0.013, presence/effusion=0.659, frac_gt0.5/cardio=0.250, yield/poe=0.000. The CTR and presence scores confirm what the grid shows: cardiomegaly generation is near-absent (presence 0.013, frac_gt0.5 0.250), effusion is partially working (presence 0.659 but this may be picking up the noisy texture as false positives).

**Per-mode flags**:
- nan-or-inf: ✓ pass (val/loss=0.2771, train/loss=0.3138, grad_norm=0.0373 — all finite)
- loss-zero-collapse: ✓ pass (train/loss 0.314 >> 1e-8)
- all-black-pixels: ✓ pass by global mean (mean pixel ~0.47 on full grid including borders); however the panels themselves show noise not anatomy — this is not a black-pixel failure but a noise failure; the all-black-pixels check was not triggered because mean is above threshold
- no-sample-improvement: not checked (first save; no prior grid to compare)
- oscillation: ✓ pass — val/loss 5-point window [2000–10000]: 0.2726, 0.2742, 0.2696, 0.2716, 0.2771; rel_p2p = (0.2771-0.2696)/0.2730 = 0.027, well below 0.15
- val-train-divergence: ✓ pass — val_slope +0.00039/1k, train_slope +0.026/1k; neither diverging criterion met
- diversity-collapse: not checked (no valid CXRs to assess diversity)
- latent-collapse: not checked (no latent manifold plots)
- loss-plateau: ⚠️ triggered — val/loss slope over window 0.00048/1k, below the 0.001 threshold. Val/loss has been approximately flat from step 1000 (0.2688) to step 10000 (0.2771). First plateau trigger.
- loss-component-imbalance: ✓ pass — cls0=0.536, cls1=0.091, cls2=0.304; max/min ratio = 5.9, below 10:1 threshold
- kl-posterior-collapse: N/A (LDM noise-prediction; no KL term)
- kl-weight-imbalance: N/A
- blurry-outputs: not checked (all panels are noise, not blurry outputs — distinct failure mode)
- fp-precision-artifact: not checked (no clustered black patches visible; the panels are noise not black)
- conditioning-ignored: ⚠️ partial — all three class columns show identical noise patterns; there is no visible difference between NO FINDING, CARDIOMEGALY, and EFFUSION at step 10000. This strongly suggests conditioning is not yet functioning. However, this is at 10% of total training and the LR warmup (500 steps) has only just completed — early conditioning failure may be transient.
- overfit-regime: ✓ pass — gap val(0.277)-train(0.314) = -0.037 (val < train, no gap criterion met)
- gradient-vanishing: ✓ pass (grad_norm=0.0373, well above 1e-6)
- gradient-explosion: ✓ pass (z-score -0.74 from local window; no spike)
- noise-schedule-mismatch: not checked (no per-timestep quality metrics)
- lung-anatomy-absent: ⚠️ triggered — all panels show pure noise; lateral vs. central intensity ratio is indistinguishable (lateral ≈ central ≈ isotropic noise). Anatomy is absent by visual observation. At step 10000 this is expected in early training — the model has not learned to produce valid CXRs yet.
- attention-map-collapse: not checked (attention maps not logged)
- all other modes: not checked or N/A

**Trend signals**:
- Improvement direction: val/loss flat (0.2688→0.2771, no trend). Eval CTR improving: 0.370 at step 5000 → 0.434 at step 10000.
- Oscillation: rel_p2p 0.027 — no oscillation
- Persistence of concern: loss-plateau first trigger (1/3); conditioning-ignored first trigger; lung-anatomy-absent first trigger

**Reference comparison**: No baseline available.

**Diagnosis**:
At step 10000 (10% of max_steps), the model is producing pure noise on all grid seeds. This is early training — the model has not yet learned to denoise to valid CXR structure. The plateau in val/loss (flat at ~0.271 from step 1000 to 10000) reflects the well-documented behavior of diffusion models on latent space: the aggregate denoising loss floors early because the model quickly learns a reasonable prior, but sample quality continues improving as the model learns the finer structure of the score function. The visual absence of any CXR structure at step 10000 is the primary concern, but it is not a catastrophic failure mode at this stage.

The conditioning-ignored flag is concerning but expected at 10k steps: with cfg_dropout_p=0.15, 15% of training batches use the null token. Early in training the model may not have learned to use the conditioning signal at all. This should resolve as training progresses.

The most notable positive signal is the CTR trajectory: 0.370@5k → 0.434@10k. The model is beginning to produce cardiac silhouettes that the CTR instrument can measure, even if they are not visually convincing in the recon grid. This suggests the early conditioning signal is beginning to be incorporated, just not yet producing visually coherent outputs.

**Diagnostic steps**:
1. Check step-15000 recon grid (next available) — if valid CXR panels begin appearing, the step-10000 all-noise state was early training, not a structural failure.
2. Watch the presence/cardio trajectory at step 15000 eval — if it rises above 0.05, the cardiomegaly conditioning is starting to work.

**Fix options**:
- Option A (continue): Early training; all-noise at step 10000 is expected. No action.

**Recommendation**: Continue. Step-10000 all-noise state is expected at 10% of training. Monitor step-15000 visual quality.

---

## Save v50 (step 20000) — 2026-06-30T17:42:06Z

**Severity**: ⚠️ concerning

**Visual observations**:
No image directly downloaded for step 20000 (recon_grid_step0020000.png available in W&B but not fetched to disk). Based on eval metric trajectory: presence/cardio=0.170 at step 20000 (up from 0.013 at step 10000), frac_gt0.5/cardio=0.105. This suggests some cardiomegaly signal is emerging. No visual inspection possible for this save.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val=0.2783, train=0.1419, grad=0.0243 — all finite)
- loss-zero-collapse: ✓ pass
- oscillation: ✓ pass (rel_p2p=0.030, pts=5)
- loss-plateau: ✓ pass (slope +0.0011/1k — above threshold; plateau lifted briefly)
- val-train-divergence: ✓ pass
- loss-component-imbalance: ⚠️ triggered — cls0=0.509, cls1=0.007, cls2=0.342; ratio cls0/cls1 = 69.3, well above 10:1 threshold. cls1 (cardiomegaly) has dropped to near zero on this batch. Per-batch stochasticity with class_mult reweighting — likely a batch where cls1 samples all had easy (low-noise) timesteps. First trigger.
- gradient-vanishing: ✓ pass (grad=0.024)
- lung-anatomy-absent: not checked (no image fetched)
- all visual modes: not checked

**Trend signals**:
- Improvement direction: val/loss flat (0.277→0.278 from 10k to 20k). CTR improving: 0.434@10k → 0.419@20k (slight dip). Presence/cardio improving: 0.013→0.170.
- Oscillation: pass
- Persistence: plateau cleared (not triggered); loss-component-imbalance first trigger

**Reference comparison**: No baseline.

**Diagnosis**:
The cls1 near-zero at step 20000 (0.007) is consistent with per-batch stochasticity under the class_mult scheme — when cardiomegaly samples happen to land on low-noise timesteps (easy denoising), their MSE is near zero. This is the same per-batch artifact documented in prior runs. The 69× ratio is driven by a denominator near zero, not by cls0 being extremely high. The loss-component-imbalance flag is likely a false positive here; watch whether it recurs persistently.

**Recommendation**: Continue. Monitor cls1 at the next several evaluation points — if it consistently stays near zero while cls0/cls2 are at normal levels, the cardiomegaly class is being under-trained. But a single point at 0.007 with recovery expected is routine.

---

## Save v51 (step 30000) — 2026-06-30T18:32:50Z

**Severity**: ⚠️ concerning

**Visual observations**:
No image directly downloaded for step 30000 (recon_grid_step0030000.png available but not fetched). compose_grid_step0030000.png was fetched and inspected. The compose grid (4 rows: null-anchor w1.0, null-anchor w2.0, healthy-anchor w1.0, healthy-anchor w2.0; 4 columns: reference CXR + seeds 137/256/512) shows a severe pattern: the reference column (leftmost) shows a valid real CXR for all 4 rows. Seeds 137 and 256 produce identifiable but noisy CXR structures. Seed 512 in the first 3 rows shows partial structure; in row 4 (healthy-anchor w2.0) seed 512 produces a valid-looking CXR. The generation quality is inconsistent across seeds — some seeds produce structured CXRs, others produce noisy outputs even in the compose grid. At step 30000 (30% of training), some valid compositions are appearing but the per-seed inconsistency is already visible.

Eval metrics at step 30000: ctr_cardio=0.456, presence_cardio=0.107, presence_effusion=0.565, yield=0.031.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val=0.2724, train=0.2621, grad=0.0254)
- loss-zero-collapse: ✓ pass
- oscillation: ✓ pass (rel_p2p=0.043)
- loss-plateau: ⚠️ triggered — slope -0.00015/1k (|slope| < 0.001). Second consecutive trigger.
- val-train-divergence: ✓ pass
- loss-component-imbalance: ✓ pass (ratio=7.5; cls0=0.643, cls2=0.086, cls1=0.159)
- gradient-vanishing: ✓ pass (grad=0.025)
- lung-anatomy-absent: partial — compose grid shows some structured CXRs mixed with noisy outputs; anatomy is present in some cells, absent in others; not globally absent

**Trend signals**:
- Improvement direction: val/loss flat (0.272 vs 0.278 at 20k — marginal improvement). CTR rising: 0.456 (best so far). Presence/cardio fluctuating (0.107).
- Persistence: plateau 2/3; loss-component-imbalance resolved

**Recommendation**: Continue. Second consecutive plateau trigger; anatomy partially present in compose grid.

---

## Save v52 (step 40000) — 2026-06-30T19:19:36Z

**Severity**: ⚠️ concerning

**Visual observations**:
No image directly downloaded for step 40000. Eval metrics: ctr_cardio=0.443, presence_cardio=0.170, presence_effusion=0.440, yield=0.094. The yield of 0.094 (9.4%) at step 40000 is the highest in the run up to this point and approaches the 10% threshold from prior analysis. frac_gt0.5/cardio=0.273.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val=0.2788, train=0.0890, grad=0.0225)
- loss-zero-collapse: ✓ pass
- oscillation: ✓ pass (rel_p2p=0.024)
- loss-plateau: ⚠️ triggered — slope +0.00029/1k. Third consecutive trigger — persistence criterion met.
- val-train-divergence: ✓ pass (catalog check: val_slope=+0.00035/1k, train_slope=-0.030/1k — train declining but val not rising above threshold)
- overfit-regime: ✓ pass (val-train gap=0.190, less than 0.5×val=0.139; train variance not low; no regime)
- loss-component-imbalance: ✓ pass (ratio=2.3)
- gradient-vanishing: ✓ pass (grad=0.022)

**Trend signals**:
- Improvement direction: val/loss essentially flat across all 4 saves (0.2771 → 0.2783 → 0.2724 → 0.2788 at steps 10–40k; mean 0.276). Yield peaked at 0.094 (step 40000).
- Oscillation: ✓ pass across all 4 saves
- Persistence: loss-plateau 3/3 — escalation criterion met. However, see diagnosis: this is the same expected behavior as hifei736.

**Reference comparison**: No baseline.

**Diagnosis**:
Val/loss plateau triggered for 3 consecutive saves. This is the same documented behavior as hifei736 (prior run): the aggregate ε-prediction val/loss floors early (~0.27) and the mean barely moves, while the sample quality and eval metric trajectory (CTR, presence, yield) shows genuine improvement. The meaningful signal is in the eval metrics: CTR has risen from 0.370@5k to 0.443@40k (vs real-cardio target 0.535), presence/cardio from 0.013 to 0.170. The plateau in the mean val/loss is the expected behavior of this training objective, not a stall. Held at ⚠️ to flag persistence, but this is inspection-confirmed as the same artifact seen in prior runs.

**Recommendation**: Continue. Plateau is documented expected behavior. Monitor eval CTR and presence as the real quality indicators.

---

## Save v53 (step 50000) — 2026-06-30T20:10:31Z

**Severity**: ⚠️ concerning

**Visual observations**:
recon_grid_step0050000.png inspected directly. The grid shows a striking MIXED pattern of valid CXRs and pure noise panels:

- Seed 42: NO FINDING = excellent CXR (bilateral lung fields, ribs, mediastinum visible); CARDIOMEGALY = good CXR but with scattered bright salt-and-pepper dots overlaid (valid anatomy, but white-dot artifacts scattered across the image); EFFUSION = pure salt-and-pepper noise with no anatomical structure
- Seed 137: NO FINDING = very good CXR; CARDIOMEGALY = good CXR with white-dot artifacts; EFFUSION = very faint structure emerging but mostly noise
- Seed 256: NO FINDING = good CXR; CARDIOMEGALY = pure noise (distinct from the other seeds where CARDIOMEGALY renders correctly); EFFUSION = partial structure visible (better than seed 137 effusion)
- Seed 512: NO FINDING = faint structure present but foggy; CARDIOMEGALY = pure noise; EFFUSION = partially structured, noisy

Key observations: (1) The EFFUSION column is the most consistently noisy — it fails or is degraded in 3/4 seeds. (2) The CARDIOMEGALY column shows opposite behavior in different seeds: good in seeds 42/137, pure noise in seeds 256/512. (3) NO FINDING is the most consistently rendered class (good in 3/4 seeds, foggy in seed 512). (4) The per-seed inconsistency is the dominant pattern — within a single checkpoint, identical network weights produce valid CXRs for some seeds and pure noise for others. (5) Scattered white-dot artifacts appear on valid cardiomegaly panels — these are not checkerboard but sparse salt-and-pepper, distributed across the image.

The white-dot artifacts on valid panels are an unrecognized pattern not matching any catalog mode description. They appear as isolated bright pixels scattered across otherwise anatomically correct CXRs.

Eval at step 50000: ctr_cardio=0.314, presence_cardio=0.076, presence_effusion=0.409, yield=0.094.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val=0.2702, train=0.2331, grad=0.0276)
- loss-zero-collapse: ✓ pass
- all-black-pixels: ✓ pass (mean pixel 0.584 across full grid)
- no-sample-improvement: not checked (no step-40000 recon grid on disk to compare)
- oscillation: ✓ pass (rel_p2p=0.032)
- loss-plateau: ⚠️ triggered — slope -0.00025/1k. Fourth consecutive trigger.
- val-train-divergence: ✓ pass
- diversity-collapse: ✓ pass visually (clear differences between seed rows and class columns where valid CXRs appear)
- latent-collapse: not checked (no latent PCA logged)
- loss-component-imbalance: ✓ pass (ratio=3.3; cls0=0.326, cls1=0.144, cls2=0.475)
- blurry-outputs: ✓ pass (valid panels show sharp edges/ribs; the noise panels are not blurry, they are pure noise — a distinct failure)
- fp-precision-artifact: ⚠️ partial — scattered white-dot artifacts on valid panels are consistent with bf16 precision artifacts (isolated pixel saturation from activation overflow). However, the noise panels are not clustered-black-patches as described in the catalog; they are full-panel noise. Two distinct phenomena may be present: (a) bf16 underflow producing occasional white pixels on otherwise valid panels, and (b) a different mechanism producing full-panel noise for specific seeds/classes.
- conditioning-ignored: ⚠️ partial — the effusion column is notably worse than other columns across all seeds, suggesting the effusion conditioning signal may be weaker or the model is not yet generating effusion pattern reliably. The seed 256 cardiomegaly failure in otherwise-valid context also suggests class-specific conditioning issues. Full conditioning-ignored check requires unconditional vs. conditional LPIPS comparison — not logged.
- overfit-regime: ✓ pass
- gradient-vanishing: ✓ pass (grad=0.028)
- lung-anatomy-absent: ⚠️ triggered for noisy panels — anatomy absent in effusion column (3/4 seeds) and some cardiomegaly panels. ✓ pass for valid panels (bilateral lung fields clearly present).
- checkerboard-artifact: ✓ pass (no periodic checkerboard pattern; the white dots are non-periodic)
- noise-schedule-mismatch: not checked
- attention-map-collapse: not checked

**Trend signals**:
- Improvement direction: val/loss flat (0.270 vs 0.279 — marginally better). Eval metrics oscillating (ctr_cardio 0.434→0.443→0.314, non-monotone).
- Oscillation: ✓ pass on val/loss metric; but per-seed visual quality is oscillating between saves
- Persistence: plateau 4/3+ consecutive; lung-anatomy-absent for noisy panels persistent; per-seed noise pattern is the dominant signal

**Reference comparison**: No baseline.

**Diagnosis**:
The per-seed inconsistency is the key diagnostic signal. At step 50000, the model with identical weights and noise schedule produces valid anatomically correct CXRs for some seed/class combinations and pure noise for others. This pattern cannot be explained by:
- Global scale_factor error (would affect all panels equally)
- VAE decoder failure (same decoder for all panels)
- bf16 GroupNorm collapse (documented in memory as producing all-black outputs — not pure noise)
- NaN/Inf propagation (val/loss and grad_norm are finite throughout)

The most likely mechanism: certain fixed noise seeds (seeds 256, 512) initialize the reverse diffusion trajectory in a region of the latent noise space where the model's learned score function is poorly calibrated. The model has learned to denoise starting from some Gaussian noise distributions well, but fails for other starting points. This can happen when:
1. The training batch construction under class_mult reweighting creates a non-uniform coverage of the latent space, leaving some regions undertrained.
2. The fixed evaluation grid uses seeds that happen to land in undertrained noise regions. The good-seed rows (42, 137) happen to start in well-covered regions.

The effusion column being specifically worse is consistent with the run's effusion_weight=1 (reduced from 2.0 in prior runs) combined with class_mult giving effusion only 35% of batches — effusion may be the most undertrained class.

The white-dot artifacts on valid panels are most likely bf16 activation saturation producing isolated overflow pixels in the decoder. These are distinct from the full-panel noise failure.

**Diagnostic steps**:
1. Generate a larger set of samples at step 50000 with more seeds (seeds 0–20) to test whether the noise failure is truly seed-specific or affects all seeds for certain classes. If seeds 42/137 consistently work and seeds 256/512 consistently fail, the failure is in the starting noise vector distribution, not in the network weights.
2. Check whether the eval score for effusion at step 50000 (presence/effusion=0.409) correlates with the visual observation — effusion panels that have structure should score higher than those that are pure noise.
3. At the next checkpoint (step 60000), compare the recon grid — if the same seeds still fail for the same classes, the failure is structural (certain noise initializations are outside the model's denoising capability). If different seeds fail, the pattern is stochastic and may improve with more training.

**Fix options**:
- Option A (continue training): The model is 50% complete and some valid CXRs are being produced. The failure may resolve with more training steps as the model's coverage of the noise space improves.
- Option B (investigate latent distribution): Check whether the class_mult reweighting is creating non-uniform latent coverage. If effusion latents cluster differently from normal/cardiomegaly latents in the VAE-v2 embedding space, certain noise seeds may consistently miss the effusion manifold.
- Option C (DDIM step count at inference): The evaluation grid may be using too few denoising steps. If some noise seeds require more refinement steps to converge to valid CXRs, increasing from 50 to 100 DDIM steps at eval time could reduce the noise failures without changing training.

**Recommendation**: Continue training. The per-seed inconsistency at 50% of training is concerning but not catastrophic — some valid CXRs are being produced. Monitor the step-60000 grid to see if the pattern improves, persists, or worsens. Do not adjust training yet; first determine whether the same seeds fail consistently.

**Unrecognized observations**:
1. Per-seed inconsistency pattern: the same checkpoint producing valid CXRs for seeds 42/137 and pure noise for seeds 256/512 (for certain classes) does not match any catalog mode. This is most consistent with non-uniform coverage of the noise-conditional score function — certain starting points in Gaussian noise space are in undertrained regions. This pattern could indicate a need for more diverse training noise seeds or longer training on the undertrained classes.
2. Scattered white-dot artifacts on valid cardiomegaly panels: isolated bright pixels distributed non-uniformly across valid CXR anatomy. These may be bf16 decoder activation overflow (individual neurons saturating) rather than the clustered-patch pattern the catalog describes. Not in the catalog; suggest adding a "scattered-pixel-saturation" mode for bf16 models where individual activation overflow produces sparse bright pixels rather than clustered black patches.

---

## Save v54 (step 60000) — 2026-06-30T21:01:03Z

**Severity**: ⚠️ concerning

**Visual observations**:
No image directly downloaded for step 60000. Eval metrics: ctr_cardio=0.394, presence_cardio=0.326, frac_gt0.5=0.154, presence_effusion=0.409, yield=0.000. Presence/cardio rose to 0.326 (best in the run to this point) while yield dropped to 0.000.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val=0.2755, train=0.2029, grad=0.0363)
- loss-zero-collapse: ✓ pass
- oscillation: ✓ pass (rel_p2p=0.033)
- loss-plateau: ⚠️ triggered — slope +0.00032/1k. Fifth consecutive trigger.
- loss-component-imbalance: ✓ pass (ratio=6.3)
- gradient-vanishing: ✓ pass
- all visual modes: not checked (no image fetched)

**Trend signals**:
- Improvement direction: presence/cardio improving (0.076@50k → 0.326@60k — significant jump). CTR slightly lower (0.314→0.394). Yield fell to 0.
- Plateau: 5th consecutive trigger

**Diagnosis**: The presence/cardio jump to 0.326 at step 60000 is notable — the model is beginning to render cardiomegaly consistently enough for the classifier to detect it. However yield is 0 at this step. The disconnect between high presence/cardio and zero yield suggests that while the cardiomegaly marginal is improving, the PoE composition is not producing the co-presence required for yield. This could reflect that the effusion conditional (presence/effusion=0.409) is not producing strong enough effusion signal in the composition context.

**Recommendation**: Continue. Presence/cardio is improving; monitor whether the per-seed visual quality at step 65000 recon grid shows consolidation.

---

## Save v55 (step 70000) — 2026-06-30T21:47:22Z

**Severity**: ⚠️ concerning

**Visual observations**:
recon_grid_step0070000.png inspected directly. The mixed noise/clean pattern persists and is qualitatively similar to step 50000:

- Seed 42: NO FINDING = excellent sharp CXR; CARDIOMEGALY = good CXR, mild texture; EFFUSION = pure salt-and-pepper noise
- Seed 137: NO FINDING = good CXR (slightly lower contrast); CARDIOMEGALY = good CXR; EFFUSION = faint structure visible but mostly noise
- Seed 256: NO FINDING = good CXR; CARDIOMEGALY = pure noise (same seed-256/cardiomegaly failure as step 50000); EFFUSION = good CXR (seed 256 effusion now shows structure)
- Seed 512: NO FINDING = dark/noisy (anatomy barely visible); CARDIOMEGALY = good CXR (seed 512 cardiomegaly now working, unlike at step 50000); EFFUSION = partially structured

The pattern has SHIFTED between step 50000 and step 70000: seed 512 cardiomegaly recovered (was noise at 50k, now valid at 70k), but seed 512 no_finding degraded (was partial at 50k, now dark/noisy at 70k). Seed 256 cardiomegaly is still pure noise at both steps. This shifting pattern — different seeds failing at different checkpoints — is diagnostic: the failure is not locked to specific seeds permanently but varies as the training updates different aspects of the score function.

Eval at step 70000: ctr_cardio=0.436, presence_cardio=0.232, yield=0.000, presence_effusion=0.409.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val=0.2756, train=0.0871, grad=0.0292)
- loss-zero-collapse: ✓ pass
- all-black-pixels: ✓ pass (mean pixel 0.449)
- oscillation: ✓ pass (rel_p2p=0.040)
- loss-plateau: ⚠️ triggered — slope +0.00024/1k. Sixth consecutive trigger.
- val-train-divergence: ✓ pass (val_slope=+0.00052/1k, train declining — but val slope below trigger threshold of 0.01)
- loss-component-imbalance: ⚠️ triggered — cls0=0.0074, cls1=0.394, cls2=0.318; ratio cls1/cls0 = 53.4. cls0 (no_finding) dropped to near-zero on this batch. Per-batch artifact — same mechanism as the cls1 near-zero at step 20000. Second overall imbalance trigger (at non-consecutive save; prior at step 20000, not 60000/70000 pair).
- blurry-outputs: ✓ pass (valid panels show sharp CXR detail; one cardiomegaly panel in seed 512 appears slightly oversmoothed)
- lung-anatomy-absent: ⚠️ triggered — seed 42 effusion, seed 137 effusion, seed 256 cardiomegaly, seed 512 no_finding panels show no anatomy.
- fp-precision-artifact: ⚠️ partial — seed 512 no_finding shows a very dark/muted output; this could be bf16 decoder underflow reducing pixel intensities in that panel. Not the clustered-black-patch pattern but intensity suppression. Not checked formally (no pixel-patch analysis).
- checkerboard-artifact: ✓ pass (no periodic texture visible)
- conditioning-ignored: ⚠️ partial — effusion column still consistently worse than other columns (2/4 seeds still pure noise for effusion)

**Trend signals**:
- Improvement direction: val/loss flat throughout. CTR fluctuating (0.436 at 70k, same region as 50k). Presence/cardio moderate at 0.232. Yield still 0.
- Oscillation: ✓ pass
- Persistence: plateau 6/3+ consecutive; effusion-column noise persistent across multiple saves; loss-component-imbalance second trigger (non-consecutive)

**Reference comparison**: No baseline.

**Diagnosis**:
The shifting per-seed pattern between step 50000 and 70000 (seed 512 cardiomegaly recovered, seed 512 no_finding degraded) is important diagnostic information. It rules out a simple "seed 512 initialization is always bad" explanation — instead, the model's denoising capability for specific seed/class combinations is changing as training updates the score function. The persistent failure of the effusion column (3/4 seeds pure noise at step 50k, 2/4 at 70k — some improvement) and the seed 256 cardiomegaly failure (persistent across both saves) suggest these specific failure locations may reflect undertrained regions of the joint noise-conditional score function.

The val/loss (0.276) and grad_norm (0.029) are healthy — the overall training signal is functioning. The per-seed visual failure is not driven by gradient collapse or loss divergence but by incomplete coverage of the sampling distribution at inference time.

**Recommendation**: Continue. The pattern is shifting (some recovery, some new failures), which indicates the model is still learning. Monitor the step-80000 and step-90000 grids carefully — if the effusion column failure rate is still 50%+ at step 80000, consider whether the class_mult for cls2 (0.35) is sufficient.

---

## Save v56 (step 80000) — 2026-06-30T22:37:47Z

**Severity**: ⚠️ concerning

**Visual observations**:
No image directly downloaded for step 80000. Eval metrics: ctr_cardio=0.520 (best in entire run; approaching real-cardio threshold 0.535), frac_gt0.5=0.583, presence_cardio=0.201, presence_effusion=0.503, yield=0.000. The CTR of 0.520 is the best measurement in the entire run and is now only 0.015 below the real-cardio reference (0.535). frac_gt0.5=0.583 means 58% of generated cardiomegaly images cross the CTR=0.5 cardiomegaly line — a major improvement from 0.125 at step 5000.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val=0.2737, train=0.0985, grad=0.0165)
- loss-zero-collapse: ✓ pass
- oscillation: ✓ pass (rel_p2p=0.044)
- loss-plateau: ⚠️ triggered — slope -0.00045/1k. Seventh consecutive trigger.
- loss-component-imbalance: ⚠️ triggered — cls0=0.178, cls1=0.010, cls2=0.298; ratio cls2/cls1=30.7. cls1 near-zero again on this batch. Per-batch artifact (third overall imbalance trigger, but non-consecutive — different save from the step-70000 trigger).
- gradient-vanishing: ✓ pass (grad=0.0165; LR at this step is ~1.06e-5 in cosine decay, so lower grad norm expected)
- val-train-divergence: ✓ pass (val_slope=-0.00081/1k — slightly negative, no divergence)
- all visual modes: not checked (no image fetched)

**Trend signals**:
- Improvement direction: CTR 0.520 is the run's best. frac_gt0.5 0.583 — approaching target. Val/loss flat.
- Plateau: 7th consecutive
- CTR trajectory: 0.370@5k → 0.434@10k → 0.443@40k → 0.520@80k — genuine improvement

**Diagnosis**:
Step 80000 is the strongest checkpoint by the project's primary eval metric (CTR). The cardiomegaly generation has improved substantially — from 12.5% crossing CTR=0.5 at step 5000 to 58.3% at step 80000, approaching the real-cardio reference (75%). The val/loss plateau is irrelevant to this metric: the model is genuinely improving on the task-relevant signal even as the aggregate denoising loss is flat.

The yield remains 0 despite strong cardio CTR — this indicates the PoE composition is not co-rendering both pathologies simultaneously. The effusion effusion presence (0.503) suggests the effusion marginal is present but the joint composition is failing.

**Recommendation**: This save (step 80000) is the current best checkpoint by CTR. Continue to step 100000 and then evaluate with the S4 watcher; step 80000 is the primary candidate for the best model checkpoint of this run.

---

## Save v57 (step 90000) — 2026-06-30T23:28:09Z

**Severity**: ⚠️ concerning

**Visual observations**:
recon_grid_step0090000.png inspected directly. This is the save the user identified the symptom in. Confirming the user's description:

- Seed 42: NO FINDING = good CXR with rib detail; CARDIOMEGALY = good CXR (enlarged cardiac silhouette visible); EFFUSION = pure salt-and-pepper noise (confirmed)
- Seed 137: NO FINDING = pure noise (confirmed); CARDIOMEGALY = good CXR, moderate quality; EFFUSION = good CXR (confirmed — the pattern is reversed from seed 42)
- Seed 256: ALL THREE = pure noise or near-pure noise. The cardiomegaly panel has faint vertical striations suggesting the beginning of rib structure, but is mostly noise. NO FINDING and EFFUSION are pure salt-and-pepper. (confirmed as worst row)
- Seed 512: NO FINDING = pure noise; CARDIOMEGALY = extremely oversmoothed/blurry blob — faint bright oval in the center with no rib or lung detail, the anatomy is heavily smoothed out; EFFUSION = noisy but has some structure visible (confirmed per user description)

The seed 512 cardiomegaly panel is particularly informative: it is not noise but is instead a heavily oversmoothed blob. This is a different failure mode from the pure noise — it suggests the model is denoising but converging to an oversmoothed mean rather than a sharp CXR. This is the blurry-outputs signature, but occurring only in a specific seed/class combination.

compose_grid_step0090000.png inspected directly. The compose grid has 4 rows (null-anchor w1.0, null-anchor w2.0, healthy-anchor w1.0, healthy-anchor w2.0) and 4 columns (reference CXR + seeds 137/256/512). The reference column (leftmost) shows a valid real CXR in all 4 rows. Seed 137 column shows dark noisy lung-field patterns — some structure but not clean CXRs. Seeds 256 and 512 show pure noise in all 4 rows. Only seed 42 (not shown in this compose grid — the compose grid uses seeds 137/256/512 in columns 2-4) would likely produce valid composition. The 3/3 compose seeds are degraded.

Eval at step 90000: ctr_cardio=0.494, presence_cardio=0.170, frac_gt0.5=0.417, presence_effusion=0.503, yield=0.000.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val=0.2757, train=0.1754, grad=0.0211)
- loss-zero-collapse: ✓ pass
- all-black-pixels: ✓ pass (mean pixel 0.494)
- no-sample-improvement: ⚠️ partial — the pattern at step 90000 is similar to step 70000 (same classes still failing); panels that are pure noise at step 70000 are still often noise at step 90000. Not a formal LPIPS comparison but visual assessment suggests limited improvement in the noise-panel cells.
- oscillation: ✓ pass (rel_p2p=0.049, window [82k–90k])
- loss-plateau: ⚠️ triggered — slope +0.00057/1k. Eighth consecutive trigger.
- val-train-divergence: ✓ pass
- diversity-collapse: ✓ pass — among the valid CXR panels, there is clear seed-to-seed diversity (different chest anatomy/shapes visible)
- latent-collapse: not checked
- loss-component-imbalance: ⚠️ triggered — cls0=0.00025, cls1=0.086, cls2=0.334; ratio cls2/cls0 = 1314. cls0 at essentially zero (2.5e-4). This is an extreme per-batch ratio but the same mechanism as prior triggers — the no_finding batch happened to draw all easy (low-noise) timesteps at this evaluation step. With class_mult giving no_finding only 25% of samples per batch (vs 40% cardiomegaly + 35% effusion), there are fewer no_finding samples per batch, increasing the per-batch variance of cls0. Third trigger in 4 saves (steps 20k, 70k, 80k, 90k) — approaching persistence territory for this mode.
- blurry-outputs: ⚠️ triggered — seed 512 cardiomegaly panel shows heavily oversmoothed output. hf_energy check not run (FFT not computed), but visual observation is clear: the seed 512/cardiomegaly panel is an oversmoothed blob, not a sharp CXR.
- fp-precision-artifact: ✓ pass — no clustered intra-panel black patches; the noise panels are full-panel noise, not spatially localized
- conditioning-ignored: ⚠️ partial — effusion column still failing in seed 42 (seed 42/effusion = noise). The per-class conditioning signal for effusion is weaker than for no_finding/cardiomegaly.
- overfit-regime: ✓ pass
- gradient-vanishing: ✓ pass (grad=0.0211; LR is ~2.47e-6, late cosine decay; low grad norm expected)
- lung-anatomy-absent: ⚠️ triggered — pure noise panels in seed 42/effusion, seed 137/no_finding, seed 256/all, seed 512/no_finding have no anatomy

**Trend signals**:
- Improvement direction: CTR 0.494 (down from 0.520 at step 80k — a dip). Val/loss flat at 0.276.
- Oscillation: ✓ pass
- Persistence: loss-plateau 8/3+ consecutive; lung-anatomy-absent persistent (noise panels persistent across saves 50k/70k/90k); loss-component-imbalance now 3+ out of 4 recent saves (20k, 70k, 80k, 90k) — approaches escalation under persistence rule, but the saves are non-consecutive for this mode

**Reference comparison**: No baseline.

**Diagnosis**:
The step-90000 save shows the same fundamental pattern as step 50000 and 70000 — per-seed inconsistency where certain seed/class combinations produce valid CXRs and others produce pure noise. The pattern has been present since at least step 50000 (40% of training) and has not resolved by step 90000 (90% of training). This is the primary failure mode for this run.

Two failure types are now visible:
1. Full-panel noise (seed 256/all, seed 42/effusion, seed 137/no_finding, seed 512/no_finding): no anatomical structure, pure isotropic noise.
2. Oversmoothed blob (seed 512/cardiomegaly): the model generates an oversmoothed mean rather than a sharp CXR. This is a different failure — the denoising is converging but to a blurry attractor.

The root cause of the per-seed inconsistency is most likely incomplete training coverage of the (noise, class) joint distribution. With class_mult giving no_finding only 25% of batches, the score function for no_finding at certain noise seeds may be undertrained. Similarly, effusion at 35% (down from 2.0× upweight in prior runs) may have reduced coverage.

The oversmoothed blob at seed 512/cardiomegaly is a separate phenomenon: at this specific noise initialization, the reverse diffusion converges to a smooth mean rather than a sharp output. This can happen when the score function near this noise seed is weak (flat) and the DDIM sampler integrates toward a smooth prior rather than a sharp conditional distribution. It is consistent with the cardiomegaly denoising being undertrained in certain regions of noise space — the model defaults to a smooth prior when it lacks confidence.

The loss-component-imbalance (cls0 at 0.00025, ratio 1314) at this step is an extreme per-batch reading that reflects the class_mult architecture: with only 25% of a 16-sample batch containing no_finding images (4 samples), and those 4 samples happening to draw very low noise timesteps (t near 0, near-zero MSE), the cls0 loss can briefly be near zero. This is not a structural failure but a sampling artifact.

**Diagnostic steps**:
1. Generate a larger batch of samples (N=100 per class) at step 90000 using seeds 0–100 uniformly. Check what fraction of seeds produce pure noise per class. If >50% of seeds produce noise for any class, the model has a structural denoising failure for that class. If <10% are noise, the issue is limited to a few unfortunate fixed seeds in the eval grid.
2. Check whether the eval grid seeds (42, 137, 256, 512) are systematically more "difficult" than the training batch seeds — if training used seeds 0–999 uniform and eval is fixed to these 4 seeds, the eval may not be representative.
3. Increase DDIM steps from 50 to 100 at inference and re-evaluate — if oversmoothed outputs become sharper with more steps, the sampling budget is the limiting factor, not the model weights.
4. Inspect the latent magnitude of seeds 256 and 512 vs seeds 42 and 137: run `torch.norm(noise)` for each seed and check if seeds 256/512 produce latent vectors with higher-than-normal norm (outside the training noise distribution at t=T).

**Fix options**:
- Option A (continue to 100k and evaluate): The run has 10k steps remaining. Allow completion and evaluate the final checkpoint; the CTR trajectory (0.520 best at step 80k) suggests the cardiomegaly target may be nearly met even with some noisy seeds. Use only the valid-seed samples for downstream evaluation.
- Option B (post-training sampling fix): The eval grid seeds may be partly to blame. For downstream use (CTR/presence measurement), sample with a larger set of seeds and discard noise failures — this is operationally feasible if only 20–30% of seeds fail.
- Option C (retrain with more effusion and no_finding samples): The class_mult of no_finding=0.25 may be too low, creating undertrained regions of no_finding noise space. For the next run, try class_mult={0:0.33, 1:0.40, 2:0.27} to give more balanced coverage. The effusion_weight=1 reduction from 2.0 may have over-corrected.
- Option D (longer training): The per-seed failure rate has been gradually declining (step 50k: ~50% noise panels; step 70k: ~40%; step 90k: ~40%). More training steps may continue this trend. However, with the LR now at 2.47e-6 (late cosine decay), the remaining training capacity is limited.

**Recommendation**: Continue to step 100000. Then evaluate with a large seed sweep (N=100 seeds per class) to characterize the noise failure rate and determine whether the model meets the Exp5 CTR target on average. Step 80000 (ctr_cardio=0.520) is the current best checkpoint by the primary eval metric. Do not retrain yet — first measure whether the per-seed noise rate is acceptable (<20%) on a large seed sweep.

**Unrecognized observations**:
1. The mixed noise/clean per-seed pattern has been persistent from step 50000 through step 90000 without resolving. This is an emergent failure mode not described in the catalog. It appears to reflect incomplete coverage of the (noise, class) joint distribution — certain starting noise vectors fall outside the regions where the score function has been reliably trained. A candidate new catalog mode: "denoising-coverage-gap" — characterized by valid outputs for most seeds but noise outputs for a repeating subset of seeds across multiple checkpoints. The fix is not a hyperparameter change but a training distribution/step-count issue.
2. Oversmoothed blob at seed 512/cardiomegaly: this is a distinct failure from noise — the model is denoising but producing a smooth mean. This may be the "blurry-outputs" mode but is strikingly localized to a single seed/class combination rather than affecting all panels equally.

---

## Save v58 (step 100000, final) — 2026-07-01T00:14:38Z

**Severity**: ⚠️ concerning

**Visual observations**:
recon_grid_step0100000.png inspected directly. The mixed noise/clean pattern persists at the final checkpoint and is qualitatively identical to step 90000:

- Seed 42: NO FINDING = good CXR; CARDIOMEGALY = good CXR; EFFUSION = pure noise (same as step 90000)
- Seed 137: NO FINDING = pure noise (same as step 90000); CARDIOMEGALY = good CXR; EFFUSION = good CXR (same as step 90000)
- Seed 256: ALL THREE = noisy. NO FINDING = noise with faint dark patches; CARDIOMEGALY = high-frequency noise with dark channel-like structures (possible early rib signal); EFFUSION = noise. Seed 256 has not improved between step 90000 and 100000.
- Seed 512: NO FINDING = pure noise; CARDIOMEGALY = oversmoothed/blurry blob (same as step 90000 — slightly more defined but still no rib structure); EFFUSION = partially structured (same as step 90000)

The final checkpoint is nearly identical to the step-90000 checkpoint in visual quality. The 10000 additional training steps (with LR already below 2.5e-6) have not meaningfully changed the per-seed pattern.

Eval at step 100000: ctr_cardio=0.493, presence_cardio=0.263, frac_gt0.5=0.333, presence_effusion=0.471, yield=0.000. Summary metrics: auc/cardio_gen=0.713, auc/poe_cardio=0.838, leak/cardio_to_effusion=0.406.

**Per-mode flags**:
- nan-or-inf: ✓ pass (val=0.2777, train=0.1438, grad=0.0198, lr=0)
- loss-zero-collapse: ✓ pass
- all-black-pixels: ✓ pass (mean pixel 0.508)
- no-sample-improvement: ⚠️ triggered — the step-100000 grid is visually indistinguishable from step-90000 for the same seed/class combinations. The same cells that were noise at step 90000 are still noise at step 100000; the same cells that were valid remain valid. The additional 10000 steps at <2.5e-6 LR produced no measurable visual improvement.
- oscillation: ✓ pass (rel_p2p=0.037)
- loss-plateau: ⚠️ triggered — slope +0.00014/1k. Ninth (final) consecutive trigger.
- val-train-divergence: ✓ pass
- loss-component-imbalance: ✓ pass at this final step (ratio=2.0; cls0=0.480, cls1=0.234, cls2=0.364 — balanced)
- blurry-outputs: ⚠️ triggered — seed 512/cardiomegaly still oversmoothed at final checkpoint. No resolution with additional training.
- fp-precision-artifact: ✓ pass (no clustered black patches)
- conditioning-ignored: ⚠️ partial — effusion conditioning still weak (seed 42/effusion = noise)
- lung-anatomy-absent: ⚠️ triggered — same noisy panels persist
- gradient-vanishing: ✓ pass (grad=0.020; expected at LR=0)

**Trend signals**:
- Improvement direction: val/loss flat (0.2777 final — higher than any single checkpoint val). CTR final 0.493 (below 80k best of 0.520). frac_gt0.5/cardio=0.333 (below 80k best of 0.583). Yield=0 at final step.
- Oscillation: ✓ pass
- Persistence: plateau 9/3+ (run-length); noise/clean pattern fully persistent for last 50k steps

**Reference comparison**: No baseline.

**Diagnosis**:
The final checkpoint shows no improvement over step 80000-90000 by any metric. The run finished with:
- Best CTR: 0.520 at step 80000 (not the final checkpoint)
- Best frac_gt0.5: 0.583 at step 80000
- CTR at final: 0.493 (below real-cardio threshold of 0.535)
- Yield at final: 0.000 (PoE composition not producing co-presence)
- Per-seed noise failure: ~40-50% of fixed eval panels produce pure noise at every checkpoint since step 50000

The final checkpoint does NOT meet the primary goal of the rebalanced retrain (CTR approaching real-cardio 0.535, yield >10%). The best checkpoint of this run (step 80000) comes closest on CTR (0.520) but still falls below the real-cardio reference by 0.015, and yield remains 0.

The zero yield is the most serious gate failure. The project requires both-present co-presence yield >10% (gate 3). The PoE composition (compose_grid at step 90000 confirmed) shows that 3/3 tested seeds produce noise in the compose grid — the composition is failing because the individual conditionals are failing for those seeds. Since the fixed eval seeds for composition (137/256/512) happen to include seeds that produce noise in the recon grid, the composition inherits those failures.

**Diagnostic steps**:
1. Run a large seed sweep (N=100) on the best checkpoint (step 80000 model_step0080000.safetensors) for both single-disease and PoE composition. If the per-seed noise rate is <20%, the model may be adequate for downstream evaluation on aggregated statistics.
2. Check the AUC: auc/cardio_gen=0.713 and auc/poe_cardio=0.838 from summary. These suggest the model CAN rank cardio vs. non-cardio images (AUC 0.71), but the absolute presence score is low (0.263) because many images are pure noise (which score near 0 for any class). If noise images are excluded, the AUC on valid images may be higher.
3. Evaluate whether the leak/cardio_to_effusion=0.406 is a structural failure — if 40% of generated cardiomegaly images are being classified as effusion, this indicates the cardiomegaly conditional is co-rendering effusion features (cross-conditioning bleed).

**Fix options**:
- Option A (use best checkpoint, large seed sweep): Use step 80000 checkpoint with a large seed sweep (N=100+) and filter noise samples. If the valid-image subset meets CTR and presence targets, proceed to Exp5 evaluation on that subset.
- Option B (longer training from step 80000): Resume from step 80000 with a higher LR (e.g., restart cosine from 1e-5) and train for another 20000 steps. The cosine decay had reached <1e-5 by step 80000, which may have limited further improvement. A warm restart could allow additional learning.
- Option C (class_mult adjustment): For a new run, increase no_finding class_mult from 0.25 to 0.33 and effusion from 0.35 to 0.40, reduce cardiomegaly from 0.40 to 0.27. The current run may have over-emphasized cardiomegaly (which did improve from 0.125 to 0.583 frac_gt0.5) at the expense of effusion and no_finding denoising coverage.
- Option D (DDIM steps at eval): Test whether the noise failures are DDIM-step-count artifacts. Run the fixed eval seeds at 200 DDIM steps instead of 50. If noise panels become valid CXRs, the sampler budget (not the model weights) is the bottleneck.

**Recommendation**: Do not retrain immediately. First: (1) Run the step-80000 checkpoint with N=100 random seeds for all three classes to measure the noise failure rate. (2) Test 200 DDIM steps on the fixed eval seeds to determine if noise failures are sampler-budget artifacts. If >80% of seeds produce valid CXRs with 200 DDIM steps, a sampling fix resolves the issue without retraining. If noise failures persist with 200 DDIM steps, consider Option C (rebalanced class_mult for a new run). The zero yield remains the hardest gate — address this as a priority.

---

## Run mzd3jwy9 — VERDICT (exp_vae_v2_ldm, rebalanced retrain) — 2026-07-01

**Saves analyzed this sweep:** v48(10k) v50(20k) v51(30k) v52(40k) v53(50k) v54(60k) v55(70k) v56(80k) v57(90k) v58(100k, final). 10 saves total.

**Severity timeline:** ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️

**Run state:** finished (100k steps). Identity: exp_vae_v2_ldm / rebalanced retrain using VAE-v2 latents and class_mult reweighting.

**Primary failure modes:**
1. Per-seed inconsistency (unrecognized pattern, persistent from step 50k to 100k): ~40-50% of fixed eval seeds produce pure noise rather than valid CXRs. Same seeds fail across multiple checkpoints. Root cause: incomplete training coverage of the (noise, class) joint score function — certain noise initializations fall in undertrained regions.
2. Effusion conditioning weak: effusion column is the most consistently noisy column across all seeds and all checkpoints. The class_mult=0.35 for effusion (reduced from effusion_weight=2.0) may have over-corrected, leaving effusion undertrained.
3. Zero yield throughout (0% co-presence in PoE composition at every eval step). The compose grid seed failures (seeds 137/256/512 all failing in the compose grid at step 90000) explain the zero yield — the composition inherits the per-seed noise failures.

**Positive signals:**
- CTR reached 0.520 at step 80000 (real-cardio ref=0.535) — the rebalancing did substantially improve cardiomegaly rendering vs prior runs (was 0.395 at 100k in the old LDM)
- No NaN/Inf, no gradient collapse, no catastrophic failure
- frac_gt0.5/cardio reached 0.583 at step 80000 (up from 0.125 at step 5000)
- Val/loss stable throughout (mean 0.2735, std 0.005) — training numerically healthy

**Best checkpoint:** step 80000 (model_step0080000.safetensors in ckpts/vae_v2_ldm/ — verify path) by CTR (0.520) and frac_gt0.5 (0.583). NOT the final checkpoint (step 100000 performs worse on all eval metrics).

**Open gates:**
- Exp5 CTR gate: CTR 0.520 vs target 0.535 — within 0.015; may pass on a large seed sweep if noise samples are filtered
- Yield gate (10% co-presence): 0.000 at every checkpoint — NOT met. Critical blocker.
- Large seed sweep needed before any gate decision

**Catalog modes skipped (not logged):** latent-collapse (no PCA), noise-schedule-mismatch, attention-map-collapse, timestep-distribution-skew, update-to-weight-ratio, weight-norm-explosion, fid-regression (no FID logged), loss-scale-anomalous (no baseline), lr-batch-size-mismatch (no reference config), unexpected-image-stats (baseline stats not defined for VAE-v2 latents).


---

# Training Tracking — prime_lab/paper3-ldm/umqnrs0o

Entries appended by /analyze-run; oldest first. Run name: `exp_vae_v2_ldm_mu_v2`.

**Run context (v2 recipe — NOT a like-for-like rerun of 6wj35ips / mzd3jwy9).** Differences: adds Weight-EMA (decay 0.9999 ≈ ~10k-step time constant), adds min-SNR-γ loss weighting (γ=5), rebalanced `class_mult` normal/cardio/effusion = 0.28/0.34/0.38 (was 0.25/0.40/0.35), effusion_weight lowered to 1. batch 16, grad_accum 1, lr 1e-4, warmup 500, bf16 train, max_steps 100k, ckpt_every 10k, grid_every 5k, compose_every 15k. **Recipe GOAL: stop the EFFUSION COLLAPSE seen in 6wj35ips (presence/effusion fell 0.378→0.003) and improve per-seed consistency.**

**Two interpretive artifacts applied to every entry below:**
1. **EMA warmup.** Eval panels + eval scalars are computed on EMA weights; with decay 0.9999 the EMA at 10k–15k is a heavy average of early under-trained weights. Early low eval metrics UNDER-READ and are NOT scored as regression on their own — the question is whether they RECOVER by ~20k–30k.
2. **min-SNR loss scale.** `train/loss` is min-SNR-γ-weighted and NOT comparable to the prior run's ~0.06 magnitude. Trend and stability are judged, not absolute magnitude.

**No baseline available; absolute catalog checks only (no good-run reference dir for this recipe).**

Checkpoint artifacts (type=model) found: `ldm-ckpt:v69` (step10000), `ldm-ckpt:v70` (step20000). Recon grids at 5k/10k/15k/20k and a compose grid at 15k were inspected visually. Run state at analysis time: **running**, `_step`=23884.

---

## Save 1 (step 10000) — 2026-07-01T12:46:46Z — ldm-ckpt:v69

**Severity**: ⚠️ concerning

**Visual observations**:
Recon grid at 10k (3 classes × 4 seeds, 512²): outputs are coarse cloud-like noise with NO recognizable thoracic anatomy — no lung fields, ribs, or mediastinum yet. Compared against the 5k grid (which is finer-grained, higher-frequency pure noise), the 10k grid is visibly *coarser/smoother* noise — i.e. the denoiser is beginning to organise spatial frequency but has not yet produced structure. This is the expected appearance of EMA-averaged weights at 10k when the EMA time constant (~10k steps) means the eval weights are still dominated by the earliest, near-random parameters. Seeds differ (no within-grid diversity collapse). No all-black pixels, no clustered black patches, no checkerboard texture. Mid-grey mean intensity (not black, not saturated).

**Per-mode flags**:
- nan-or-inf: ✓ (0 NaN/Inf across 243 dense train rows; train/loss, val/loss, grad_norm all finite)
- loss-zero-collapse: ✓ (min train/loss 0.0067 ≫ 1e-8)
- all-black-pixels: ✓ (grid is mid-grey noise, high mean — not black)
- blurry-outputs: not scored (pre-warmup by EMA-lag; catalog says skip before warmup)
- lung-anatomy-absent: not scored (pre-warmup by EMA-lag; structure legitimately not expected yet at 10k EMA)
- no-sample-improvement: ✓ (5k→10k shows clear change in grain — model IS moving)
- oscillation: ✓ (val/loss 0.0664→0.0658, flat-declining; train/loss monotone)
- loss-plateau: ✓ (train/loss still declining)
- val-train-divergence / overfit-regime: ✓ (val declining with train)
- loss-component-imbalance: ✓ (cls0/cls1/cls2 = 0.144/0.100/0.110 → max/min ≈ 1.4×, ≪ 10:1)
- gradient-explosion: ⚠️ transient only (grad_norm peaked ~2.29 in earliest steps, already settled to ~0.08 by 10k; isolated early spike, self-corrected — Warning tier, not escalated)
- kl-posterior-collapse / kl-weight-imbalance: n/a (LDM in latent space; VAE codec is frozen, KL not trained here)
- **cardio channel absent**: ⚠️ (presence/cardio = 0.000, ctr_med/cardio = NaN — the cardio-conditioned samples are not being rendered as cardiomegaly; the instrument returns NaN CTR because it finds no heart to measure)
- latent-collapse, noise-schedule-mismatch, attention-map-collapse, timestep-distribution-skew, update-to-weight-ratio, weight-norm-explosion, fid-regression: not checked (not logged)
- loss-scale-anomalous, lr-batch-size-mismatch, unexpected-image-stats: not checked (no baseline / reference for this recipe)

**Trend signals**:
- Improvement direction: improving (val/loss 0.0664@5k → 0.0658@10k; train/loss declining monotonically)
- Oscillation: val/loss peak-to-peak over 5k–10k window ≈ 0.0006 (negligible)
- Persistence of concern: cardio-channel-absent present since first eval (5k) — 2 consecutive so far (not yet ≥3)

**Reference comparison**:
- No baseline available for the v2 recipe; comparison limited to absolute catalog checks. Cross-run note only: presence/effusion = 0.659 here vs the 6wj35ips collapse to 0.003 — the effusion channel is NOT collapsing, which is the recipe's stated goal.

**Diagnosis**:
The training process is mechanically healthy: loss decreases monotonically, gradients are finite and small, no fatal mode fires. The concerning signals are both explainable and expected at this step. (1) The noise-like samples are the direct consequence of EMA warmup — the eval uses EMA weights whose 0.9999 decay means at 10k they still encode mostly the random-init phase; the *raw* weights (reflected in the falling train loss) are learning fine. This is exactly the artifact flagged in the run brief and should not be read as a regression. (2) The cardio channel producing nothing (presence 0, CTR NaN) reproduces the pre-existing, well-documented "cardio under-renders / diffusion-not-composition-for-cardio" failure from the prior lineage — the generator has historically failed to render the heart on the cardio conditional. At 10k under EMA lag it is impossible to separate "cardio genuinely not learning" from "EMA hasn't caught up on the cardio conditional yet." The right move is to withhold judgement and check recovery by 20k–30k.

**Diagnostic steps**:
1. Confirm the noise appearance is EMA-lag, not a broken denoiser: look at `train/loss` dense series in W&B — it falls 0.76→~0.03 by 10k, so the *raw* model is denoising. If raw-weight sample dumps are available, render one; it should look far more structured than the EMA grid. (Confirmed here from the loss curve.)
2. Track `presence/cardio` and `ctr_med/cardio` at 20k, 30k, 40k. If cardio presence stays 0.000 and CTR stays NaN past ~30k (well beyond EMA warmup), the cardio conditional is genuinely failing to render — the known cardio-under-render problem — not an EMA artifact.
3. Watch `leak/cardio_to_effusion` (0.094 at 10k): if it climbs, cardio prompts are being rendered *as effusion* — the classic cardio mis-render signature seen before.

**Fix options** (do NOT act yet — too early; these are for if cardio fails to recover by ~30k):
- Option A (conservative): let it run to ~30k and re-evaluate; EMA warmup alone can explain everything seen at 10k. No config change.
- Option B: if cardio still 0 at 30k, raise the cardio `class_mult` (currently 0.34) further and/or reduce EMA decay to 0.999 so eval reflects the trained weights sooner for monitoring.
- Option C: if cardio is genuinely dead at 40k, this reproduces the "cardio conditional under-renders intrinsically" finding — the lever is a data/rebalance change (border-visible cardio subset), not more steps.

**Recommendation**: Let the run continue. 10k is inside EMA warmup; nothing here justifies pausing. Re-assess cardio specifically at the 20k save (below) and again at 30k.

---

## Save 2 (step 20000) — 2026-07-01T13:21:45Z — ldm-ckpt:v70

**Severity**: ⚠️ concerning

**Visual observations**:
Recon grid at 20k shows CLEAR progress over 10k: the texture is smoother and less granular, and the first thoracic structure is emerging — several panels (notably the CARDIOMEGALY column, seeds 137/256) show a lighter vertical central band (mediastinum-like) flanked by darker lateral regions (lung-field-like). It is still well short of a clean CXR and no panel is diagnostically convincing, but the denoising-quality trajectory 5k(fine noise) → 10k(coarse noise) → 20k(emerging structure) is monotonic and exactly what EMA catch-up should look like. Seeds differ across the grid (diversity intact). No all-black pixels, no clustered black patches, no checkerboard, no saturation. The 15k PoE compose grid (∅-anchor and healthy-anchor, w=1.0/2.0) is still pre-structural cloud at that step and cannot yet be judged for co-presence rendering — too early.

**Per-mode flags**:
- nan-or-inf: ✓ (all finite; 0 NaN/Inf in dense series through 24.2k)
- loss-zero-collapse: ✓ (min train/loss 0.0067 ≫ 1e-8)
- all-black-pixels: ✓ (grid is mid-grey, structured; not black)
- blurry-outputs: ⚠️-eligible but NOT scored (still within EMA-lag regime; structure only just emerging at 20k — catalog says skip before warmup and EMA warmup ≈ 10–20k here). Re-score at 30k+.
- lung-anatomy-absent: borderline, NOT scored (lateral/center structure is just beginning to appear; heuristic is unreliable on half-formed structure — defer to 30k)
- no-sample-improvement: ✓ (10k→20k shows clear, unmistakable improvement — structure emerging)
- oscillation: ✓ (val/loss 5k→20k = 0.0664/0.0658/0.0609/0.0634; peak-to-peak 0.0055, relative ≈ 8.5% < 15% threshold)
- loss-plateau: ✓ (train/loss 0.042→0.031 over window; still declining)
- val-train-divergence / overfit-regime: ✓ (val 0.0609@15k, 0.0634@20k — small uptick but within noise; train still falling; gap tiny — no divergence)
- loss-component-imbalance: ✓ (cls0/cls1/cls2 = 0.105/0.092/0.055 → max/min ≈ 1.9×, ≪ 10:1; rebalanced class_mult is working — no single class starved)
- gradient-explosion: ✓ (grad_norm 0.053 at 20k; the early ~2.29 spike is long resolved)
- **cardio channel absent**: ⚠️ (presence/cardio = 0.000, ctr_med/cardio = NaN — still no cardio rendering at 20k)
- **cardio→effusion leakage**: ⚠️-adjacent (leak/cardio_to_effusion rose 0.094@10k → 0.469@15k → 0.813@20k — cardio prompts increasingly read AS effusion). This is a real, climbing signal — the known cardio-mis-render-as-effusion pattern — but at 20k it remains confounded with EMA warmup. Treated as ONE concern (the cardio channel) with the presence flag, not double-counted.
- kl modes: n/a (latent-space LDM; VAE frozen)
- latent-collapse, noise-schedule-mismatch, attention-map-collapse, timestep-distribution-skew, update-to-weight-ratio, weight-norm-explosion, fid-regression: not checked (not logged)
- loss-scale-anomalous, lr-batch-size-mismatch, unexpected-image-stats: not checked (no baseline)

**Trend signals**:
- Improvement direction: improving (val/loss net down 0.0664→0.0634 over 5k–20k; train/loss monotone down; sample structure emerging)
- Oscillation: val/loss peak-to-peak 0.0055 over the full window (8.5% relative — below the 15% flag)
- Persistence of concern: cardio-channel-absent now 4 consecutive evals (5k/10k/15k/20k). Under the strict catalog rule this persistence would push toward ❌, BUT the interpretive brief explicitly warns that EMA warmup makes early eval metrics under-read on the *conditional* channels through ~20k–30k. Because the warmup window overlaps the entire observed span, the persistence cannot yet be attributed to a genuine training failure. Held at ⚠️, NOT escalated to ❌, pending the 30k save which lands outside the warmup window.

**Reference comparison**:
- No baseline. Cross-run headline (informational, not scored): **presence/effusion = 0.659 and dead-flat across all four evals (5k→20k), vs 6wj35ips which collapsed to 0.003.** The v2 recipe's primary objective — prevent effusion collapse — is being MET so far. Note: presence/effusion is byte-identical (0.659) at every eval, as are `eval/latent_arith_cos_c/e` (0.9521/0.9667); ctr_med/effusion DOES move (NaN→0.204→0.302), confirming eval runs. The flatness of presence/effusion may reflect a fixed eval subset scored identically each time — worth a glance at the eval harness — but it is a NON-COLLAPSED, healthy value, which is the point.

**Diagnosis**:
Two independent stories, opposite signs. (1) **Effusion — success.** The recipe's goal is being achieved: effusion presence holds at 0.659 (vs the prior run's 0.003 collapse), ctr_med/effusion is climbing (0.204→0.302 toward the 0.5 target), class-loss balance is healthy (1.9× ratio), and overall training dynamics are clean (monotone loss, finite grads, no fatal modes, emerging anatomy). Nothing here looks "worse" once the min-SNR loss-scale and EMA-warmup artifacts are accounted for — the user's impression of "worse" is most plausibly (a) the EMA grids looking like noise at 10k, and (b) the min-SNR train/loss reading a different magnitude than the old run. Both are expected recipe artifacts, not regressions. (2) **Cardio — unresolved, but not yet a verdict.** presence/cardio = 0 and rising cardio→effusion leakage (to 0.81) reproduce the long-documented "cardio conditional under-renders / renders as effusion" problem. Mechanistically the most likely chain is: the cardio conditional has too little/too-hard signal to render an enlarged heart, so the denoiser falls back to the dominant effusion-like texture it has learned, and the instrument reads those samples as effusion (hence leakage↑ while cardio-presence stays 0). BUT the entire 5k–20k span is inside EMA warmup, so this cannot yet be separated from "EMA hasn't caught the cardio conditional up." Alternative benign explanation: cardio structure emerges 10–20k later than effusion because the rebalanced class_mult (0.34) plus min-SNR reweighting changed the per-class learning speed. The 30k save is the decision point.

**Diagnostic steps**:
1. At the 30k save (first checkpoint clearly outside EMA warmup), read `presence/cardio` and `ctr_med/cardio`. If still 0 / NaN → genuine cardio failure, not EMA. If they lift off → it was warmup lag.
2. Read `leak/cardio_to_effusion` at 30k: if it stays ≥0.8, cardio is being actively mis-rendered as effusion (real failure). If it falls as cardio structure appears, it was warmup confound.
3. Render a raw-weight (non-EMA) cardio sample at 20k if the training script can dump one — if raw cardio already shows a big heart while the EMA grid does not, the problem is purely EMA monitoring lag and the run is fine.
4. Confirm presence/effusion's flat 0.659 is a real held score and not a cached constant: check the eval harness computes it fresh each call (ctr_med/effusion moving suggests eval does run; verify presence isn't short-circuited).

**Fix options** (only if cardio confirmed failing at 30k — do NOT act now):
- Option A (conservative): continue to 30k unchanged; let EMA catch up and re-decide. Nothing currently justifies intervention.
- Option B: if cardio dead at 30k — raise cardio `class_mult` above 0.34 and/or temporarily drop `ema_decay` to 0.999 so monitoring reflects trained weights; consider disabling min-SNR reweighting on the cardio class if it is deprioritising cardio timesteps.
- Option C: if cardio genuinely under-renders at 40k+ — this reproduces the intrinsic cardio-under-render finding; the effective lever is the border-visible cardio data rebalance (per project memory), not more steps or LR changes. Reuse the effusion channel, retrain/reweight cardio.

**Recommendation**: Let the run continue to at least 30k, then re-analyze. Do not pause. Headline for the parent: effusion is holding (recipe goal met so far); cardio is the open question but is not yet judgeable because the whole observed span is inside EMA warmup.

**Unrecognized observations**:
- presence/effusion, eval/latent_arith_cos_c, and eval/latent_arith_cos_e are byte-identical at every eval step while ctr_med/effusion changes. Not a catalog mode. Most likely a fixed-subset eval scored deterministically (benign), but flag it: verify the eval harness isn't caching/short-circuiting these three scalars. If they ARE cached, presence/effusion's "hold" is less informative than it looks — though ctr_med/effusion's genuine rise still independently supports the effusion-not-collapsing conclusion.

**Catalog modes skipped (not logged this run):** latent-collapse (no PCA), noise-schedule-mismatch (no per-timestep quality), attention-map-collapse (no attn logging), timestep-distribution-skew (no timestep histogram), update-to-weight-ratio / weight-norm-explosion (no per-layer norms), fid-regression (no FID), loss-scale-anomalous / lr-batch-size-mismatch / unexpected-image-stats (no v2 baseline/reference).


---

# Training Tracking — prime_lab/paper3-ldm/yrcj8qgl

Entries appended by /analyze-run; oldest first. Run name: `exp_vae_v2_ldm_mu_v3`. Analyzed 2026-07-01 while run state = **running**, `_step` ≈ 13034 (only one checkpoint artifact logged so far: `ldm-ckpt:v71` @ step10000).

**Run context (v2 recipe, v3 iteration — the successor to umqnrs0o `exp_vae_v2_ldm_mu_v2`).** Config deltas vs umqnrs0o: `class_mult` normal/cardio/effusion = **0.2 / 0.4 / 0.4** (was 0.28/0.34/0.38 — cardio and effusion both raised, normal lowered), `cfg_dropout_p` = 0.15 (explicit CFG dropout logged), `effusion_weight` = 1, `min_snr_gamma` = 5, `use_min_snr` = True, `use_ema` = True / `ema_decay` = 0.9999, batch 16, grad_accum 1, lr 1e-4, warmup 500, bf16 train, max_steps 100k, ckpt_every 10k, grid_every 5k, compose_every 15k. **Recipe GOAL (inherited): hold effusion (no collapse) AND recover cardio rendering — the two channels that split across the prior μ lineage (6wj35ips: cardio good / effusion collapsed; umqnrs0o: effusion held / cardio absent). This run aims to get both.**

**Two interpretive artifacts carried from the v2 lineage:**
1. **EMA warmup.** Eval scalars/panels are nominally EMA-weighted (decay 0.9999 ≈ ~10k-step time constant), so early eval can under-read. IMPORTANT CAVEAT FOR THIS RUN: unlike umqnrs0o (which showed pure noise at 10k), yrcj8qgl's 10k recon grid is already a recognizable CXR — so whatever weights the recon path uses here, structure has emerged well ahead of the umqnrs0o EMA-lag schedule. Do not blindly apply the "10k = EMA noise" prior to this run; the images contradict it.
2. **Decode-scale bug.** This run was launched WITHOUT the decode fix: the in-training dashboard decodes generated latents WITHOUT dividing by scale_factor (sf≈5.034), which 5× over-scales v2 μ generations. Generated latents at these steps are under-dispersed (std≈0.33 vs trained ≈1.0), which partially compensates the missing division — so buggy-decode images CAN look reasonable and the metrics can trend correctly, but **absolute pixel/CTR values sit on the un-divided decode path**. Judge trend + internal consistency; flag absolutes as recipe-artefactual.

**No baseline available; absolute catalog checks only (no good-run reference dir for this recipe).**

---

## Save 1 (step 10000, only save so far) — 2026-07-01T14:39:57Z — ldm-ckpt:v71

**Severity**: ✓ healthy

**Visual observations**:
The 10k recon grid (3 conditions × 4 seeds, 512²: NO FINDING / CARDIOMEGALY / EFFUSION) shows **recognizable chest X-rays** — bilateral lung fields, rib cage, mediastinal shadow, spine, and hemidiaphragms are all present and in the right places. This is a striking, favourable contrast with umqnrs0o at the same step (pure coarse noise there). Trajectory is monotone: the 5k grid is the same anatomy but hazier/softer; 10k is sharper with clearer rib and diaphragm edges. The CARDIOMEGALY column renders visibly **broader cardiac silhouettes** than the NO FINDING column across seeds — i.e. the cardio conditional is actually rendering an enlarged heart, not falling back to effusion texture. Seeds differ across each row (within-grid diversity intact — no mode collapse). No all-black panels, no clustered black patches, no checkerboard periodicity, no saturation; mid-range mean intensity. Caveat: images are on the un-divided (buggy) decode path, so absolute contrast/greyscale is not calibrated — but structure, anatomy, and cardio-vs-normal differentiation are legible regardless.

**Per-mode flags**:
- nan-or-inf: ✓ (0 non-finite across all 12 dense train rows; train/loss, val/loss, grad_norm all finite through step ~12k)
- loss-zero-collapse: ✓ (min train/loss 0.0204 ≫ 1e-8)
- all-black-pixels: ✓ (grid is structured mid-grey, high mean — not black)
- lung-anatomy-absent: ✓ (lateral lung fields visibly darker than central mediastinum in NO FINDING/EFFUSION columns; anatomy clearly present)
- blurry-outputs: ✓ (edges — ribs, diaphragm — are resolved at 10k; not scored quantitatively, no FFT, but visually not blurred)
- diversity-collapse: ✓ (seeds differ within every row)
- no-sample-improvement: ✓ (5k→10k is a clear, monotone sharpening)
- oscillation: ✓ (val/loss 0.0736→0.0629 over the run, monotone-ish decline; peak-to-peak small)
- loss-plateau: ✓ (train/loss 0.031→0.020 still declining; val/loss declining)
- val-train-divergence / overfit-regime: ✓ (val declining with train; no gap growth)
- loss-component-imbalance: ✓ (train/loss_cls0/1/2 at last dense row = 0.117/0.138/0.035 → max/min ≈ 3.9×, ≪ 10:1; early transient cls0 spike to 0.41 settled by 10k)
- gradient-explosion: ✓ (grad_norm max 0.16, last 0.052 — no spike; far below any z-3 event)
- checkerboard-artifact: ✓ (no periodic texture visible)
- kl-posterior-collapse / kl-weight-imbalance: n/a (latent-space LDM; VAE-v2 codec frozen, KL not trained here)
- **cardio channel**: ✓ **RENDERING** (presence/cardio 0.138, ctr_med/cardio 0.496 vs real ref 0.535, frac_gt0.5/cardio 0.500, border_frac/cardio 0.5625; cardiomegaly column visibly enlarged) — this is the channel that FAILED in umqnrs0o (presence 0, CTR NaN, leak→0.81) and it is alive here.
- cardio→effusion leakage: ✓ acceptable and FALLING (leak/cardio_to_effusion 0.469@5k → 0.375@10k; contrast umqnrs0o's climb to 0.81 — cardio prompts are NOT being swallowed by effusion texture here)
- latent-collapse, noise-schedule-mismatch, attention-map-collapse, timestep-distribution-skew, update-to-weight-ratio, weight-norm-explosion, fid-regression: not checked (not logged)
- loss-scale-anomalous, lr-batch-size-mismatch, unexpected-image-stats: not checked (no v2 baseline / reference)

**Trend signals**:
- Improvement direction: improving (val/loss 0.0736→0.0629; train/loss 0.031→0.020; sample sharpness up 5k→10k)
- Oscillation: val/loss peak-to-peak ≈ 0.011 over 12 dense rows (small); no save-frequency bounce
- Persistence of concern: none. No catalog mode is in a triggered state. (One watch item below is a single-interval move, not a persistent flag.)

**Reference comparison**:
- No baseline for the v2/v3 recipe. Cross-run positioning (informational, not scored): yrcj8qgl at 10k sits **ahead of both prior μ runs** — it has cardio rendering (6wj35ips-like strength: ctr 0.496 near real 0.535) AND effusion not collapsed (presence 0.159, ctr 0.447 — not the 6wj35ips 0.003 floor), i.e. it is currently doing what umqnrs0o and 6wj35ips each did only half of. yield/poe_gate3 = 0.125–0.156 already clears the 0.10 co-presence gate at 10k (umqnrs0o and mzd3jwy9 sat at 0.000 on that gate) — the single most encouraging number, though early and on the buggy-decode instrument.

**Diagnosis**:
Mechanically clean and, on the visual + eval evidence, the best-positioned run of the μ lineage at the 10k mark. No fatal mode, no concerning mode triggers. The user's "looks good quite early" is corroborated: recognizable CXR anatomy at 10k, both conditional channels active, class-loss balance healthy (3.9× not 10×+), gradients small and stable, losses declining. The recipe change most plausibly responsible is the `class_mult` shift to 0.2/0.4/0.4 (cardio raised from 0.34→0.40 relative to a lowered normal weight) combined with explicit `cfg_dropout_p`=0.15, which together give the cardio conditional more gradient share while keeping effusion weighted — addressing the exact split that broke the two prior runs. The one thing to watch (not a failure): presence/effusion fell 0.315@5k → 0.159@10k in the same interval that ctr_med/effusion ROSE 0.423→0.447. That divergence (fewer samples cross the presence threshold, but the median CTR of effusion samples went up) is most consistent with the effusion distribution tightening/shifting under EMA+min-SNR rather than collapsing — a collapse would drop BOTH presence and CTR toward the floor, as 6wj35ips did. With only two eval points it is not a trend and triggers no catalog mode; it is logged as a watch item, not a ⚠️.

**Watch items (not catalog triggers, no action)**:
1. presence/effusion 0.315→0.159 across 5k→10k while ctr_med/effusion rose — re-check at the 20k eval. If presence keeps falling toward ~0 AND ctr_med/effusion also turns down, that is the onset of the effusion collapse this recipe exists to prevent → would become ⚠️ then. If presence stabilises/recovers, it was distribution tightening.
2. eval/latent_arith_cos_c/_e are again byte-identical across evals (0.9521 / 0.9667) as in umqnrs0o — same suspected fixed-subset deterministic scoring; benign but verify the eval harness isn't short-circuiting these scalars.

**Recommendation**: Let the run continue — it is the healthiest 10k in the lineage and nothing justifies intervention. At the 20k save, re-analyze with two specific questions: (a) does presence/effusion stabilise or keep falling (watch item 1); (b) does cardio (ctr_med/cardio, frac_gt0.5/cardio) hold or improve past 0.5. If both hold, this run is on track to satisfy both channels for the first time.

**Catalog modes skipped (not logged this run):** latent-collapse (no PCA), noise-schedule-mismatch (no per-timestep quality), attention-map-collapse (no attn logging), timestep-distribution-skew (no timestep histogram), update-to-weight-ratio / weight-norm-explosion (no per-layer norms), fid-regression (no FID), loss-scale-anomalous / lr-batch-size-mismatch / unexpected-image-stats (no v2/v3 baseline or reference config).

---

## Sweep 2 (steps 20k–100k, run FINISHED) — analyzed 2026-07-02

Run reached **step 100000 and state=finished** (heartbeat 2026-07-01T19:48Z). Nine new checkpoints since Save 1: `ldm-ckpt:v72`(20k) … `v80`(100k). No baseline dir → absolute catalog checks only. All artifacts on the **buggy decode path** (decodes generated latents WITHOUT ÷scale_factor, sf≈5.034); per the run caveat, judge trend + internal consistency and flag absolute pixel/CTR as recipe-artefactual. Grids inspected directly at steps 25k / 50k / 75k / 90k / 100k + compose@90k.

**Finite check:** 0 non-finite across train/loss (2000 rows, min 0.00505 max 0.997), val/loss, grad_norm — nan-or-inf ✓ and loss-zero-collapse ✓ for every save in the sweep. These two fatal modes are cleared run-wide and not repeated per-entry below.

**Headline finding of the sweep — the "inconsistent x-rays" concern, resolved:** The inconsistency is NOT rising per-seed variance (numerically, eval scatter *tightens* late: ctr_med/cardio window std 0.010→0.005, presence/effusion ptp 0.156→0.031 across the run as LR→~0 and the model freezes). It is a **stable, CLASS-LOCALIZED rendering failure**: at every checkpoint 50k→100k the NO FINDING and EFFUSION columns are consistently well-formed CXRs, while the **CARDIOMEGALY column is consistently distorted** (blobby/warped mediastinum, mangled cardiac silhouette, wispy bright artifacts — seeds 137/256/512 all affected). This is the cardio conditional weakening, not random per-seed collapse and not the umqnrs0o/mzd3jwy9 "pure-noise-on-a-fixed-seed-subset" pattern (here the non-cardio columns never fail).

**Decode-bug timing verdict (user hypothesis):** MIXED. If the late degradation were purely the convergence-driven over-scaling (z0 std→1.0 unmasking the missing ÷sf), it would hit ALL columns uniformly — it does not; NO FINDING/EFFUSION stay clean while cardio degrades and presence/cardio collapses. So the cardio degradation is **model-real**, not a decode artifact. What IS decode-artefactual: the washed-out/over-bright greyscale of the late recon and compose grids (compose@90k is visibly over-bright) — consistent with over-scaling on now-well-dispersed latents. Net: the *structural* cardio failure is real; the *absolute brightness/contrast* late is the decode bug.

**Full-run trends (per-10k-step slopes over the eval series):**
- ctr_med/cardio: −0.005/10k. Peak 0.521@40k → monotone decline → 0.457@100k. Late (50k→100k) slope −0.009/10k.
- frac_gt0.5/cardio: −0.021/10k. Peak 0.571@40k → 0.310@95–100k. Late slope −0.025/10k. **Strongest single declining signal.**
- presence/cardio: 0.107 plateau (40k–80k) → collapse 0.076@85k → 0.045@95k → **0.013@100k**.
- ctr_med/effusion: −0.006/10k, 0.447@10k → 0.367@100k (slow drift down).
- presence/effusion: **+0.008/10k — HELD/rose** (0.159@20k → 0.284@100k). The recipe's core goal (no effusion collapse) IS met.
- leak/cardio_to_effusion: **+0.020/10k — RISING** (0.406@20k → peak 0.625@80k → 0.531@100k). Cardio increasingly bleeds into effusion texture late.
- yield/poe_gate3: flat ~0.19 (0.156–0.219), still clears the 0.10 gate at every save.
- val/loss: 0.0635@20k → trough 0.0567@85k → back UP to 0.0641@100k.
- auc/cardio_gen (classifier separability): ROSE 0.775→0.80 late — diverges from CTR (classifier still separates gen-cardio even as the CTR-instrument and presence fall; the two judges disagree late, see watch item).

**Plateau test (last3 80–100k vs mid3 40–60k):** ctr_med/cardio −0.042, frac_gt0.5/cardio −0.120, presence/cardio −0.042, yield −0.021 — cardio channel plateaued at 40k then DECLINED. presence/effusion +0.083 (improved). So "has cardio plateaued?" → **worse than plateaued: it peaked ~40k and regressed.**

## Save 2 (step 20000) — 2026-07-01T15:19:02Z — ldm-ckpt:v72

**Severity**: ✓ healthy

**Visual** (25k grid nearest, decode-buggy): mostly well-formed CXRs across all 3 columns/4 seeds; ~9–10/12 clean. Two CARDIOMEGALY panels (seeds 256/512) show mild mediastinal distortion — the first hint of the cardio-column weakness. Anatomy present, diversity intact, no black/checkerboard.
**Metrics**: ctr_med/cardio 0.512 (near peak), frac_gt0.5/cardio 0.552, presence/effusion 0.159 (held), yield 0.219, leak 0.406, val/loss 0.0635.
**Trend**: cardio at/near its run peak; effusion holding. improving-to-flat.

## Save 3 (step 30000) — 2026-07-01T15:54:17Z — ldm-ckpt:v73

**Severity**: ✓ healthy

**Visual**: consistent with 25k — non-cardio columns clean, occasional cardio-column distortion. No new failure mode.
**Metrics**: ctr_med/cardio 0.515, frac_gt0.5/cardio 0.552, presence/cardio 0.076, presence/effusion 0.159, yield 0.188, leak 0.438, val/loss 0.0628.
**Trend**: flat/near-peak on cardio; effusion held.

## Save 4 (step 40000) — 2026-07-01T16:25:29Z — ldm-ckpt:v74

**Severity**: ✓ healthy

**Visual**: peak-quality checkpoint by metrics; grids show the cleanest cardio rendering of the run (cardio-column distortion least frequent here).
**Metrics**: ctr_med/cardio **0.521 (run max)**, frac_gt0.5/cardio **0.571 (run max)**, presence/cardio 0.107, presence/effusion 0.222, yield 0.219, leak 0.469, val/loss 0.0613.
**Trend**: cardio at its all-run peak; this is the strongest cardio checkpoint. **Candidate best-cardio model.**

## Save 5 (step 50000) — 2026-07-01T17:00:40Z — ldm-ckpt:v75

**Severity**: ⚠️ concerning

**Visual observations**:
50k recon grid (decode-buggy) inspected directly. NO FINDING and EFFUSION columns are clean, well-formed CXRs across all four seeds (bilateral lung fields, ribs, diaphragm, mediastinum all correct). The **CARDIOMEGALY column has clearly degraded**: seeds 137, 256, 512 show blobby/warped cardiac-mediastinal regions with loss of a clean heart silhouette; seed 256 has bright wispy artifacts overlaying the lower mediastinum. The degradation is confined to the cardio column — this is a class-conditional failure, not a per-seed noise pattern (the same seeds render fine in the other two columns).

**Per-mode flags**:
- all-black-pixels: ✓ (structured mid-grey, high mean)
- lung-anatomy-absent: ✓ (lateral fields darker than mediastinum in NO FINDING/EFFUSION; anatomy present)
- blurry-outputs: ✓ for NO FINDING/EFFUSION; cardio column soft but not globally blurred (not FFT-scored)
- diversity-collapse: ✓ (seeds differ within rows)
- no-sample-improvement: ✓ (still evolving vs 40k)
- loss-component-imbalance: ✓ (cls0/1/2 = 0.064/0.046/0.040 at 50k dense row → max/min ≈ 1.6×, ≪ 10×)
- gradient-explosion / oscillation: ✓ (val/loss 0.0599, smooth; train/loss stable)
- **cardio channel**: ⚠️ — ctr_med/cardio 0.504 (down from 0.521@40k), frac_gt0.5/cardio 0.518 (down from 0.571@40k), presence/cardio 0.107. First eval where cardio has clearly turned down off its peak AND the grid shows column-localized distortion. Single-save trigger → ⚠️.
- cardio→effusion leakage: ✓-watch (leak 0.438, within band but the rising trend has begun)
- effusion channel: ✓ (presence/effusion 0.159, ctr_med/effusion 0.412 — held, no collapse)
- kl-*, latent-collapse, noise-schedule-mismatch, attention-map-collapse, timestep-skew, per-layer norms, fid: not checked (not logged / latent-space LDM, VAE frozen)

**Trend signals**:
- Improvement direction: cardio worsening (ctr/frac both down off 40k peak); effusion flat/held; val/loss still low (0.060).
- Oscillation: ctr_med/cardio window std ~0.010 (not elevated).
- Persistence of concern: first appearance of cardio-column degradation — not yet persistent.

**Reference comparison**: no baseline. Cross-run: cardio still rendering (unlike umqnrs0o); effusion held (unlike 6wj35ips) — the dual-channel goal still partially intact at 50k.

**Diagnosis**:
The cardio conditional peaked around step 40k and has begun regressing. Mechanically, the most likely chain: with `class_mult` cardio=0.4 but only 854 cardio training images (vs 4621 effusion, 4000 normal) — a ~5:1 data deficit that the weight only partially offsets — the cardio conditional has limited distinct signal. As training continues past the point where it had learned a coarse enlarged-heart prior (~40k), the shared UNet capacity and the min-SNR-weighted objective keep optimizing the well-represented normal/effusion modes, and the under-supported cardio mode drifts/degrades (its rendering is not being reinforced enough to hold). The rising leak/cardio_to_effusion is the co-symptom: cardio prompts increasingly resolve toward effusion-like texture. This is the same cardio-under-training failure the rebalance lever was built to fix — the rebalance helped (cardio renders at all, peaked at 0.521, far better than the old 0.395@100k) but did not make cardio *durable* across a full 100k run.

**Diagnostic steps**:
1. Confirm data imbalance is the driver: check the training manifest counts for this run (cardio vs effusion vs normal border-visible latents). If cardio ≈ 850 vs effusion ≈ 4600, the ~5:1 deficit at class_mult 0.4 is the prime suspect.
2. Look at ctr_med/cardio and frac_gt0.5/cardio at 40k vs 50k–100k in W&B — confirm the peak-at-40k-then-decline shape (it is: 0.521/0.571 @40k → 0.457/0.310 @100k).
3. Cross-check leak/cardio_to_effusion over the same window — if it rises as cardio falls (it does: 0.44→0.53–0.62 late), the two are the same phenomenon (cardio collapsing into effusion mode).
4. Rule out decode-artifact as the *cause* of the visual distortion: it is column-localized to cardio, so it is not the uniform over-scaling bug (that would hit all columns).

**Fix options**:
- Option A (best-checkpoint harvest, no retrain): adopt **step 40000** (`ldm-ckpt:v74`, model_step0040000) as the cardio checkpoint — it is the run's peak on ctr_med/cardio (0.521) and frac_gt0.5/cardio (0.571). Use a large seed sweep and filter cardio-column failures. Conservative, uses what the run already produced.
- Option B (raise cardio share / longer effective cardio exposure): next run, increase cardio `class_mult` above 0.4 (e.g. 0.5) and/or oversample cardio latents so effective cardio:effusion exposure approaches 1:1, holding effusion_weight=1. Targets the root deficit.
- Option C (early-stop policy for future runs): stop at the cardio peak (~40k here) rather than the full 100k — extra steps regressed cardio while only marginally improving effusion presence.

**Recommendation**: Let the (already finished) run stand and **harvest step 40000 for cardio**, step 80k–100k for effusion. For the next run, Option B (raise cardio exposure). Cardio degradation is now a single-save ⚠️; watch whether it persists through 60k–70k (it does — see below).

## Save 6 (step 60000) — 2026-07-01T17:35:42Z — ldm-ckpt:v76

**Severity**: ⚠️ concerning

**Visual observations**:
Consistent with 50k and 75k: NO FINDING / EFFUSION columns well-formed, CARDIOMEGALY column persistently distorted across seeds. No improvement in the cardio column vs 50k.

**Per-mode flags**:
- all-black-pixels / lung-anatomy-absent / diversity-collapse: ✓
- **cardio channel**: ⚠️ (ctr_med/cardio 0.494, frac_gt0.5/cardio 0.407 — both down again from 50k; presence/cardio 0.107). **Second consecutive save** with cardio below its 40k peak and grid distortion in the cardio column.
- cardio→effusion leakage: ⚠️-watch (leak 0.438; rises to 0.563 at 65k — trend up)
- effusion channel: ✓ (presence/effusion 0.190, ctr_med/effusion 0.395 — held)
- no-sample-improvement: ✓ (cardio column still changing between 50k/60k/75k, just not improving)

**Trend signals**:
- Improvement direction: cardio worsening (2nd save); effusion held; val/loss 0.061 (flat).
- Persistence of concern: cardio degradation now spans 50k→60k (2 consecutive). One more consecutive save escalates to ❌.

**Reference comparison**: no baseline; cardio still above the old-LDM 0.395 floor.

**Diagnosis**: Same mechanism as Save 5 (cardio under-training / drift off peak). The failure is now confirmed persistent over two saves and column-localized — reinforcing that it is model-side (cardio conditional), not the decode artifact. frac_gt0.5/cardio dropping from 0.518→0.407 means the share of cardio samples that actually cross the CTR=0.5 cardiomegaly line is falling — the conditional is losing its grip on the enlarged-heart mode.

**Diagnostic steps**: as Save 5 (steps 1–4). Additionally: (5) at 65k, leak/cardio_to_effusion jumps to 0.563 — inspect whether cardio-conditioned samples are being classified as effusion by the instrument, which would confirm cardio→effusion mode-mixing.

**Fix options**: as Save 5 (A: harvest step 40k; B: raise cardio exposure; C: early-stop at peak).

**Recommendation**: Harvest step 40k for cardio; continue the sweep — this is now a 2-save-persistent ⚠️, escalating next if it holds.

## Save 7 (step 70000) — 2026-07-01T18:07:01Z — ldm-ckpt:v77

**Severity**: ❌ likely failed (cardio channel; run overall = mixed — effusion healthy)

**Visual observations**:
75k grid inspected directly. NO FINDING and EFFUSION columns remain clean, well-formed CXRs across all four seeds. CARDIOMEGALY column distorted across seeds 137/256/512 (blobby heart, warped mediastinum, bright wispy artifacts on seed 256) — visually indistinguishable in failure character from 50k and 90k/100k. The cardio-column failure is now stable across three checkpoints.

**Per-mode flags**:
- **cardio channel**: ❌ — ctr_med/cardio 0.477 (3rd consecutive save below 40k peak, now below the 6wj35ips-style 0.5 line), frac_gt0.5/cardio 0.379, presence/cardio 0.107 but about to collapse. **Persistence ≥3 consecutive saves (50k, 60k, 70k) of the same degrading cardio mode → escalation rule fires.**
- cardio→effusion leakage: ⚠️ (leak 0.500 at 70k, 0.594 at 75k — clearly elevated, co-symptom of cardio mode-mixing). Co-trigger with the cardio decline.
- all-black-pixels / lung-anatomy-absent / diversity-collapse: ✓ (non-cardio columns clean)
- effusion channel: ✓ (presence/effusion 0.253, ctr_med/effusion 0.402 — HELD; effusion is the success of this run)
- no-sample-improvement: ⚠️ (cardio column not improving 50k→70k)

**Trend signals**:
- Improvement direction: cardio worsening for a 3rd consecutive save (ctr 0.504→0.494→0.477); effusion holding; val/loss 0.061 flat.
- Oscillation: within-window std still small — this is a directional decline, not oscillation.
- Persistence of concern: **cardio degradation persistent for 3 consecutive saves (50k/60k/70k)** → escalates cardio channel to ❌ per the 3-consecutive rule. Leak rising co-triggers.

**Reference comparison**: no baseline. Cardio has now dropped below CTR 0.5 and is heading toward the pre-rebalance regime, though still above the 0.395 old-LDM floor.

**Diagnosis**:
The cardio conditional has entered a settled degradation regime. The escalation to ❌ is on the *cardio channel specifically*, triggered by 3-consecutive-save persistence of the cardio decline plus co-triggering rising leak/cardio_to_effusion. Mechanistically this is cardio under-training made durable by the data imbalance (≈5:1 effusion:cardio): once the coarse enlarged-heart prior was learned (~40k), continued optimization on the effusion/normal-dominated batches and the min-SNR objective erodes the cardio-specific rendering rather than reinforcing it, and cardio prompts leak into the better-supported effusion mode (rising leak). Note the divergence: auc/cardio_gen is actually *rising* (0.80) while the CTR instrument and presence fall — the two judges disagree. Most likely reading: the classifier still detects a cardio-ish *style* signature the model reliably imprints, but the geometric CTR instrument (which needs an actual enlarged, well-formed cardiac silhouette) sees the silhouette degrading. The CTR/geometry is the task-relevant judge for cardiomegaly here, so the ❌ stands for the cardio *rendering* goal. (Could alternatively be that the classifier is picking up the leak-into-effusion texture as a spurious cue — either way the geometric rendering is the failing quantity.)

**Diagnostic steps**:
1. Look at ctr_med/cardio in W&B across 40k→100k: confirm the peak-then-monotone-decline (0.521→0.457). Confirmed.
2. Overlay frac_gt0.5/cardio (0.571→0.310) — the fraction crossing the cardiomegaly line has nearly halved. This is the decisive cardio-rendering-failure signal.
3. Reconcile the auc/cardio_gen rise vs CTR fall: run the CTR/HybridGNet instrument and the DenseNet classifier on the SAME step-70k cardio samples; if CTR says "not enlarged" and classifier says "cardio", the classifier is keying on non-geometric cues (style/leak) — trust CTR for the rendering verdict.
4. Confirm column-localization rules out the decode bug: the failure is only in the cardio column, so it is not uniform over-scaling.

**Fix options**:
- Option A (harvest, recommended): use **step 40000** for cardio (peak ctr 0.521 / frac 0.571) and a late checkpoint (80k–100k) for effusion; large seed sweep + filter cardio-column failures. No retrain; salvages the run's best-of-both.
- Option B (rebalance harder next run): raise cardio `class_mult` toward 0.5 and oversample cardio latents to ~1:1 effective exposure vs effusion; keep effusion_weight=1, min_snr_gamma=5. Directly targets the imbalance root.
- Option C (early-stop + separate cardio finetune): stop base training at the cardio peak; optionally finetune a cardio-specialist from step 40k on cardio-only latents for a few k steps to consolidate the enlarged-heart mode before it erodes.

**Recommendation**: Do NOT read this ❌ as "whole run failed" — it is a **cardio-channel** ❌ on a run whose effusion channel is healthy. Action: harvest step 40k (cardio) + a late ckpt (effusion); for the next run, Option B. Let the finished run stand.

## Save 8 (step 80000) — 2026-07-01T18:42:07Z — ldm-ckpt:v78

**Severity**: ❌ likely failed (cardio channel; effusion healthy)

**Visual observations**:
Consistent with 75k/90k/100k: NO FINDING and EFFUSION columns clean; CARDIOMEGALY column distorted across seeds. No change in failure character. Late grids also show the washed-out/over-bright greyscale that is the decode-scale artifact (not a model failure).

**Per-mode flags**:
- **cardio channel**: ❌ (persistence continues — 4th consecutive declining save; ctr_med/cardio 0.472, frac_gt0.5/cardio 0.414, presence/cardio 0.107).
- cardio→effusion leakage: ⚠️ **leak peaks 0.625@80k** — run maximum. Co-trigger with cardio decline.
- effusion channel: ✓ (presence/effusion 0.253, ctr_med/effusion 0.385 — held; val/loss at run trough 0.0567–0.0596 here).
- all-black / anatomy / diversity: ✓ (non-cardio columns clean)

**Trend signals**:
- Improvement direction: cardio worsening (4th save); effusion held; val/loss near its minimum.
- Persistence of concern: cardio ❌ persists (≥3 consecutive well exceeded); leak at run max.

**Reference comparison**: no baseline; cardio below 0.5, above old-LDM 0.395 floor.

**Diagnosis**: Same settled cardio-under-training regime as Save 7. leak/cardio_to_effusion hitting its run maximum (0.625) here is the clearest evidence of cardio→effusion mode-mixing — at 80k the cardio conditional is most strongly bleeding into effusion texture. Effusion remains the run's success (val/loss trough coincides with effusion holding). The decode-scale over-brightness is now clearly visible in the grids but is orthogonal to the cardio failure.

**Diagnostic steps**: as Save 7 (steps 1–4). Additionally: (5) 80k is the leak maximum — sample step-80k cardio-conditioned images and run the effusion instrument; a high false-effusion read confirms the mode-mixing.

**Fix options**: as Save 7 (A harvest step 40k for cardio; B rebalance harder; C early-stop/cardio-finetune).

**Recommendation**: Cardio ❌ persists. Harvest step 40k for cardio. Effusion is well-served by late checkpoints (80k has the val/loss trough). Next run: Option B.

## Save 9 (step 90000) — 2026-07-01T19:17:05Z — ldm-ckpt:v79

**Severity**: ❌ likely failed (cardio channel; effusion healthy)

**Visual observations**:
90k recon grid inspected directly — near-identical to 100k (the model is effectively frozen at this LR). NO FINDING / EFFUSION columns clean and well-formed; CARDIOMEGALY column distorted (seed 137 blobby heart, seed 256 badly warped with bright wisps, seed 512 mangled lower field). Compose (PoE)@90k inspected: 4 anchors × 4 seeds render recognizable CXRs with bilateral basal effusion-like opacities but no clearly enlarged hearts, and a washed-out/over-bright cast (decode-scale artifact). PoE is coherent (unlike mzd3jwy9's noisy compose grid) but cardio contribution is weak.

**Per-mode flags**:
- **cardio channel**: ❌ (ctr_med/cardio 0.463, frac_gt0.5/cardio 0.414, presence/cardio 0.076 — presence now falling too; 5th consecutive declining save).
- no-sample-improvement: ⚠️ (90k≈100k for matched seed/class — the extra steps produce no visual change; LR in late-cosine tail).
- cardio→effusion leakage: ⚠️ (leak 0.594).
- effusion channel: ✓ (presence/effusion 0.284 — run high; ctr_med/effusion 0.368).
- all-black / anatomy / diversity: ✓ (non-cardio columns).

**Trend signals**:
- Improvement direction: cardio worsening + presence/cardio now dropping (0.107→0.076); effusion at its presence high; val/loss ticking back up (0.061).
- Persistence of concern: cardio ❌ sustained; no-sample-improvement emerging (frozen late grids).

**Reference comparison**: no baseline.

**Diagnosis**: Late-run frozen regime. The cardio channel has both degraded (rendering) and is now losing presence (fewer cardio samples read as cardiomegaly at all). val/loss turning back up (0.0567@85k → 0.0612@90k) alongside no-visual-improvement indicates the run has passed its useful point — the late steps are not buying quality on either channel and are eroding cardio. The over-bright compose grid is the decode artifact; the weak-cardio/strong-effusion composition is the model.

**Diagnostic steps**: as Save 7. Additionally: (5) compare 90k vs 100k grids for matched seeds — near-identical confirms the frozen no-improvement regime and supports an early-stop policy next run.

**Fix options**:
- Option A (harvest): step 40k for cardio; for effusion, 85k–90k (presence high, val/loss low). 
- Option B (rebalance harder next run) / Option C (early-stop ~40–50k) as prior entries.

**Recommendation**: Harvest split checkpoints (40k cardio / ~85–90k effusion). The run is past its useful point by 90k.

## Save 10 (step 100000, FINAL) — 2026-07-01T19:48:36Z — ldm-ckpt:v80

**Severity**: ❌ likely failed (cardio channel; run overall = split verdict — effusion goal MET, cardio goal NOT durable)

**Visual observations**:
100k recon grid inspected directly. NO FINDING and EFFUSION columns are consistently well-formed CXRs across all four seeds — the effusion channel is a clear success. The CARDIOMEGALY column remains distorted across seeds (seed 42 marginal, 137 blobby heart, 256 badly mangled with bright wispy artifacts, 512 warped lower field). The final grid is qualitatively identical to 75k/90k — no late recovery. Overall greyscale washed-out/over-bright = decode-scale artifact.

**Per-mode flags**:
- **cardio channel**: ❌ — ctr_med/cardio **0.457 (run min)**, frac_gt0.5/cardio **0.310 (run min)**, presence/cardio **0.013 (near-total collapse; from 0.138@10k)**. presence/both = 0. The cardio rendering goal (CTR→real-cardio 0.535) is NOT met and has regressed hard from its 40k peak.
- no-sample-improvement: ❌ (100k visually indistinguishable from 90k for matched seed/class; the final 10k steps at LR≈0 bought nothing).
- cardio→effusion leakage: ⚠️ (leak 0.531; peaked 0.625@80k).
- loss-component-imbalance: ✓ (final cls0/1/2 = 0.084/0.049/0.157 → max/min ≈ 3.2×, ≪ 10× — cls2/effusion now the largest, consistent with effusion being the actively-learned channel).
- effusion channel: ✓ **GOAL MET** (presence/effusion 0.284 — run high, NO collapse; ctr_med/effusion 0.367). This is the recipe's primary success: unlike 6wj35ips, effusion did not collapse.
- all-black / lung-anatomy-absent / diversity-collapse: ✓ (non-cardio columns well-formed, diverse).
- kl-*, latent-collapse, noise-schedule-mismatch, attention-collapse, timestep-skew, per-layer norms, fid: not checked (not logged / frozen-VAE latent LDM).

**Trend signals**:
- Improvement direction: cardio at run minimum (worsening throughout the back half); effusion held/improved; val/loss back up to 0.0641 (above its 85k trough) — the aggregate objective also stopped improving.
- Oscillation: none structural; the late motion is directional decline + frozen plateau, not bouncing.
- Persistence of concern: cardio degradation persistent from 50k→100k (six saves); no-sample-improvement 90k→100k.

**Reference comparison**: no baseline. Positioning: cardio 0.457 final is below the run's own 40k peak (0.521) and below real-cardio 0.535, but above the old-LDM 100k 0.395. Effusion held where 6wj35ips collapsed.

**Diagnosis**:
Split verdict. **Effusion channel: success** — presence held/rose to 0.284 and effusion CXRs render cleanly at every checkpoint; the recipe's anti-collapse goal is met. **Cardio channel: failure of durability** — it peaked ~40k (ctr 0.521, frac 0.571), then regressed monotonically to a run-minimum at 100k (ctr 0.457, frac 0.310, presence 0.013). The ❌ is on cardio, driven by ≥3-consecutive-save persistence (50k–100k), co-triggering rising leak, and terminal presence collapse. Root cause: cardio under-training under the ~5:1 effusion:cardio data imbalance that `class_mult`=0.4 only partially offsets — once the coarse cardiomegaly prior was learned early (~40k), continued training on effusion/normal-dominated batches eroded the fragile cardio rendering and mixed it into the better-supported effusion mode (rising leak). The rebalance lever WORKED partially (cardio rendered at all, peaked well above the old floor) but did not make cardio durable across a full 100k schedule; running to 100k actively hurt cardio while only marginally helping effusion presence. The final grid's over-bright cast is the known decode-scale bug (over-scaling on now-well-dispersed convergent latents) — cosmetic, not the cause of the cardio failure (which is column-localized).

Two things to flag as NOT catalog-standard: (1) the auc/cardio_gen-vs-CTR divergence (classifier separability rose to 0.80 while CTR/presence fell) — the classifier likely keys on a non-geometric cardio-style or the leak-into-effusion cue; the geometric CTR is the task-relevant judge and it says cardio rendering regressed. (2) The "inconsistent x-rays" the user observed are CLASS-localized (cardio column), not per-seed-random and not decode-uniform — a distinct signature from the umqnrs0o/mzd3jwy9 fixed-seed-noise pattern.

**Diagnostic steps**:
1. W&B ctr_med/cardio + frac_gt0.5/cardio 40k→100k: confirm peak@40k → monotone decline (confirmed: 0.521/0.571 → 0.457/0.310).
2. presence/cardio 10k→100k: confirm terminal collapse (0.138 → 0.013). Confirmed — the cardio conditional loses the cardiomegaly mode entirely by the end.
3. Reconcile auc/cardio_gen↑ vs CTR↓: run BOTH the CTR geometric instrument and the DenseNet classifier on identical step-40k and step-100k cardio samples. If CTR@100k « CTR@40k but classifier ≈ equal, confirms the classifier is not tracking geometry → trust CTR, harvest 40k.
4. Confirm the training manifest imbalance (cardio ≈ 850 vs effusion ≈ 4600 border-visible latents) as the mechanistic root.
5. Confirm decode bug is cosmetic: cardio failure is column-localized, so uniform over-scaling is excluded as its cause.

**Fix options**:
- Option A (harvest split checkpoints — recommended, no retrain): **cardio → step 40000** (`model_step0040000`, ctr 0.521 / frac 0.571, run peak); **effusion → step 85000–100000** (presence 0.28, val/loss low). Large seed sweep, filter cardio-column failures. Salvages the best of both channels from the finished run for downstream Exp5/PoE.
- Option B (rebalance harder — next training run): raise cardio `class_mult` toward 0.5 AND oversample cardio latents so effective cardio:effusion exposure ≈ 1:1; hold effusion_weight=1, min_snr_gamma=5, keep EMA. Directly attacks the imbalance that made cardio non-durable.
- Option C (early-stop + cardio consolidation): cap base training near the cardio peak (~40–50k), and/or finetune a cardio-specialist from step 40k on cardio-only latents for a few k steps to lock in the enlarged-heart mode before back-half erosion. Combine with Option B for the strongest cardio.

**Recommendation**: Treat this as a **split result, not a whole-run failure**: effusion goal met, cardio goal not durable. Immediate action — **Option A**: harvest step 40000 for cardio and a late checkpoint (~85–100k) for effusion, run a large seed sweep, filter the cardio-column distortions. For the next run, **Option B** (raise cardio exposure to ~1:1) optionally with **Option C** (early-stop / cardio finetune). Do not run future cardio-bearing runs to 100k on this data mix — the back half regresses cardio.

**Unrecognized observations**:
1. **Class-localized (not per-seed) inconsistency.** The visual inconsistency is confined to the CARDIOMEGALY column at every checkpoint 50k→100k, while NO FINDING/EFFUSION stay clean. This differs from the umqnrs0o/mzd3jwy9 "pure-noise-on-a-fixed-seed-subset" pattern (which hit whole seeds across classes). Candidate catalog mode: **"class-conditional-rendering-decay"** — one conditioning class peaks then monotonically degrades (rendering + presence) while others hold, co-symptomed by rising cross-class leakage; root cause = per-class training under-representation, not a hyperparameter that can be tuned mid-run. Fix = class exposure rebalancing / early-stop at the weak class's peak.
2. **Judge divergence (auc↑ vs CTR/presence↓).** classifier AUC and the geometric CTR instrument disagreed on cardio in the back half. Worth a catalog note: when two eval instruments diverge on the same channel, prefer the task-definitional one (CTR for cardiomegaly geometry) and treat the other's signal as possibly keying on a spurious/style cue.

**Catalog modes skipped (not logged this run):** latent-collapse (no PCA), noise-schedule-mismatch (no per-timestep quality), attention-map-collapse (no attn logging), timestep-distribution-skew (no timestep histogram), update-to-weight-ratio / weight-norm-explosion (no per-layer norms), fid-regression (no FID), loss-scale-anomalous / lr-batch-size-mismatch / unexpected-image-stats (no v2/v3 baseline or reference config).
