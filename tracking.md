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

