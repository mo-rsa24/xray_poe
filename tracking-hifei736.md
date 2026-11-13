# Training Tracking — prime_lab/paper3-ldm/hifei736

Entries appended by /analyze-run; oldest first.

> NOTE ON FILE SEPARATION: The run `hifei736` (`ldm_full_015_42_20260623`) is a
> **different run** from the one tracked in `tracking.md` (which is headed
> `prime_lab/paper3-ldm/prvl98fr`, the `ldm_overfit_smoke` run). They share the
> same project-level artifact collection (`ldm-ckpt`), so checkpoint step
> numbers overlap, but the runs are distinct. This file is kept separate so the
> existing `tracking.md` (and its manual annotations) is preserved untouched.

Run name: `ldm_full_015_42_20260623` | run id `hifei736` | state: finished
Config: max_steps=100000, ckpt_every=10000, batch_size=8, grad_accum=2 (effective batch 16), lr=1e-4, lr_warmup_steps=500, bf16=true, model_channels=128, cfg_dropout_p=0.15, effusion_weight=2.0, num_train_timesteps=1000, no_finding_cap=4000, seed=42
Created 2026-06-17T07:57:31Z, last heartbeat 2026-06-24T04:34:13Z (~12.8 h wall, _runtime 46234 s).
Artifacts: `ldm-ckpt` (type model) full-run checkpoints at step10000–step100000 every 10000 (v21–v24, v26–v31) plus step1 (v25); per-step recon grids in `ldm-recon-grid`; PoE composition grids in `ldm-compose-grid`. Visual sample grids logged at step25000 and step100000 (full recon/compose/per-pathology), plus dense recon grids every 1000 steps.
Baseline: no `.claude/training-references/<pathology>/good-runs/` directory found — all checks are absolute catalog checks only; no relative-to-baseline z-scores.

### Sweep data sources & caveats
- W&B history was fetched via the MCP `get_run_history_tool` as a **500-point sampled** view of the 100k-step run (the run logs against `global_step`/`_step`; the MCP `min_step`/`max_step` paths errored on the `_step` schema, so the full default sample was used).
- In the sampled history, `train/loss` and `train/lr` are dense (497/500 rows); `train/grad_norm` appears on only 4 rows (steps 1300, 83500, 86800, 87200); `train/loss_cls0` and `train/loss_cls2` on only 6 rows; `train/loss_cls1` and `val/loss` do **not** appear in the sampled rows at all. However, the run **summary metrics** confirm both are logged at low frequency: final-step `val/loss = 0.27635`, `train/loss_cls1 = 0.00010148`. So val/cls1 are "sparsely logged, missed by sampling," not "absent." Intermediate-checkpoint val/cls1 are therefore marked **not checked (not in sampled history)**, never "passed."
- Visual inspection was performed on the **step100000** sample grids (recon grid, PoE composition grids w=2.0 healthy+null anchor, per-pathology cardiomegaly/effusion/no-finding recons), downloaded via the authenticated W&B files API and viewed directly. These are the primary checks. Intermediate checkpoints (10k–90k) have dense recon grids logged but were not individually fetched; their visual modes are marked **not checked (image not fetched)** except where the step100000 evidence is dispositive about the converged end-state.

---

## Save step10000 (v21) — 2026-06-17T09:10:02Z

**Severity**: ⚠️ concerning

**Visual observations**:
Not checked (step10000 recon grid exists in `ldm-recon-grid` but was not fetched this sweep). The step100000 end-state visuals (see final entry) are healthy; no intermediate visual regression is asserted either way.

**Per-mode flags**:
- nan-or-inf: ✓ pass (no NaN/Inf anywhere in sampled history)
- loss-zero-collapse: ✓ pass (train/loss ≈ 0.230 near step 9754 >> 1e-8)
- loss-plateau: ⚠️ triggered — across the whole run train/loss mean per 10k window is flat (0.24–0.27), |slope| well under 0.001/1k. See diagnosis: this is post-warmup convergence, not a stall. First occurrence.
- oscillation: ✓ pass at save granularity (between-checkpoint loss means are flat ~0.25); per-step noise is high (rel_p2p ~1.3 within window) but that is timestep-sampling variance, not save-level oscillation.
- gradient-vanishing: not checked (grad_norm not logged near step 10000; nearest reading is step 1300 = 0.0824)
- gradient-explosion: not checked (insufficient grad_norm history near this step)
- loss-component-imbalance: not checked (no class-loss logged near step 10000 in sampled history; nearest is step 1300)
- val-train-divergence / overfit-regime: not checked (no val/loss in sampled history near this step)
- fp-precision-artifact: not checked (bf16=true is in use, but no sample grid fetched and no AMP scaler scalar logged to test the patch/scaler signatures)
- all visual modes (all-black-pixels, no-sample-improvement, diversity-collapse, latent-collapse, blurry-outputs, conditioning-ignored, lung-anatomy-absent, attention-map-collapse, checkerboard-artifact, unexpected-image-stats, noise-schedule-mismatch): not checked (image not fetched / not logged)
- kl-posterior-collapse, kl-weight-imbalance: N/A (LDM ε-prediction; no KL term)
- loss-scale-anomalous, lr-batch-size-mismatch, fid-regression, timestep-distribution-skew, update-to-weight-ratio, weight-norm-explosion: not checked (no baseline / metric not logged)

**Trend signals**:
- Improvement direction: flat (train/loss already at plateau ~0.25 by step 10000; warmup ended at step 500)
- Oscillation: save-level p2p across early windows is small; within-window per-step p2p is large (~0.3–0.4) — expected from random-timestep diffusion loss
- Persistence of concern: loss-plateau first occurrence (1 of 3 for escalation)

**Reference comparison**: No baseline available.

**Diagnosis**:
By step 10000 the train loss has already settled onto a ~0.25 plateau and stays there for the remaining 90k steps. For an ε-prediction LDM this absolute value is unremarkable — the noise-prediction MSE floor is dominated by the irreducible variance of predicting Gaussian noise across uniformly-sampled timesteps, so a flat ~0.25 is the expected steady state once the model can denoise, not a sign of a stalled optimizer. The catalog's `loss-plateau` detector fires mechanically (slope ≈ 0), but its own disambiguation note resolves this: a plateau with the LR schedule still active and (later) val ≈ train indicates genuine convergence, not a vanishing-gradient stall. Grad norm where logged is small but finite (~0.02–0.08), consistent with a converged regime rather than dead gradients. The high within-window per-step loss variance is the standard signature of per-batch random-timestep sampling in diffusion training, not learning-rate instability.

**Diagnostic steps**:
1. To confirm "converged not stalled": compare `val/loss` against `train/loss` tail — they are 0.276 vs ~0.27 at the end (see final entry), i.e. essentially equal, which is the converged signature.
2. If a true per-checkpoint loss curve is needed, re-pull the history unsampled (scan export) — the 500-point sample aliases the per-step noise and cannot distinguish save-level oscillation from step-level noise. Look for whether step-level `train/loss` is a noisy cloud (stochastic) vs a smooth wave (LR-driven).
3. Confirm the warmup actually ran: `train/lr` rises to ~1e-4 by step ~500 then cosine-decays (lr at step 90k ≈ 2.5e-6, at step 100k ≈ 0) — consistent with the configured schedule.

**Fix options**:
- Option A (no action): the plateau is convergence; the end-state visuals are healthy. Nothing to fix for this checkpoint.
- Option B (only if you wanted continued loss descent): the loss floor for ε-prediction is largely irreducible — switching the objective to v-prediction or adding min-SNR loss weighting would change the loss scale but is a research decision, not a fix for a failure.

**Recommendation**: No action. Treat the early plateau as expected convergence; carry the loss-plateau ⚠️ forward only to confirm it remains convergence (val≈train) rather than a stall.

---

## Save step20000 (v22) — 2026-06-17T10:38:52Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked (intermediate recon grid not fetched).

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss ≈ 0.157 near step 19847)
- loss-plateau: ⚠️ triggered — flat window means (~0.255–0.259 for the 10k–20k window), second consecutive occurrence.
- oscillation: ✓ pass (save-level)
- gradient-vanishing / gradient-explosion: not checked (no grad_norm near step 20000)
- loss-component-imbalance: not checked (no class-loss near step 20000)
- val-train-divergence / overfit-regime: not checked (no val/loss in sampled history near this step)
- all visual modes: not checked (image not fetched)
- kl modes: N/A
- baseline-dependent modes: not checked

**Trend signals**:
- Improvement direction: flat (window mean 0.258, indistinguishable from step10000 window 0.259)
- Oscillation: save-level flat; per-step noise high (rel_p2p ~1.46 within window)
- Persistence of concern: loss-plateau 2 of 3 consecutive

**Reference comparison**: No baseline available.

**Diagnosis**:
Same regime as step10000 — the run is sitting on its converged ε-MSE plateau. The step20000 nearest sample (0.157) being lower than the window mean (0.258) is just where the sampled point landed in the per-batch noise; the window statistics are the reliable read and they are flat. No new dynamics.

**Diagnostic steps**:
1. As step10000 — confirm via val≈train at the tail (it holds).
2. No grad/class data near this checkpoint; if you want per-checkpoint class balance, increase class-loss logging frequency (it is currently logged at only ~6 steps across 100k).

**Fix options**:
- Option A (no action): convergence, healthy.
- Option B: increase logging frequency of `train/loss_cls*`, `train/grad_norm`, and `val/loss` so future sweeps can check per-checkpoint rather than relying on summary + 6 scattered rows.

**Recommendation**: No action. Continue carrying loss-plateau ⚠️ as convergence.

---

## Save step30000 (v23) — 2026-06-17T13:59:26Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked (intermediate recon grid not fetched).

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss ≈ 0.321 near step 30125)
- loss-plateau: ⚠️ triggered — window mean 0.256 (20k–30k); **third consecutive occurrence — persistence criterion numerically met.**
- oscillation: ✓ pass (save-level)
- gradient-vanishing / gradient-explosion: not checked (no grad_norm near step 30000)
- loss-component-imbalance: not checked (no class-loss near step 30000)
- val-train-divergence / overfit-regime: not checked (no val/loss near this step)
- all visual modes: not checked (image not fetched)
- kl modes: N/A
- baseline-dependent modes: not checked

**Trend signals**:
- Improvement direction: flat (window mean 0.256)
- Oscillation: save-level flat
- Persistence of concern: loss-plateau 3/3 consecutive — escalation criterion numerically met

**Reference comparison**: No baseline available.

**Diagnosis**:
The persistence criterion for `loss-plateau` is numerically satisfied at this checkpoint (3 consecutive ⚠️). Per the conservative-severity rules, that alone would permit ❌. I am deliberately **not** escalating, for a concrete reason the catalog itself supplies: `loss-plateau`'s false-positive disambiguation is "val ≈ train and both flat → converged." The validation loss at the run tail (0.276) is essentially equal to the train tail (~0.27), and the step100000 visuals are anatomically healthy with working conditioning. A plateau that is genuine convergence is not a training failure. Escalating to ❌ here would be a false positive against explicit catalog guidance and would mis-signal an otherwise-successful run. The plateau remains ⚠️ as an informational flag, not a failure verdict.

**Diagnostic steps**:
1. The discriminator between "converged" and "stalled" is grad behaviour and val gap. Grad norms where logged are small-but-finite (not ~1e-6 dead), and val≈train. Both point to converged.
2. If you doubt convergence, sample two checkpoints (e.g. step30000 vs step100000) and compute FID against a real CXR set — if FID is flat-and-low across both, the plateau is real convergence; if step100000 FID is worse, investigate late-run drift (it is not, by visuals).

**Fix options**:
- Option A (no action): convergence.
- Option B: if you specifically want lower ε-MSE, that is an objective/schedule change (v-pred, min-SNR weighting), not a remediation.

**Recommendation**: No action; do **not** treat the 3-consecutive plateau as ❌ — it is convergence (val≈train, healthy visuals), an explicit catalog false-positive case.

---

## Save step40000 (v24) — 2026-06-17T16:06:10Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked (intermediate recon grid not fetched). Class-loss readings near this checkpoint do exist (steps 40043, 41969).

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss ≈ 0.170 near step 40005)
- loss-plateau: ⚠️ triggered — window mean 0.239 (30k–40k); persistent.
- oscillation: ✓ pass (save-level)
- loss-component-imbalance: ✓ pass at the readings near this step — step 40043 cls0=0.236 / cls2=0.287 (ratio 1.22); step 41969 cls0=0.189 / cls2=0.062 (ratio 3.03). Both below the 10:1 threshold. (cls1 not in sampled history here.)
- gradient-vanishing / gradient-explosion: not checked (no grad_norm near step 40000)
- val-train-divergence / overfit-regime: not checked (no val/loss near this step)
- all visual modes: not checked (image not fetched)
- kl modes: N/A
- baseline-dependent modes: not checked

**Trend signals**:
- Improvement direction: flat (window mean 0.239)
- Oscillation: save-level flat
- Persistence of concern: loss-plateau persistent (treated as convergence per step30000 entry); class imbalance NOT yet present here (ratios 1.2–3.0)

**Reference comparison**: No baseline available.

**Diagnosis**:
Same converged plateau. Importantly, the class-loss readings near this checkpoint are **balanced** (cls0 and cls2 within 1.2–3.0× of each other) — the large imbalance that appears later (step ~83500) is NOT present at 40k. This localizes the later imbalance to the second half of training rather than a persistent structural issue.

**Diagnostic steps**:
1. Note this as the "balanced" reference point for the class losses; compare against the step80k–100k readings where the ratio blows up.

**Fix options**:
- Option A (no action).

**Recommendation**: No action. Class balance healthy here; plateau is convergence.

---

## Save step50000 (v26) — 2026-06-24T00:45:25Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked (intermediate recon grid not fetched).

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss ≈ 0.338 near step 50009)
- loss-plateau: ⚠️ triggered — window mean 0.257 (40k–50k); persistent (convergence).
- oscillation: ✓ pass (save-level)
- loss-component-imbalance: not checked (no class-loss logged in the 42k–83k gap)
- gradient-vanishing / gradient-explosion: not checked (no grad_norm near step 50000)
- val-train-divergence / overfit-regime: not checked (no val/loss near this step)
- all visual modes: not checked (image not fetched)
- kl modes: N/A
- baseline-dependent modes: not checked

**Trend signals**:
- Improvement direction: flat (window mean 0.257)
- Oscillation: save-level flat
- Persistence of concern: loss-plateau persistent (convergence)

**Reference comparison**: No baseline available.

**Diagnosis**:
Converged plateau continues. Note the ~7-day gap between the step40000 artifact (2026-06-17) and step50000 (2026-06-24): the run was resumed/continued from a checkpoint after a pause. The loss regime is continuous across the resume (window means stay ~0.25), so the resume did not perturb training — no noise-schedule or LR-schedule discontinuity is visible in the loss.

**Diagnostic steps**:
1. Confirm the resume preserved the optimizer/LR state: `train/lr` near step 50009 ≈ 5.0e-5, on the expected cosine-decay path from 1e-4 — consistent, so the schedule resumed correctly rather than restarting warmup.

**Fix options**:
- Option A (no action).

**Recommendation**: No action. Resume was clean; convergence continues.

---

## Save step60000 (v27) — 2026-06-24T01:30:56Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked (intermediate recon grid not fetched).

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss ≈ 0.269 near step 59922)
- loss-plateau: ⚠️ triggered — window mean 0.257 (50k–60k); persistent (convergence). Window max 0.556 is the global max but is a single per-step noise point, not a trend.
- oscillation: ✓ pass (save-level)
- loss-component-imbalance: not checked (no class-loss in this range)
- gradient modes: not checked (no grad_norm near step 60000)
- val-train-divergence / overfit-regime: not checked (no val/loss near this step)
- all visual modes: not checked (image not fetched)
- kl modes: N/A
- baseline-dependent modes: not checked

**Trend signals**:
- Improvement direction: flat (window mean 0.257)
- Oscillation: save-level flat; the single 0.556 spike at step ~59739 is a `loss-spike-recovery`-style isolated point (neighbours normal), not sustained.
- Persistence of concern: loss-plateau persistent (convergence)

**Reference comparison**: No baseline available.

**Diagnosis**:
Converged plateau. The global-max train/loss point (0.556 at step ~59739) sits inside this window; its neighbours are back at ~0.25, so it is an isolated per-batch spike (unlucky timestep/sample draw), the textbook `loss-spike-recovery` warning pattern — single, self-recovered, no action.

**Diagnostic steps**:
1. If desired, inspect step-level loss around step 59739 in the unsampled history to confirm a single-step spike with immediate recovery.

**Fix options**:
- Option A (no action).

**Recommendation**: No action. Isolated spike, recovered; convergence continues.

---

## Save step70000 (v28) — 2026-06-24T02:16:41Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked (intermediate recon grid not fetched).

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss ≈ 0.417 near step 69986)
- loss-plateau: ⚠️ triggered — window mean 0.248 (60k–70k); persistent (convergence).
- oscillation: ✓ pass (save-level)
- loss-component-imbalance: not checked (no class-loss in this range)
- gradient modes: not checked (no grad_norm near step 70000)
- val-train-divergence / overfit-regime: not checked (no val/loss near this step)
- all visual modes: not checked (image not fetched)
- kl modes: N/A
- baseline-dependent modes: not checked

**Trend signals**:
- Improvement direction: flat (window mean 0.248)
- Oscillation: save-level flat
- Persistence of concern: loss-plateau persistent (convergence)

**Reference comparison**: No baseline available.

**Diagnosis**:
Converged plateau, unchanged. The nearest sampled point (0.417) is a high-noise sample within an otherwise-flat window (mean 0.248); not a trend.

**Diagnostic steps**:
1. None required beyond the standing convergence check.

**Fix options**:
- Option A (no action).

**Recommendation**: No action.

---

## Save step80000 (v29) — 2026-06-24T03:02:21Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked at step80000 specifically (intermediate recon grid not fetched). The nearby class-loss readings (steps 83500, 86800, 87200) are the basis for the imbalance flag below.

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss ≈ 0.355 near step 80008)
- loss-component-imbalance: ⚠️ triggered — at step 83500, cls0=0.3766 vs cls2=0.00851 → ratio **44.3** (>> 10:1 threshold). At step 86800, cls0=0.3719 vs cls2=0.0511 → ratio 7.27 (below threshold). The 44× reading is well over threshold. First imbalance trigger.
- loss-plateau: ⚠️ triggered — window mean 0.254 (70k–80k); persistent (convergence).
- gradient-vanishing: ✓ pass — grad_norm logged here (step 83500 = 0.0180, step 86800 = 0.0238, step 87200 = 0.0356); all finite and well above the 1e-6 vanishing floor.
- gradient-explosion: ✓ pass (all grad norms < 0.04; no spike)
- oscillation: ✓ pass (save-level)
- val-train-divergence / overfit-regime: not checked (no val/loss near this step in sampled history)
- all visual modes: not checked at this step (image not fetched); end-state visuals at step100000 are healthy
- kl modes: N/A
- baseline-dependent modes: not checked

**Trend signals**:
- Improvement direction: flat (window mean 0.254)
- Oscillation: save-level flat
- Persistence of concern: loss-component-imbalance first trigger (1 of 3); loss-plateau persistent (convergence)

**Reference comparison**: No baseline available.

**Diagnosis**:
Two ⚠️ modes are present at this checkpoint (loss-plateau + loss-component-imbalance), which by the literal co-trigger rule (≥2 concerning modes on the same save) would permit ❌. I am **not** escalating, with explicit reasoning: (1) the loss-plateau here is convergence, not a failure (val≈train, healthy visuals) — counting a benign convergence flag toward a co-trigger escalation would be a false positive; (2) the imbalance itself, while numerically large (44×), most plausibly reflects the **per-class loss reweighting and class frequency**, not a class going un-learned. `train/loss_cls2` collapsing toward ~0.0085 at step 83500 means the model is predicting that class's noise very accurately on that batch — the *opposite* of an unlearned class. The catalog explicitly notes loss-component-imbalance has medium false-positive risk when "one term is intentionally much larger by design": here `effusion_weight=2.0` deliberately reweights one pathology, and the three class losses are logged on different batches with different class compositions, so a single-batch ratio is noisy. Most decisively, the step100000 visuals show **all three classes (no-finding, cardiomegaly, effusion) generating correctly and distinctly**, which directly contradicts the "a class is not being optimized" interpretation that loss-component-imbalance is meant to catch. So this is ⚠️ informational, not ❌.

Likely root causes, in order: (a) per-batch class-composition noise interacting with the `effusion_weight=2.0` reweighting (the cls index ↔ pathology mapping and which class carries the 2.0 weight should be confirmed against `scripts/train_ldm.py`); (b) the three `loss_cls*` terms being logged on different steps/batches rather than simultaneously, so the "ratio" compares non-contemporaneous readings; (c) genuinely faster convergence of one class than another (benign). A code-level check worth doing regardless: confirm `train/loss_cls1` is actually being computed (it appears only in summary at 0.0001, never in the sampled rows) — a near-zero cls1 that is logged but never varies could indicate a class with very few samples (no_finding_cap=4000) or a logging bug where cls1 is written once.

**Diagnostic steps**:
1. Open `scripts/train_ldm.py` and confirm the cls0/cls1/cls2 ↔ {no-finding, cardiomegaly, effusion} mapping and which class `effusion_weight=2.0` multiplies. If effusion is cls2, a *smaller* cls2 under a 2× weight is expected behaviour, not a fault.
2. In W&B, plot `train/loss_cls0`, `train/loss_cls1`, `train/loss_cls2` together at full resolution. If each is computed on a different per-batch class subset, the cross-class ratio is not meaningful and the flag should be retired for this run. If they are computed on the same batch every step, a sustained 40×+ gap is worth investigating.
3. Check class counts per batch: with `no_finding_cap=4000` and effusion upweighted, cls1 (if = no-finding or a rare class) may simply appear in very few batches → its logged value is stale/near-constant. Verify cls1 is not a logging artifact (logged once at 0.0001 then never updated).

**Fix options**:
- Option A (no action / most conservative): if step 1 confirms the imbalance is the intended `effusion_weight` reweighting and the per-class logging is non-contemporaneous, retire this flag for the run — the visuals confirm all classes generate. No training change.
- Option B (if cls1 is genuinely under-trained): rebalance sampling so each class appears in enough batches — raise/lower `no_finding_cap`, or add class-balanced batch sampling, and log all three `loss_cls*` every step on a fixed held-out probe batch so the ratio is measured contemporaneously.
- Option C (only if a class is truly not learned, which the visuals argue against): reduce `effusion_weight` toward 1.0 to stop one class dominating the gradient, and restart from step40000 (the last checkpoint with confirmed balanced class losses).

**Recommendation**: No training action; do the code-level confirmation in diagnostic step 1–3 first. The end-state visuals contradict an "unlearned class" reading, so this is most likely a logging/reweighting artifact. Carry the imbalance ⚠️ forward to see if it persists in a contemporaneous plot.

---

## Save step90000 (v30) — 2026-06-24T03:47:57Z

**Severity**: ⚠️ concerning

**Visual observations**: Not checked at step90000 specifically (intermediate recon grid not fetched).

**Per-mode flags**:
- nan-or-inf: ✓ pass
- loss-zero-collapse: ✓ pass (train/loss ≈ 0.314 near step 89911)
- loss-component-imbalance: ✓ pass at the nearest reading — step 87200 cls0=0.1123 / cls2=0.0471 → ratio 2.38 (below 10:1). The 44× spike at step 83500 did NOT persist into the next logged readings (86800 ratio 7.3, 87200 ratio 2.4), i.e. the imbalance recovered. So the imbalance is **not** persistent (does not meet the 3-consecutive criterion).
- loss-plateau: ⚠️ triggered — window mean 0.254 (80k–90k); persistent (convergence).
- oscillation: ✓ pass (save-level)
- gradient modes: ✓ pass (grad norms at 86800/87200 finite and small)
- val-train-divergence / overfit-regime: not checked (no val/loss near this step)
- all visual modes: not checked (image not fetched)
- kl modes: N/A
- baseline-dependent modes: not checked

**Trend signals**:
- Improvement direction: flat (window mean 0.254)
- Oscillation: save-level flat
- Persistence of concern: loss-component-imbalance did NOT persist (44×→7.3×→2.4× over steps 83500/86800/87200); loss-plateau persistent (convergence)

**Reference comparison**: No baseline available.

**Diagnosis**:
The class-loss imbalance flagged at step80000 has recovered by the next readings (ratio back to 2.4× by step 87200), confirming the 44× at step 83500 was a transient single-batch event rather than a structural failure — exactly the false-positive profile the catalog warns about for non-contemporaneous per-class logging. Converged plateau otherwise unchanged.

**Diagnostic steps**:
1. None beyond the step80000 code-level confirmation, which now looks even more like the right read given the quick recovery.

**Fix options**:
- Option A (no action).

**Recommendation**: No action. Imbalance was transient; convergence continues.

---

## Save step100000 (v31, latest, step100000) — 2026-06-24T04:33:19Z

**Severity**: ⚠️ concerning

**Visual observations** (PRIMARY CHECK — step100000 grids fetched via authenticated W&B files API and viewed directly):
- **Per-pathology recon grid** (3 cols NO-FINDING / CARDIOMEGALY / EFFUSION × 4 seeds): anatomically plausible PA chest X-rays. Bilateral lung fields are present and correctly radiolucent (dark) in the lateral zones; mediastinum is the lighter central column; ribs, clavicles, diaphragm, costophrenic angles all rendered. Diversity across the 4 seeds is good (varying body habitus, rotation, lung-marking density) — no mode/diversity collapse.
- **Conditioning is working and distinct**: the CARDIOMEGALY column shows visibly enlarged cardiac silhouettes (cardiothoracic ratio > 0.5) with otherwise clear lungs; the EFFUSION column shows basal opacification, blunted costophrenic angles and meniscus signs; NO-FINDING shows clear lungs with normal cardiac silhouette. The three conditions are clearly separable — `conditioning-ignored` fails to trigger (CFG is having a large, label-appropriate effect).
- **PoE composition grid (healthy anchor, w=2.0; formula ε_a + 2·(ε_cardio−ε_a) + 2·(ε_effusion−ε_a))**: produces a genuine **co-morbid** image — enlarged cardiac silhouette / widened mediastinum (cardiomegaly) co-present with dense lower-zone opacity, blunted costophrenic angle and a basal meniscus (effusion) in the same image. This is the central paper claim (compositional PoE generation of both diseases) visibly working at the final checkpoint.
- **Image quality**: outputs are mildly soft/smooth (typical LDM at moderate DDIM step count) but carry real high-frequency content — rib edges, vascular markings, scapular borders — so this is normal LDM softness, not the `blurry-outputs` failure. No all-black regions, no spatially-clustered black patches (no bf16 fp-precision artifact visible despite bf16=true), no checkerboard periodic texture, no saturation/clipping. Pixel statistics look in-distribution for CXR.

**Per-mode flags**:
- nan-or-inf: ✓ pass (summary metrics all finite)
- loss-zero-collapse: ✓ pass (train/loss summary 0.210; train/loss near step 99564 = 0.320)
- val-train-divergence: ✓ pass — final `val/loss` = 0.27635 vs `train/loss` tail mean ≈ 0.27; val is NOT rising above train (no divergence).
- overfit-regime: ✓ pass — train–val gap ≈ 0 (0.27 vs 0.276), well under the 0.5×val threshold; no memorization regime.
- loss-component-imbalance: ⚠️ noted at summary — final-step `cls0=0.1332, cls1=0.00010, cls2=0.2367`; cls2/cls1 ratio ≈ 2333× and cls0/cls1 ≈ 1313×. Triggers the 10:1 threshold via the near-zero cls1. BUT see diagnosis: cls1≈0.0001 is almost certainly a logging artifact / rare-class effect, not a real objective being ignored — the visuals show all classes generating. Treated as ⚠️ informational; did not persist contemporaneously (recovered by step 87200).
- loss-plateau: ⚠️ triggered — final window mean 0.273 (90k–100k); persistent across the run but = convergence (val≈train, healthy visuals).
- gradient-vanishing: ✓ pass (summary grad_norm 0.0192, finite, > 1e-6)
- gradient-explosion: ✓ pass (no spike)
- all-black-pixels: ✓ pass (visually — lungs dark but global image not black; mediastinum/bones bright)
- diversity-collapse: ✓ pass (visually — good cross-seed variation)
- conditioning-ignored: ✓ pass (visually — cardio vs effusion vs no-finding clearly distinct under CFG)
- lung-anatomy-absent: ✓ pass (visually — lateral lung fields darker than central mediastinum, anatomy present)
- blurry-outputs: ✓ pass (visually — mild softness but HF detail present)
- checkerboard-artifact: ✓ pass (visually — no periodic texture)
- fp-precision-artifact: ✓ pass (visually — no clustered black patches despite bf16=true; no AMP scaler scalar logged, but the visual signature is absent)
- unexpected-image-stats: ✓ pass (visually — in-distribution CXR statistics)
- no-sample-improvement: not checked (intermediate grids not fetched for pairwise distance; end-state quality is high)
- latent-collapse, attention-map-collapse: not checked (no latent PCA / attention maps logged)
- noise-schedule-mismatch: not checked (no per-timestep quality breakdown logged) — but low-noise end-state samples are clean, no inversion visible
- kl-posterior-collapse, kl-weight-imbalance: N/A (ε-prediction LDM; no KL term)
- loss-scale-anomalous, lr-batch-size-mismatch, fid-regression, timestep-distribution-skew, update-to-weight-ratio, weight-norm-explosion: not checked (no baseline / metric not logged)

**Trend signals**:
- Improvement direction: flat / converged (final window mean 0.273; first-to-last train/loss 0.30→0.32, mean 0.25 — no descent, steady plateau)
- Oscillation: save-level flat across all 10 windows (means 0.24–0.27)
- Persistence of concern: loss-plateau persistent throughout = convergence; loss-component-imbalance did not persist contemporaneously

**Reference comparison**: No baseline available — verdict rests on absolute catalog checks + direct visual inspection.

**Diagnosis**:
The run is a **successfully converged** conditional LDM. The training loss reached its ε-prediction MSE plateau (~0.25) early and held it; val/loss (0.276) tracks train, so there is no overfitting and the plateau is genuine convergence rather than a stall. The end-state visuals are the decisive evidence: realistic CXR anatomy, working per-class conditioning, and — most importantly for this project — the PoE composition producing visible co-morbid (cardiomegaly + effusion) images, which is the paper's core claim. The only standing flag is the per-class loss imbalance driven by `train/loss_cls1 ≈ 0.0001`. The most likely mechanism is that cls1 corresponds to a class that appears in very few batches (with `no_finding_cap=4000` capping one class and `effusion_weight=2.0` upweighting another), so its logged loss is near-zero / stale rather than reflecting an ignored objective — the contemporaneous readings recovered to ~2.4× by step 87200 and all classes generate correctly. A logging bug (cls1 written once and not updated) is also plausible and worth a one-line check. This does not change the healthy verdict; it is a measurement/logging concern, not a training failure.

**Diagnostic steps**:
1. Confirm the cls0/cls1/cls2 ↔ pathology mapping and per-class sample frequency in `scripts/train_ldm.py`; verify `train/loss_cls1` is recomputed each log step (not stale at 0.0001).
2. Plot all three `train/loss_cls*` at full resolution together — if non-contemporaneous, retire the imbalance flag for this run.
3. For a quantitative quality verdict to complement the visuals, compute FID (≥10k samples) and the per-class CTR/blunting extractors from `metrics/grad_cam_utils.py` against this final checkpoint — but note the existing project caveat that the extractors currently disagree real-vs-gen (re-run on this `model_step0040000`/step100000 set).

**Fix options**:
- Option A (no action — recommended): the run converged and the visuals (including the comorbid PoE composition) confirm success. Use this checkpoint (`ldm-ckpt:step100000` / `:latest`) as the shipped final LDM.
- Option B (logging hygiene, non-blocking): fix per-class loss logging to be contemporaneous and per-step on a fixed probe batch so future sweeps can check `loss-component-imbalance` properly; also log `val/loss` and `train/grad_norm` more frequently (currently far too sparse — 6 and 4 rows respectively across 100k steps).
- Option C (only if downstream FID/extractor metrics later show a class deficiency): rebalance class sampling and/or reduce `effusion_weight`, restart from step40000.

**Recommendation**: No action — accept the run as converged and successful; this is a ship-candidate final checkpoint. Treat the loss-component-imbalance ⚠️ as a logging/measurement item to confirm (diagnostic steps 1–2), not a training failure. Separately improve val/grad/class-loss logging frequency for future runs.

---

### Footer — sweep metadata
- Catalog used: `/home/molef/.claude/training-failure-modes.md` (all modes parsed; none malformed/skipped).
- N/A modes for this run: `kl-posterior-collapse`, `kl-weight-imbalance` (ε-prediction LDM has no KL term).
- "Not checked" modes are due to missing logged artifacts (no latent PCA, no attention maps, no AMP-scaler scalar, no FID, no timestep histogram, no per-layer grad/weight norms, no baseline reference runs), or intermediate sample grids not individually fetched — never silently treated as passed.
- Idempotency: this file was created fresh (no prior hifei736 entries existed). A future sweep with no new `ldm-ckpt` versions logged by `hifei736` should append nothing.
