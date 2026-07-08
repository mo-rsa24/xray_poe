# 🧪 Improve — frozen no-retrain composition operators (GATED on O1)

## Description
If O1's diagnostics say there is something to combine (cardio-alone ceiling is NOT empty AND coverage is NOT
near-zero), try frozen no-retrain operators in cost order and score them on one shared scorecard on a
yield-per-compute axis. Model: **frozen v4-60k**. **GATED: do not run if O1 shows an empty ingredient /
unreachable region** — in that case record the skip as the finding (no combiner can help) and route the
signal-injector (concept-algebra, parked) instead.

## Purpose
Serves O4 and DoD-3. This is the Track-B "can we recover the null for free" arm; either outcome is publishable.

## Goal
clean-ε / SuperDiff / best-of-N each report co-presence yield-per-NFE on the shared scorecard, with a verdict:
**recoverable-for-free** vs **operator-limited** vs **manifold/support-limited**.

## Tasks
- [ ] ⚠️ **GATE CHECK** — confirm O1's cardio-alone ceiling is non-empty and coverage is non-zero before spending
  any compute here. If not, STOP: record "no ingredient to combine" as the O4 verdict and skip to the parked
  concept-algebra injector.
- [ ] ⚠️ **Clean-unconditional-ε anchor (first, cheapest)** — replace the ~75%-diseased `null` anchor with a
  clean unconditional ε in `cfg_compose` (`scripts/cfg_anchor_sweep.py` Part 1). If the anchor is the problem,
  the fancier operators would be fixing the wrong thing — so this runs first.
  - [ ] ⚠️ Remedy: if no clean unconditional ε is available (the model's null is intrinsically diseased), sample
    a no-finding-conditioned anchor as the proxy and label it as such.
- [ ] ⚠️ **SuperDiff AND (best-first operator)** — per-step Itô log-density bookkeeping + a 2×2 solve for weights
  κ so both conditions gain log-density at equal rates (upweights the under-satisfied cardio). Both "models" are
  v4-60k at two conditions. Re-measure yield with `gate3_yield.py`.
  - [ ] ⚠️ Remedy: if the Itô density estimate is numerically unstable, fall back to the paper's discretized
    estimator (SuperDiff §Prop 6) and validate on the plan-02 toy before trusting the CXR number.
- [ ] ⚠️ **Best-of-N control (truth anchor)** — vanilla PoE-CFG, N draws, keep both-disease by the PSPNet-CTR
  verifier (NOT the DenseNet classifier — its joint-confound would corrupt selection). If best-of-N matches the
  fancy operators at equal NFE, the operators aren't earning their keep — a real finding.
- [ ] ⚠️ **Shared scorecard + verdict** — for every arm: (1) co-presence yield, (2) cardio CTR distribution,
  (3) effusion presence must HOLD vs effusion-alone, (4) off-manifold/artifact guard (LPIPS-to-real), (5) NFE.
  Report **yield-per-NFE** (fixed budget or a yield-vs-NFE curve). Verdict: recoverable-for-free / operator- /
  manifold-limited.

## Engagement Instructions
```bash
# clean-ε anchor:
python scripts/cfg_anchor_sweep.py ...        # Part 1 clean-unconditional anchor
# each operator's yield on the shared instrument:
python scripts/gate3_yield.py --ckpt ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors ... \
  --out results/poe_composition/operators/<arm>/
```
Verify: the GATE decision is recorded; each arm reports the full 5-field scorecard + yield-per-NFE; SuperDiff
validated on the plan-02 toy first; best-of-N selection uses PSPNet-CTR not the classifier; a single verdict
(recoverable/operator-/manifold-limited) is written.

## Recommended skill
— custom; `scripts/cfg_anchor_sweep.py`, `scripts/gate3_yield.py`, + the SuperDiff Itô-weighting / ULA hooks on
   frozen v4-60k.
