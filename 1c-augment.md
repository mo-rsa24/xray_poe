# Augment compile: breaking the two-disease composition null on frozen v4-60k

Source: `/augment` session, 2026-07-08. Input idea: PoE-CFG fails to compose cardiomegaly + effusion; mine alternative composition methods that could break the null, judged against our constraints (squashed ~1-of-4-channel VAE latent, few real comorbid films, a working v4-60k LDM we do not want to retrain from scratch).

Prior diagnosis carried in (load-bearing): composition was measured *faithful*; the cardio channel under-renders (gen cardio CTR 0.395 vs real 0.547 on the 100k checkpoint). So the null is a deficient-ingredient problem, not (only) a bad-combiner problem. This is the fact that ranks every method below.

---

## 1. The idea, strengthened

Run a **four-way, inference-time composition comparison on the frozen v4-60k LDM**, gated on a cardio-alone ceiling probe, scored by one shared scorecard on a yield-per-compute axis.

**The gate (run first, non-negotiable).** Before any composition run, sample v4-60k *cardio-alone* (single condition, no effusion) and measure heart size with the existing PSPNet CTR instrument. This is now the sole disambiguator between "bad combiner" and "empty ingredient," because we chose v4-60k-only (no before/after arm from the rebalanced arm1 checkpoint). The number it returns sets the ceiling that makes the whole table legible. It costs one sampling batch and reuses tooling we already have.

**The four arms.**

1. **SuperDiff AND (best first bet).** Per denoising step, track a running per-model log-likelihood of the current sample with SuperDiff's Itô density estimator, then solve a tiny linear system for weights κ (summing to 1) so the two conditionings, ε(x_t, c_cardio) and ε(x_t, c_effusion), gain log-density at *equal rates*. This upweights whichever condition is currently least satisfied, which is structurally our under-rendered cardio. The two "models" are the same v4-60k queried at two conditions; SuperDiff composes vector fields and does not care that they share weights. Cost is essentially free relative to PoE-CFG (we already run both conditional passes; we add density bookkeeping and a 2×2 solve). No data, no retrain.

2. **Unadjusted Langevin predictor-corrector.** After each denoise step, run k Langevin corrector steps (k≈2–5) on the *composed* score to pull the sample onto the joint high-density region. We use the *unadjusted* version (no Metropolis accept/reject), because valid accept/reject needs a scalar energy whose gradient is the score, which ε-prediction does not give us on a frozen model. Cheapest bolt-on. Best value as a companion to SuperDiff, not standalone. No retrain.

3. **Best-of-N control (the truth anchor).** Sample N films from vanilla PoE-CFG, score each with the verifier, keep the ones with both diseases. This is the canonical inference-time-scaling baseline. If best-of-N matches the fancy operators at equal compute, the operators are not earning their keep, and that is a real finding.

4. **Concept algebra (the only signal-injector).** Estimate a linear cardiomegaly direction in the score representation and inject it during sampling so cardio presence is an explicit additive knob rather than an emergent property of conditioning. This is the one arm that can add heart signal the composition itself cannot produce. It is also the one our constraints punish hardest (squashed latent, few comorbid films). **Estimate the direction from real single-cardio NIH films, not from v4-60k's own cardio-alone samples**, to avoid injecting the model's deficient notion of "heart" back into itself. Needs a lightweight probe, no LDM retrain. Gated on building the single-cardio pool; runs after the frozen three, in parallel with pool collection.

**Scorer decision.** Primary scorer = PSPNet-CTR + blunting extractors (the less-confounded, geometry-based judge, validated on real films). It drives both the yield metric and the best-of-N selection. The DenseNet classifier is secondary (second opinion only), never the selector, because it was trained mutually-exclusive, suppresses joint logits, and would systematically reject true both-disease samples, corrupting the best-of-N control.

**Shared scorecard (all arms).** (1) co-presence yield (primary), (2) cardio CTR distribution, (3) effusion presence must HOLD vs effusion-alone, (4) an off-manifold/artifact guard (LPIPS-to-real or classifier confidence, since FID is demoted per the supervisor reconcile), (5) NFE. Report **yield per unit compute** (yield vs total NFE), either at a fixed shared NFE budget or as a yield-vs-NFE curve, so a raw yield column does not silently reward whoever spent the most forward passes.

**What the experiment resolves.** If the frozen combiners plateau below target while the ceiling probe shows near-empty cardio, the honest read is that the rebalanced retrain (arm1, already working) is the true lever and no inference-time operator substitutes for it. If SuperDiff AND moves yield materially, the null was recoverable for free. Either outcome is publishable.

---

## 2. What was grafted, and from where

- **SuperDiff AND weighting.** Source: Skreta et al., *SuperDiff*, arXiv:2412.17762 (ICLR 2025). Confidence: high; mechanism verified from full text (Theorem 1 Itô estimator, Proposition 6 equal-density-change AND weights), not assumed. Strengthened: gives the one operator whose mechanism targets our exact failure (upweights the under-satisfied cardio condition), at near-zero cost over PoE-CFG. Trade-off accepted: it is still a combiner, so its ceiling is set by v4-60k's cardio capacity; it cannot manufacture heart the model never learned.

- **Unadjusted Langevin corrector.** Sources: Song et al. 2021 (predictor-corrector sampler) for the frozen-valid form; Du et al., *Reduce, Reuse, Recycle*, arXiv:2302.11552 (ICML 2023) for the Metropolis-corrected form we deliberately do not use. Confidence: high. Strengthened: cheap mixing companion. Trade-off accepted: modest alone; valid accept/reject would need an energy parameterization we are not building.

- **Best-of-N control.** Source: inference-time-scaling-of-diffusion literature, diffusion-inference-scaling.github.io (2025). Confidence: high that it is the correct honest control. Strengthened: converts "did the operator help or did we just spend compute?" into a measurable answer. Trade-off accepted: N× base sampling for the control arm.

- **Yield-per-NFE axis.** Same source as above. Confidence: high. Strengthened: makes four methods with wildly different per-sample cost comparable. Trade-off accepted: sampler bookkeeping.

- **Concept algebra / cardio direction.** Source: Wang et al., *Concept Algebra for (Score-Based) Text-Controlled Generative Models*, arXiv:2302.03693 (NeurIPS 2023); CASL SAE variant, arXiv:2601.15441. Confidence: medium that it transfers to our squashed-latent, low-data setting. Strengthened: the only arm that can add signal rather than recombine. Trade-off accepted: high variance under our constraints; requires a real single-cardio pool we must still build; risk of circularity if seeded from generated cardio-alone.

---

## 3. What was left native

- **The deficient-ingredient diagnosis stays the ranking principle.** We did not let any borrowed operator override the measured fact that composition is faithful and cardio under-renders. That is why three of four arms are framed as combiners with a hard ceiling, and why the ceiling probe gates everything. Borrowing an operator's optimism here would have hidden the real bottleneck.

- **The rebalanced retrain (arm1) stays the accepted true lever.** We explicitly did not graft any inference-time method as a *replacement* for it. The four-way table tests how much is recoverable frozen; it does not relitigate the retrain decision.

- **v4-60k-only scope stays native.** We considered a before/after arm on the rebalanced checkpoint and dropped it per the user's call, accepting that the cardio-alone ceiling probe carries the full disambiguation load instead.

- **PSPNet-CTR as primary judge stays native.** We resisted the convenient graft of using the DenseNet classifier as the best-of-N selector, because its joint-confound would corrupt the control.

---

## 4. Open grafts (surfaced, not yet decided)

- **HMC on a composed energy.** Du et al. 2023, arXiv:2302.11552. Confidence: high on mechanism. Buys better mixing, but needs an energy distillation/parameterization of v4-60k (effectively a retrain-flavored step) and several leapfrog evals per step. Gate on the ceiling probe: if cardio is near-empty, HMC buys mixing we cannot use. Park until the frozen three report.

- **Greedy / beam search over best-of-N.** Inference-time-scaling literature (2025) shows both beat plain best-of-N at equal budget. A second-order refinement of the control; adds table width without sharpening the core question. Park.

- **Real single-cardio sourcing for concept algebra.** NIH has cardiomegaly-only cases; confirm a usable pool exists and is large enough to estimate a stable direction before committing to arm 4 as a real bet rather than a fallback. Confidence: medium that the pool is adequate.

- **SuperDiff is not our dead sampling-reweight.** Framing note to carry into the writeup so the method is not dismissed by pattern-match: the dead lever was a *fixed global* weight on the two scores; SuperDiff is a *per-step, per-sample* weight solved for equal log-density gain. Adaptive, not a constant knob.

---

### Un-augmented / still open

- Whether v4-60k is pre- or post-rebalance is unknown; the ceiling probe (arm gate) is what will tell us, so it is answered by running the experiment rather than before it.
- Concept algebra (arm 4) remains the least-tested arm; its real-data pool is not yet built.
