# Augment: turning the PoE co-presence null into a strong paper section

Compiled from an augment session (2026-07-08). Input: the Act 9 mechanistic null
(`docs/DECISION_TIMELINE.md` Act 9, `results/vae_v2/downstream_eval/`). This file records the
strengthened idea plus a provenance trail: what was borrowed, from where, how sure, what it buys,
what it costs. Novelty was never the question; every entry is a part grafted onto the idea, not a
verdict on it.

---

## 1. The idea, strengthened

**The claim, scoped narrowly and defensibly.** Additive classifier-free guidance used as a
product-of-experts operator, `ε_null + w·(ε_cardio − ε_null) + w·(ε_effusion − ε_null)`, sampled with
ordinary reverse diffusion (ancestral sampling), fails to compose cardiomegaly and effusion into one
chest X-ray. Floor-corrected co-presence yield is 2.6% at N=2000 against a pre-registered 30% mark,
and it stays flat (2.6 to 4.1 to 3.1%) as `w_cardio` goes 2 to 4 to 6. The single-disease cardio fix
(best-of-8 rejection sampling, detector 0.875) does not carry into composition. The model renders a
big heart OR fluid, not both in one image.

**What makes it a finding, not a shortfall.** Two gates dissociate cleanly: single diseases render
faithfully (effusion 0.865 native, cardio 0.875 via best-of-8), yet co-presence collapses and does
not respond to the very lever that rescues single-disease cardio. The failure is in the combining,
not the drawing.

**The spine of the section is a 2x2 that self-calibrates.** A minimal toy reproduction sits beside
the clinical instance, both sampled two ways:

| | ancestral CFG-sum (unadjusted) | corrected sampler (Metropolis-adjusted MCMC) |
|---|---|---|
| **toy** (2D, known ground-truth product) | fails, as predicted by theory | recovers the product (positive control) |
| **CXR PoE** | 2.6% null | the punchline cell |

The toy uses Du et al.'s setting re-skinned to the failure geometry: two factors on separate
coordinates (factor A = "x large" for heart, factor B = "y large" for fluid), so the AND-target
requires both present (top-right corner). Train two one-factor score models, compose with the exact
`ε_null + w(ε_A − ε_null) + w(ε_B − ε_null)` operator, sample ancestrally. If samples collapse onto
the axes ("x large, y small OR x small, y large"), that is "big heart or fluid" in a system where the
true product is known exactly. The same corrected sampler that recovers the toy product is then run
on the real CXR PoE, so the CXR punchline cell has a calibrated reference.

**Running the corrected sampler on CXR, done so it counts.** A standard epsilon-prediction model
admits only unadjusted samplers (ULA, U-HMC), which carry systematic discretisation bias. To get a
Metropolis-adjusted sampler without retraining, define an approximate energy from the existing score
net, `f_θ(x,t) = −½‖s_θ(x,t)‖²`, and Metropolis-correct with it; also run the line-integral MH
acceptance rule as a second corrected sampler and compare. If both corrected samplers agree (both
recover the toy, both behave the same on CXR), the CXR verdict is robust to sampler choice.

**Why the sweep is flat, from first principles.** Joint compositional ability behaves like a product
of the component abilities. If composition-ability is near zero, joint yield ≈ cardio × effusion ×
composition ≈ 0 no matter how high `w_cardio` is pushed. Cranking one channel multiplies by a
near-zero factor, so the sweep cannot move. This is the mechanism behind the money plot (abstain
drops 52.8 to 33.4% while yield stays flat).

**The explanation is written as a subtraction.** A PoE-CFG failure can live in exactly four places,
and the evidence closes three:

1. **Components** (each single-disease marginal score is bad): ruled out, Exp-09 passes.
2. **Operator** (additive score is not the score of the product): adjudicated by the toy.
3. **Sampler** (ancestral vs Metropolis-adjusted, the non-conservative-field problem): adjudicated by
   the 2x2.
4. **Target support** (the conjunction is not learnable from the data, e.g. the co-occurrence base
   rate and the known 7:1 imbalance): the residual, and the localized cause if the corrected sampler
   still fails on CXR.

Lead with the elimination. The section's strength is that it positively rules out components and
adjudicates operator and sampler, leaving support as the last thing standing. It is a diagnosis, not
a menu of maybes.

**Qualitative evidence, made falsifiable.** The CTR-vs-CP scatter (`ctr_cp_scatter.png`) is the map;
PSPNet heart/thorax + costophrenic-angle overlays and GradCAM exemplars are callouts pinned to
clusters (the abstain pile, the "big heart, no fluid" corner). Each exemplar carries its reading and
a competing reading (for example "we read this as cardio-channel dominance; a competing account is
instrument saturation at CP ≈ 80 degrees"). Overlays do both jobs, illustrate the phenotype and
propose a mechanism, while staying honest that saliency is qualitative, not a formal explanation.

**Pre-registration resolves the identity.** The reading rule is fixed before the CXR corrected
sampler is run. Two abstracts are declared in advance:

- **Abstract-N (null):** corrected sampling fails on CXR, so the composition failure survives correct
  sampling and localises to target support (locus 4).
- **Abstract-M (method):** corrected sampling recovers co-presence, so naive additive-CFG PoE fails
  but Metropolis-corrected sampling composes.

The section's identity is a pre-registered composition investigation. The null is the hypothesis, not
a forced conclusion, and declaring the fork in advance is what converts "we got a null" into "we
pre-registered and the null held."

---

## 2. What was grafted, and from where

**G1. Sampler-not-model theory of composition failure.**
Source: Du et al., *Reduce, Reuse, Recycle*, ICML 2023, arXiv 2302.11552
(https://arxiv.org/abs/2302.11552). Confidence: high (landmark, widely cited).
Strengthened: the mechanism. The additive-CFG PoE under ancestral sampling is exactly the regime
their theory says cannot sample the product distribution, so an "A or B" mode-selecting outcome is
predicted, not idiosyncratic.
Trade-off accepted: it scopes the claim to "additive-CFG under ancestral sampling," and it obligates
running a corrected sampler (handled by G3), otherwise the paper is refutable on "did you try MCMC?"

**G2. The toy control, re-skinned to the failure geometry.**
Source: Du et al. toy setup (2D: GMM of 8 Gaussians on a ring radius 0.5 std 0.3; product composition;
Figure 2 columns = components, ground-truth product, reverse diffusion, HMC; code at
https://github.com/yilundu/reduce_reuse_recycle). Confidence: high that it is implementable and cheap.
Strengthened: converts the mechanism from correlational (measured through detectors on real X-rays) to
causal (measured against known ground truth). The single biggest upgrade to the section.
Trade-off accepted: one extra small experiment, plus a stated analogy gap (the toy is an existence
proof of the sampling failure mode, not a model of thoracic anatomy).

**G3. Metropolis-adjusted sampling without retraining.**
Sources: Du et al. energy-from-score trick `f_θ = −½‖s_θ‖²`; and the line-integral MH acceptance rule,
*MCMC-Correction of Score-Based Diffusion Models for Model Composition*, arXiv 2307.14012
(https://arxiv.org/abs/2307.14012). Confidence: medium-high (both reuse a pretrained score model).
Strengthened: makes the CXR corrected-sampler experiment count. Without a Metropolis-adjusted sampler,
a raw annealed-Langevin failure is dismissable as "you used the biased sampler."
Trade-off accepted: `f = −½‖s‖²` is an approximate energy, so chain diagnostics (acceptance rate,
autocorrelation) must be reported or the deeper-null claim wobbles. Running both samplers and comparing
buys robustness against "your approximate energy just mixed badly."

**G4. Reason the guidance sweep is flat.**
Source: *Compositional Abilities Emerge Multiplicatively*, NeurIPS 2023, arXiv 2310.09336
(https://arxiv.org/abs/2310.09336). Confidence: medium on the exact mapping (their setting is
synthetic).
Strengthened: gives the flat sweep a first-principles reason. Joint yield ≈ cardio × effusion ×
composition, so a near-zero composition factor makes `w_cardio` sweeps flat.
Trade-off accepted: cited as a named hypothesis, not proof, because it is a synthetic-task result.

**G5. Failure-locus taxonomy as a subtraction.**
Sources: the four-locus decomposition is assembled from Du et al. (operator, sampler) plus the
compositional-generalization line for the support locus (arXiv 2310.09336; *Does Data Scaling Lead to
Visual Compositional Generalization?* arXiv 2507.07102, https://arxiv.org/pdf/2507.07102).
Confidence: high on the structure, medium on locus-4 attribution.
Strengthened: lets the discussion be written as an elimination that positively rules out components
(Exp-09) and adjudicates operator and sampler (the 2x2).
Trade-off accepted: must be presented as subtraction, not a "could be any of these" menu, or it throws
away the Exp-09 evidence.

**G6. Medical overlay convention plus falsifiable annotation.**
Source: Saporta et al., *Benchmarking saliency methods for chest X-ray interpretation*, Nature Machine
Intelligence 2022 (https://www.nature.com/articles/s42256-022-00536-x). Confidence: high.
Strengthened: overlays get a citation-backed convention (attribution vs expected anatomical region)
and an inoculation against the over-interpretation reviewer. Their documented "effusion correctly
classified but poorly localised" pattern is reusable phrasing for the abstain pile.
Trade-off accepted: overlays are demoted from evidence-of-mechanism to illustration-of-phenotype with
falsifiable annotations. The CTR/CP numbers carry the mechanism.

**G7. Negative-result framing and pre-registered conditional outcomes.**
Source: Karl et al., *Position: Embracing Negative Results in Machine Learning*, ICML 2024,
arXiv 2406.03980 (https://arxiv.org/abs/2406.03980). Confidence: high.
Strengthened: legitimises the section (pre-registration is what makes a null publishable) and dissolves
the identity tension via two pre-declared abstracts and a fixed reading rule.
Trade-off accepted: gives up asserting "this is a negative-result paper" up front; keeps honesty by
letting the punchline cell decide.

---

## 3. What was left native

- **The dissociation design and the ablation-of-the-fix.** Single-disease passes, composition fails,
  and the best-of-8 lever that rescues single cardio does nothing to joint yield. This is the paper's
  core and it is the user's own. No graft touches it.
- **The rigor scaffolding.** Floor-correction, Wilson confidence intervals, the pre-registered 30%
  pass mark, the dose-response sweep. Borrowing rigor here would have added nothing.
- **The money plot.** Abstain drops 52.8 to 33.4% while yield stays flat 2.6 to 4.1 to 3.1%. Kept as
  is; G4 only supplies its reason, it does not change the plot.
- **The CTR-vs-CP scatter as the quantitative backbone.** Native. The overlays hang off it rather than
  replacing it.

Why native: these are where the evidence already is. Grafting onto them would have diluted a clean
mechanistic isolation into a patchwork.

---

## 4. Open grafts (surfaced, not committed)

- **Measure the training co-occurrence base rate of cardiomegaly + effusion (NIH/VinDr).** Turns
  locus-4 (target support) from borrowed synthetic-task theory into the user's own measured fact.
  Cheap (one query over labels). Pairs with arXiv 2507.07102 ("data scaling alone does not fix
  composition"). Left open by choice; flagged because the subtraction lead has a load-bearing
  dependency on locus-4 being the last thing standing, and without this measurement locus-4 rests on
  analogy. Small risk it undercuts the framing if co-occurrence turns out common; either way it is
  information worth having before writing.
- **Escalation ladder if the energy-trick chain mixes badly on CXR.** Score-distillation into an EBM
  to learn the conservative component of the pretrained score field (heavier, a fallback only if both
  no-retrain corrected samplers give ambiguous diagnostics). Confidence: medium; not needed unless G3
  diagnostics are poor.
- **Region-masking / MultiDiffusion as an explicit out-of-scope note.** Deliberately not grafted (it
  is a spatial-ownership method for a different claim and would pull the paper toward "here is a fix").
  Recorded here so it can be named as out-of-scope if a reviewer asks, rather than built upon.

---

## Reading order for the section (the compiled spine)

1. Setup and the pre-registered fork (two abstracts, reading rule fixed in advance): G7.
2. The single-vs-two-disease dissociation and the ablation-of-the-fix: native.
3. The toy 2x2, ancestral fails vs corrected recovers against known ground truth: G1, G2, G3.
4. The CXR 2x2 punchline cell, corrected sampler(s) on the real PoE: G3.
5. Why the sweep is flat: G4.
6. Discussion as subtraction: components ruled out, operator and sampler adjudicated, support the
   residual: G5 (and the open base-rate graft).
7. Qualitative map plus falsifiable overlay callouts: G6.
