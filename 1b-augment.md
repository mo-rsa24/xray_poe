# Augment compile: bottleneck-localization for PoE co-presence

Question: are we *limiting* the PoE, or hitting a true wall? Localize the bottleneck among
four candidates (VAE codec / DATA / composition operator / LDM) with four cheap, independent
instruments, run in a fixed order so each result narrows blame before the next is paid for.

Central reframing that survived augmentation: the hypothesis "VAE-v2 latent is entangled ->
concepts not linearly separable -> PoE can't navigate" is now a **measurement, not an
assertion**, and it is explicitly separated from the rival diagnosis "the valid region is just
small and rare" (consistent with the r=-0.18 real coupling). The instruments are built so
entanglement and ecological rarity cannot be mistaken for each other.

---

## 1. The idea, strengthened: fixed order of execution

### Step 1 (first, ~1 day) - Codec-delta separability probe -> candidate (a) VAE
Run one linear CAV probe (cardio-only vs no-finding; effusion-only vs no-finding) at three
stages: raw pixels -> VAE-v2 latent `z` (4x128x128) -> UNet h-space.

Report per stage:
- probe accuracy (separability)
- cosine between the two class CAVs (entanglement geometry)
- CAV weight-mass per latent channel (does it pile into the known dead channels)
- a permutation null (so "not separable" is a rejected hypothesis, not an assertion)

**The pixels -> `z` accuracy drop is the codec's contribution.**

Ruling:
- separable at `z` with near-orthogonal CAVs -> **(a) OUT**
- collapse specifically at `z`, or CAV mass in dead channels -> **(a) IN**
- the *same* anti-alignment already present in pixels -> it is anatomy/data, not codec

First because it is cheapest (reuses the existing probe code) and settles the central
hypothesis before any further investment.

### Step 2 (second, ~1 day) - Coverage / density on the real comorbid manifold -> candidate (b) DATA / ecology
Build the real both-disease manifold as k-NN balls (in an xrv embedding or `z`, **never pixels**)
around real comorbid films. Measure:
- **coverage**: fraction of real-comorbid neighborhoods reached by >=1 generated comorbid film
- **density**: whether generated comorbids pile into one corner vs spread across the region

Ruling:
- decent coverage -> the rare region is reachable; "too few real films" is not the wall
- near-zero coverage -> the target region is genuinely unreached

Second because it is also cheap and it frames how to read the sampler test.

### Step 3 (third, implementation + N x sampling wall-clock) - ULA sampler-swap -> candidate (c) OPERATOR
Freeze the LDM and both conditionals; replace naive CFG-PoE score-addition with unadjusted
annealed Langevin (ULA) composition.

Read **asymmetrically** (the LDM is epsilon-prediction, so no scalar energy -> only the
*unadjusted* sampler is available; there is no Metropolis accept/reject):
- yield rises -> **(c) IN, decisively** (you were limiting the PoE; even a weak sampler unsticks it)
- yield flat -> **(c) only PROVISIONALLY out** (ULA-without-MH is a weak instrument; the clean
  MH version needs an energy reparameterization that would reconflate c and d)

Third because it costs implementation plus wall-clock, and its null is only interpretable once
(a) and (b) are known.

### Step 4 (residual) - candidate (d) LDM
If the region is reachable (b ok), the latent is separable (a out), and unsticking the operator
gives little lift (c provisional), the remaining blame is the under-trained joint conditional.
No new instrument: the retrain monitor and during-training probe curves already speak to it.

---

## 2. What was grafted, and from where

- **CAV linear probe as the separability instrument** - TCAV (Kim et al.,
  https://arxiv.org/pdf/1711.11279); chest-x-ray precedent + counterfactual traversal
  (https://arxiv.org/html/2506.04058). Strengthened: "entangled" -> per-stage accuracy +
  CAV-cosine. Trade-off: needs clean single-disease sets (already on disk:
  `images_cardio_only`, `images_effusion_only`). Confidence: high.
- **Permutation-null significance** - corrected TCAV testing
  (https://arxiv.org/html/2509.24058v1, shows the original t-test is not robust).
  Strengthened: "not separable" becomes a rejected hypothesis. Trade-off: many probe
  re-fits (cheap). Confidence: high.
- **Cross-stage attribution (pixel -> z -> h-space delta)** - activation-patching /
  causal-mediation framing (Basu et al., https://arxiv.org/pdf/2310.13730). Strengthened:
  isolates the codec's contribution as a difference, not a narrative. Trade-off: probe wired
  at each stage (alignment is by NIH ID, confirmed). Confidence: medium-high.
- **ULA / annealed-MCMC sampler-swap** - Du & Liu, Reduce/Reuse/Recycle
  (https://arxiv.org/abs/2302.11552, code https://github.com/yilundu/reduce_reuse_recycle);
  their result "the sampler, not the model, causes composition failure" is the (c)-vs-(d) cut.
  Strengthened: isolates the operator without retraining. Trade-off: epsilon-prediction gives
  only the unadjusted sampler -> asymmetric reading. Confidence: high on method, medium on how
  much an unadjusted sampler moves this specific yield.
- **Coverage / density for the comorbid manifold** - Naeem et al.
  (https://arxiv.org/pdf/2002.09797, code https://github.com/clovaai/generative-evaluation-prdc);
  improved P/R lineage Kynkaanniemi (https://arxiv.org/pdf/1904.06991). Strengthened: closest
  thing to a formal "ecological validity" metric. Trade-off: k-NN sensitive to feature space -
  must use an embedding, not pixels. Confidence: high.
- **Active-units / KL-rate framing** - posterior-collapse instruments
  (https://www.emergentmind.com/topics/posterior-collapse;
  http://papers.neurips.cc/paper/9138-dont-blame-the-elbo-a-linear-vae-perspective-on-posterior-collapse.pdf).
  Folds the "3.4% signal, one of four channels" observation into a comparable number that
  cross-checks the CAV channel-mass result in Step 1. Confidence: high.

---

## 3. What was left native (deliberately not borrowed)

- **The four-candidate decomposition** - your own skeleton; already maps onto the attribution
  component-graph, so no borrow reshaped it.
- **The 9.8% real co-presence as the target** - kept as the ecological anchor, not swapped for a
  distribution-matching objective; coverage/density measure *reaching* it, they do not redefine it.
- **The `z`-stage focus for candidate (a)** - kept upstream of the UNet on purpose. The existing
  `eval/hspace_probe_acc` (train_ldm.py:550) is at the wrong stage (downstream, label-conditioned);
  reading it as (a)'s verdict would inherit the conditioning signal. Re-point the probe at `z`
  and at pixels instead.

---

## 4. Open grafts (surfaced, not adopted)

- **Energy-reparameterized LDM for a decisive (full-MH) sampler test.** Would make a sampler-swap
  null decisive instead of provisional, but changes the model and reconflates (c)/(d). Deferred;
  only worth it if Step 3 returns a null that needs hardening. (Du/Liu EBM parameterization.)
- **h-space difference-of-means directions** (Haas et al., https://arxiv.org/html/2303.11073v2)
  as an alternative CAV estimator if logistic-regression CAVs prove unstable across
  random-negative draws.
- **DCI / MIG / SAP - explicitly REJECTED, not open.** They assume ground-truth axis-aligned
  factors chest films do not have, and would mismeasure correlated anatomy as entanglement.

---

## Repo grounding (confirmed during augmentation)

- LDM is epsilon-prediction + MSE (src/models/ldm.py:138) -> not energy-based -> ULA-only.
- Single-disease splits already exist: data/nih/images_cardio_only, images_effusion_only,
  images_both, plus no-finding sets.
- Latents (data/latents_v2/train/00000468_048.pt) align to PNGs by NIH ID -> pixel-vs-latent
  probe needs no re-encoding.
- An h-space linear probe is already logged (scripts/train_ldm.py:550,
  eval/hspace_probe_acc + per-class) - reuse it, but re-point to `z` for candidate (a).

Next step suggestion: /triage-plan to route these four instruments into the vae-v2 plan tree.
