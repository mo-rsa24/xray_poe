# The Evaluation Philosophy, From First Principles

> Authoritative spec for Paper 3 (PoE composition of single-disease diffusion experts
> to synthesize pixel-verifiable co-morbid chest X-rays). Compiled 2026-06-24.
> Proxy = cardiomegaly + effusion on NIH ChestX-ray14. Target = silicoTB.

The one-sentence thesis this whole structure defends:

> **PoE composition of single-disease diffusion experts synthesizes pixel-verifiable,
> ecologically-valid co-morbid chest X-rays from *no* co-morbid data; gated through
> CTR + costophrenic + radiologist validation, these images augment a data-starved
> classifier and improve co-morbid detection on the pixel-verifiable subset —
> outperforming naive overlay. Entangled presentations and the de-tangling mechanism
> are scoped out as future work.**

Everything below is what you'd need to rebuild that sentence and know exactly why each
word is load-bearing.

---

## 1. Definitions

- **Cardiomegaly** — an enlarged heart. *Operationally defined* as **CTR > ~0.5 on a PA film**. Not a label opinion — a geometric fact.
- **CTR (cardiothoracic ratio)** — maximal horizontal cardiac width ÷ maximal horizontal inner-thoracic width, measured on a frontal film. The canonical cardiomegaly criterion.
- **Pleural effusion** — fluid in the pleural space. On a frontal film it **blunts the costophrenic angle** (the sharp corner where diaphragm meets ribcage) and, when large, forms a rising meniscus.
- **Costophrenic (CP) angle** — that lower-lateral corner. Sharp = clear; blunted/filled = effusion.
- **PA vs AP** — *posteroanterior* (standing, beam back-to-front; heart near detector, minimal magnification — CTR 0.5 valid) vs *anteroposterior* (portable/supine; heart far from detector, magnified — CTR 0.5 over-calls). **Threshold is view-dependent.**
- **Separable both** — a co-morbid film where *both* signs are independently visible (heart border readable **and** CP angle assessable).
- **Entangled both** — a co-morbid film where one sign obscures the other (e.g. effusion whiteout hides the cardiac border). CTR unmeasurable.
- **Pixel-intrinsic ground truth** — a label you can verify *from the image alone*, with no lateral / prior / CT.
- **Well-posedness** — a learning problem is well-posed only if the label is a function of variables *present in the input*. Pixel→label where the label needs off-pixel evidence is **ill-posed**.
- **Construct validity** — the detector fires *for the right reason* (right anatomy), not a shortcut.
- **Triangulation** — trust = multiple **independent** instruments converging. Independence is the whole point; shared sources = circularity = no information.
- **PoE (Product of Experts)** — composing two single-disease diffusion experts so a sample satisfies **both** conditions: in score form, `ε_compose = ε_uncond + w_c·(ε(x,cardio) − ε_uncond) + w_e·(ε(x,effusion) − ε_uncond)`. Each expert contributes its disease's denoising direction.
- **VAE / LDM** — VAE encodes a 512×512 image to a 4×128×128 latent and decodes back; the LDM is the diffusion model operating *in that latent*.
- **FID** — Fréchet Inception Distance; a **population** distance between two image sets. Cannot see per-image properties.
- **AUC** — ranking quality of a detector over a population.
- **Oracle baseline** — a classifier trained on *abundant real co-morbid data*. Buildable on the **proxy** (cardio+effusion), impossible on the **target** (silicoTB).
- **Overlay baseline** — crude composition: paste an effusion-base region onto a cardio film. The rival PoE must beat.
- **Starved baseline** — classifier trained on only the scarce real co-morbid data.
- **Proxy / target** — cardio+effusion = the proxy you prove the method on (data exists); **silicoTB** = the scarce target you apply it to.

---

## 2. Background (the minimum)

NIH ChestX-ray14: ~112k single **frontal** films, labels **NLP-mined from radiology
reports**. The reporting radiologist had the **lateral view, priors, sometimes CT** —
none shipped in the dataset. So a label certifies the *patient*, not the *pixels*.
Cardio and effusion co-occur (φ=0.13) because both are heart-failure signs — a *causal*
link, which is why they're the most clinically-meaningful pair and why they sometimes
physically interfere.

The real motivation: **silicoTB**, where co-morbid data is genuinely too scarce to train
a reliable classifier. If you can manufacture valid co-morbid training data by composing
single-disease experts, you fix that — but you can only *prove the mechanism* where an
oracle exists, i.e. on the proxy.

---

## 3. Components

| Component | What it is | One-line role |
|---|---|---|
| **CTR instrument** | heart+thorax segmentation → ratio (PSPNet/HybridGNet) | canonical cardio measure; *teaches the classifier* |
| **CP-angle instrument** | intensity/sharpness read at the costophrenic corner | canonical effusion measure (the *robust* witness) |
| **Two binary detectors** | independent sigmoid heads, one per disease | the classifier being improved; co-presence-capable |
| **GradCAM witness** | saliency of a detector's call | confirms *right-reason* (looks at the heart) |
| **Radiologist gate** | human per-image read | ecological validity, final authority |
| **PoE composer** | two single-disease diffusion experts, summed scores | manufactures co-morbid images from single-disease only |
| **Overlay baseline** | naive region paste | the rival that justifies PoE's machinery |
| **Dynamics study** | characterize the both-set | defines ground truth + the gap PoE fills |
| **Stratified downstream test** | AUC by presentation stratum | proves utility *and* (on proxy) the mechanism |

**Independence bookkeeping (the rule that prevents circularity):** **CTR teaches** the
cardio classifier (label source). The classifier's **independent witnesses are GradCAM +
radiologist** — GradCAM is independent of the *teacher* (it reads where the net looks,
not the CTR signal), the radiologist is independent of everything. CTR is *never* used to
both teach and validate the same model.

---

## 4. The gate pipeline (the trust chain, in order)

Each gate is a filter. An image — real or synthetic — flows through; failing a gate
routes it out (to "scoped-out" or "discarded"), it does not silently pass.

**Gate 0 — Well-posedness.** Is the label verifiable from the pixels? PA view, heart
border visible. *Entangled-both fails here → scoped out as future work.* This gate defines
your study population.

**Gate 1 — Construct validity (localization).** Does the detector's positive call land on
the right anatomy? GradCAM saliency-mass inside a heart box (cardio) / CP region
(effusion) ≥ threshold; spot-checked visually. Kills shortcut learning.

**Gate 2 — Criterion agreement.** Does the detector agree with its canonical measure — via
an *independent* path? Cardio detector (CTR-taught) confirmed by **GradCAM + radiologist**,
not by CTR again.

**Gate 3 — Co-presence certificate (per-image).** On a both-image, do **both**
single-disease gates pass *simultaneously*? This — not FID — is the "both" claim.

**Gate 4 — Distributional sanity floor (FID).** Are these coherent, in-distribution chest
films? Necessary, never sufficient. Catches garbage; certifies nothing about co-presence.

**Gate 5 — Ecological validity.** Radiologist accepts each image as a plausible real
patient. Per-image, not distributional.

**Gate 6 — Downstream utility (stratified).** Does the PoE-augmented classifier beat
**starved** *and* **overlay** baselines on the pixel-verifiable subset — and on the
**proxy**, how does it sit against the **oracle**? Report AUC **by stratum** (normal /
cardio-only / effusion-only / both-separable).

**The symmetry rule:** Gates 0–5 run on **both** real test films *and* synthetic training
images. The same CTR+CP+radiologist filter that certifies your test set also *filters which
generated images are allowed into augmentation*. A synthetic image that fails the gate is
discarded, not trained on.

---

## 5. Concrete example, traced end to end

**Real film A** — `00012345_000.png`, NIH label `Cardiomegaly|Effusion`, View=PA.

1. **Gate 0:** PA ✓. Heart border visible ✓ → eligible.
2. **CTR instrument:** segment heart (max width 178px) and thorax (inner width 300px) → CTR = 0.59 > 0.5 → **cardio present, separable** ✓.
3. **CP instrument:** left CP angle filled, meniscus present, right angle sharp → **effusion present** ✓.
4. **Gate 3 (co-presence):** both fire, neither obscures → **SEPARABLE BOTH** → enters the **pixel-verifiable test set**.

**Real film B** — massive left effusion, whiteout to mid-thorax.
- **Gate 0:** PA ✓, but cardiac border invisible → CTR unmeasurable → **fails Gate 0 → ENTANGLED → scoped out.** (Label may be correct — radiologist saw the prior — but the pixels can't witness it. Ill-posed; excluded by principle.)

**Synthetic film C** — PoE compose cardio-expert + effusion-expert:
1. Sample `z_T` (4×128×128), denoise with `ε_compose = ε_uncond + 2.0·(ε(z,cardio) − ε_uncond) + 1.0·(ε(z,effusion) − ε_uncond)` → `z_0` → VAE-decode → 512×512 image.
2. **Gate 4 (FID floor):** looks like a real CXR, features in-distribution ✓.
3. **CTR:** 0.56 > 0.5 ✓; **CP:** left angle blunted ✓ → **Gate 3 co-presence ✓**.
4. **Gate 5:** radiologist — "plausible CHF film, both signs coherent" ✓ → **enters augmentation set.**

**Synthetic film D** — PoE sample where the heart didn't enlarge: CTR = 0.44 → **fails
Gate 3 → discarded** (this is exactly the cardio-under-rendering; the gate catches it
automatically).

**Downstream (Gate 6):** train three classifiers — **starved** (scarce real both),
**+overlay**, **+PoE** (real both + gated synthetic C-type). Test on the pixel-verifiable
set, AUC **by stratum**. On the proxy, also train the **oracle** (abundant real both).
Claim holds if **+PoE > +overlay > starved**, with gains concentrated on **both-separable**
(and ideally the lonely single-disease strata).

---

## 6. Build each component from first principles

- **CTR instrument.** Input: PA film. Op: semantic-segment heart + lung-field/thorax (existing PSPNet); take max horizontal extents; ratio. Output: CTR ∈ (0,1) + "border-visible?" flag (the Gate-0 signal). Threshold 0.5 (PA); shift or exclude for AP.
- **CP-angle instrument.** Input: film + lung mask. Op: locate the lateral diaphragm-ribcage corner; read intensity sharpness / angle fill along a profile. Output: blunted? (effusion present) + severity. This is the *robust* witness — survives cardiomegaly.
- **Two binary detectors.** Input: image. Op: two **independent sigmoid** heads (NOT softmax — softmax forbids "both"), cardio head trained on **CTR-derived** labels, effusion head on CP-derived labels. Output: P(cardio), P(effusion), each free to be high.
- **GradCAM witness.** Input: trained detector + image. Op: gradient-weighted activation map; compute pointing-game (saliency-mass in heart box). Output: right-reason score. Independent of the CTR teacher.
- **PoE composer.** Input: two single-disease experts + noise. Op: summed CFG-style scores (§5 step 1); denoise; VAE-decode. Output: candidate co-morbid image (then gated).
- **Overlay baseline.** Input: a cardio film + an effusion-base patch. Op: paste/blend the patch at the base. Output: crude co-morbid image (then gated identically). The control that asks "does diffusion buy anything?"
- **Dynamics study.** Input: all both-labelled NIH films. Op: stratify PA/AP → run CTR+CP → label separable/entangled → **count**. Output: the fraction separable = your test-set size *and* the gap PoE fills.
- **Stratified downstream test.** Input: 3–4 trained classifiers. Op: AUC per stratum on the pixel-verifiable set. Output: lumpy-vs-flat gain profile = utility + (on proxy) mechanism.

---

## 7. One small exercise per component

1. **CTR:** 10 PA films — run the segmenter, compute CTR, plot vs NIH label. Find the disagreements; those are why you don't trust the label.
2. **CP-angle:** 10 effusion films — plot the intensity profile across each CP corner; see blunting as a measurable dip.
3. **Well-posedness:** on 100 both-labelled films, flag "cardiac border visible?" and count separable vs entangled. That single fraction is your foundation.
4. **Binary detectors:** train one sigmoid cardio head on CTR-derived labels; confirm P(cardio) and P(effusion) can *both* exceed 0.5 on one image (the property softmax denied you).
5. **GradCAM:** run it on 20 cardio-positives; compute the heart-box pointing-game fraction. Is it > 80%? Now you've turned the round-1 vibe into a number.
6. **PoE:** compose two experts, decode 10 images, eyeball — then run Gate 3 and see how many pass (you'll rediscover the cardio under-rendering as a *measured* pass-rate).
7. **Overlay:** build 10 naive overlays, push them through the identical gate, compare pass-rate and realism to PoE.
8. **Downstream:** train starved / +overlay / +PoE; plot AUC by stratum on one figure. If +PoE beats +overlay, the machinery earned its keep.

---

## Where this is still thin (flagged honestly)

- **Effusion's instrument** is less crisp than CTR — CP-angle blunting is a softer measurement than a ratio with a canonical threshold. Tighten its operational definition before it's a gate.
- **The oracle on the proxy** must actually be built, or Paper-1-vs-2 stays unfalsifiable — the utility framing needs the oracle to make "as good as having the data" a real comparison.
- **silicoTB transfer:** the proxy leans on CTR; the target lacks an equivalent. Design Gates 1–5 so they don't *secretly* require a CTR-like oracle, or the method won't carry over.

---

### Reference: the refined claim (for citation in downstream plans)

> PoE composition of single-disease diffusion experts synthesizes pixel-verifiable,
> ecologically-valid co-morbid chest X-rays from no co-morbid data; gated through
> CTR + costophrenic + radiologist validation, these images augment a data-starved
> classifier and improve co-morbid detection on the pixel-verifiable subset —
> outperforming naive overlay composition. Entangled presentations and the de-tangling
> mechanism are scoped out as future work requiring multi-view decomposition.

### Provenance note (Paper 1 vs Paper 2)

The headline is the **utility** framing (Paper 2): synthetic co-morbid data improves a
data-starved classifier, proven by downstream AUC. The **mechanism** is stated narrowly —
PoE augments the scarce co-morbid class with *separable* examples ("spatial
de-entanglement within the co-morbid class"), **not** a claim to de-bias disease
co-occurrence (PoE adds to the both-class, so it cannot reduce co-occurrence). The
*contribution* is the compositional source (co-morbid images from single-disease-only
inputs), defended against the **overlay baseline**. The **mechanistic** Paper-1-vs-2
question (does PoE match or beat the oracle?) is testable only on the **proxy**, where the
oracle can be built; the **target** (silicoTB) can only show the practical claim (beats the
starved baseline).
