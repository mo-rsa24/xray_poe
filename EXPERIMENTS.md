<div align="center">

# 🫁 Experiments — Composing Correlated Diseases by PoE, and Proving When It Breaks

**When we compose two single-disease conditions at generation time, do we get a _realistic_ both-disease X-ray — or just two diseases pasted together? And does the answer depend on whether the two diseases are _correlated_?**

`scope: characterize the gap and isolate its cause (correlation)` · `compute: local RTX 4070 12GB` · `pinned: 2026-06-05`

</div>

---

## 🧩 Background — why this needs testing

Composable Diffusion (Liu et al., 2022) builds a both-concepts image by **multiplying** the two single-concept distributions — a *product of experts* (PoE). At each denoising step you run the model once per concept and **add** their score predictions; adding scores = multiplying distributions, so you sample the **product** `p(x|c₁)·p(x|c₂)/p(x)`.

Multiplying is only exactly correct when the two concepts are **conditionally independent** — loosely, separate in the image *and* unrelated in cause. *Cat* + *grass* qualifies; so does *mailbox* + *sandstorm*. That is why PoE famously composes things never seen together.

> **Assumption (A):** spatial non-overlap ≈ conditional independence ≈ the joint is the product of the parts.

**The research question:** chest diseases can break this. The paper is a *controlled* test of what happens when they do — correlated pair vs. independent pair, same model, same method.

### Two kinds of "separate" — the idea the whole project turns on

> ### **Cardiomegaly and effusion are separate in space but linked in statistics.**

- **Separate in space** — the big heart sits centrally, the fluid sits at the lower edges. They don't overlap on the image.
- **Linked in statistics** — both come from heart failure, so they appear together far more than chance.

**Why the split matters — it divides the labor of the two models, and the two must not be confused:**

| Axis | Governed by | The concern |
|------|-------------|-------------|
| **Space** — do the signatures overlap? | the **VAE** | can the codec *draw* both at once? |
| **Statistics** — do they co-occur for a reason? | **PoE** | can composition get the *relationship* right? |

Spatial separability makes the VAE's job easy: it paints a heart-shape it knows next to a fluid-shape it knows, so it needs no both-disease training. It does **nothing** for PoE's job — the statistical link — which spatial separation cannot fix. **The trap is letting "the VAE is fine" imply "PoE is fine." Different axes, different problems.**

### The exactness rule, and why correlation breaks it

By Bayes, the PoE product equals the true joint **only** when the labels are conditionally independent given the image:

```
true joint:   p(x|c₁,c₂)         ∝ p(c₁,c₂|x) · p(x)
PoE product:  p(x|c₁)·p(x|c₂)/p(x) ∝ p(c₁|x)·p(c₂|x) · p(x)
                                       └── equal iff  p(c₁,c₂|x) = p(c₁|x)·p(c₂|x) ──┘
```

The mechanism is sharper than that abstract condition, though. Because we train each condition on **single-disease** images, the model never learns `p(x|c₁)` — it learns `p(x | c₁, **not** c₂)`.

- **Independent pair:** "c₁ alone" looks like "c₁ with c₂." The bias vanishes → product matches the joint → **PoE works.**
- **Correlated pair:** the shared cause makes "c₁ alone" systematically *milder* than "c₁ with c₂." Both factors miss the severe-coupled appearance → the product can't recover it → **PoE fails.**

> **Correlation is what makes the single-disease marginals biased in the first place.** That one sentence is the engine of the paper: same model, same method — the control pair reproduces the joint and the treatment pair shows a measurable gap *because of correlation alone.*

---

## 🎯 The hypotheses

**H1 — Marginals are preserved.** *(expected true — the gate)*
Composition reproduces each disease's own appearance.
→ Presence and single-disease measurements on composed images match real single-disease images.

**H2-control — The joint is reproduced for an independent pair.** *(the affirmative claim)*
For two *uncorrelated* diseases, composition reproduces the statistical relationship.
→ Composed vs real sits at the floor. This establishes that PoE is sound when its independence assumption holds.

**H2 — The joint shows a measurable gap for the correlated pair.** *(the limiting case)*
When diseases are correlated, composition does not fully reproduce the statistical relationship.
→ The joint distribution of (heart size, fluid blunting) in composed images differs from real both-disease X-rays, by more than the floor. Control reproduces the joint + treatment shows a gap ⇒ correlation is the cause, not the VAE or the training.

**H3 — The break is structural.** *(rules out the easy explanation)*
The H2 failure is intrinsic to the independence assumption — not a fixable imbalance.
→ No choice of composition weights closes the gap.

**H-anchoring — The composition anchor matters.** *(refinement)*
Anchoring at *health* approximates the joint better than anchoring at the *mixture*, because comorbid pathology is health plus added deviations.
→ Smaller H2 gap when we subtract `ε(z, normal)` than when we subtract the unconditional `ε(z, ∅)`. *(May also land "no difference" — the correlation gap can swamp it. Both publish.)*

In one line: **H1 holds · H2-control holds on the independent pair · H2 shows a measurable gap on the correlated pair · H3 confirms it's structural · H-anchoring asks if a smarter anchor helps.** That is the paper.

---

## ⚖️ The controlled comparison

Pick **two** disease pairs by their measured correlation. Same VAE, same LDM, same method — the *only* difference is correlation.

| Pair | Correlation | Prediction |
|------|-------------|------------|
| **Control** — an independent pair (TBD via Exp 1) | ≈ 0 (no shared cause) | PoE reproduces the joint — gap at floor |
| **Treatment** — cardiomegaly + effusion | high (shared cause: heart failure) | measurable gap above floor |

If the control pair reproduces the joint while the treatment pair shows a gap, no skeptic can blame the codec or the training run. Correlation is isolated as the cause.

---

## 🧪 Two ways the composition can fail — and how we tell them apart

When composed images don't match real both-disease images, **two different culprits look identical from the outside:**

1. **The composition can't build it.** PoE adds two single-disease experts; neither saw the diseases together, so neither knows the *coupling* (severe heart failure → bigger heart **and** heavier fluid, together). The math has no slot for it. *This is the failure we want to demonstrate.*
2. **The latent space can't hold it.** If the VAE can't represent the combined appearance, even a *perfect* composition decodes to something wrong — the destination isn't on the map.

A type-2 failure wears a type-1 costume. To attribute the headline result we **must** separate them — that is the job of the **ceiling check** (experiment 3): reconstruct *real* both-disease images through the VAE and see whether the latent can hold the joint at all.

---

## 📏 Metrics, in plain words

Read once and the experiments below are self-explanatory.

- **Correlation (φ / odds ratio)** — for a label pair, how much more they co-occur than chance. `0` = independent. Picks the treatment and control pairs.
- **Presence rate** — how often a disease classifier says "yes, it's here." Tests whether a disease shows up *at all*.
- **The floor** — split the real both-disease set in half, measure the halves against each other. Real-vs-real, so the *smallest gap measurable*. Everything is judged against it.
- **Two-sample test (the key idea)** — train a small classifier to tell *generated* from *real*.
  - Can't tell apart → same distribution → composition worked.
  - Can tell apart → they differ → composition failed.
  - Reported `0.5`–`1.0`: `0.5` = indistinguishable *(good)*, `1.0` = trivially separable *(bad)*. *(Formally "C2ST AUC"; you only need the 0.5-vs-1.0 intuition.)*
- **Distribution distance** — one number for how far two image sets are apart; `0` = identical. *(Formally "MMD"; a second angle.)*
- **FID** — standard image-realism distance; lower = more realistic.
- **Reconstruction quality (SSIM/LPIPS)** — how faithfully the VAE rebuilds an image it was given. Powers the ceiling check.

---

## 📋 At a glance

| # | Experiment | Tests | What it answers | Cost |
|---|------------|-------|------------------|------|
| **1** | Correlation matrix *(gate)* | — | Which pairs are correlated? Is there a gap to find at all? | ⚡ no GPU |
| **2** | Train the VAE | — | Can we compress/rebuild X-rays well? | 🏋️ train |
| **3** | Ceiling check | — | Can the latent even *hold* the both-disease look? | 💨 cheap |
| **4** | Train the single-disease LDM | — | Can we generate convincing single-disease X-rays? | 🏋️ train |
| **5** | Marginals check *(gate)* | **H1** | Does each disease show up on its own? | 💨 cheap |
| **6** | Joint-structure test *(headline)* | **H2 · H2-control · H-anchoring** | Is the correlation reproduced — and only for the correlated pair? | 💨 cheap |
| **7** | Reweighting control | **H3** | Can tuning weights fix the gap? | 💨 cheap |
| **8** | Floor baseline *(sanity)* | — | Does PoE beat just averaging two images? | 💨 cheap |

**Run order (by dependency):** `1 (gate) → 2 → 3 (ceiling) → 4 → 5 (gate) → 6 (headline) → 7 + 8`.
Two gates: experiment 1 decides whether the project is worth running; experiment 5 decides whether experiment 6 is interpretable. The ceiling check (3) must clear before any composition claim.

<sub>Cost: ⚡ labels only · 🏋️ training-bound · 💨 inference-only. Hypothesis tags double as the `skeleton-paper` claim links (no `CLAIMS.md` yet, so tagged `local:`).</sub>

---

## 1 · Correlation matrix  ⚡ → gate + pair selection

> **Which disease pairs actually co-occur more than chance — and is our intended pair one of them?** *(do this before any modeling)*

- **Build:** over the 19 labels, compute the pairwise co-occurrence matrix (φ-coefficient or odds ratio) from the label table alone.
- **Triple duty:** picks the **treatment** pair (strongest correlation, ideally a known clinical interaction) · picks the **control** pair (correlation ≈ 0) · is the **go/no-go gate**.
- **Decision:**
  - ✅ **go** — a strongly correlated pair *and* a near-zero pair both exist → the controlled comparison is possible.
  - ❌ **pivot** — if the intended pair is only weakly correlated, there is no gap to find; choose a stronger pair or rethink the framing.
- **Figure:** the 19×19 correlation heatmap with treatment and control pairs ringed.
- **Compute:** labels only — zero GPU-hours.

> 💡 The both-disease labels are load-bearing **three** ways: they define experiment 6's target (the floor), select treatment + control, and gate the project here.

## 2 · Train the VAE  🏋️

> **Can we compress an X-ray to a latent and rebuild it faithfully?** *(prerequisite, not a hypothesis)*

- **Build:** one VAE, **label-blind** — the network takes *no* disease label. Labels are used only to *curate* the training set so it covers every appearance the LDM will later generate.
- **Both-disease images:** **not** held out by default. The hold-out belongs to the *LDM* (the prior), not the codec. Including both-disease images can only *raise* the ceiling. Whether you need them is decided by experiment 3 — for spatially separable diseases you likely don't.
- **Measure:** reconstruction quality (SSIM/LPIPS).
- **Gate:** poor reconstructions → everything downstream is built on sand. Fix first.
- **Figure:** real vs reconstructed panel.
- **Compute:** 256² · f=8 → 32×32×4 latent · bf16.

> ⚠️ **The >11GB memory risk on the 4070.** Fallback: train at 128², or use a frozen pretrained CXR VAE. Checkpoints → `$HOME/PhD/Paper3/ckpts` with a `df` disk guard.

## 3 · Ceiling check  💨 → separates the two failure modes

> **Can the latent space even hold the both-disease appearance?** *(without this, the headline result is unattributable)*

- **Do:** take *real* both-disease images, encode → decode through the VAE, compare reconstructions to the originals (SSIM/LPIPS + a two-sample score).
- **Why:** this is the line between "composition can't build it" (the result we want) and "the latent can't hold it" (a codec ceiling masquerading as a composition failure).
- **Decision:**
  - ✅ **ceiling is high** — reconstructions match → the latent holds the joint → any later gap in *composed* images is a genuine composition failure. Proceed.
  - ❌ **ceiling is low** — reconstructions already off → fix the VAE (feed it both-disease images, or use a better codec) **before** any composition claim, and re-run.
- **Figure:** real vs VAE-reconstructed both-disease panel, side by side.
- **Compute:** inference only — needs the VAE only, not the LDM.

## 4 · Train the single-disease LDM  🏋️

> **Can the model generate convincing single-disease X-rays — and support both composition anchors?** *(prerequisite, not a hypothesis)*

- **Build:** one conditional LDM on single-disease latents only — conditions = `normal`, `cardiomegaly`, `effusion`, … (whatever pairs you'll compose). **It never sees a both-disease image** — this is *the* hold-out that makes composition a real test.
- **CFG dropout:** also train with the label randomly dropped, so the model learns an unconditional null `ε(z, ∅)`. Now **one model supports two anchors at inference** — subtract `ε(z, ∅)` *or* `ε(z, normal)` — no retraining (this powers H-anchoring).
- **Measure:** single-disease sample quality (FID vs real single-disease images).
- **Gate:** unconvincing single-disease samples → the composition test means nothing.
- **Figure:** single-disease sample grid.
- **Compute:** RunPod A4000/A5000 · batch 16 bf16 · ~8h for 100k steps.

> 📄 **Detailed operational design → [`plans/single-disease-ldm/EXPERIMENTS.md`](plans/single-disease-ldm/EXPERIMENTS.md)**
> Covers: hypothesis + three-way falsification, class config + weighted sampling spec,
> W&B logging spec (metrics, artifacts, intervals, success criteria), RunPod job config
> (GPU choice, disk guard, kill criteria), CFG weight sweep for OOD compositional eval.

## 5 · Marginals check  💨 → tests **H1**  *(the gate before the headline)*

> **Does each disease show up correctly on its own?**

- **Compare:** single-disease composed images vs real single-disease images.
- **Measure:** presence rate + the single-disease feature (heart size, or blunting).
- **Also validate here:** the heart-size / blunting extractor behaves the same on generated and real images.
- **Decision:**
  - ✅ **H1 supported** — presence within 5 points of real for *both* diseases, and two-sample score ≤ 0.60.
  - ❌ **gate trips** — marginals not preserved → experts too weak; fix experiment 4, don't read experiment 6 as a joint result.
  - 🔁 **inconclusive** — two-sample score 0.55–0.60 → more samples.
- **Samples:** ≥2000 per disease, with confidence intervals.

## 6 · Joint-structure test  💨 → tests **H2 · H2-control · H-anchoring**  ⭐

> **Does the composed both-disease distribution match real both-disease X-rays — and does that depend on correlation, and on the anchor?** *(the headline)*

- **Compare:** PoE-composed images vs real both-disease images, both passed through the VAE (removes codec distortion as a confound — the ceiling check already proved this reference is sound).
- **Measure:** two-sample score + distribution distance on the **joint** (heart size, fluid blunting) — the *pair*, not each alone.
- **Why this metric:** presence can't reveal a broken correlation — both diseases can be present yet combined unnaturally. Only the *joint* catches it.
- **Three arms, one experiment:**
  - **Control pair, null = `∅`** → tests **H2-control** (same method, uncorrelated pair — the affirmative case).
  - **Treatment, null = `∅`** → tests **H2** (standard Liu PoE on the correlated pair — the limiting case).
  - **Treatment, null = `normal`** → tests **H-anchoring** (deviations-from-health anchor). One model, anchor swapped at inference.
- **Decision (per arm, vs the floor):**
  - ✅ **gap is real** — two-sample score ≥ 0.65, above the floor's 95% upper bound.
  - ❌ **no gap** — score ≤ 0.55, indistinguishable from real → PoE reproduces the joint *(for the control arm: expected; for the treatment arm: a clean publishable null)*.
  - 🔁 **inconclusive** — 0.55–0.65, or overlapping the floor → more samples, don't over-read.
- **What a result looks like:** treatment/`∅` ≈ **0.78**, control/`∅` ≈ **0.57** (≈ floor 0.55), treatment/`normal` ≈ **0.69** → *joint broken only when correlated; health anchor helps but doesn't cure.*
- **Figures:**
  - **corroboration** — (heart size, blunting) scatter: real vs composed, floor overlaid. Each axis overlaps; the diagonal coupling differs.
  - **anti-corroboration** — the same plot with composed points *inside* the real cloud (score ≈ 0.5). That would refute H2.

> ⚠️ **Power flag:** if the real both-disease set is only a few hundred images, the floor is wide and a test can land "inconclusive" on sample size alone. Check N first.

## 7 · Reweighting control  💨 → tests **H3**

> **Can tuning the two composition weights close the gap?** *(structural, or just a knob?)*

- **Vary:** weights `(w₁, w₂)` over a small grid `{0.5, 0.75, 1.0, 1.5, 2.0}²` (incl. 1.0), best chosen on a validation split.
- **Logic:** weights can only rescale how loud each disease is — they cannot invent the *interaction* features neither expert ever saw. If the gap survives the best weights, it is structural.
- **Decision:**
  - ✅ **H3 supported (structural)** — even the best weights leave the test score ≥ 0.65.
  - ❌ **H3 rejected** — some setting brings it ≤ 0.55 → the gap was just imbalance *(still interesting — reshapes the story)*.
  - 🔁 **inconclusive** — between → more samples.
- **Note:** tuning two numbers on validation injects a *minimal* amount of joint information — deliberate; it's the cheapest possible "fix" and it's the control.

## 8 · Floor baseline  💨 → sanity

> **Is PoE actually better than just averaging two images?**

- **Compare:** PoE-composed vs a naive overlay (average two single-disease images).
- **Measure:** presence rate + FID.
- **Decision:**
  - ✅ **PoE is meaningful** — beats overlay on *both* presence and FID, beyond the confidence interval.
  - ❌ **PoE adds nothing** — if it can't beat trivial mixing, reconsider the method before any joint claims.

---

## 🔒 Pre-registration · 2026-06-05

Decided *before* any run, so results can't be narrated after the fact.

| Outcome | What we will see |
|---|---|
| **Claim holds** | Exp 5 marginals clean · Exp 6 control ≈ floor **and** treatment ≥ 0.65 above floor · Exp 7 gap survives reweighting · Exp 8 PoE beats overlay |
| **Claim wrong** | Exp 6 treatment ≤ 0.55, indistinguishable from real → PoE reproduces the joint even when correlated *(clean null, still publishable)* |
| **Anchoring** | smaller treatment gap under `normal` than `∅` → health anchor helps; or no difference → null choice is second-order. Either is reported. |
| **In-between** | inconclusive → rerun with more samples. Not reinterpreted. |

---

## ✅ Before the first run

- [ ] **Run experiment 1 first** — it's labels-only and it gates everything. Confirm a strongly correlated pair *and* a near-zero control pair exist in *your* data.
- [ ] Report the real both-disease **N** for each pair → power check for the experiment 6 floor.
- [ ] Confirm the **heart-size + blunting extractor** works on your data, and name the coupling it must capture (e.g. heart size and fluid volume rising together with severity). *Experiment 6 is not testable without it.*
- [ ] Decide the VAE — train from scratch (256²/128²) or use a frozen pretrained CXR VAE; let experiment 3 decide whether both-disease images go in.

---

## ⚡ Fastest derisk — the 2-D toy ladder, before any GPU-hour

Run these in order. They exercise the entire study in a day, with no VAE and no extractor — and rung 1 *is* the project's real go/no-go gate.

1. **Correlation matrix.** Build the 19×19 label co-occurrence matrix; print the top-5 and bottom-5 pairs. *This is experiment 1 — do it for real here.*
2. **Toy, independent.** Two uncorrelated labels on 2-D points, trained separately. PoE them → composed cloud lands on the true joint (score ≈ 0.5). *PoE works under independence.*
3. **Toy, correlated.** Make the two labels share a latent factor. Re-run PoE → the gap appears against the floor. *Your whole paper in 2-D — and you'll see the single-label clouds are biased (the marginal-bias mechanism).*
4. **Both anchors in 2-D.** Add a "mixture" null and a "normal" cluster; compose subtracting each; compare the gaps. *That's H-anchoring before any X-ray.*
5. **Ceiling vs composition in 2-D.** Cripple the toy "decoder" so it can't represent the joint region → a gap appears *even for the independent pair*; fix the decoder → it vanishes. *Feel the difference between a representational ceiling and a composition failure.*

> **If the gap doesn't appear in rung 3, it won't appear on X-rays either.**

---

## 📌 Still thin (flagged honestly)

- **The control pair is unverified.** Cardiomegaly + fracture is a plausible independent pair, but experiment 1 must confirm it's actually ≈ 0 in *your* data. If nothing is near-zero, the controlled comparison weakens — surface it, don't paper over it.
- **The interaction feature for the figure** — "heart size and fluid rise together with severity" is the predicted coupling, but you still need an extractor that measures it reliably on generated images (validated in the experiment 5 gate).
