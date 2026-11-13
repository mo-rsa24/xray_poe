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

> 📄 **Detailed operational design → [`plans/01-single-disease-ldm/EXPERIMENTS.md`](plans/01-single-disease-ldm/EXPERIMENTS.md)**
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

---

# 🛠 Fix-and-recheck plan · 2026-06-24

This section was added after running experiments 5 and 6. It does not re-open the diagnosis below — it just lays out what to fix, what to re-measure, and how to decide.

**A few words used a lot, in plain terms:**
- **The detector** — the small classifier that looks at an X-ray and says which disease is present.
- **Both-disease X-ray** — a real chest X-ray that has *both* a big heart and chest fluid.
- **Heart-width ratio** — how wide the heart is compared to the chest. A geometric measurement of the picture, so it can't be fooled the way the detector can.
- **Guidance strength** — how hard we push the model toward a disease while it draws. Higher = pushed harder.
- **The two claims this protects** — **N1**: composed X-rays show both diseases as often as real ones do. **N2**: composing beats just pasting two X-rays on top of each other.

## What went wrong (settled — do not re-investigate)

Two separate problems. They are different and must not be merged.

1. **The detector can't see both diseases at once.** It was only ever trained on X-rays with *one* disease (or none) — never on X-rays with both. So it learned to pick one answer, not two. On our composed X-rays it almost never reports both. But here's the giveaway: even on *real* both-disease X-rays it only reports both about 1 in 10 times. That ceiling is the detector's fault, not the generator's — **any** detector trained this way would do the same. This problem has nothing to do with which disease pair we pick.

2. **The model draws weak hearts.** When asked for a big heart, the heart it draws barely scores higher than a normal one — far below a real enlarged heart. This one *is* specific to the heart: hearts are a whole-chest shape and are hard to draw, and it shows up whether or not fluid is also requested.

> ## 🔒 The one thing we must not break (claim N2)
>
> Our headline result is "composing beats pasting two X-rays on top of each other." Pasting only looks bad when one disease changes the **whole chest shape** and the other sits in **one region**. A big heart is the only common whole-chest finding. So **big heart + chest fluid** is the pair the whole paper rests on — pasting them gives an obvious double-exposure ghost, and it's the classic heart-failure picture (1063 real examples). If we swap the big heart for another *region* disease (e.g. patchy opacities), pasting works almost as well and our advantage disappears. So the pair is not a free choice. Replacing the big heart is a last resort, never called a "fix," and we never swap to a pair where the two diseases sit apart and don't interact.

**Order — all three are cheap, and all run before we decide anything about the pair:** E1 → E2 → E3 → decide.

---

## EXP-E1: Retrain the detector so it can report both diseases
- claim_id: `local:N1`
- scope: Independent of which pair we use — needed either way. Until this is fixed, every "both diseases present" number (real and composed) is unreliable.
- independent_var: what the detector is trained on — add a group of real both-disease X-rays so it learns that "both at once" is allowed.
- lever: retrain the detector with that both-disease group included. (Build step: the folder of both-disease images doesn't exist yet — only the one-disease folders do. Make it from the real both-disease list `data/nih/partition_testboth.parquet`. See `plans/02-pre-evaluation/plans/03-classifier-finetune.md`.)
- dependent_var: how often the fixed detector reports both diseases on *real* both-disease X-rays.
- metric: that "both reported" rate on real X-rays. If the old low ceiling was the training's fault, a detector that has seen both-disease X-rays should clear it easily. If it stays stuck low even now, then counting "both present" is simply the wrong headline measure.
- sample_size: all real both-disease X-rays (report how many; about 1000), plus about 1000 composed ones for the re-check. Report the rate with a confidence interval.
- falsify_condition: **fixed (good)** — the rate on real X-rays rises above roughly 15 in 100 → the measure is now fair, go re-check the composed X-rays against it. **Not fixed** — stays at about 10 in 100 or below → counting "both present" is the wrong headline; switch N1 to a different measure. **Unclear** — lands between (about 10–15 in 100) → add more images, re-check calibration, run again; don't spin it.
- figures: bar chart of the "both reported" rate on real X-rays, old detector vs fixed detector, with confidence bars. (Anti-figure: the fixed detector still stuck at the old low line — that would mean the measure itself is the problem.)
- compute: hippo, one GPU. Retraining is short; reading the rate is just inference. Save to `$HOME/PhD/Paper3/ckpts` after a disk-space check.
- status: ⚠️ pending

## EXP-E2: Measure the composed heart by its shape, not the detector
- claim_id: `local:N1` (also de-risks the N2 pair)
- scope: Specific to the heart. Asks a simple question: are the composed hearts actually drawn wide, even if the detector missed them?
- independent_var: how we measure the heart — the detector vs. the heart-width ratio measured directly from the picture (using the HybridGNet outline tool in the inspection harness; `figures/visual_inspection/`, `plans/02-pre-evaluation/plans/04-visual-inspection.md`).
- lever: outline the heart and lungs, compute the heart-width ratio from geometry. This can't be fooled by the detector's one-answer habit.
- dependent_var: heart-width ratio of composed hearts vs real enlarged hearts vs normal hearts.
- metric: heart-width ratio. It cleanly splits the two possibilities: if composed hearts measure wide (close to real enlarged hearts), the **model was fine and the detector was the problem**. If they measure normal-width, the **model really can't draw a big heart**.
- sample_size: about 1000 composed hearts, plus real enlarged and real normal hearts to compare against. Report the typical value and spread for each group.
- falsify_condition: **was measurement** — composed hearts measure clearly wider than normal, most of the way toward real enlarged hearts. **model really fails** — composed hearts measure no wider than normal. **Unclear** — somewhere in between → more images and cross-check with E3.
- figures: the three groups (normal · composed · real enlarged) as overlaid distributions; composed landing on real-enlarged means it was a measurement problem. (Anti-figure: composed landing on normal — proof the model floors the heart.)
- compute: hippo, inference only, cheap. Reuses the existing inspection harness. No training.
- status: ⚠️ pending

## EXP-E3: Try pushing the heart harder
- claim_id: `local:N1`
- scope: Specific to the heart. We've only ever drawn it at the default push. A whole-chest shape may simply need a harder push than a small fluid region does.
- independent_var: guidance strength on the heart — try a few settings from the default up to much stronger (default, then progressively harder, five settings in all).
- lever: turn up the guidance strength on the heart while drawing. No retraining needed — the model already supports this.
- dependent_var: the composed heart-width ratio (measured the E2 way, not the detector) at each push setting.
- metric: heart-width ratio vs push strength. If a harder push is the answer, the heart should get wider as we push harder, up toward real enlarged hearts. If the width doesn't move no matter how hard we push, pushing isn't the answer.
- sample_size: about 500 composed hearts at each of the five settings. Also read the fluid at each setting, so we catch it if a harder heart push starts ruining the fluid or realism.
- falsify_condition: **pushing works** — the heart widens as we push harder and reaches the real-enlarged range at some setting. **pushing doesn't work** — width stays flat across all settings → the cause is something else. **Unclear** — width rises but stalls short, or a harder push wrecks the fluid/realism → report the trade-off, don't force a setting.
- figures: a curve of heart-width vs push strength with the real-enlarged range shaded, fluid level drawn alongside to show the cost. (Anti-figure: a flat line — pushing harder changes nothing.)
- compute: hippo, drawing 500 images at each of five settings — the main GPU cost here. Run the settings one after another (no job scheduler on hippo). Save under `$HOME/PhD/Paper3` after a disk-space check.
- status: ⚠️ pending

---

## 🔒 The decision, decided now (before we look at results)

Read this only after E1, E2, and E3 are all done. E1 decides whether our "both present" measure can be trusted at all. E2 and E3 together decide whether the model can draw a big heart.

| If… | Then | Keeps safe |
|---|---|---|
| **Keep the pair (the win)** — E2 shows the heart *is* drawn wide, **or** E3 shows a harder push widens it | **Keep big heart + chest fluid.** The "failure" was bad measurement and/or too gentle a push, not the model. Lock in the fixed detector and the working push, and re-run experiments 5 and 6. | Both N1 and N2 — nothing swapped |
| **Swap the heart (last resort only)** — **all three** still say the model genuinely can't draw a big heart | Replace **only the heart**, keep the **fluid**, and pick a replacement that still **physically interacts** with the fluid so pasting still fails: collapsed-lung + fluid (3275 examples) or pneumonia/consolidation + fluid (1287). **Never** a pair where the two sit apart and don't interact — that quietly kills the paper. | N2 — pasting must still fail |
| **Change the measure (separate, depends only on E1)** — E1 says the detector still can't report both even after retraining | Drop "count how often both are present" as the headline. Use the joint heart-size-and-fluid comparison from experiment 6 instead. This is independent of the heart question. | N1 — keeps a measure we can trust |

**Either way we keep the pair:** re-measure the "fluid works, heart doesn't" comparison with the fixed detector (E1) and the shape measurement (E2). Fluid's apparent success is partly just the old detector being freed to report it, so it has to be re-checked before we claim it.

**For the write-up (decided now):**
- Always report "both present" *relative to what real X-rays score*, never as a bare number.
- Say plainly in the methods that the first detector was trained on one-disease X-rays only, that this is why the first numbers were unreliable, and that E1 fixed it.
- If we swap the heart, say plainly it was dropped because the *model* couldn't draw it (E1+E2+E3), and that the replacement still keeps one whole-chest disease and one region disease so pasting still fails.
