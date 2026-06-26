<div align="center">

# 🫁 Experiments — PoE Composition of Single-Disease Experts for Pixel-Verifiable Co-Morbid X-Rays

**Can we manufacture valid co-morbid chest X-rays by composing two single-disease diffusion experts — with no co-morbid training data — and use them to improve a data-starved classifier, beating a naive overlay?**

`scope: prove the utility claim on the proxy (cardiomegaly + effusion)` · `cluster: hippo` · `tracking: W&B` · `latent: 512²/f=4 → 4×128×128` · `pinned: 2026-06-24`

> Authoritative source: [`docs/EVAL_PHILOSOPHY.md`](docs/EVAL_PHILOSOPHY.md). This file is the runnable contract; the philosophy doc is the why. If they ever disagree, the philosophy doc wins and this file is wrong.

</div>

---

## 0 · What we are claiming (read this first)

The question the whole paper investigates (the **lead is exposure, not a win**):

> **Can single-disease diffusion experts, composed via Product-of-Experts from *no* co-morbid training data, synthesize co-morbid chest X-rays faithful enough to help a data-starved classifier — and if not, where exactly does it break?** We test this on a proxy (cardiomegaly + effusion) where every claim is pixel-verifiable via CTR + costophrenic geometry. The paper leads by **exposing the comorbidity gap**: validate individual-disease synthetics first, then compositional ones, then measure downstream utility against an oracle ceiling. A utility gain is the upside case; a clean null — *"composition in latent space does not translate to composition in disease space"* — is an equally informative, pre-registered result (§5a). Entangled presentations and the de-tangling mechanism are scoped out as future work.

Roles to keep straight, because they are easy to merge by accident:

- **The frame (lead)** is *mechanistic exposure*: does compositional generation actually produce classifiable co-morbid signal? We report where it succeeds and where it fails — **before** any utility claim — and our own evidence (cardio under-renders; PoE yield NULL pre-retrain) says the honest current answer is "not yet."
- **The contribution** is the *source*: co-morbid training images built from single-disease-only inputs, defended against the **overlay** baseline.
- **The utility test** is whether gated synthetic co-morbid data makes a starved classifier better — measured by **accuracy + sensitivity + specificity *and* AUC-by-stratum** on the pixel-verifiable subset. A null here is publishable, not a failure.
- The **mechanism** question — does PoE *match the oracle*? — is only answerable on the **proxy** (cardio + effusion), because that is the only place an oracle can be built. The real target, **silicoTB**, has no abundant co-morbid data and so can only ever show the practical claim. **silicoTB is not an experiment here.** We have no access to that data. It is named only as the motivating application in the paper's discussion / future-work.

**Reporting discipline (supervisor-aligned, 2026-06-26).** Every synthetic result is rendered on **both** the pre-retrain checkpoint (`model_step0040000`, test-patient-contaminated) and the rebalanced-cardio checkpoint — a **before/after** that visualizes the retrain fix. Before-model numbers are valid for *problem-exposure* figures; before-model *utility* numbers carry a contamination/leakage label and are illustrative only — clean utility comes from the after-model. The reconciliation that set this scope is `docs/SUPERVISOR_RECONCILE.md`.

**Proxy for everything below:** cardiomegaly + effusion on NIH ChestX-ray14. Every experiment runs on the proxy, on **hippo**, tracked in **W&B**.

---

## 1 · The protocol — Gate 0–6, pre-registered

This is the trust chain. Pre-registering it means we commit *now* to the order and the pass/fail rules, before any run, so a result can't be narrated into a win after the fact. An image — real test film **or** synthetic training image — flows through the gates in order; failing a gate routes it out (to "scoped-out" or "discarded"), it does not silently pass.

| Gate | Question | Pass rule |
|---|---|---|
| **0 — Well-posedness** | Is the label verifiable from the pixels alone? | PA view, heart border visible. *Entangled-both fails here → scoped out as future work.* Defines the study population. |
| **1 — Construct validity** | Does the detector fire on the right anatomy? | GradCAM saliency-mass inside the heart box (cardio) / CP region (effusion) ≥ threshold; spot-checked. Kills shortcut learning. |
| **2 — Criterion agreement** | Does the detector agree with its canonical measure, by an *independent* path? | Cardio detector (CTR-taught) confirmed by **GradCAM + radiologist**, never by CTR again. |
| **3 — Co-presence certificate** | On a both-image, do **both** single-disease gates pass *at the same time*? | CTR > 0.5 (PA) **and** CP angle blunted, simultaneously. *This — not FID — is the "both" claim.* |
| **4 — Distributional floor (FID)** *(demoted — appendix sanity check)* | Are these coherent, in-distribution chest films? | FID reported in the appendix only — catches garbage, certifies nothing about co-presence. **Not a headline gate.** |
| **5 — Ecological validity** | Would a radiologist accept this as a real patient? | Per-image human read. Final authority. |
| **6 — Downstream utility** | Does the PoE-augmented classifier beat **starved** *and* **overlay** — and on the proxy, how close to the **oracle**? | **Accuracy + sensitivity + specificity *and* AUC by stratum** (normal / cardio-only / effusion-only / both-separable). A null vs overlay is an accepted, pre-registered outcome. |

**The symmetry rule (load-bearing):** Gates 0–5 run on **both** the real test films *and* the synthetic training images. The same CTR + CP + radiologist filter that certifies the test set also decides which generated images are allowed into augmentation. A synthetic image that fails the gate is **discarded, not trained on**.

**The independence rule (prevents circularity):** CTR *teaches* the cardio classifier (it is the label source). The classifier's independent witnesses are **GradCAM + radiologist** — GradCAM reads where the net looks (not the CTR signal), the radiologist is independent of everything. **CTR is never used to both teach and validate the same model.**

---

## 2 · The seven experiments, and how they sit on the gates

Every experiment below maps onto a part of the gate pipeline. The gates are the protocol; the experiments are how we build and test each link.

| # | Experiment | Builds / tests | Gate it serves | Cost |
|---|---|---|---|---|
| **1** | Dynamics study — count separable vs entangled | the study population + the size of the gap PoE fills | **Gate 0** | ⚡ labels + instruments, no training |
| **2** | VAE recon-fidelity at heart-border + CP-angle | the codec ceiling for separable films | feeds **Gate 3/4** | 💨 inference |
| **3** | LDM expert-cleanliness / separable-both **yield** | how many PoE samples actually pass co-presence | **Gate 3** | 💨 inference |
| **4** | Deconfounded single-disease split construction | clean single-disease training sets for the experts | feeds the experts behind **Gate 3** | ⚡ labels + curation |
| **5** | Reuse-vs-retrain decision gate (VAE / LDM) | whether existing checkpoints are good enough | decides the inputs to **Gate 3** | ⚡ decision (💨/🏋️ if retrain) |
| **6** | Oracle baseline on the proxy | the "as good as having the data" ceiling | **Gate 6** | 🏋️ one classifier train |
| **7** | **Headline** — overlay-vs-PoE stratified AUC | the utility claim | **Gate 6** | 🏋️ classifier arms + 💨 gating |

**Run order (by dependency):**
`1 → 4 → 2 → 3 → 5 → 6 → 7`

- **1** defines the population and must run first (it is also the go/no-go: if almost nothing is separable, there is no pixel-verifiable claim to make).
- **4** builds the clean expert training data.
- **2** and **3** measure whether the codec can hold, and the experts can render, the separable-both look.
- **5** reads 2 + 3 and decides reuse-or-retrain before anything expensive.
- **6** builds the oracle ceiling so **7** is interpretable.
- **7** is the headline.

<sub>Cost legend: ⚡ labels/instruments only · 💨 inference-bound · 🏋️ training-bound. No `CLAIMS.md` exists yet, so claims are tagged `local:` and anchored to the philosophy-doc thesis; a later `CLAIMS.md` reconciles against them.</sub>

---

## 3 · Metrics, in plain words

Read once and the experiment blocks are self-explanatory.

- **CTR (cardiothoracic ratio)** — heart width ÷ inner-chest width on a frontal film. **> ~0.5 on a PA film = cardiomegaly.** The canonical, geometric cardio measure. **Instrument: `scripts/ctr_overlay.py`** — `torchxrayvision` ChestX-Det **PSPNet** anatomical segmentation (heart-mask width ÷ lung-mask width), *not* the presence classifier. ⚠️ **Do not use `metrics/extractors.py` `heart_size` here** — it is a GradCAM proxy read *off the classifier*, so using it as the gate would be circular (independence rule). The proxy stays a *correlated feature*, never a Gate-0/2/3 witness.
- **CP-angle read** — sharpness/fill at the lower-lateral corner where diaphragm meets ribcage. **Blunted = effusion.** The robust witness — it survives a big heart. **Instrument: derive geometrically from the lung mask of the same PSPNet segmentation (`ctr_overlay.py`)** — *not* the GradCAM `blunting` proxy. (This also tightens the §8 "effusion instrument is thin" gap.)
- **Separable both** — a co-morbid film where *both* signs are independently readable (heart border visible **and** CP angle assessable). The opposite is **entangled** (one sign hides the other; CTR unmeasurable) → scoped out.
- **Yield (Gate-3 pass-rate)** — of the PoE samples we generate, the fraction that show *both* signs at once. This is where cardio under-rendering shows up as a number.
- **Presence rate** — how often a disease detector says "yes, here." Tests whether a disease shows up at all (a marginal check, weaker than co-presence).
- **AUC** — ranking quality of a detector over a population. The downstream outcome.
- **AUC by stratum** — AUC computed separately on normal / cardio-only / effusion-only / both-separable. A real utility gain should be **lumpy** (concentrated on both-separable), not flat.
- **Accuracy · sensitivity · specificity** (supervisor 7.5) — reported per disease category alongside AUC, threshold-based (Youden), in Tables 1–2. Accuracy alone hides class imbalance; sensitivity (catch the disease) and specificity (don't over-call it) are the rigorous pair the supervisors asked for, and they expose where the classifier fails on comorbidity specifically.
- **Bootstrap CI** — the uncertainty band on an AUC or a ΔAUC. "Disjoint CIs" = the difference is real, not noise.
- **FID** — population-level realism distance; lower = more realistic. A floor, never a co-presence certificate.
- **Recon fidelity (SSIM / LPIPS)** — how faithfully the VAE rebuilds an image it was given. Powers the ceiling check — but here we read it *locally*, at the heart border and the CP corner, not as a whole-image average.

---

## 4 · The ablation matrix (the headline, Exp 7)

Each row is one trained classifier. The only thing that changes is the **augmentation source**. Every synthetic image — PoE or overlay — passes through the *identical* Gate 0–5 filter before it is allowed in (symmetry rule).

| Configuration | Real both-disease | Gated PoE synthetic | Gated overlay synthetic | Expert training data | What it answers |
|---|---|---|---|---|---|
| **Starved** | scarce only | — | — | — | floor: what you get with no help |
| **+Overlay** | scarce | — | ✓ | — | does diffusion buy anything over pasting? |
| **+PoE** | scarce | ✓ | — | deconfounded (Exp 4) | **the claim** |
| **Oracle** | abundant | — | — | — | the ceiling — proxy only, impossible on silicoTB |

**Evaluated by stratum** (the same pixel-verifiable test set from Exp 1): normal / cardio-only / effusion-only / both-separable.

Claim holds if **+PoE > +overlay > starved**, with the gain concentrated on **both-separable**, and on the proxy +PoE sits closer to the oracle than overlay does.

---

## EXP-01: Dynamics study — count separable vs entangled
> **In plain words:** go through every NIH film labelled with *both* diseases and check, one by one, whether you can actually see both signs (a big heart **and** a blunted fluid corner). Count how many are "separable" like that. That count *is* your test set — and if it's tiny, there's nothing to prove.

- **claim_id:** `local:` population-exists — "a pixel-verifiable separable-both subset exists in NIH and is large enough to be a test set."
- **independent_var:** presentation type (separable vs entangled), stratified by view (PA vs AP).
- **dependent_var:** per-film CTR + CP-angle read; the fraction of both-labelled films that are *separable*.
- **ablation_rows:** n/a (characterization, not arms) — produces the **study population** every later experiment uses.
- **metric:** fraction separable. Why it must move: the whole pixel-verifiable claim rests on this subset existing. If almost no film is separable, there is nothing to test and no gap for PoE to fill. (AP films use a shifted/relaxed CTR threshold or are excluded — the threshold is view-dependent.)
- **sample_size:** *all* NIH both-labelled (cardio ∧ effusion) films. Report **N total, N PA, N AP, N separable, N entangled** with the fraction and a CI.
- **falsify_condition:** **support** if separable fraction ≥ 30% (enough films for a test set *and* a measurable gap); **null/pivot** if < 10% (population too small — the pixel-verifiable framing is not feasible, revisit scope); **inconclusive** if 10–30% → check whether AP-threshold adjustment or a looser border-visibility rule recovers films before deciding.
- **figures:** **corroboration** — stacked PA/AP bar of separable vs entangled counts, with the separable slice (= test-set size) called out. **anti-corroboration** — the same bar dominated by entangled, showing the pixel-verifiable subset is too thin to support a claim.
- **compute:** hippo, CPU/GPU light — segmentation + CP read over a few thousand films, no training. Cache CTR/CP per film to W&B as an artifact. Wall-clock ~hours.
- **disposition:** **reuse** `scripts/ctr_overlay.py` (PSPNet CTR + lung-mask CP); **new** = separable/entangled orchestration + count. Inputs: `data/nih/` + `Data_Entry_2017.csv` (cols: Image Index, Finding Labels, View Position). **Demote** `extractors.py` proxy.
- **procedure (calibrate → automate → audit; see §6):** (1) pull the cardio ∧ effusion film list (~N ≈ 1063), split PA/AP; (2) **hand-label ~30 PA films** with the `ctr_overlay` red/blue overlay open — *separable* (heart border **and** CP corner both readable) vs *entangled* (one hides the other) — this is the only by-hand labelling; (3) **validate the instrument** against those 30, require **≥ 90% agreement** before trusting it; (4) **run on all ~1063**, write per-film CSV (CTR, CP, view, separable flag); (5) **audit** a random 30 + every borderline (CTR 0.48–0.52, ambiguous CP); (6) **freeze** the separable set as the pixel-verifiable test set, log entangled as scoped-out (not deleted). You are **not** eyeballing 1063 by hand — you eyeball ~30 to *trust the instrument*, the instrument does the 1063.
- **status:** ⚠️ pending

## EXP-02: VAE recon-fidelity at heart-border + CP-angle
> **In plain words:** squeeze real both-disease films through the VAE and back out, then check the heart edge and the fluid corner still read the same. If the codec itself breaks them, a later "composition failed" verdict would be blaming the wrong thing — so we rule that out first.

- **claim_id:** `local:` codec-ceiling — "the 4×128×128 latent can hold the *separable* both-disease appearance, so a later composition gap is genuinely a composition failure, not a codec ceiling."
- **independent_var:** the image set passed through the VAE round-trip (real separable-both vs matched real single-disease).
- **dependent_var:** *localized* reconstruction fidelity — heart-border SSIM/LPIPS and CP-angle sharpness — **and** whether the canonical reads survive the round-trip (|ΔCTR|, CP class flip-rate).
- **ablation_rows:** n/a — this is the **ceiling check** that protects the attribution of Exp 7.
- **metric:** |ΔCTR| and CP-class-preservation across encode→decode, plus local LPIPS at the two witnesses. Why it must move: if the round-trip itself flips a sign read, no composition claim built on that latent is interpretable. We read *locally* because a good whole-image SSIM can still smear the exact border/corner that the gate depends on.
- **sample_size:** all separable-both films from Exp 1 + a matched single-disease control set (report N each).
- **falsify_condition:** **support (ceiling high)** if post-recon reads agree with originals within instrument tolerance (|ΔCTR| ≤ 0.02 and CP class preserved ≥ 95%) and local LPIPS below the single-disease control's level; **null (ceiling low)** if the round-trip flips the sign read (CTR crosses 0.5 **or** CP class flips) on > 10% of films → the VAE must be fixed/retrained before any composition claim (feeds Exp 5); **inconclusive** between → add films / inspect failures.
- **figures:** **corroboration** — real vs VAE-reconstructed separable-both panel, with CTR/CP read printed under each. **anti-corroboration** — the same panel where the reconstruction has smeared the CP corner or shrunk the heart so the read flips.
- **compute:** hippo, inference only — VAE only, no LDM. Reuse the canonical `vae_step0025000` checkpoint. Wall-clock ~hours.
- **disposition:** **extend** `scripts/vae_recon_pilot.py` (add localized heart-border / CP fidelity + |ΔCTR| / CP-flip metrics — current pilot is whole-image); **reuse** `ckpts/vae_step0025000.pt`; CTR/CP reads via `scripts/ctr_overlay.py`.
- **procedure (see §6):** input = the **separable-both set frozen by Exp 1** + a matched single-disease control; encode→decode each, run the CTR/CP instrument on original vs reconstruction, count sign-flips. Eyeball ~20 real-vs-recon pairs first (overlay on) to confirm the localized metric matches what you see, then run all.
- **status:** ⚠️ pending

## EXP-03: LDM expert-cleanliness / separable-both YIELD gate
> **In plain words:** generate a couple thousand composed images and count how many actually show *both* signs at once. That pass-rate ("yield") tells you whether the experts can draw a big heart and the fluid together — and it's exactly where the known cardio under-rendering shows up as a number.

- **claim_id:** `local:` yield — "PoE composition of the two experts produces separable-both images that pass Gate 3 (co-presence) at a high-enough rate to supply augmentation volume."
- **independent_var:** composition source = PoE-composed samples (using the experts from Exp 4); secondarily, expert cleanliness (deconfounded vs mixed-trained).
- **dependent_var:** **yield** — the Gate-3 pass-rate: fraction of PoE samples with CTR > 0.5 **and** CP blunted *simultaneously*.
- **ablation_rows:** feeds the **+PoE** arm of the Exp 7 matrix — yield sets how much gated synthetic data that arm can draw on.
- **metric:** yield (%) with CI, plus the per-sample CTR and CP distributions. Why it must move: this is exactly where the known **cardio under-rendering** surfaces as a number (the cardio expert renders CTR ≈ 0.395 « real ≈ 0.547). A low yield means the experts can't co-render the heart and the fluid in one separable image, which is the direct trigger for the rebalanced-cardio retrain lever (Exp 5).
- **sample_size:** ≥ 2000 PoE samples per composition setting, with CIs. Composition: `ε_compose = ε_uncond + w_c·(ε(z,cardio) − ε_uncond) + w_e·(ε(z,effusion) − ε_uncond)`.
- **falsify_condition:** **support** if yield ≥ 30% (enough gated images to augment); **null** if yield < 10% (experts cannot co-render separably as-is → composition source unusable without a fix → fire the rebalanced-cardio retrain lever in Exp 5); **inconclusive** if 10–30% → try a small expert-weight adjustment (`w_c`, `w_e`) and more samples before deciding retrain.
- **figures:** **corroboration** — CTR-vs-CP scatter of PoE samples with the Gate-3 pass region shaded; the passing fraction is the yield. **anti-corroboration** — the same scatter with the cloud collapsed below CTR 0.5 (heart never enlarges), the under-rendering signature.
- **compute:** hippo, inference only — VAE + LDM, sampling ~2k images per setting. Reuse `model_step0040000` set unless Exp 5 says retrain. Wall-clock ~hours per setting.
- **disposition:** **reuse** `scripts/generate.py` + `src/inference/cfg.py` (`cfg_compose`) for PoE sampling and `ckpts/model_step0040000.safetensors` (W&B lineage `prime_lab/paper3-ldm/runs/hifei736`); **new** = yield-gate harness (Gate-3 pass-rate + CTR×CP scatter); gate reads via `scripts/ctr_overlay.py`.
- **procedure (see §6):** generate a small batch (~50) first, run the Gate-3 instrument, eyeball the CTR×CP scatter + a few sample images to confirm the gate behaves; only then sample the full ≥ 2000 and compute yield. Watch the kill-criterion (yield → 0 early = stop, don't burn the full 2k).
- **status:** ⚠️ pending

## EXP-04: Deconfounded single-disease split construction
> **In plain words:** build clean training sets — cardio films with *no* effusion, effusion films with *no* cardio — so each expert only ever sees its own disease. Then check the cardio expert doesn't secretly draw fluid (and vice-versa). Otherwise "composition" isn't really composing anything; the leak did it.

- **claim_id:** `local:` clean-experts — "single-disease experts trained on de-confounded splits (cardio ∧ ¬effusion, effusion ∧ ¬cardio) render their own disease without leaking the other, so composition is the only place the two diseases meet."
- **independent_var:** the training-set definition for each expert — `cardio ∧ ¬effusion` and `effusion ∧ ¬cardio` vs the naive `cardio-any` / `effusion-any`.
- **dependent_var:** cross-leakage — does the cardio-only expert render effusion signs (CP blunting), and vice-versa — plus the size of each clean split.
- **ablation_rows:** supplies the **expert training data** for the **+PoE** arm; the mixed-trained variant is the contrast that shows deconfounding was necessary.
- **metric:** on ≥ 1000 samples from each expert, the presence rate of the *other* disease (CP blunting for the cardio expert; CTR > 0.5 for the effusion expert). A clean expert keeps cross-presence near the real single-disease base-rate. Why it must move: if the "cardio" expert was trained on cardio films that also carried effusion (they co-occur, φ = 0.13), it silently learns effusion too, and "composition" is no longer testing composition.
- **sample_size:** report **N** for each deconfounded split (these are smaller than the naive splits — that is the cost of cleanliness); ≥ 1000 expert samples for the leakage check.
- **falsify_condition:** **support** if deconfounded experts show cross-presence ≤ the real single-disease base-rate + a small margin, *and* the splits are large enough to train a usable expert; **null** if deconfounding does not reduce cross-presence vs mixed-trained (leakage comes from composition, not the training data — re-frame what the experts contribute); **inconclusive** between → more samples / inspect.
- **figures:** **corroboration** — bar of cross-presence: deconfounded vs mixed-trained expert, deconfounded near base-rate. **anti-corroboration** — the two bars equal, showing the split made no difference.
- **compute:** hippo, labels + curation to build the splits (⚡); the leakage check needs ≥1k samples from each trained expert (💨). If the experts must be retrained on the clean splits, that cost lands in Exp 5.
- **status:** ⚠️ pending

## EXP-05: Reuse-vs-retrain decision gate (VAE / LDM)
> **In plain words:** decide — *before* spending big compute — whether the VAE/LDM checkpoints we already have are good enough to reuse, or whether the cardio model needs a rebalanced retrain. The rule is committed up front so the call isn't made after seeing a disappointing number.

- **claim_id:** `local:` reuse-or-retrain — "the pre-refinement VAE/LDM checkpoints are either good enough to reuse for the refined scope, or they are not and must be retrained (rebalanced cardio)."
- **independent_var:** model provenance — reuse the shipped `vae_step0025000` / `model_step0040000` checkpoints vs retrain on the Exp 4 deconfounded splits with cardio rebalanced.
- **dependent_var:** whether the reused models clear the Exp 2 ceiling threshold **and** the Exp 3 yield threshold.
- **ablation_rows:** decides which models feed every downstream arm — a gate, not a sweep.
- **metric:** the decision rule itself — **reuse iff** (Exp 2 ceiling = high) **and** (Exp 3 yield ≥ 30%); otherwise **retrain**. Why it must move: retraining is the single most expensive action in the plan, so we commit the trigger in advance rather than deciding it after seeing a disappointing number.
- **sample_size:** reads Exp 2 and Exp 3 outputs; no new sampling for the decision. If retrain triggers, the LDM retrain uses the Exp 4 splits with the cardio:effusion imbalance corrected (the 7:1 data imbalance + effusion_weight that under-trained the cardio conditional).
- **falsify_condition:** **reuse justified** if both upstream gates pass; **retrain required** if either fails (low ceiling → VAE; low yield → rebalanced-cardio LDM); **inconclusive** if either upstream result is itself inconclusive → resolve upstream (more samples) before committing compute.
- **figures:** **corroboration** — a small 2×2 decision table (ceiling pass/fail × yield pass/fail) with the chosen cell ringed. **anti-corroboration** — n/a (this is a decision record, not a measurement); the honest-flag is logging *which* gate forced a retrain.
- **compute:** the decision is ⚡ free. If retrain triggers: 🏋️ LDM retrain on hippo (single box — sequential, not an array job), checkpoints to `$CKPT_DIR` behind a `df` disk guard, W&B-logged. Wall-clock: hours-to-overnight for the retrain.
- **status:** ⚠️ pending

## EXP-06: Oracle baseline on the proxy
> **In plain words:** train one classifier on *lots* of real both-disease data — the "if only we had the data" ceiling, which is only possible on the proxy. If even this ceiling barely beats the starved baseline, the test set has no room to show any improvement, and the headline test below wouldn't mean anything.

- **claim_id:** `local:` oracle-headroom — "an oracle trained on abundant real co-morbid data sets a meaningful ceiling above the starved baseline, so the test set has headroom for any method to demonstrate utility."
- **independent_var:** training data = abundant *real* both-disease films (possible only on the proxy).
- **dependent_var:** downstream AUC on the pixel-verifiable test set, by stratum.
- **ablation_rows:** the **Oracle** row of the Exp 7 matrix — the ceiling that makes "as good as having the data" a real comparison.
- **metric:** oracle AUC, and the oracle − starved gap on both-separable. Why it must move: without the oracle, Paper-1-vs-2 (does the synthetic data match having the real data?) is unfalsifiable; and if the oracle barely beats starved, the test set is saturated and Exp 7 cannot show utility no matter how good PoE is.
- **sample_size:** all available real both-disease proxy films for training (report N); test on the Exp 1 separable set; ≥ 3 seeds, bootstrap CIs.
- **falsify_condition:** **support (headroom exists)** if oracle − starved ΔAUC ≥ 0.05 on both-separable; **null (no headroom)** if ΔAUC < 0.02 → the test set is saturated, Exp 7 is uninterpretable as a utility test → revisit the test set / stratum before running Exp 7; **inconclusive** between → more seeds.
- **figures:** **corroboration** — AUC-by-stratum bars for oracle vs starved, headroom visible on both-separable. **anti-corroboration** — oracle ≈ starved everywhere, showing no room for any method to help.
- **compute:** hippo, one classifier train (🏋️) + inference. Checkpoints to `$CKPT_DIR` behind the `df` guard. Wall-clock: hours.
- **status:** ✅ **done 2026-06-26 — NULL (no headroom).** Oracle (555 real both-films, ×3 seeds) − Starved (50 films, ×3 seeds), both-separable **ΔAUC = −0.012** (cardio −0.010 [−0.018,−0.002]; effusion −0.013 [−0.027,−0.000]) → **< 0.02 ⇒ null, test set saturated.** The CIs sit entirely below 0 — abundant real comorbid data does *not* beat the scarce floor. Sens/spec filed (oracle cardio 0.773/0.500, effusion 0.675/0.784). Construct-validity: GradCAM pointing-game **0.23 < 0.80 FAIL** (the CTR-relabelled cardio detector does not attend to the heart); Gate-2 radiologist read still pending. Artifacts: `results/exp06_oracle/{verdict,detector_construct_validity,patient_disjoint_check}.md`, `figures/exp06_oracle_vs_starved_auc.png`. **Consequence (per falsify_condition + D10): the saturated test set makes any Exp 7 utility null *uninterpretable* → revisit the test set / stratum BEFORE running Exp 7.** Routes to §5 "upstream blocks the headline."

## EXP-07: Headline — overlay-vs-PoE stratified downstream AUC  ⭐
> **In plain words:** train four classifiers — no help, overlay-augmented, PoE-augmented, and the oracle — and compare how well each spots disease on every kind of film. The claim wins only if PoE beats *both* the overlay and the no-help baseline, specifically on the films where both diseases are visible.

- **claim_id:** `local:` headline-utility — the full thesis: gated PoE synthetic data improves the data-starved classifier on the pixel-verifiable subset, beating overlay and starved, and on the proxy closing toward the oracle.
- **independent_var:** the augmentation source for the data-starved classifier — none (**starved**) / **overlay** / **PoE** — with **oracle** as the proxy-only ceiling.
- **dependent_var:** AUC by stratum (normal / cardio-only / effusion-only / both-separable).
- **ablation_rows:** **Starved · +Overlay · +PoE · Oracle** — the full §4 matrix. Every synthetic image (PoE and overlay alike) passes the identical Gate 0–5 filter before entering its arm.
- **metric:** ΔAUC on the **both-separable** stratum. Why it must move: this is the only stratum where the co-morbid composition can help; presence-level or whole-set AUC can stay flat while the co-morbid signal is broken. A genuine gain is *lumpy* — it spikes on both-separable and is near-zero on the single-disease strata.
- **sample_size:** ≥ 3–5 seeds per arm (10+ if AUC is noisy); bootstrap CIs on every ΔAUC; test on the Exp 1 pixel-verifiable set (pre-committed, never tuned on).
- **falsify_condition (the single headline rule for the whole plan):**
  - **support** — ΔAUC(PoE − overlay) ≥ 0.03 **and** ΔAUC(PoE − starved) ≥ 0.05 on both-separable, with disjoint 95% bootstrap CIs, and the gain concentrated on both-separable (not flat across strata).
  - **null** — ΔAUC(PoE − overlay) ≤ 0.00 → the diffusion machinery is unjustified, overlay wins. A clean, publishable null.
  - **inconclusive** — 0.00 < ΔAUC(PoE − overlay) < 0.03, or CIs overlap → rerun with more seeds. **Not** reinterpreted.
- **figures:** **corroboration** — AUC-by-stratum grouped bars, four arms, with +PoE spiking on both-separable and the oracle line drawn in. **anti-corroboration** — the same chart with a *flat* profile (PoE ≈ overlay across all strata) or PoE *below* overlay — either refutes the headline.
- **compute:** hippo. Cost = classifier training × arms × seeds (🏋️) + generating and gating the synthetic augmentation sets (💨). Sequential or a simple GNU-parallel loop over arms — **no Slurm array** (hippo is one box). Checkpoints to `$CKPT_DIR` behind the `df` guard. Wall-clock: overnight for the full grid.
- **status:** ⚠️ pending

---

## 5 · Pre-registration · 2026-06-24

Committed *before* any run. The in-between case has a pre-decided action (rerun with more seeds) so it can't be narrated into a result.

| Outcome | What we will see |
|---|---|
| **Claim holds** | Exp 1 separable fraction ≥ 30% · Exp 2 ceiling high · Exp 3 yield ≥ 30% (or retrain in Exp 5 recovers it) · Exp 6 oracle shows headroom · **Exp 7: +PoE > +overlay > starved on both-separable, disjoint CIs, gain lumpy on both-separable, +PoE closer to oracle than overlay.** |
| **Claim wrong** | Exp 7: ΔAUC(PoE − overlay) ≤ 0.00 → PoE does not beat naive overlay. The diffusion machinery is unjustified. Reported as a clean null. |
| **Upstream blocks the headline** | Exp 1 < 10% separable (no population) · or Exp 2 ceiling low (codec can't hold the joint) · or Exp 3 yield < 10% even after the Exp 5 retrain (experts can't co-render) · or Exp 6 no headroom (saturated test set). Each routes to its own fix before Exp 7 is read. **⟵ FIRED 2026-06-26: Exp 6 returned NULL (oracle−starved ΔAUC = −0.012 on both-separable, CIs entirely < 0). The both-separable test set is saturated → Exp 7 cannot be read as a utility test until the test set / stratum is revisited. Also: Exp 3 yield was already NULL (9%) — two upstream gates now block the headline.** |
| **In-between** | Any experiment lands inconclusive → rerun with more seeds/samples per that experiment's rule. Not reinterpreted. |

### 5a · Outcome-independent narrative (the lead, supervisor §6)

The paper's contribution does **not** depend on the headline landing positive. Both directions advance the field, and the framing is committed before the result:

- **Positive (upside):** real data augmented with gated synthetic compositional images yields a net gain on comorbidity classification **without** degrading the single-disease strata → evidence that compositional diffusion produces clinically useful synthetic data.
- **Negative (equally informative):** augmentation fails to improve comorbidity, or degrades overall performance → the conclusion *"compositional generation in latent space does not translate to compositional generation in disease space,"* a real finding that the approach needs fundamental revision. (Devon: *"then we just conclude that… even if we trained on this, it didn't help. Good to check."*)

The absence of a slam-dunk is **not** failure. Section 5 of the paper adapts to the result; the contribution — a rigorous, pixel-verifiable account of whether synthetic compositional data can improve comorbidity classification — holds either way. The oracle (Exp 6) is what keeps a null *interpretable*: it proves the test set had headroom, so a null means "the method didn't help," not "the test was saturated."

---

## 6 · Running it on hippo

Single box, no scheduler. Sequential runs or a simple GNU-parallel loop over configs — **never a Slurm array** (there is no array scheduler here). Tracking is **W&B**.

**W&B run-naming convention** — searchable, sortable:
```
{exp}_{arm}_{seed}_{date}
# e.g.  exp7_poe_s01_20260624   exp7_overlay_s01_20260624   exp6_oracle_s03_20260624
```
Tag every run with the gate it serves (`gate0`…`gate6`) and the checkpoint it used (`vae_step0025000`, `model_step0040000`, or the retrain id).

**Disk guard (pre-flight, abort early)** — a full disk silently kills checkpointing and you lose the run's tail. Guard *before* writing:
```bash
#!/bin/bash
# hippo: one box, run configs sequentially or via a parallel loop — NOT a Slurm array.
set -euo pipefail

CKPT_DIR=${CKPT_DIR:-/home/molefe/PhD/Paper3/ckpts}
USED=$(df --output=pcent "$CKPT_DIR" | tail -1 | tr -dc '0-9')
if [ "${USED:-100}" -ge 90 ]; then
  echo "ABORT: ${CKPT_DIR} is ${USED}% full — checkpointing will fail. Free space or repoint CKPT_DIR." >&2
  exit 1
fi

CONFIGS=(configs/exp7_*.yaml)
for CONFIG in "${CONFIGS[@]}"; do
  for SEED in 1 2 3; do
    RUN="$(basename "${CONFIG%.yaml}")_s$(printf '%02d' "$SEED")_$(date +%Y%m%d)"
    python run_experiment.py --config "$CONFIG" --seed "$SEED" \
      --ckpt-dir "$CKPT_DIR" --wandb-name "$RUN"
  done
done
```
(To run arms concurrently, wrap the inner body in a `parallel` loop instead of the nested `for`, capped to the GPU count.)

**Kill criteria** — stop a run early on: loss diverges / NaN, OOM that doesn't recover, or yield (Exp 3) trending toward 0 across the first few thousand samples (don't burn the full 2k to learn the experts can't co-render).

**Crashes ≠ model failure (learned 2026-06-25, the hard way).** On the rebalanced retrain, *every* crash so far was a **monitoring-arm artifact, not a training failure** — the model was healthy (no NaN/Inf, grad-norm fine, val at a run-min) at each one. A **process/OOM crash is recoverable and orthogonal to model quality**; always ask "is this the model or the tooling?" before reacting. Two concrete bugs to avoid:
- **The auto loss-divergence guard must be EMA-smoothed + persistent.** A single-snapshot guard (kill if loss > 2× the *raw* loss at step 10k) FALSE-killed a healthy run at step 10294 on per-batch DDPM noise. Fixed: kill only if `EMA(loss) > 2× the step-10k EMA` for ≥200 consecutive steps (`scripts/train_ldm.py`).
- **Don't run heavy CTR/PSPNet eval concurrently with training.** Hippo is one GPU (~32.6 GB) with tiny host RAM (~33.5 GB); training (~18.6 GB) + a heavy eval pass (PSPNet + classifier + ~192 generations) OOMs the box. Run `monitor_retrain.py` / `watch_retrain.sh` **only when the GPU is free** (offline on saved checkpoints), never alongside the live run.
- Resume now **forks to a fresh W&B run** (W&B forking is gated behind a private preview), with `resumed_from_run`/`resumed_from_step` lineage in the config — the step axis starts mid-run by design, not a bug.

**In-training generation-quality dashboard (new, `metrics/eval_checkpoint.py`).** Logs live to the *same* W&B run as loss: `ctr_med/{cardio,effusion,poe}`, `frac_gt0.5/*`, `border_frac/*`, `presence/*` (floor-subtracted *trend*, not absolute — classifier domain-shift), `eval/instrument_ok`, `ref/cardio_real=0.535` + `ref/threshold=0.5`, plus burned + W&B-layered overlay panels every 5k steps. It is the single shared evaluator (also called offline by `monitor_retrain.py`). ⚠️ Heavy (192 generations / 1000 steps) → throughput cost + OOM risk; gate behind a flag and/or prefer the offline path.

**Canonical checkpoints to start from** (do not use the decoy `ckpts/ldm/*` overfit run): VAE `ckpts/vae_step0025000.pt`, LDM `ckpts/model_step0040000.safetensors` (W&B lineage `prime_lab/paper3-ldm/runs/hifei736`), classifier `ckpts/presence_classifier_finetuned.pt`. Reuse them unless Exp 5 fires the retrain trigger.

### Artifacts & reuse map (wired 2026-06-24)

| Item (hippo `~/PhD/Paper3`) | Disposition |
|---|---|
| `ckpts/vae_step0025000.pt` | **reuse** (gated by Exp 2) |
| `ckpts/model_step0040000.safetensors` · W&B `prime_lab/paper3-ldm/runs/hifei736` | **reuse** (gated by Exp 3/5) |
| `ckpts/presence_classifier_finetuned.pt` (sigmoid multi-label) | **extend** — retrain on CTR-derived labels + co-presence data |
| `data/latents/{train,val}` + `scale_factor.pt` | **reuse** (already VAE-encoded) |
| `data/latents/test_both` | **reuse w/ caution** — OLD scope; **re-filter to separable-only** (Exp 1) before any test use |
| `data/nih/` + `Data_Entry_2017.csv` | **reuse** (raw images + labels/view) |
| `scripts/ctr_overlay.py` (xrv PSPNet) | **reuse** — the canonical CTR + CP instrument + red/blue inspection overlay |
| `scripts/generate.py` + `src/inference/cfg.py` | **reuse** — PoE composer (`cfg_compose`) |
| `metrics/grad_cam_utils.py` + `scripts/grad_cam.py` | **reuse** — GradCAM witness (Gate 1 only) |
| `scripts/naive_overlay.py` / `diag_compose_vs_overlay.py` | **reuse** — overlay baseline (Gate 6 rival) |
| `scripts/vae_recon_pilot.py` | **extend** — localized fidelity (Exp 2) |
| `scripts/finetune_classifier.py` | **extend** — detector retrain on CTR labels |
| `configs/ldm_rebalanced.yaml` (+ `ckpts_rebalanced/` in tag `v1.0-pre-refined-scope`) | **extend** — conditional rebalanced retrain (Exp 5) |
| `metrics/extractors.py` `heart_size`/`blunting` | **demote** — GradCAM proxy, circular; correlated-feature only, never a gate witness |
| dynamics · yield · deconfound · 4-arm-AUC harnesses · inspection notebook | **build new** |

### Human-in-the-loop execution pattern (Exp 1–3, and any manual gate)

The manual experiments are **not** "eyeball every film." They follow **calibrate → validate-instrument → automate → audit → freeze**:

1. **Calibrate (small, by hand):** open ~20–30 films in the viewer with the relevant overlay (`ctr_overlay.py` for CTR/CP; `grad_cam.py` for saliency). Hand-label them. *This is the only by-hand labelling.*
2. **Validate the instrument:** run the automated instrument on those same 20–30; require **≥ 90% agreement** with your hand labels before trusting it. Disagreement → inspect, fix the rule/threshold, do **not** proceed.
3. **Automate (all N):** run the validated instrument over the full set; write a per-film CSV.
4. **Audit:** spot-check a random ~30 of its calls **plus every borderline case**; fix systematic errors.
5. **Freeze:** the passing set becomes the frozen artifact (e.g. the pixel-verifiable test set); routed-out films are **logged, not deleted**.

**How you know you're doing it right:** the ≥ 90% hand-vs-instrument agreement on the calibration set, and the overlay lets you *see* each measurement (the red/blue CTR lines, the saliency map). **Use the segmentation CTR (`ctr_overlay.py`) for the separable/entangled call — not classifier + GradCAM** (that would be circular for *defining the population*; GradCAM is for Gate 1 only).

---

## 7 · Future work — silicoTB, LoRA composition, monolithic baseline (named, not run)

**silicoTB** is the real-world application that motivates the whole method: co-morbid data there is genuinely too scarce to train a reliable classifier, which is exactly the gap PoE composition is meant to fill. **We have no access to that data**, so it is *not* an experiment in this paper. It appears only in the discussion / future-work, framed honestly: the proxy proves the mechanism *and* the utility because an oracle can be built; the target can only ever show the practical claim (beat the starved baseline), and the transfer carries a real risk — the proxy leans on CTR and silicoTB has no equivalent canonical instrument, so Gates 1–5 must not *secretly* require a CTR-like oracle or the method won't carry over.

**LoRA for compositional generation** (supervisor §5) — a named future direction: rather than CFG-weighted ε-composition, fine-tune lightweight LoRA adapters per disease and compose them, which may co-render the joint case more faithfully than additive guidance. Discussion-only here.

**Monolithic joint-condition diffusion baseline** (supervisor 1.2/1.3/7.8) — **specified but deferred** (`docs/SUPERVISOR_RECONCILE.md`, item D12). It answers the reviewer question *"why compose at all instead of training one diffusion model on joint conditions?"* It is the generator-side analogue of the oracle (it needs comorbid training data, so it is proxy-only). We motivate it and acknowledge it explicitly; we execute it only if time allows. Not running it is a scoped limitation, not a silent omission.

---

## 8 · Still thin (flagged honestly)

- **Effusion's instrument is softer than CTR.** CP-angle blunting has no canonical numeric threshold the way CTR has 0.5. Tighten its operational definition before it is trusted as a gate (Gates 1–3 lean on it).
- **The oracle must actually be built (Exp 6)**, or the Paper-1-vs-2 "as good as having the data" framing stays unfalsifiable.
- **The separable population size (Exp 1) is unknown until measured.** If it is small, the Exp 7 floor is wide and tests can land inconclusive on sample size alone — check N before reading any AUC.
- **Cardio under-rendering is expected, not hypothetical.** Exp 3 yield is likely low on the reused checkpoints; the plan already routes that to the Exp 5 rebalanced-cardio retrain rather than treating it as a surprise. *(Update 2026-06-25: the rebalanced retrain is recovering it — gen-cardio CTR crossed 0.5 mid-run.)*
- **The border filter cleans the cardiac border, not the basal/CP region.** `border_visible` (peri-cardiac opacity) excludes whiteout *over the heart*, but cardio films still carry **basal/diaphragm haze** (elevated hemidiaphragm, AP technique) that the LDM faithfully reproduces — and that sometimes climbs over the border in generation. This is **not effusion leakage** (deconfounded split; the instrument over-reads basal haze at the 34% floor). Don't reflexively tighten the cardio filter for it — over-reaches and shrinks the already-scarce cardio class; diagnose (audit training films) before adding any basal/CP-clarity requirement.

---

<sub>Schema note: no canonical `EXPERIMENTS.md` format was found in the repo, so each EXP-block uses the experiment-planner default schema (claim_id · independent_var · dependent_var · ablation_rows · metric · sample_size · falsify_condition · figures · compute · status). Reconcile against the paper-orchestration spec if one is adopted. Status markers: ⚠️ pending · ✅ done · ❌ removed.</sub>
