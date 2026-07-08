# Decision Timeline — how each result set up the next decision

**What this is.** A plain-language, chronological record of the *gates* in this project: at
each step, a script produced a number, that number forced a choice, and that choice decided
what we ran next. It exists so that the reasoning behind "we think the generator is the
problem" is written down and checkable, instead of living implicitly across scattered
`verdict.md`, `.txt`, and `.csv` files.

**How to read it.** Every entry has the same five parts:

- **Question** — what we were trying to decide.
- **What we ran** — the script / command.
- **What it said** — the measured number.
- **What we decided** — the gate outcome.
- **Fired next** — the step that decision triggered.

Numbers are pulled from the cited files. Where an exact count matters, read it from the
cited audit file — the counts there are the source of truth, and a few files disagree at
the ±30-film level because they were written at different points in the cleaning pipeline.

There is a short **glossary** at the end for the recurring terms (CTR, LPIPS, μ-latent,
signal fraction, guidance `w`, PoE).

**Two reading levels.** The Acts below are the plain-language spine — a tired reader gets the
decision chain in one pass. Underneath, a **Methods & Rationale** appendix (M1–M5, before the
glossary) answers the expert questions an ML/DL/software/data-science reader will ask —
sigmoid vs softmax, which library, how the overlays and Grad-CAM are drawn, whether PoE
composition is theoretically sound in a continuous latent, how we even decide a generated
image "has cardiomegaly", and which metrics (AUC/ROC, sensitivity, specificity, FPR) we
report. The spine cross-links to these notes as `→ [M3]`. Depth lives in the appendix so the
spine stays readable; both are kept plain-spoken (plain means *clear*, not *shallow*).

---

## The one-line spine

```
paper question: does synthetic data help detect BOTH-disease cases when real both-disease is scarce?
      │
  [G0] measure the generator ............ cardio hearts come out too small (CTR 0.40 vs real 0.54)
      │
  [G1] is it the composition? the codec?  no + no — the CARDIO channel is the bottleneck
      │
  [G2] retrain cardio (rebalance) ........ cardio liftable but not DURABLE (peaks then regresses)
      │
  [G3] clean the data ................... drop obscured borders + basal haze; cardio floor too thin
      │
  [G4] add external cardio (VinDr) ...... 2.3× more clean cardio, breaks the haze correlation
      │
  [G5] upgrade the codec (VAE-v2) ....... sharper images BUT posterior collapse → train on μ, not samples
      │
  [G6] rebuild the LDM (v1→v4) .......... isotropic latents + early-stop; v4-60k is current
      │
  [G7] why does cardio STILL under-render? checks A/B/C ↓
      │
  [A] guidance knob ..................... YES, w=4 reaches real CTR 0.538  (free fix)
  [B] is the data borderline? ........... NO, training data is 0.539 — model under-shoots good data
  [C] is a big heart reachable? ......... YES — best-of-8 hits 0.571 (100% >0.5); NOT capped
      │
  [VERDICT] utility gap fully closable at inference (w4 best-of-8 → detector 0.875 > real 0.806); no retrain
            lever comparison woven in below (free-lever table + [G9a]/[G9v]); co-presence null is the headline
```

---

## Act 0 — The paper's question

The paper asks a narrow, honest question: when you have very few *both-disease* (comorbid)
chest X-rays, does adding *synthetic* X-rays — either single-disease images or
product-of-experts (PoE) compositions of two disease experts — help a detector spot the
comorbid cases?

The paper leads with a **mechanistic null**: a sober "here is exactly where and why the
synthetic pipeline falls short," not a "synthetic data wins" claim. Everything below is the
evidence chain behind that framing.

Source: `MASTER_PLAN.md`, memory `supervisor-reconcile-decisions`.

---

## Act 1 — Finding the fault: the generator under-renders enlarged hearts

### [G0] Does the generator actually draw the diseases?  → methods [M2] (overlays), [M4] (how we read cardiomegaly)

- **Question.** Do generated cardiomegaly images actually have enlarged hearts?
- **What we ran.** The CTR instrument (PSPNet heart/lung segmenter) on generated vs real
  images — `scripts/ctr_overlay.py`, `metrics/eval_checkpoint.py`.
- **What it said.** Generated cardio CTR median **0.395**; real cardio **0.536**. The
  generated hearts were smaller than even real *normal* chests (0.456). The original 100k
  model (`hifei736`) was also trained on a 7:1 effusion:cardio imbalance **and** on
  test-patient-contaminated data.
- **What we decided.** The under-rendering is real, not a measurement artifact. This is the
  seed of the mechanistic-null story.
- **Fired next.** A set of experiments (Exp 01–05) to localise *which stage* is at fault.

Source: `configs/ldm_rebalanced.yaml` (header), `tracking.md` (hifei736 block), memory
`extractor-real-vs-gen-disagree`.

### [G1] Is the fault in the composition, or the codec, or the cardio channel?  → methods [M3] (is PoE sound in a continuous latent?)

- **Question.** Three suspects — the PoE *composition* step, the *VAE* codec, or the
  *cardio* conditional itself. Which one?
- **What we ran / what it said.**
  - **Exp 02 (codec ceiling).** Encode→decode a real image and re-measure. The VAE barely
    moves the anatomy: median |ΔCTR| ≈ **0.011**. → The codec is faithful. *Do not retrain
    the VAE for fidelity reasons.*
  - **Exp 03 (PoE yield).** Co-presence yield was NULL (≈9.3%, below the 10% gate). PoE
    rendered cardio at 25.4% vs 4.2% for cardio-alone — so composition *helps*, but the
    cardio channel barely clears CTR 0.5.
  - **Exp 04 (cross-leakage).** Generated cardio self-renders at ~4% vs ~49% for real — a
    fidelity failure concentrated in cardio.
- **What we decided.** Composition is faithful; the VAE is faithful; **the cardio conditional
  is the bottleneck.** Reuse the VAE and the effusion expert, retrain cardio.
- **Fired next.** The rebalanced cardio retrain (Act 2).

Source: memories `exp02-codec-ceiling`, `exp03-poe-yield`, `exp04-cross-leakage`,
`ldm-cardio-undertrained-rebalance-lever`.

---

## Act 2 — Retrain cardio: the "rebalance lever"

### [G2] Can cardio be lifted by fixing the training mix?

- **Question.** The original model saw cardio in ~6% of batches (7:1 imbalance plus an
  `effusion_weight=2.0` legacy setting). If we give cardio a fair share, does it render
  bigger hearts?
- **What we ran.** A from-scratch conditional LDM with class balancing
  (`class_mult` 0.25 / 0.40 / 0.35 for normal/cardio/effusion), on border-visible training
  data — `configs/ldm_rebalanced.yaml`, run `mzd3jwy9`.
- **What it said.** Cardio CTR climbed to **0.521 at step 40k** (crossed the 0.5 line for
  the first time), but then **regressed to 0.457 by 100k**, and cardio presence collapsed at
  the end. Effusion held.
- **What we decided.** Split verdict: the lever *works* (cardio is liftable), but the result
  is **not durable** — cardio peaks mid-training then decays. Capacity and the VAE were
  re-confirmed innocent; the cheap unused lever (weight-EMA) and early-stopping became the
  next things to try.
- **Fired next.** Two parallel moves: clean the training data (Act 3), and rebuild the codec
  + LDM properly with EMA and early-stop (Acts 5–6).

Source: `tracking.md` (mzd3jwy9 block), memory `rebalanced-retrain-operational`.

---

## Act 3 — Clean the data: obscured borders and basal haze

### [G3a] Border-visible filter — match training to test

- **Question.** Many NIH films have a lower cardiac border hidden by whiteout (edema,
  atelectasis). If we train on those, PoE compositions come out with an *unmeasurable* border,
  which makes the Gate-3 co-presence test undefined. Should we keep them?
- **What we ran.** Scored `border_visible` on the deconfounded splits —
  `scripts/wire_border_visible_cache.py`, annotations in
  `results/exp05_decision/split_border_visible.csv`.
- **What it said.** Only **66%** of NIH cardio and **41%** of NIH effusion had a visible
  border. So 59% of effusion films were obscuring the very thing we measure.
- **What we decided.** Hard-drop obscured-border films. Kept roughly **cardio 854 /
  effusion 4,621 / normal 4,000** (normal capped for balance).
- **Fired next.** Exposed that the clean cardio pool (~854) is thin — feeds the basal-haze
  cleaning and the VinDr decision.

Source: `plans/complete/rebalanced-cardio-retrain/plans/02-split-wiring.md`,
`results/exp05_decision/split_border_visible.csv`.

### [G3b] Basal-haze cleaning — a scorer bake-off

- **Question.** NIH cardio patients are mostly symptomatic, so basal opacity ("haze" at the
  lung bases) is endemic. A cardio model trained on them learns to *paint haze as part of
  cardiomegaly*. Which scorer best separates clean from hazy films?
- **What we ran.** A two-candidate bake-off — Scorer A (the validated PSPNet-based
  peri-cardiac-opacity + costophrenic-angle instrument) vs Scorer B (a raw-pixel Laplacian
  sharpness score) — `scripts/basal_clean_bakeoff.py`, `scripts/basal_clean_score.py`.
- **What it said.** Both agreed on the obvious extremes; on the disagreement set Scorer B was
  confounded by NIH-vs-VinDr *texture* rather than actual cleanliness. Scorer A won.
- **What we decided.** Use Scorer A. Keep only films that are basal-clean **and** genuinely
  enlarged (CTR > 0.5).
- **Fired next.** Scored the whole cardio pool — this is the count that drove the VinDr
  decision (Act 4).

Source: `plans/complete/rebalanced-cardio-retrain/plans/07-cleanliness-scorer-bakeoff.md`,
`results/clean_cardio/SCORER.md`.

---

## Act 4 — Add external cardio: VinDr

### [G4] Is NIH cardio enough on its own?

- **Question.** After cleaning, the NIH clean-cardio floor is only ~854. A from-scratch
  cardio conditional trained on that few collapses the lower chest to white mush, and it
  learns the cardio↔basal-haze correlation. Can we break that with *external* cardio that
  doesn't share NIH's patient bias?
- **What we ran.** Pulled VinDr-CXR (VinBigData, 512px), derived cardio∧¬effusion labels,
  border-filtered, and checked patient-disjointness — `scripts/build_vindr_*`,
  `scripts/build_vae_diet.py`, `data/vae_v2_diet/leakage_check.txt`.
- **What it said.** VinDr adds **1,952 deconfounded cardio** films (2.3× NIH's 854), and the
  patient-ID intersection with NIH is **0** (structurally disjoint — NIH integer IDs vs VinDr
  hex IDs). Combined border-visible diet ≈ **10,409 NIH + 6,671 VinDr**.
- **What we decided.** Adopt VinDr as an external cardio base. It roughly doubles the clean
  cardio and dilutes the haze correlation.
- **Fired next.** The combined NIH+VinDr diet became the training set for the VAE-v2 codec and
  the rebuilt LDM. (Note: an earlier *two-stage transfer* plan was superseded on 2026-07-01 —
  VinDr now feeds a single joint-from-scratch codec, not a pretrain→finetune sequence.)

Source: `additional_plans/vindr-transfer/MASTER_PLAN.md`,
`additional_plans/vae-v2/plans/02-diet-preparation.md`,
`data/vae_v2_diet/{audit_nih.txt, leakage_check.txt}`, memory `external-transfer-scope`.

---

## Act 5 — Upgrade the codec: VAE-v2, and the collapse we nearly missed

### [G5a] Is the VAE good enough perceptually?

- **Question.** The V1 VAE passed the *spatial* fidelity gate (|ΔCTR| 0.009), but were the
  reconstructions perceptually sharp enough for a diffusion model to learn fine cardiac
  texture?
- **What we ran.** `scripts/eval_vae_v2.py` — four gates (ΔCTR, lower-zone SSIM, full-image
  LPIPS, cardiac-crop LPIPS) on the held-out border-visible set, V1 vs V2.
- **What it said.** V1 *failed* perceptually: LPIPS **0.152** full-image (target < 0.10),
  **0.229** cardiac crop (target < 0.08). The fine-tuned **VAE-v2 (step 75k)** passed all
  four with room to spare: |ΔCTR| **0.0027**, SSIM **0.976**, LPIPS **0.022** full,
  **0.020** cardiac. n=854 films.
- **What we decided.** **GO** on VAE-v2 for reconstruction. Freeze `vae_v2_step0075000.pt`.
- **Fired next.** A fifth gate that the four reconstruction gates never checked (next entry).

Source: `results/vae_v2/eval_gate/verdict.md`, `docs/vae_v2_architecture_decision.md`,
`additional_plans/vae-v2/plans/04-vae-eval-gate.md`.

### [G5b] Is the VAE's latent actually usable for diffusion? (the posterior-collapse gate)

- **Question.** All four gates above measure *reconstruction*, which uses the posterior mean
  μ. None checked whether the latent still carries signal when you *sample* from it — which is
  what a diffusion model does when it denoises from pure noise.
- **What we ran.** Read the posterior parameters with `vae.encode_moments` and computed the
  **signal fraction** = μ_std² / (μ_std² + σ_mean²).
- **What it said.** VAE-v2 sampled signal fraction was **3.4–3.6%** (vs **21%** for V1) — the
  posterior had collapsed toward the prior. The collapse was channel-concentrated: one of the
  four latent channels carried ~12%, the others 0.1–1.4%.
- **What we decided.** VAE-v2 is fine for reconstruction but **broken for sampled latents**.
  The fix is to train the LDM on **μ** (the deterministic mean, `encode_moments`), not on
  sampled latents. Per-channel normalization is *not* the fix (it would amplify the empty
  channels); global scaling on μ is.
- **Fired next.** Re-encode NIH+VinDr as μ-latents; this gate is *why* the LDM retrains use μ.
  It also explains two earlier LDM runs (`mzd3jwy9`, `gtsw5bq7`) that burned before the
  collapse surfaced.

Source: `additional_plans/vae-v2/plans/04-vae-eval-gate.md` (Gate 5),
`additional_plans/vae-v2/MASTER_PLAN.md`, memory `vindr-latent-scale-bug`.

---

## Act 6 — Rebuild the LDM: the v1→v4 recipe ladder

Each version fixed the failure the previous one exposed. All train in VAE-v2 μ-latent space.

| Ver | Run | Change from previous | Result | Verdict |
|-----|-----|----------------------|--------|---------|
| **v1** | `mzd3jwy9` | class-balance the mix (0.25/0.40/0.35) | cardio CTR 0.521 @40k → **regressed** to 0.46 @100k | effusion ✓, cardio not durable |
| **v2** | `umqnrs0o` | + weight-EMA, + min-SNR, mix 0.28/0.34/0.38 | cardio **absent** (presence ~0, leaks to effusion 0.81); effusion held 0.66 | effusion ✓, cardio ✗ |
| **v3** | `yrcj8qgl` | restore cardio share (0.20/0.40/0.40) | **both channels active**; cardio CTR 0.496, PoE yield clears 0.10 for the first time; but washed-out greyscale late | **best so far** |
| **v4** | (current) | per-channel isotropic latents, early-stop at 60k, min-SNR off, decode ÷scale-factor fixed | crisp images, cardio held near peak | **current canonical** |

### [G6a] Why v2 failed — the mix starved cardio

Lowering cardio's share from 0.40 to 0.34 to make room for effusion made cardio disappear
entirely. Lesson: cardio needs ≥0.40 share to survive. → v3 restored it.

### [G6b] Why v3 was best but still imperfect — two hidden bugs

- **Posterior anisotropy.** VAE-v2's per-channel latent std was ~[0.05, 0.36, 0.09, 0.12]
  (a 7.6× spread). With a single global scale factor, three channels sat far from unit
  variance, so the diffusion noise schedule drowned them → under-dispersed, washed-out
  images late in training.
- **Decode scale-factor bug.** The in-training preview dashboards decoded generated latents
  *without* dividing by the scale factor, over-brightening the previews by ~5×. This was
  cosmetic (it did not corrupt the final `generate.py` output), but it masked how the real
  images looked. Caught and fixed.

### [G6c] v4 — the fixes

- **Per-channel scale factor** makes all four latent channels unit-variance (isotropic), which
  matches the diffusion prior. Reconstruction fidelity is preserved (LPIPS still 0.022).
- **Early-stop at 60k**, because cardio peaks around 40–50k and running to 100k actively hurt
  it in every prior run.
- **min-SNR removed** as a non-helping confound.
- Current canonical checkpoint: `ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors`.

Source: `tracking.md` (mzd3jwy9, umqnrs0o, yrcj8qgl blocks),
`configs/ldm_vae_v2*.yaml`, `configs/ldm_vae_v2_v4.yaml`, memory
`ldm-decode-scalefactor-and-recipe`.

---

## Act 7 — Why so many classifiers, and what each answers  → methods [M1] (sigmoid vs softmax), [M5] (metrics)

Three *different* measuring tools are in play. Keeping them separate is essential — never
blur the judge with the judged.

- **The CTR/CP instrument (the "judge").** A PSPNet anatomical segmenter that measures the
  cardiothoracic ratio geometrically. It is *not* a classifier. It answers: "is the heart
  actually enlarged?" Real cardio median CTR **0.536**. Files: `scripts/ctr_overlay.py`,
  `metrics/eval_checkpoint.py`.

- **The presence classifier (the "judged").** A DenseNet-121 (NIH-pretrained) + small MLP
  head, two sigmoid outputs [cardio, effusion]. Both-disease images are **held out of its
  training on purpose**, so it is honest about comorbidity rather than tuned to it. It answers:
  "does a synthetic single-disease image read like a real one?" (Exp 09 body-or-no-body gate).
  Real performance: cardio sens 0.79 / spec 0.82, effusion sens 0.77 / spec 0.84. File:
  `ckpts/presence_classifier_finetuned.pt`.

- **The downstream augmentation classifiers (the "utility test").** Four arms × seeds, all the
  same architecture, trained on different data to answer the paper's actual question:
  - **Oracle** — sees all real both-disease data (headroom guard: if augmentation can't beat
    this, is the test just saturated?).
  - **Starved** — sees only 50 both-disease images (simulates the real scarcity).
  - **+Individual** — starved plus synthetic single-disease images.
  - **+PoE** — starved plus synthetic PoE compositions.

  **Table 2 result:** on the comorbid stratum, +PoE cardio gave Δ **+0.032** [+0.018, +0.048]
  — above noise but below the pre-registered +0.05 support threshold → **pre-registered null**,
  consistent with cardio being under-rendered. Files: `scripts/train_detector.py`,
  `results/tables/table2_augmentation.md`, `results/exp06_oracle/`.

Source: memories `classifier-identity`, `presence-classifier-joint-confound`,
`exp09-individual-gate`, `table2-seed-plan`.

---

## Act 8 — The current gate: why does cardio *still* under-render, and is it fixable for free?

Against the v4-60k model, generated cardio CTR is ~**0.482** vs real **0.536** — a 0.054 gap.
Three checks (`scripts/cardio_enhance_diag.py`) decide whether this is a free inference-time
fix, a retrain, a data-curation job, or a codec job — *before* spending any GPU on a retrain.

### [A] Does turning up the guidance knob make bigger hearts?

- **Question.** Diffusion sampling has a guidance strength `w`. Does raising it push CTR up?
- **What we ran.** `cardio_enhance_diag.py a --w-list 1 2 3 4 --n 60`.
- **What it said.** CTR climbs monotonically: **0.477 (w=1) → 0.507 (w=2) → 0.519 (w=3) →
  0.538 (w=4)**. At w=4, 69% of hearts clear CTR 0.5 (real is 77%).
- **What we decided.** The guidance knob **is** the lever. Reaching the real reference at w=4
  is a **free inference-time fix** — no retrain needed to lift heart size.
- **Fired next.** Confirm on a regenerated final set (the swept images weren't saved), then
  re-run the Exp 09 detector + CTR gate at w=3–4.

Source: `results/section4/cardio_enhance/cfg_ctr_sweep.csv`.

### [B] Is the model under-drawing, or is the training data itself borderline?

- **Question.** Maybe the model draws small hearts because the *training* hearts are small
  (the border-visible filter may have kept only mild cases). Which is it?
- **What we ran.** `cardio_enhance_diag.py b --latent-dir data/latents_v2/joint_perch --n 200`.
- **What it said.** Training cardio CTR median **0.539** (essentially equal to real 0.536);
  80% above 0.5, 41% above 0.55.
- **What we decided.** The data is **not** borderline — it has plenty of genuinely enlarged
  hearts. So the model is **under-shooting good data**. That points the fix at the generator /
  inference (Check A), not at re-curating the data.
- **Fired next.** Confirms Check A is the right lever; makes Check C the remaining question.

Source: `results/section4/cardio_enhance/train_cardio_ctr.txt`.

### [C] Is a big heart even reachable at sampling time? — RESOLVED: yes, via best-of-8  → see [best-of-n], [cfg-guidance]
<!-- gate: id=G8c status=accepted date=2026-07-06 plan-ref=additional_plans/vae-v2/plans/07-ldm-enhance.md supersedes= superseded-by= evidence=reachability-table -->

- **Question.** Can sampling-time tricks reach the real reference (~0.536) without changing the
  model — anchoring on a *normal* chest ("normal → sick" is the enlargement direction), or
  best-of-8 (draw 8, keep the biggest heart)?
- **What we ran.** `cardio_enhance_diag.py c --ckpt $CKPT --vae $VAE --scale-factor $SF --n 80 --best-of 8`
- **What it said.** At default w=1: null anchor 0.477, normal anchor 0.477 (**identical** — the
  anchor does nothing on its own at low guidance), **best-of-8 median 0.571**, 100% enlarged.
  <!-- claim: id=c-best8 kind=quantitative value=0.571 cell="best-of-8,ctr_median" table=reachability-table -->
- **What we decided.** Big hearts are strongly REACHABLE by rejection sampling — the learned
  distribution is **NOT capped**. This rules out the "range is capped → data/codec" heavy branch.
  The `normal` anchor is inert here.
- **Fired next.** The full lever comparison (every free inference knob × both instruments) → the
  utility verdict. **⚠ Not yet woven into this timeline — see coverage gaps.**

<!-- evidence: reachability-table -->
**Exhibit — reachability probe (anchor + best-of-8)**  ·  tier: quantitative

| arm | n | valid | CTR median | % > 0.5 | % > 0.53 | % > 0.55 |
|---------------|----|-------|-----------|---------|----------|----------|
| anchor=null   | 80 | 77 | 0.477 | 33.8 | 19.5 | 16.9 |
| anchor=normal | 80 | 77 | 0.477 | 33.8 | 19.5 | 16.9 |
| best-of-8     | 80 | 80 | **0.571** | 100.0 | 83.8 | 68.8 |

*Caveat:* best-of-8 CTR is high **by construction** (we keep the max-CTR draw) — whether the
*detector* fires on the kept images is the real utility question, answered by the lever
comparison. Real cardio reference: 0.536.

```bash
# reproduce:
CKPT=ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors
VAE=ckpts/vae_v2/vae_v2_step0075000.pt
SF=data/latents_v2/joint_perch/scale_factor.pt
python scripts/cardio_enhance_diag.py c --ckpt $CKPT --vae $VAE --scale-factor $SF --n 80 --best-of 8
# → results/section4/cardio_enhance/reachability.txt
```
<!-- exhibit-meta: source=results/section4/cardio_enhance/reachability.txt generator=scripts/cardio_enhance_diag.py git_sha=4c6d12f generated=2026-07-06 status=fresh -->
<!-- notes: (human-owned; survives regeneration) -->
<!-- /evidence -->

Source: `results/section4/cardio_enhance/reachability.txt`,
`additional_plans/vae-v2/plans/07-ldm-enhance.md`.

### [VERDICT] Which free knob closes both gates — and do we retrain?  → see [cfg-guidance], [best-of-n]
<!-- gate: id=G8v status=accepted date=2026-07-06 plan-ref=additional_plans/vae-v2/plans/07-ldm-enhance.md supersedes=where-this-leaves-us superseded-by= evidence=lever-comparison-table -->

- **Question.** Checks A/B/C all point to an inference fix. Which specific free knob closes *both*
  the geometric CTR gate **and** the detector gate (sensitivity ≥ 0.70), and does anything still
  need a retrain?
- **What we ran.** The full lever comparison — every free inference knob × both instruments,
  n=160/arm (`cardio_enhance_compare.py` → `compare_levers.csv`), plus detector confirmation at w3/w4
  (`ctr_vs_detector.py`).
- **What it said.** `w4 best-of-8` reaches detector sensitivity **0.875 — which beats real cardio
  (0.806)** — at CTR 0.610. <!-- claim: id=v-best8-det kind=quantitative value=0.875 cell="w4 best-of-8,detector_sens" table=lever-comparison-table --> `w6 null` clears the 0.70
  gate at **1× cost** (detector 0.750). The `normal` anchor is a **dead lever** — it raises
  geometric CTR but *lowers* detector sensitivity.
- **What we decided.** The too-small-heart utility gap is **fully closable at inference — no
  retrain.** Recommended final-set setting: `w4 best-of-8` (best, 8× cost) or `w6 null` (clears the
  gate at 1× cost). The retrain rungs (data curation / cardio-only fine-tune / VAE un-squash) are
  **PARKED** — not needed for the utility win.
- **Fired next.** Regenerate the final cardio set at the chosen setting, re-run plan-08's Exp-09 +
  augment probe, write before→after into the plan-08 verdict. **🔄 IN PROGRESS — the augment-probe
  set is generating now: w6-natural 10k + w4 10k (~2.5h), then w4 best-of-8 10k overnight (~15h);
  each writes `manifest_all.csv` + `manifest_ctr_gt0.5.csv` for the plan-08 A0/A1/A2 detector probe.**
  > ✅ **RESOLVED (2026-07-07)** → the augment probe came back NULL ([G9a]) and the downstream co-presence
  > check came back a mechanistic null, overall NO-GO ([G9v]). The 10k best-of-8 overnight pile was killed (a
  > cleaner pile can't move a flat null). See Act 9 below.

<!-- evidence: lever-comparison-table -->
**Exhibit — free-lever comparison (both instruments, n=160/arm)**  ·  tier: quantitative

| arm | CTR median | % enlarged | detector sens | of-enlarged caught |
|-----|-----------|-----------|---------------|--------------------|
| REAL cardio (ref)          | 0.538 | 79.9% | **0.806** | 0.866 |
| w1 null (baseline default) | 0.483 | 39.4% | 0.350 | 0.492 |
| w4 null                    | 0.542 | 76.6% | 0.631 | 0.712 |
| **w6 null** (1× cost gate-clearer) | 0.561 | 86.5% | **0.750** | 0.807 |
| w4 + normal-anchor (dead lever) | 0.568 | 82.5% | 0.594 | 0.646 |
| **w4 best-of-8** (winner) | 0.610 | 100% | **0.875** | 0.875 |
| w4 normal-anchor best-of-8 | 0.643 | 100% | 0.756 | 0.756 |

*Caveat:* best-of-N CTR is high by construction (keep max-CTR draw), but the detector is read on the
kept set and selection is by the *independent* CTR, so the sensitivity number is honest.

```bash
# reproduce:
python scripts/cardio_enhance_compare.py   # → results/section4/cardio_enhance/compare_levers.csv
python scripts/ctr_vs_detector.py --w 4     # → results/section4/cardio_enhance/detector_w4/
```
<!-- exhibit-meta: source=results/section4/cardio_enhance/compare_levers.csv generator=scripts/cardio_enhance_compare.py git_sha=4c6d12f generated=2026-07-06 status=fresh -->
<!-- notes: (human-owned) -->
<!-- /evidence -->

Source: `additional_plans/vae-v2/plans/07-ldm-enhance.md` (Check-A block: lever comparison + verdict),
`results/section4/cardio_enhance/compare_levers.csv`.

---

## Act 9 — The downstream verdict: single diseases pass, two-disease composition is a mechanistic null  → see [poe-composition], [ctr], [cardiomegaly-readout]

Act 8 fired: regenerate the final cardio set with the free fix, then re-run the plan-08 downstream
checks. Act 9 is what those checks said. Two gates: an exploratory side-check (do the fakes help a
real detector?), and the headline (does the fixed cardio rescue *two-disease* composition?).

### [G9a] Do the model's own best fake hearts help a real detector? — NULL  → see [presence-classifier], [best-of-n]
<!-- gate: id=G9a status=accepted date=2026-07-07 plan-ref=additional_plans/vae-v2/plans/08-downstream-eval.md supersedes= superseded-by= evidence=augment-probe-table -->

- **Question.** A low-stakes extra tied to Objective 7: if we add the model's *well-drawn* fake
  cardio to a detector's training set, does the detector get better at spotting *real* cardio?
- **What we ran.** A frozen-backbone probe (keep the trained image-reader fixed — it *is* the
  measuring instrument — and retrain only the small top layer). Three training sets ×
  3 seeds, tested on the same leakage-held-out real cardio set: **A0** = real only, **A1** = real +
  all fake, **A2** = real + big-heart-filtered fake. Ran for two fake-image sources (`w6`, `w4`).
  Harness `scripts/augment_probe.py` (caches the reader's features → the 9-run sweep is minutes).
- **What it said.** Flat. Cardio AUC (a threshold-free accuracy score, 0.5 = coin-flip, 1.0 = perfect)
  sits at **0.874–0.878 across all three sets** <!-- claim: id=g9a-auc kind=quantitative value=0.877 cell="A2,auc" table=augment-probe-table -->, every set inside every other's 95% band, for both
  sources. The catch rate at 90% clear rate is flat at ~0.62–0.64.
- **What we decided.** **NULL — the fakes don't help.** The flatness across *both* the sets and the
  sources points at a ceiling in the frozen reader, not in the fake-heart quality — so it is "no
  measured help," not "proven useless" (the plan warned about exactly this). Does not touch the
  paper's story.
- **Fired next.** Nothing — recorded and parked. The full 10k best-of-8 pile was **not** built (a
  cleaner pile can't move a flat, reader-limited null); killed to free the GPU for [G9v].
- **Exhibits.** `results/vae_v2/exhibits/reuse_fig_augment_probe.png` (flat AUC — adding synthetic
  cardio adds no signal a real detector can use). Rendered in illuminate Act 6.

<!-- evidence: augment-probe-table -->
**Exhibit — augment-probe (frozen-backbone, 3 sets × 3 seeds × 2 sources)**  ·  tier: quantitative

_Summary._ ![augment-probe](../results/section4/augment_probe/fig_augment_probe.png)
Headline: **A2 AUC 0.877 vs A0 0.874** — a null. *Adding fake cardio to a detector's training set,
filtered or not, does not improve how well it spots real cardio; the frozen reader is the ceiling.*

| set | source | cardio AUC (mean ± 95%) | catch-rate @ 90% clear |
|-----|--------|-------------------------|------------------------|
| A0 real only        | — | 0.874 ± 0.006 | 0.640 |
| A1 real + all fake  | w6 | 0.878 ± 0.007 | 0.639 |
| A2 real + big-heart | w6 | 0.877 ± 0.001 | 0.636 |
| A1 real + all fake  | w4 | 0.876 ± 0.005 | 0.617 |
| A2 real + big-heart | w4 | 0.877 ± 0.006 | 0.634 |

<details><summary>↓ full exhibit</summary>

Figure: two panels (AUC + catch-rate), one thin line per seed across A0/A1/A2 + bold per-set mean,
w6 solid / w4 dashed, y-axis pinned 0–1 (never zoomed, so noise can't look like signal).

```bash
# reproduce:
python scripts/augment_probe.py --synth-dir results/section4/augment_probe/synth/w6 --tag w6
python scripts/augment_probe.py --synth-dir results/section4/augment_probe/synth/w4 --tag w4
python scripts/fig_augment_probe.py   # → results/section4/augment_probe/fig_augment_probe.png
```
</details>
<!-- exhibit-meta: source=results/section4/augment_probe/probe_runs_w6.csv generator=scripts/augment_probe.py git_sha=4c6d12f generated=2026-07-07 status=fresh -->
<!-- notes: (human-owned) -->
<!-- /evidence -->

Source: `results/section4/augment_probe/verdict.md`, `additional_plans/vae-v2/plans/08-downstream-eval.md`.

---

### [G9v] Does the fixed cardio rescue two-disease composition? — NO. The headline mechanistic null.  → see [poe-composition], [ctr], [metrics]
<!-- gate: id=G9v status=accepted date=2026-07-07 plan-ref=additional_plans/vae-v2/plans/08-downstream-eval.md supersedes= superseded-by= evidence=copresence-verdict -->

- **Question.** Single-disease cardio is now fixable at inference ([G8v]: `w4 best-of-8` → detector
  0.875). Does that fix carry into *two-disease* images — do composed X-rays finally show an enlarged
  heart AND lung fluid together (the PoE co-presence yield ≥ 30%)?
- **What we ran.** Exp-03 at the full **N=2000** (`gate3_yield.py`), then the deciding follow-up: turn
  the cardio guidance knob up *inside the two-disease composition* — `w_cardio` 2 → 4 → 6 (n=500 each,
  `exp03_wc{4,6}/`) — the plan-07 fix applied to the cardio channel of the PoE.
- **What it said.** Two-disease share (floor-corrected — raw rate minus the false-alarm floor on
  healthy images) is **2.6% at N=2000** <!-- claim: id=g9v-yield kind=quantitative value=2.6 cell="w_cardio=2,floor_corrected" table=copresence-verdict -->, far under the 30% mark and *below* both the n=200
  preview (4.6%) and the old model (9.3%). Turning the knob up sharpens the heart — too-close-to-call
  drops 52.8% → 43.0% → 33.4% — **but the share stays flat: 2.6% → 4.1% → 3.1%.** The same lever that
  lifts single-disease cardio to 0.875 does nothing here.
- **What we decided.** **Overall NO-GO, recorded as the paper's headline.** The two checks split:
  single diseases draw faithfully, but two-disease images collapse and **do not respond to the cardio
  fix.** So the failure is in the *combining*, not the drawing quality — the model draws a big heart
  **or** fluid, not both in one X-ray. No checkpoint release for a co-presence claim; the null is the
  registered mechanistic finding.
- **Fired next.** Filed the verdict (`downstream_eval/verdict.md`) and the pre-registration row
  (`results/tables/table2_augmentation.md`, 2026-07-07). Closes plans 07 + 08. This is the result the
  paper leads with.
- **Exhibits.** `results/vae_v2/exhibits/`: `overlay_copresence.png` (a big heart CTR 0.83 with a
  SHARP corner CP 51° — the null made visible), `reuse_guidance_sweep.png` (yield flat as cardio
  guidance rises), `reuse_ctr_cp_scatter.png` (films cluster one-disease-or-the-other). The cardio
  inference-fix that this gate tested is exhibited in `chart_detector_bar.png` (best-of-8 0.875 > real
  0.806) + `overlay_gradcam_cardio.png` (detector heat on the heart, 0.93), and the residual small-heart
  finding in `chart_ctr_hist.png` + `overlay_cardio_ctr.png` (gen CTR 0.482 vs real 0.536). illuminate
  Acts 3–5.

<!-- evidence: copresence-verdict -->
**Exhibit — two-disease composition under the new model**  ·  tier: quantitative

_Summary._ ![guidance sweep](../results/vae_v2/downstream_eval/guidance_sweep.png)
Headline: **two-disease share 2.6% (30% needed), flat at 2.6→4.1→3.1% as the knob goes 2→4→6.**
*The heart sharpens as we push guidance, but the two-disease share never leaves the floor — the model
cannot place both diseases in one image, and the single-disease fix does not change that.*

| w_cardio | too-close-to-call | confident share | floor-corrected share |
|----------|-------------------|-----------------|-----------------------|
| 2.0 (N=2000) | 52.8% | 34/943 = 3.6% (95% ≈ [2.6, 5.0]) | **2.6%** |
| 4.0 (n=500)  | 43.0% | 4.2% | 4.1% |
| 6.0 (n=500)  | 33.4% | 4.2% | 3.1% |

Old-model baseline: 9.3% (also a null). Pass mark: ≥ 30%.

<details><summary>↓ full exhibit</summary>

Sample triplet (composed two-disease X-rays, ordered by heart width):
![triplet](../results/vae_v2/downstream_eval/copresence_triplet.png)
p25 CTR=0.45 · p50 CTR=0.49 · p75 CTR=0.54 — most composed hearts sit *under* 0.5 (not enlarged),
which is why the share is a floor. Scatter of all 2000 (heart-width × corner-angle):
![scatter](../results/vae_v2/downstream_eval/exp03/ctr_cp_scatter.png) — the pass corner (top-right,
both diseases) is nearly empty.

```bash
# reproduce (N=2000 baseline + guidance sweep):
python scripts/gate3_yield.py --ckpt ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors \
  --vae ckpts/vae_v2/vae_v2_step0075000.pt --scale-factor data/latents_v2/joint_perch/scale_factor.pt \
  --n 2000 --out results/vae_v2/downstream_eval/exp03/
for WC in 4 6; do python scripts/gate3_yield.py --w-cardio $WC --n 500 \
  --ckpt ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors \
  --vae ckpts/vae_v2/vae_v2_step0075000.pt --scale-factor data/latents_v2/joint_perch/scale_factor.pt \
  --out results/vae_v2/downstream_eval/exp03_wc${WC}; done
```
Verified 2026-07-07: N=2000 CSV → abstain 0.528, confident 34/943=0.0361, composed CTR median 0.495.
</details>
<!-- exhibit-meta: source=results/vae_v2/downstream_eval/exp03/exp03_results.csv generator=scripts/gate3_yield.py git_sha=4c6d12f generated=2026-07-07 status=fresh -->
<!-- notes: (human-owned) -->
<!-- /evidence -->

Source: `results/vae_v2/downstream_eval/verdict.md`, `additional_plans/vae-v2/plans/08-downstream-eval.md`,
`results/tables/table2_augmentation.md` (pre-registration row, 2026-07-07).

---

## Where this leaves us

> ⚠️ **SUPERSEDED by [VERDICT] (2026-07-07)** — Check C and the lever comparison resolved this:
> the fix is inference-only (`w4 best-of-8` → detector 0.875, or `w6 null` at 1× cost), **no
> retrain**. The paragraph below predates that result and is kept for the record, not as current
> status.

Checks A and B together already say a lot: the training data is good, and simply generating
with higher guidance (w≈4) puts cardio heart size on top of the real reference. That makes the
most likely conclusion **"the generator under-shoots good data, and it's fixable at inference,"**
rather than a data or codec failure. Check C is the last probe needed to confirm that the fix
is truly free (guidance / anchor / best-of-K) and not a retrain.

**Next action:** run Check C, then regenerate the final cardio set at w=3–4 and re-run the
Exp 09 detector + CTR gate to confirm the fix survives end-to-end.

---

## Methods & Rationale (for the expert reader)

> 📁 **Migrated to the concept KB (`docs/concepts/`) on 2026-07-07.** These notes are now atomic,
> hyperlinked concept files (canonical), each with three depths — **Gist** → **Explainer** →
> **Deep-dive**. The spine cites them by name (`→ see [PSPNet]`). The full text is kept below as the
> migrated copy; the concept files are the source of truth going forward.

| was | now | what it is |
|-----|-----|-----------|
| M1 | [presence-classifier](concepts/presence-classifier.md) | DenseNet two-sigmoid classifier (the "judged") |
| M2 | [pspnet](concepts/pspnet.md) + [grad-cam](concepts/grad-cam.md) | CTR segmenter (judge) + effusion localisation |
| M3 | [poe-composition](concepts/poe-composition.md) | ε-space CFG composition, the manifold question |
| M4 | [cardiomegaly-readout](concepts/cardiomegaly-readout.md) | how we call a synthetic "cardiomegaly" |
| M5 | [metrics](concepts/metrics.md) | sens/spec/FPR/AUC/ROC, Youden, bootstrap CI, strata |
| new | [cfg-guidance](concepts/cfg-guidance.md), [best-of-n](concepts/best-of-n.md), [ctr](concepts/ctr.md) | the free levers + CTR definition |

These notes answer the "how exactly?" and "why is that sound?" questions the spine glosses
over. Each is code-verified with a file:line so a supervisor can check it.

### [M1] The presence classifier — sigmoid, not softmax, and why

- **Architecture.** A torchxrayvision DenseNet-121 (`densenet121-res224-nih` weights) backbone
  with a small MLP head (Linear→ReLU→Dropout→Linear) producing **two logits** — one for
  cardiomegaly, one for effusion. Input is 224×224 grayscale, xrv-normalized
  (`presence_classifier.py:75-82`). Backbone mostly frozen; only the last block (`denseblock4`
  + `norm5`) is fine-tuned.
- **Sigmoid, not softmax** — the load-bearing choice. The head output goes through
  `torch.sigmoid` (`presence_classifier.py:112`), giving **two independent probabilities**.
  Cardiomegaly and effusion are *not mutually exclusive* — a patient can have both, either, or
  neither — so the two heads must be independent Bernoulli decisions (sigmoid + binary
  cross-entropy). A softmax would force the two classes to *compete* for one probability mass,
  which is exactly wrong for comorbidity. This is why the classifier can even represent a
  both-disease image.
- **Comorbidity held out on purpose.** Both-disease images are excluded from training, so the
  classifier is honest about co-presence rather than tuned to it (it is the *judged*, not the
  *judge*). See [M4] and [M5].

### [M2] The overlays and Grad-CAM — how the pictures are drawn

- **Cardio (CTR lines).** The cardiothoracic ratio is measured geometrically, not by the
  classifier. A pretrained PSPNet (torchxrayvision ChestX-Det) segments heart and both lungs;
  CTR = widest horizontal heart span ÷ widest thorax span, drawn as two horizontal rules on the
  overlay (`ctr_overlay.py`). CTR > 0.5 is the classic cardiomegaly sign.
- **Effusion ("the circles").** The effusion localisation is a **Grad-CAM heat-map** over the
  presence classifier, plus a bounding box at 50 % of the map's peak
  (`grad_cam_utils.py:142-148`). It highlights the basal / costophrenic region where fluid
  collects.
- **How Grad-CAM works here** (`grad_cam_utils.py:105-137`). Take the effusion logit,
  back-propagate to the DenseNet's final feature map (`model.features`, the 7×7 post-`norm5`
  map), global-average-pool those gradients into per-channel weights, take the weighted sum of
  the activation channels, ReLU (keep only evidence *for* the class), upsample to 512², and
  normalise to [0,1]. The bright region is "what the network looked at to say effusion."
- **The one non-obvious fix.** The raw effusion CAM had a spurious hot band along the *top*
  border (padding artifact in the 7×7 map) that anchored the min-max normalisation and bled
  colour into the lung apices. Fix: zero the top row before upsampling (apex is neither where
  cardiomegaly nor effusion localises, so it's anatomically safe) and normalise by the 99th
  percentile instead of the max, so no single hot cell sets the scale
  (`grad_cam_utils.py:13-26,125-136`).

### [M3] PoE composition — is it sound in a continuous latent? (the manifold question)

The premise "the VAE latent is continuous, so PoE traverses straight lines, and a curved
manifold means those lines pass through holes → nonsensical images" is *almost* the right
worry, but the mechanism is subtly different — worth stating precisely.

- **What the code actually does** (`generate.py:181-182,282`). PoE is **not** interpolation
  between two latent points, and **not** a product of two Gaussians in latent space. It is
  composition in **noise-prediction (ε) space** via classifier-free guidance. One UNet predicts
  the noise under each conditioning; at every denoising step the guidance vectors are added:

  ```
  ε = ε_anchor + w·(ε_cardio − ε_anchor) + w·(ε_effusion − ε_anchor)
  ```

  This is the composable-diffusion / product-of-experts-of-scores formulation (Liu et al.): each
  expert contributes a score (gradient of a log-density), and adding scores multiplies the
  distributions. The sampler then follows this *combined score field* backward through the
  reverse-diffusion trajectory.
- **So where does the "holes" risk really live?** Not on a straight line in latent space, but in
  the *trajectory* the sampler takes under the summed field. When the two experts disagree — one
  pulls toward "big heart", the other toward "basal fluid" — their added guidance can steer the
  path into a region the training data never populated (off the data manifold), and the decoded
  image is then implausible or renders only one disease. **That is exactly our measured failure:**
  the co-presence yield was NULL (~9 %, [G1]/[Exp03]) because the cardio channel under-renders,
  so the combined field lands in a low-density region for cardiomegaly. The intuition (composition
  can go off-manifold) is right; the fix is not a different geometry but a *stronger, better-
  trained cardio score* (Acts 2, 6) and higher guidance (Act 8, Check A).
- **Why a continuous VAE latent at all?** Diffusion needs a continuous, smooth space to add and
  remove Gaussian noise and to define scores; a discrete/VQ latent would not support the ε-space
  score arithmetic above. The VAE is KL-regularized with a Gaussian posterior
  (`encode_moments → μ, σ`); we train the LDM on **μ** to avoid the posterior-collapse noise
  ([G5b]). The cost of continuity is the very off-manifold risk above — accepted because [Exp02]
  showed the codec is faithful (|ΔCTR| ≈ 0.011), so composition failures are the *generator's*,
  not the codec's.
- **Entanglement.** VAE channels specialise to whatever varies in the training data; with a 7:1
  effusion:cardio imbalance, capacity skewed toward effusion features and the cardio direction
  was under-formed and partly entangled with effusion (we measured generated cardio *leaking* to
  the effusion head, [G6a]). Rebalancing the class mix (Acts 2, 6) is, in this framing, an attempt
  to disentangle the cardio factor by giving it enough exposure to claim its own capacity.

### [M4] How do you know a generated image "has cardiomegaly"? (and what a label even means)

- **Two independent judges must agree.** (1) The *geometric* judge: the PSPNet segmenter measures
  CTR; > 0.5 = enlarged ([M2]). (2) The *appearance* judge: the DenseNet presence classifier's
  cardio sigmoid ([M1]). We only trust "this synthetic image is cardiomegaly" when both agree —
  which is why [Exp09] uses both and reports where they diverge.
- **What the *label* means is itself limited.** An NIH "Cardiomegaly" label is a radiologist's
  read (or an NLP mining of the report), not ground-truth anatomy — inter-rater noisy, and on AP
  films the heart is magnified so CTR over-reads. So "does the generated image have cardiomegaly?"
  is answered against an *instrument* (CTR + classifier calibrated on real films), not against a
  perfect oracle. The honest claim is "it matches the real-cardio CTR distribution (median 0.536)
  and reads positive to a classifier calibrated on real cardio", not "it is definitively
  diseased". This epistemic caveat is why the paper leads with a mechanistic null rather than a
  clinical claim.

### [M5] Metrics & distributions — what we report and why

- **Sensitivity / specificity / FPR.** Sensitivity (recall) = TP/(TP+FN) — of real diseased
  cases, how many we catch; the operative metric for the [Exp09] body-gate, because the question
  is "does a synthetic disease image *read as diseased*". Specificity = TN/(TN+FP); FPR = 1 −
  specificity. Real-classifier operating point: cardio sens 0.79 / spec 0.82, effusion sens 0.77 /
  spec 0.84.
- **AUC / ROC.** Threshold-free ranking quality, stored per head in the checkpoint
  (`val_auc_cardio`, `val_auc_effusion`) and used as the headline for the Table 2 augmentation
  arms (ΔAUC on the comorbid stratum). The decision threshold itself is **Youden-optimal**
  (max sensitivity + specificity − 1), stored in the checkpoint, not hardcoded
  (`presence_classifier.py:178-179`).
- **Confidence intervals.** Bootstrap, 1000 resamples, 2.5/97.5 percentiles
  (`presence_classifier.py:155-161`) — this is where the Table 2 `[+0.018, +0.048]`-style
  intervals come from, and why +PoE cardio's +0.032 is called a *null* (CI clears 0 but sits
  below the pre-registered +0.05 support line).
- **Distribution / stratification.** Evaluation is stratified into `normal`, `cardio_only`,
  `effusion_only`, `both_separable`; the comorbid `both_separable` stratum is the actual claim
  criterion, and real both-disease is scarce (~1063 films, the reason the whole synthetic-
  augmentation question exists).

---

## Glossary

- **CTR (cardiothoracic ratio).** Heart width ÷ chest width, measured from an anatomical
  segmenter. > 0.5 is the classic cardiomegaly sign. Real cardio median ≈ 0.536.
- **CP.** Costophrenic angle — the sharp corner at the lung base; blunting of it is a sign of
  effusion.
- **LPIPS.** A perceptual image-similarity score (lower = more similar to the original).
  Used to check the VAE reconstructs fine texture, not just gross shape.
- **SSIM.** Structural similarity (higher = closer); here measured on the lower third of the
  image where effusion lives.
- **μ-latent (posterior mean).** The deterministic centre of the VAE's latent for an image.
  Training the LDM on μ (instead of a random sample from the latent) sidesteps the
  posterior-collapse problem.
- **Signal fraction.** How much of a *sampled* latent is real signal vs prior noise. Below
  ~15% the latent is too noisy for a diffusion model to learn from. VAE-v2 sampled = 3.4%
  (bad); V1 = 21% (fine); VAE-v2 μ = effectively 100%.
- **Guidance `w` (classifier-free guidance).** A sampling knob that sharpens how strongly the
  model follows the disease label. Higher `w` → stronger disease features (bigger hearts here).
- **PoE (product of experts).** Composing two single-disease diffusion experts at sampling
  time to make a both-disease image.
- **Border-visible.** A film whose lower cardiac border is not hidden by whiteout, so its CTR
  can actually be measured.
- **Deconfounded split.** Cardio-only (cardio ∧ ¬effusion) or effusion-only films, so a
  single-disease expert isn't secretly trained on the other disease.

---

### [G6d] Does v4-60k render blunted effusion corners on its own? — NOT YET at w=1.0 (plan 06 stays open)  → see [ctr], [pspnet], [cfg-guidance]

- **Question.** Plan 06's last open sanity: do generated effusion-only films from the winning
  v4-60k LDM (`4vg2h5i4`) show a blunted costophrenic angle in ≥80% of samples? The cardio half
  of the sanity was confirmed 2026-07-03; only the effusion-only corner-blunting artifact remained,
  and it had been GPU-blocked on hippo.
- **What we ran.** `scripts/effusion_blunt_check.py --n 200 --w 1.0` on the mscluster110 Blackwell
  GPU (RTX PRO 6000, sm_120) — generate 200 effusion-only films, score each with
  `metrics.eval_checkpoint.score_decoded_array → _row_from_score` (costophrenic max-angle; blunted
  = CP > 80°). Output → `results/vae_v2/ldm_sanity/v4_60k_final/effusion_blunt_rate.txt` +
  `effusion_panel.png`. The first run reported `valid=0/200, nan%` — a **harness bug**: the driver
  read `effusion_present` / `cp_max_angle` straight off `score_decoded_array`, which returns only the
  geometry dict; those keys come from `_row_from_score`. Fixed the driver to the canonical two-step
  (invocation-level; the scoring math was not touched — Guardrail 3) and re-ran → `valid=200/200`.
- **What it said.** blunt-rate **32.0%** (< 80% mark → **FAIL**), median CP max-angle **70.5°**,
  200/200 films scored. The instrument is sound, not the cause: on the same scorer, exp03's 2000
  films separate cleanly — 1648 non-effusion at median CP **61°**, 352 blunted at median **92°**
  (min 81°, just above the 80° cutoff, max 195°) — so the 80° threshold partitions correctly. The
  effusion-only films at w=1.0 sit at median 70.5°, a real shift up from the 61° normal baseline but
  mostly short of 80°. Repro: `python scripts/effusion_blunt_check.py --n 200 --w 1.0` (seed 42).
- **What we decided.** Do NOT force a pass and do NOT close plan 06. Recorded 32% as-is. The diagnosis
  points to effusion **rendering strength at low guidance (w=1.0)**, not an instrument failure:
  conditioning shifts the corner but weakly. Consistent with the strong classifier evidence (Exp-09 v4
  effusion synth sensitivity 0.865 PASS — final downstream verdict; earlier per-model passes read
  0.905/0.960 — [G6c] family) — the disease is present, just not
  geometrically pronounced at w=1.0. Plan 06 stays ⚠️ open; DoD item 7 stays open.
- **Fired next.** A diagnostic CFG-weight sweep (w = 1, 2, 4, 7 at n=200; the plan-06 smoke test used
  w=4.0) to separate "weak at low guidance" from "genuine rendering shortfall", and a decision on
  whether the plan-06 pass-mark should be evaluated at w=1.0 at all. No pass claimed until that resolves.
- **Exhibits.** `results/vae_v2/exhibits/`: `chart_blunt_curve.png` (blunt-rate 32%→66% across w=1→7,
  never crossing 0.80) + `overlay_effusion_cp.png` (a generated effusion film, both costophrenic corners
  flagged blunted, CP 150°/109°). illuminate Act 7.

---

### [G6e] Does higher guidance rescue the effusion blunt-rate? — Monotonic but NO, plateaus at 66% by w=7 (plan 06 still open)  → see [cfg-guidance], [ctr]

- **Question.** [G6d] left open whether the 32% blunt-rate at w=1.0 was just weak guidance or a real
  rendering shortfall. Does raising the CFG guidance weight lift the effusion-only blunt-rate to the
  ≥80% mark?
- **What we ran.** A controlled CFG sweep on v4-60k: `w ∈ {1, 2, 4, 7}`, effusion-only, n=200 each,
  same seed (42) so only guidance varies. Same instrument as [G6d]
  (`score_decoded_array → _row_from_score`; blunted = CP > 80°). Output →
  `results/vae_v2/ldm_sanity/v4_60k_wsweep/` (`wsweep_summary.csv` + per-w most-blunted panels).
  Repro: the sweep driver reuses `generate_disease` + the same scorer at four w values.
- **What it said.** A clean, monotonic dose-response:
  `w=1 → 32.0%, median CP 70.5°` · `w=2 → 44.5%, 77.0°` · `w=4 → 57.8%, 84.0°` · `w=7 → 66.3%, 94.0°`.
  Guidance genuinely drives effusion blunting: both the blunt-rate and the median CP angle rise at
  every step, and the **median film crosses the 80° blunting threshold by w=4 (84°)**. The most-blunted
  panels show real basal effusion morphology (dense lower-zone opacity obscuring the costophrenic angle),
  anatomically coherent even at w=7 — not CFG over-saturation artifact. **But the ≥80%-of-samples bar is
  not reached at any tested guidance (max 66.3% at w=7).**
- **What we decided.** The 32% at w=1.0 was partly a low-guidance measurement effect (real,
  guidance-responsive effusion), but not merely that — the geometric ≥80% blunt-rate bar is unmet even
  at aggressive w=7. Still NOT forcing a pass; plan 06 stays ⚠️ open. This is now a scoped decision for
  the human: (a) accept a known partial-geometric-blunting limitation and close plan 06 with that caveat
  (effusion is classifier-detectable — Exp-09 sens 0.865 PASS, final downstream verdict — and the median film blunts at w≥4);
  (b) carry effusion blunting to plan 07-ldm-enhance as a rendering target alongside the cardio
  small-hearts finding; or (c) revise the acceptance criterion (≥80% at w=1.0 vs median-blunts-at-
  deployment-guidance).
- **Fired next.** Await the human's decision on (a)/(b)/(c). No further guidance escalation (w>7 risks
  realism loss for diminishing geometric return).

---

### [G6f] DECISION: close plan 06 with the effusion blunt-rate as a recorded caveat, not a pass  → see [cfg-guidance]

- **Question.** Given [G6e] (effusion blunt-rate is guidance-responsive but never reaches the ≥80%
  bar), what do we do with plan 06 — accept and close, carry to an enhancement plan, or revise the
  criterion?
- **What we ran.** No new compute — a reconciliation of the evidence against the tree. Checked what
  the vae-v2 master actually requires (DoD-6 = "border-readable hearts + W&B linked", both met; the
  effusion-only blunt overlay is an extra sanity artifact, not a DoD-6 gate), and the state of the
  downstream plans (07-ldm-enhance CLOSED — cardio bigger-hearts target MET at w4 best-of-8;
  08-downstream-eval CLOSED — Exp-09 effusion sens 0.865 PASS, Exp-03 co-presence NO-GO already filed).
- **What it said.** Nothing downstream hinges on the geometric ≥80% blunt-rate: effusion is
  classifier-detectable and the co-presence verdict is already recorded. The blunt-rate shortfall is
  the same class of characterized rendering property as the cardio small-hearts finding, which plan 07
  resolved inference-only at w4 — not a blocker in either case.
- **What we decided.** **Close plan 06 with a caveat, do NOT force a pass, do NOT reopen 07/08.** The
  effusion blunt-rate is filed as a recorded curve (32%@w1 → 66%@w7, median film blunts at w≥4) with a
  known-limitation caveat, mirroring how cardio CTR was handled. Rejected a criterion-revision that
  would manufacture a pass (even the best reframe is 66% at w=7). vae-v2 plan 06 → ✅ (closed with
  caveat); vae-v2 DoD-6 → ✅. This also completes the mscluster-setup finish-progress-track objective;
  its DoD-7 "blunt-rate ≥0.80" sub-clause is **superseded** by this decision (recorded-as-curve), the
  progress-track + sync steps being the actual completion.
- **Fired next.** `/sync-plan-tree` to roll plan 06 ✅ up both the vae-v2 and mscluster-setup masters.
  vae-v2 scope is now fully closed (all plans ✅; downstream verdict = co-presence null).

---

## How this log is maintained (supersession, not deletion)

A decision log is only useful if it's honest about the paths that turned out wrong — the
test-contaminated 100k run, the two LDM runs that burned on collapsed latents, the decode
bug that made previews 5× too bright. Those aren't embarrassments to hide; they're the
reasoning a supervisor most wants to see. So the rule is:

- **Entries are append-only.** You never rewrite the reasoning of a finished entry.
- **When a later result invalidates an earlier gate, you supersede it — you don't delete it.**
  Add one forward-pointing banner to the old entry and leave everything else intact:

  > ⚠️ **SUPERSEDED by [G6b]** (2026-07-06) — the decode preview was 5× over-bright; the
  > washed-out images were a display bug, not a model failure.

- **The "is this still necessary?" pass is deliberate, not automatic.** Periodically (e.g.
  when you run `/sync-plan-tree`) re-walk the gates and stamp each one `STILL-VALID` or
  `SUPERSEDED-BY-Gxx`. This is the same idea as reconciling plan status against evidence —
  a human/LLM judgement call, never something a hook does silently.

### The staging inbox and the hook

`.claude/hooks/timeline_stub.sh` runs on session **Stop**. It sweeps `results/` (and the diet
audit files) for decision artifacts that changed since the last sweep and appends a **stub**
to [`decisions/INBOX.md`](decisions/INBOX.md) — never to this file. A stub is just a pointer
("this verdict changed, go write it up"). Promoting a stub into a real entry here, and running
`/plain-speak` on it once, is always your deliberate step. The hook stages; a person promotes.
You can also run the sweep by hand any time: `bash .claude/hooks/timeline_stub.sh`.

---

*This document is a hand-authored decision log. It is appended to at each new gate and never
rewritten; earlier gates are superseded with a banner, not deleted. The `Stop` hook only
stages stubs into `decisions/INBOX.md` — it never edits this file.*
