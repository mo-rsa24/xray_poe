# Make the fake cardiomegaly hearts bigger (utility push on the v4-60k model)

**Words this plan can't avoid — defined once, plain:**
- **CTR** — heart width ÷ chest width, measured off the X-ray. Above 0.5 means an enlarged heart (cardiomegaly). Real sick hearts sit at about 0.536.
- **Detector / sensitivity** — a trained classifier (DenseNet) that looks at an image and says "cardio: yes/no". Sensitivity = the share of true cardio images it flags. Real cardio scores 0.80.
- **Guidance strength `w`** — a sampling knob. Higher `w` pushes the fake image harder toward "cardio".
- **Anchor** — the reference the model pushes *away* from. `null` = a blank reference; `normal` = a healthy chest, so the push is "healthy → sick" (which is the enlargement direction).
- **Best-of-N** — draw N fake images for one slot, keep the one with the biggest heart, throw the rest away.
- **Latent / codec** — the VAE's compressed code for an image. Ours is squashed (uses about 1 of its 4 channels), which limits how much heart-size detail it can hold.

## Background
The winning model is **v4-60k** (`ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors`). It draws clean
normal, cardio, and effusion chest X-rays. The one problem: the cardio hearts come out too small.
Measured heart size (CTR) sits at **0.482**, but real sick hearts are at **0.536**. Only **40%** of the fakes
have an enlarged heart (CTR above 0.5); in real cardio it's **77%**. The detector agrees — it flags only
**36.5%** of the fake cardio images, versus **80%** of real ones. Two separate tools (the geometric CTR and
the trained detector) both say the same thing: the model under-draws the heart.

That under-drawing is a real paper FINDING (it lives in plan 08 downstream and the "mechanistic null" story).
**This plan is the separate utility job: make the fake cardio detectable.** The realistic win is a detector
score of **0.70–0.79** (or a CTR median near 0.53) — not a 0.90 dream.

Why the hearts come out small — three possible causes:
- **a.** Very few unique cardio training films (~2,774), repeated a lot.
- **b.** The squashed VAE code (~1 of 4 channels) can't hold much heart-size detail.
- **c.** The training data itself is borderline — the "border-visible" filter kept mild, not severe, cases.

## Description
Run three cheap checks first (A, B, C — all free, about 15 minutes total) to find out *which* cause it is:
a sampling knob, the data, or the codec. Only then spend GPU on a fix. One script does all three:
`scripts/cardio_enhance_diag.py`. Then apply the cheapest fix the checks point to. Also run `/augment` to
borrow proven tricks from other people's work.

## Purpose
Serves Objective 7 (make the fake cardio bodies faithful). Comes after: plan 06 (the v4-60k model) and
plan 08 downstream (which found the gap). Feeds the final verdict — re-run plan 08 once a fix lands.
Order: `06 → 09 → 07 → 08`.

## Goal
Fake cardio CTR median near **0.53**, and/or detector sensitivity **0.70–0.79** (up from 0.365). The cheapest
working fix from A/B/C applied, the fake cardio set regenerated with it, and the before→after numbers written down.

## Tasks

- [x] ✅ **Check A — does turning up the guidance knob make bigger hearts?**  (free, ~5 min) — **POSITIVE: the knob works.**
  - **Run:**
    ```bash
    python scripts/cardio_enhance_diag.py a --ckpt $CKPT --vae $VAE --scale-factor $SF --w-list 1 2 3 4 --n 60
    ```
  - **What it does:** draws 60 cardio X-rays at each guidance strength `w` = 1, 2, 3, 4 and prints the CTR median
    for each. (An earlier sweep only tested *sharpness* and was flat — heart *size* is a different thing.)
  - **What to do with the result:** if CTR climbs from 0.48 toward 0.53 by about `w`=3, that's a free fix —
    generate the final cardio set at that `w` and you're done. If it stays flat, the knob isn't the lever; go to B and C.
  - **Done when:** `results/section4/cardio_enhance/cfg_ctr_sweep.csv` exists with a CTR median per `w`.
  - **RESULT (n=60, 58 valid per w):** CTR climbs monotonically with `w` and reaches the real reference (0.536):
    w=1 → 0.477 (34.5% enlarged); w=2 → 0.507 (51.7%); w=3 → 0.519 (63.8%); **w=4 → 0.538 (69.0%)**.
    The guidance knob IS the lever — a free inference-time fix. This is the geometric CTR gate only, so the
    detector gate was then confirmed on saved images (below).

  - **DETECTOR CONFIRMATION (n=200 synth vs 200 real, `scripts/ctr_vs_detector.py --w {3,4}`,
    `results/section4/cardio_enhance/detector_w{3,4}/`):**
    - **w=4 (chosen):** synth CTR median **0.540** (real 0.536 — geometric gap CLOSED, 77% enlarged like real);
      detector sensitivity **62.5%** (real 80.0%), up from **36.5%** at default guidance. Of the geometrically
      enlarged fakes the detector now catches 71.6% (real 86.4%). No saturation: CTR-valid 192/200, std 0.061
      like real — w=4 does not degrade the films.
    - **w=3:** synth CTR median 0.521, detector 55.0% — worse on both, so w=4 wins.
    - **NOTE:** w=4 alone met the CTR gate but left the detector at 0.625 (< 0.70). The full lever comparison
      below then CLEARED the detector gate for free too.

  - **LEVER COMPARISON — every free inference-time knob, both instruments (`scripts/cardio_enhance_compare.py`,
    n=160/arm, `results/section4/cardio_enhance/compare_levers.csv`):**

    | arm | CTR median | % enlarged | detector sens | of-enlarged caught |
    |-----|-----------|-----------|---------------|--------------------|
    | REAL cardio (ref)          | 0.538 | 79.9% | **0.806** | 0.866 |
    | w1 null (baseline default) | 0.483 | 39.4% | 0.350 | 0.492 |
    | w4 null                    | 0.542 | 76.6% | 0.631 | 0.712 |
    | **w6 null**                | 0.561 | 86.5% | **0.750** | 0.807 |
    | w4 + normal-anchor         | 0.568 | 82.5% | 0.594 | 0.646 |
    | **w4 best-of-8**           | 0.610 | 100%  | **0.875** | 0.875 |
    | w4 normal-anchor best-of-8 | 0.643 | 100%  | 0.756 | 0.756 |

    - **WINNER — w4 best-of-8: detector sensitivity 0.875, which BEATS real cardio (0.806); CTR median 0.610.**
      Both gates smashed, zero retraining. best-of-8 = draw 8, keep the biggest-heart (highest-CTR) draw; the
      detector is read on the KEPT set (selection is by the independent geometric CTR, not the detector, so the
      number is honest). Cost: 8× generation.
    - **Cheap gate-clearer — w6 null: detector 0.750, CTR 0.561, at 1× generation cost.** If 8× is not worth it,
      plain hotter guidance already clears the 0.70 target.
    - **Dead lever — the `normal` anchor.** It raises geometric CTR but LOWERS detector sensitivity (w4_normal
      0.594 < w4_null 0.631; and adding it to best-of-8 drops 0.875 → 0.756). Do not use the anchor for this.
    - **VERDICT: the too-small-heart utility gap is fully closable at inference — no retrain needed.** Recommended
      final-set setting = **w4 best-of-8** (best quality, 8× cost) or **w6 null** (clears the gate at 1× cost).
      Retrain rungs (data curation / cardio-only fine-tune / VAE un-squash, plan 07 rungs 2–4 + plan 09) are
      PARKED — not needed for the utility win. Next: regenerate the final cardio set at the chosen setting and
      re-run plan 08's Exp-09 + augment-probe, writing before→after into the plan-08 verdict.

- [x] ✅ **Check B — is the model under-drawing, or is the training data itself borderline?**  (free, ~2 min) — **UNDER-DRAWING confirmed; data is good.**
  - **Run:**
    ```bash
    python scripts/cardio_enhance_diag.py b --vae $VAE --scale-factor $SF \
        --latent-dir data/latents_v2/joint_perch --n 200
    ```
  - **What it does:** decodes 200 of the actual *training* cardio images and measures their CTR. This is the
    ceiling the model is trying to copy.
  - **What to do with the result:** if the training median is ~0.53 but the fakes are ~0.48, the model is
    under-drawing its own data — fix with A, longer training, or more codec room. If the training median is
    itself ~0.48, the *data* is borderline — no amount of training makes big hearts, so you must build a
    severe-only subset. (Best guess: B shows borderline data.)
  - **Done when:** `results/section4/cardio_enhance/train_cardio_ctr.txt` has the training CTR median plus the
    share above 0.5 / 0.53 / 0.55.
  - **RESULT (n=200):** TRAIN cardio CTR median **0.539** (mean 0.542; 80% >0.5, 41% >0.55) — matches real
    (0.536). So the training data is NOT borderline: it holds genuinely big hearts. The generated set at default
    guidance sits at 0.477, so **the model under-draws its own good data.** This is the fixable case: turn up the
    guidance knob (Check A), not curate the data. A + B together say the same thing — free inference fix at higher
    `w`, no data curation, no retrain needed.

- [x] ✅ **Check C — is a big heart even reachable at sampling time?**  (free, ~5 min) — **YES, strongly reachable via best-of-N; the distribution is NOT capped.**
  - **Run:**
    ```bash
    python scripts/cardio_enhance_diag.py c --ckpt $CKPT --vae $VAE --scale-factor $SF --n 80 --best-of 8
    ```
  - **What it does:** two probes. (i) draws cardio with the `normal` anchor (healthy → sick is the enlargement
    direction) and compares to the `null` anchor. (ii) best-of-8: draws 8 per slot, keeps the biggest-heart one.
  - **What to do with the result:** if either probe reaches ~0.53, big hearts *are* reachable — a free
    sampling-time fix (use the `normal` anchor, or keep the best of several draws). If neither gets there, the
    model's range is capped, and the real lever is data curation or the codec.
  - **Done when:** `results/section4/cardio_enhance/reachability.txt` has the CTR median for null vs normal
    anchor and for best-of-8.
  - **RESULT (n=80, at default w=1):** anchor=null 0.477; anchor=normal **0.477 (identical — the `normal` anchor
    does nothing on its own at low guidance)**; **best-of-8 median 0.571 (100% >0.5, 84% >0.53, 69% >0.55)**.
    So big hearts are strongly REACHABLE by rejection sampling — the learned distribution is NOT capped, the model
    can draw big hearts, it just doesn't always. This rules out the "range is capped → data/codec" branch.
    Caveat: best-of-8 CTR is high BY CONSTRUCTION (we keep the max-CTR draw); whether the DETECTOR fires on the
    kept images is the real utility question → answered by the lever comparison below.

- [x] ✅ **The evidence that started this — fake vs real, both tools (DONE)**  ✓ `scripts/ctr_vs_detector.py`,
  `results/section4/ctr_vs_detector_v4/`: fake cardio CTR median 0.482 vs real 0.536; of the fakes that *do*
  have an enlarged heart (CTR above 0.5), the detector catches only 54.5% (real: 86.4%). Two independent tools
  (geometry + detector) agree the model under-draws the heart. This is why the push exists; also cited in plan 08.

- [x] ❌ ~~**Borrow proven tricks — run `/augment` (with Opus or Fable 5)**~~ — **SKIPPED: the utility gate was
  cleared without any borrowed tricks** (best-of-N rejection sampling, already in our toolbox, took the synth-cardio
  detector to 0.875 > real 0.806). No need to mine other people's work for a gain we already have. The `/augment`
  prompt below is kept only as a record of the levers considered.
  - **Done when:** ~~the `/augment` compile lists at least 3 tricks~~ — n/a, task retired.

- [x] ✅ **Apply the cheapest fix A/B/C (and `/augment`) point to** — winner = inference-only, no retrain.
  DONE: chosen fix (best-of-8, detector 0.875 > real 0.806) proven; before→after filed in the plan-08 verdict
  (cardio 0.365→0.875). The full 10k best-of-8 premium pile was deliberately **PARKED** (killed — not needed:
  the co-presence verdict is a mechanistic null and single-disease detectability is already established on
  160/200 samples). Below is the original in-progress note / fix ladder, kept for the record.
  DONE: **w6 natural 10k** (CTR med 0.562, 85% big) + **w4 10k** (CTR med 0.545, 76% big) generated
  (`scripts/build_augment_probe_set.py` → `results/section4/augment_probe/synth/{w6,w4}/`, each with
  `manifest_all.csv` → A1 and `manifest_ctr_gt0.5.csv` → A2). Augment-probe (`scripts/augment_probe.py`,
  feature-cached frozen backbone) run on both → **robust NULL** (see plan 08 + `augment_probe/verdict.md`).
  RUNNING (overnight ~15h): **w4 best-of-8 10k** premium pile + confirmatory probe (one chained job, seed 900000).
  `/augment` skipped — the utility gate was cleared without borrowed tricks. Retrain rungs stay PARKED.
  1. Sampling knob only — `w` = 2–3, or the `normal` anchor, or best-of-N. Zero retraining.
  2. Curate the data — keep training cardio with CTR above 0.55, oversample the big-heart films. Most likely the
     real lever if B shows borderline data.
  3. Short cardio-only fine-tune from v4-60k on that severe subset (~20k steps).
  4. Re-train the VAE to un-squash the code — only if 1–3 stall. This is plan 09 Task 5.
  - **Done when:** the fake cardio set is regenerated with the chosen fix, plan 08's Exp-09 + CTR are re-run, and
    the before→after gain is written into the plan-08 verdict (target: detector sensitivity ≥ 0.70 or CTR median ≥ 0.53).

## Recommended skill
`/augment` (Opus or Fable 5) for the borrow-tricks task. Everything else is custom and already built:
`scripts/cardio_enhance_diag.py` (Checks A/B/C), `scripts/ctr_vs_detector.py`, `scripts/generate.py`.

## Engagement Instructions

```bash
# env: micromamba activate jaxstack (on hippo); GPU free.
# Set these once, then the three Check commands above paste straight in:
CKPT=ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors
VAE=ckpts/vae_v2/vae_v2_step0075000.pt
SF=data/latents_v2/joint_perch/scale_factor.pt

# Check A — guidance sweep:
python scripts/cardio_enhance_diag.py a --ckpt $CKPT --vae $VAE --scale-factor $SF --w-list 1 2 3 4 --n 60
# Check B — training-data CTR (no model needed, just the VAE):
python scripts/cardio_enhance_diag.py b --vae $VAE --scale-factor $SF --latent-dir data/latents_v2/joint_perch --n 200
# Check C — reachability (anchor + best-of-N):
python scripts/cardio_enhance_diag.py c --ckpt $CKPT --vae $VAE --scale-factor $SF --n 80 --best-of 8

# All three write to results/section4/cardio_enhance/. Read the three files, then pick a rung on the fix ladder.
# Then run /augment (below) with Opus or Fable 5 for borrowed tricks.
```

**`/augment` prompt (paste this to run the borrow-tricks task):**

> `/augment` I want to make **fake cardiomegaly easier to detect** on our finished chest-X-ray model
> (checkpoint **v4-60k**, `ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors`) **without training a new model
> from scratch.** Where things stand: it's a latent diffusion model on VAE-v2 codes (per-channel whitened), and
> it draws clean normal / cardio / effusion X-rays. **The problem:** the cardio hearts come out too small —
> heart-size ratio (CTR) median **0.482** vs real **0.536**; only 40% enlarged vs 77% real; the trained detector
> flags only **36.5%** of fake cardio vs 80% real; and two independent tools (geometric CTR + the classifier)
> agree it under-draws. **Likely causes:** very few unique cardio training films (~2,774), a squashed VAE code
> (~1 of 4 channels usable), and possibly borderline training data (a filter kept mild cardiomegaly). **Already
> ruled out / off the table:** min-SNR weighting (no effect), guidance for *sharpness* (flat), and a full VAE
> re-train (too costly now). **What I want:** mine other people's work for **cheap, high-confidence tricks** that
> can push fake-cardio detectability toward a realistic **0.70–0.79** — a genuine win, not a 0.90 fantasy. Focus
> on: generation-time tricks (classifier / CTR-reward guidance, best-of-N or rejection sampling toward a bigger
> heart, `normal`-anchor guidance = the healthy→sick enlargement direction), oversampling the severe films, a
> short cardio-only fine-tune, and severity conditioning. **For each trick:** name it and its source (cite, flag
> how sure you are), what it buys, what it costs, and its *realistic* gain **given a squashed ~1-channel code**
> (don't assume a healthy code). **Prefer tricks that almost surely give a small real gain over risky big swings**
> — I'd rather stack three reliable +5% moves than chase one uncertain +30%. Run a few rounds; I'll say `compile`.
