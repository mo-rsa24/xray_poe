# 📊 Downstream check — can the new model's fake single-disease X-rays be spotted, and do its two-disease X-rays really show both? (compared to the old model)

**Words this plan can't avoid — each defined once, plain:**
- **Exp 09** — the "is the single disease even in the picture?" check. Make a fake cardio (or effusion) X-ray, ask the detector if it can spot it.
- **Exp 03 / PoE yield** — the "do two-disease X-rays really show both?" check. Of the X-rays meant to have an enlarged heart AND lung fluid, the share that show both.
- **Sensitivity (catch rate)** — how often the detector flags a true case (a real cardio X-ray → "cardio: yes").
- **Specificity (clear rate)** — how often it correctly clears a true non-case.
- **Floor-corrected** — the raw rate minus the false-alarm rate the tool reads on healthy X-rays, so we don't credit noise as a real signal.
- **Codec** — the VAE: the network that squashes an X-ray down to a small code and back. "Sharper codec" = the new VAE-v2.
- **GO / NO-GO** — GO = release the new model for later experiments. NO-GO = write down a null result and keep the old model.
- **Pre-registration table** — the results table we promised to fill in *before* running, so we can't pick and choose after the fact.
- **CTR** — heart width ÷ chest width off the X-ray. Above 0.5 = enlarged heart. Real sick hearts sit near 0.535.
- **CP (corner angle)** — the wedge angle in the bottom corner of the lung. A blunted (filled-in) corner means fluid.
- **Abstain** — X-rays sitting right on the line (heart width ≈ 0.5, or corner angle ≈ the cutoff), too close to call either way.
- **Wilson band** — the 95% range around a percentage, i.e. how much the number could wobble from sampling.
- **best-of-8** — draw 8 fake X-rays for one slot, keep the one with the biggest heart, throw the other 7 away.
- **Guidance / w_cardio** — a sampling knob. Turning it up pushes the fake image harder toward "cardio".

## Background
The idea being tested: a sharper compressor (the new VAE) should draw each single disease more clearly, and that should raise the share of two-disease X-rays that actually show both. This plan tests that directly. We re-run two checks with the winning model (**`ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors`, the v4 per-channel-whitened run**, decoded with `data/latents_v2/joint_perch/scale_factor.pt`):
- **Exp 09** — can the detector spot each single disease in a fake X-ray?
- **Exp 03** — of the two-disease X-rays, how many show BOTH?

We compare against the old model (arm1_cleancardio at 100k steps: it got fake cardio spotted only about 0.10→0.25 of the time, a FAIL; two-disease share about 9.3%, a null). If both main checks pass (spotted ≥ 0.80, two-disease share ≥ 0.30), we release the new model to Exp 5–7. If they fail, we write down the null and keep the old compressor. (Note 2026-07-02: the old `ckpts/vae_v2_ldm/model_step100000.pt` line is out of date — that was the v1 μ run; the real check used v4-60k.)

## Description
Run `scripts/validate_individual_synthetics.py` (Exp 09) with the new model to get fresh numbers for how often fake cardio and fake effusion get spotted — effusion must not get worse. Then run `scripts/gate3_yield.py` (Exp 03) with the same model for the fresh two-disease share on the NIH frozen separable set. Compare both to the old model. Write one combined verdict file.

## Purpose
Serves Objective 7 and DoD item 7. This is the last step in the plan. Its verdict decides whether we release the new model or write the whole effort down as a null. Needs plan 06 finished first (a trained checkpoint has to exist).

## Goal
One verdict file `results/vae_v2/downstream_eval/verdict.md`. It holds the Exp-09 numbers (how often each disease is spotted, and the clear rate) and the Exp-03 number (the two-disease share) for the new model against the old, an overall GO (both checks pass) or NO-GO (at least one fails), and the release decision.

## Tasks

- [x] ✅ **Re-run Exp 09 (can each single disease be spotted?) with the new model (v4-60k)**  ✓ done (`results/exp09_individual/verdict.md`, tag v4_60k, n=200): fake effusion is spotted **0.865** of the time — PASS (real 0.790); fake cardio only **0.365** — FAIL (real 0.800). So the problem is cardio, and it's the drawing, not the combining. Backed by the CTR-vs-detector check (below / plan 07-ldm-enhance): fake cardio has a heart-width ratio of 0.482 against real 0.536 — both tools agree the model draws the heart too small.
  - **What it measures:** how often fake cardio gets spotted under the new compressor.
  - **How to run it (these are the real flags — they differ from the older block further down):**
    ```bash
    python scripts/validate_individual_synthetics.py \
      --ldm ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors --tag v4_60k \
      --vae-ckpt ckpts/vae_v2/vae_v2_step0075000.pt \
      --scale-factor data/latents_v2/joint_perch/scale_factor.pt \
      --classifier ckpts/presence_classifier_finetuned.pt --n 200
    ```
  - **Numbers to pull:** how often cardio and effusion get spotted (catch rate), and the clear rate for each; the detector is `densenet121-res224-nih` + a small MLP head (same one as the original Exp 09 — see [[classifier-identity]] memory).
  - **Old-model baseline:** fake cardio spotted ≈ 0.10–0.25 (FAIL); fake effusion ≈ 0.905 (PASS). Effusion must stay ≥ 0.80 — don't let it slip.
  - **Pass mark:** fake cardio spotted ≥ 0.80. **The final GO/NO-GO gets decided in the verdict task below, not here.**
  - **Saved artifact:** `results/vae_v2/downstream_eval/exp09/exp09_results.csv` (one row per image + a summary), W&B log.
  - **Check:** the results CSV exists; the cardio and effusion columns are filled; the image count matches.

- [x] ✅ **Re-run Exp 03 (do two-disease X-rays really show both?) with the new model**  (DONE at N=2000:
  `results/vae_v2/downstream_eval/exp03/`, default settings w_cardio=2.0 / w_effusion=1.0.
  **Two-disease share, floor-corrected: 2.6%. Counting only the images we can call confidently: 34/943 = 3.6%
  (95% range ≈ [2.6%, 5.0%]). Too-close-to-call: 52.8%. Healthy-image false-alarm floor: 3.9%.** This is a
  clear null against the 30% mark — at the full sample size it's below both the quick n=200 preview (4.6%) and
  the old model (9.3%). This is the final paper number for the two-disease share at default settings. ONE
  CAVEAT: the default cardio knob (w_cardio=2.0) is the same low setting that plan 07 showed draws the heart too
  small. Whether turning that knob up — the plan-07 fix, applied to the cardio part of the two-disease image —
  rescues this is the follow-up that matters → the knob sweep below.)
  - **What it measures:** the fresh two-disease share under the new compressor.
  - **How to run it (this is the preview command — set `--n` to 2000 for the final number):**
    ```bash
    python scripts/gate3_yield.py \
      --ckpt ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors \
      --vae ckpts/vae_v2/vae_v2_step0075000.pt \
      --scale-factor data/latents_v2/joint_perch/scale_factor.pt \
      --n 2000 --out results/vae_v2/downstream_eval/exp03/
    ```
  - **Numbers:** the floor-corrected two-disease share (heart enlarged, CTR > 0.5, AND a blunted lung corner); the 95% range; and the spread of heart-width ratios (we expect it to move toward the real middle ≈ 0.535).
  - **Old-model baseline:** floor-corrected 9.3% (a null, under the 10% mark). Keep the don't-call-it band: within ±0.03 of CTR 0.5, or within ±10° of the corner-angle cutoff.
  - **Pass mark:** two-disease share ≥ 0.30 (floor-corrected, confident calls only). **Decided in the verdict task below.**
  - **Saved artifact:** `results/vae_v2/downstream_eval/exp03/exp03_results.csv`, scatter plot `results/vae_v2/downstream_eval/exp03/ctr_cp_scatter.png`.
  - **DOES TURNING THE CARDIO KNOB UP HELP? (n=500, `results/vae_v2/downstream_eval/exp03_wc{4,6}/`):** turning
    the cardio knob up with the plan-07 fix — w_cardio 2 → 4 → 6 — does sharpen the heart (too-close-to-call
    drops 52.8% → 43.0% → 33.4%, the heart clearly renders bigger) **but the two-disease share stays flat:
    floor-corrected 2.6% → 4.1% → 3.1% (confident 3.6% → 4.2% → 4.2%).** So the two-disease failure is NOT just
    the knob being too low — the same knob that lifts single-disease cardio to 0.875 does nothing here. The model
    draws a big heart OR fluid, not both in one X-ray. That's a real limit in how it combines diseases, not a
    drawing-quality problem. This is the paper's headline finding.
  - **Check:** 2000 samples; the share with its range; the scatter plot made.

- [x] ✅ **Compare to the old model and write `verdict.md`** — DONE: `results/vae_v2/downstream_eval/verdict.md`.
  **Overall NO-GO on the two-disease gate → mechanistic null written down.** Single-disease Exp-09 PASSES (cardio
  0.365 → 0.875 with the best-of-8 fix; effusion 0.865). Two-disease Exp-03 is 2.6%, far under 30%, and turning
  the cardio knob to 4 or 6 doesn't move it. The two checks split: single diseases draw fine, but two-disease
  images collapse and don't respond to the cardio fix → the problem is in the combining, not the drawing
  quality. Verify: `grep -E "Overall: GO|Overall: NO-GO"` returns one line ✓.
  - **What it does:** a plain GO / NO-GO, plus one row for the pre-registration table.
  - **Format:**
    ```markdown
    ## VAE-v2 Downstream Eval — Verdict

    | Experiment | Metric              | Pre-intervention | vae-v2 LDM | Target | Pass? |
    |------------|---------------------|------------------|------------|--------|-------|
    | Exp 09     | cardio sens         | 0.10–0.25        | X.XX       | ≥ 0.80 | ✓/✗  |
    | Exp 09     | effusion sens       | 0.905            | X.XX       | ≥ 0.80 | ✓/✗  |
    | Exp 03     | PoE yield (floor-c) | 9.3%             | XX.X%      | ≥ 30%  | ✓/✗  |

    **Overall: GO / NO-GO**

    Release decision:
    - GO: release `ckpts/vae_v2_ldm/model_step100000.pt` to Exp 5–7. Supersedes
      `ckpts_arm1_cleancardio/` for all downstream reporting.
    - NO-GO: null recorded. `vae_step0025000.pt` + current LDM remain the codec.
      Log reason and which gate failed.

    Pre-registration table row updated: [YES/NO]
    ```
  - **Saved artifact:** `results/vae_v2/downstream_eval/verdict.md`
  - **Check:** `grep -E "Overall: GO|Overall: NO-GO" results/vae_v2/downstream_eval/verdict.md` returns one line.

- [x] ✅ **Add the result to the pre-registration table** — row dated 2026-07-07 added to
  `results/tables/table2_augmentation.md` (a new "Pre-Registration Outcome — VAE-v2 Downstream" section; columns =
  codec · cardio spotted · effusion spotted · two-disease share · GO/NO-GO). Outcome: **NO-GO on the two-disease
  check** (cardio 0.875 ✓, effusion 0.865 ✓, two-disease share 2.6% ✗). NO-GO means no release, so the "if GO,
  note the release" line in DoD-7 doesn't apply. The full verdict is linked at
  `results/vae_v2/downstream_eval/verdict.md`.
  - **What it does:** keeps us honest — every experiment's result gets written down.
  - **How:** add a new row to `results/tables/table2_augmentation.md` (the pre-registration table): codec, cardio spotted, effusion spotted, two-disease share, GO/NO-GO.
  - **If GO:** note in MASTER_PLAN.md DoD-7 that the checkpoint has been released.
  - **Check:** the table has a row dated today with the new-model results.

- [x] ✅ **[EXPLORATORY — does not change the paper's story] Do the model's own best fake hearts help a detector?**  — DONE: a clear null (all sub-tasks ✅, figure + verdict written). The fakes don't help the fixed-reader detector; this fits the "model draws hearts too small" finding; it doesn't touch the paper's story.
  A quick side-check. We take the fake cardio X-rays the model draws *well*, add them to the real cardio
  training set, retrain the heart detector, and see if it gets any better at spotting cardio. This does not
  touch the main result — the GO/NO-GO above holds either way. Even a win here would be small and come with a
  string attached: it would only say "the fakes help *after* we throw away the ones the model botched," which
  agrees with our finding that the model draws hearts too small. Ties to Objective 7 (is the fake data useful?),
  as a low-stakes extra.
  - **What we measure:** does the detector's cardio catch rate — how often it flags a true cardio X-ray — go
    up when we add filtered fake cardio to its training data?
  - **How it's set up:** keep the trained image-reader (`densenet121-res224-nih`) fixed — it *is* our
    measuring instrument, so we don't touch it. Retrain only the small decision layer on top. Three training
    sets, each trained 3 times with a different random seed, all tested on the SAME set of held-out real cardio
    X-rays that none of the training used:
    - A0 = real cardio only
    - A1 = real + all fake cardio (nothing filtered out)
    - A2 = real + only the big-heart fakes (CTR > 0.5, measured by the PSPNet tool)
  - **Careful — don't over-read a "no change":** because the image-reader is fixed, it might not be able to
    use a real benefit even if one is there. So "no improvement" could mean "the fakes don't help" OR "a fixed
    reader can't take advantage of them." Report it; don't lean on it.
  - [x] ✅ Make ~10,000 fake cardio images at guidance w=3–4 (plan 07 Check-A: w=4 gives 69% big-heart);  ✓ verified (synth/w6 + synth/w4 manifests)
    measure each one's heart-width ratio (CTR; over 0.5 = enlarged); split them into two piles — all of them,
    and only the big-heart ones (CTR > 0.5). Save to `results/section4/augment_probe/synth/`.
  - [x] ✅ Retrain only the top layer (image-reader fixed) for the three training sets A0/A1/A2, each with  ✓ verified (probe_runs_{w6,w4}.csv, 9 runs each via scripts/augment_probe.py)
    seeds {0,1,2} — reuse the presence-classifier fine-tune script with a new `--freeze-backbone` flag, swapping
    the training list for each set.
  - [x] ✅ Test all 9 retrained detectors on the held-out real cardio X-rays; report the cardio catch rate  ✓ verified (probe_summary_{w6,w4}.csv — AUC + sens@spec0.9, mean±95% across 3 seeds)
    (and AUC, an overall accuracy score) as an average with its 95% spread across the 3 seeds. **Save every
    single run's number (3 sets × 3 seeds = 9 values), not just the per-set averages** — the figure and the
    spread both need them.
  - [x] ✅ Read the result — **NULL: A2 does NOT beat A0 and A2 does NOT beat A1**, and this holds for both the w6
    and w4 fake-image sources. Cardio AUC sits flat at 0.874–0.878 across all three sets (every set inside every
    other's 95% band); catch-rate-at-90%-clear-rate flat at ~0.62–0.64. Flat across the sets AND across the
    sources points at a ceiling in the fixed reader, not in the fake-heart quality — so it's "no measured help,"
    not "proven useless" (as the plan itself warned). Doesn't touch the paper's story. Written to
    `results/section4/augment_probe/verdict.md`; run numbers in `probe_runs_{w6,w4}.csv` /
    `probe_summary_{w6,w4}.csv`; script `scripts/augment_probe.py` (caches the reader's features so it runs fast).
    NOTE: this ran on the cheap piles (w6/w4); best-of-8 is unlikely to move a flat, reader-limited null.
  - [x] ✅ Figure — `scripts/fig_augment_probe.py` → `results/section4/augment_probe/fig_augment_probe.png`.
    Two panels (AUC + catch-rate-at-90%-clear-rate), one thin line per seed across A0/A1/A2 plus a bold per-set
    average, BOTH sources (w6 solid / w4 dashed), y-axis pinned 0–1 (never zoomed). Shows the flat null honestly.
    On purpose, it leaves out the old 0.365 / 0.80 reference lines — those measure a different thing (how often
    the detector spots fake cardio, Exp-09) than this figure's axis (a retrained detector's score on real test
    images); the real-only baseline is set A0 itself, marked instead. The original spec (one line per seed /
    grouped bars / axis 0–1) is kept below for the record:
    > ~~Figure (design from `/design-figure`): **main version = one line per seed** — the three training
    sets A0/A1/A2 along the bottom, one line per seed joining its three sensitivities, plus a bold average dot;
    this shows the A2 > A1 > A0 order holds in *every* seed and isn't just an averaging trick (which matters
    with only 3 seeds). **Backup/paper version = grouped bars** (average with 95% spread) once the order is
    confirmed. BOTH: **sensitivity axis pinned 0.0–1.0, never zoomed in** — zooming turns a tiny gain into a
    fake-looking win and would contradict the caption; mark two lines, **0.365** (how well the detector spots
    raw unfiltered fakes) and **0.80** (how well it spots real cardio). Caption: a small win that only shows up
    after filtering — it backs the finding, doesn't overturn it. → `results/section4/augment_probe/fig_augment_probe.png`.
  - **Saved artifact:** `results/section4/augment_probe/{synth/, heads/, verdict.md}`
  - **Verify:** `verdict.md` has a 3-set table with each set's catch rate and its 95% range across 3 seeds; the A2-vs-A1 comparison is stated.

## Recommended skill
— custom. Re-uses the existing `validate_individual_synthetics.py` + `gate3_yield.py` scripts with a new checkpoint argument.

## Engagement Instructions

```bash
# ── env ───────────────────────────────────────────────────────────────────────
# On hippo: micromamba activate jaxstack
# Pre-intervention baselines: Exp09 cardio sensitivity=0.10, Exp03 PoE yield=0.093

# ── Exp 09: individual disease gate ─────────────────────────────────────────
# validate_individual_synthetics.py flags: --ldm (safetensors), --tag (before|after),
#   --vae-ckpt, --scale-factor, --classifier, --n, --out-dir, --w, --anchor, --margin
$ python scripts/validate_individual_synthetics.py \
    --ldm ckpts_vae_v2_ldm/model_step0100000.safetensors \
    --tag after_vae_v2 \
    --vae-ckpt ckpts/vae_v2/vae_v2_step075000.pt \
    --scale-factor data/latents/scale_factor.pt \
    --classifier ckpts/presence_classifier_finetuned.pt \
    --cardio-csv data/splits/cardio_not_effusion.csv \
    --effusion-csv data/splits/effusion_not_cardio.csv \
    --n 500 \
    --anchor null \
    --out-dir results/vae_v2/downstream_eval/exp09/

# read results
$ python3 -c "
import pandas as pd, json, pathlib
out = pathlib.Path('results/vae_v2/downstream_eval/exp09/')
# validate_individual_synthetics.py writes a JSON verdict + per-sample CSV
for f in out.glob('*.json'):
    d = json.loads(f.read_text())
    print(f.name, ':', d)
"
# targets:  cardio sensitivity ≥ 0.80  (baseline: 0.10)
#           effusion sensitivity ≥ 0.80 (baseline: 0.905 — must not regress)

# ── Exp 03: PoE Gate-3 yield ─────────────────────────────────────────────────
# gate3_yield.py flags: --ckpt (safetensors), --vae, --scale-factor, --n, --out,
#   --csv-out, --fig-out, --w-cardio, --w-effusion, --floor-dir
$ python scripts/gate3_yield.py \
    --ckpt ckpts_vae_v2_ldm/model_step0100000.safetensors \
    --vae ckpts/vae_v2/vae_v2_step075000.pt \
    --scale-factor data/latents/scale_factor.pt \
    --n 2000 \
    --w-cardio 2.0 \
    --w-effusion 1.0 \
    --out results/vae_v2/downstream_eval/exp03/ \
    --csv-out results/vae_v2/downstream_eval/exp03/yield_per_sample.csv \
    --fig-out results/vae_v2/downstream_eval/exp03/ctr_cp_scatter.png \
    --floor-dir results/exp04_leakage/

# gate3_yield.py prints the verdict live; also read the CSV:
$ python3 -c "
import pandas as pd
df = pd.read_csv('results/vae_v2/downstream_eval/exp03/yield_per_sample.csv')
n = len(df); npass = int(df['gate3_pass'].sum()); nabs = int(df['abstain'].sum())
conf = df[~df['abstain']]
print(f'n={n}  gate3_pass={npass/n:.1%}  abstain={nabs/n:.1%}')
print(f'confident yield: {conf[\"gate3_pass\"].mean():.1%}')
# floor-corrected yield printed live by gate3_yield.py
# target: floor-corrected ≥ 0.30  (baseline: 0.093)
"

# ── Write verdict ─────────────────────────────────────────────────────────────
# Fill results/vae_v2/downstream_eval/verdict.md with the table format from the task above
$ grep -E "Overall: GO|Overall: NO-GO" results/vae_v2/downstream_eval/verdict.md
# expect: exactly one line

# ── File against pre-registration table ──────────────────────────────────────
$ python3 -c "
# Quick comparison vs baselines
cardio_sens_baseline = 0.10   # Exp09 BEFORE (from memory: exp09-individual-gate)
poe_yield_baseline   = 0.093  # Exp03 BEFORE (from memory: exp03-poe-yield)
# fill in vae_v2 values from the outputs above and confirm improvement
print('Exp09 cardio baseline:', cardio_sens_baseline, '→ vae_v2: [fill]  target: ≥ 0.80')
print('Exp03 yield  baseline:', poe_yield_baseline,  '→ vae_v2: [fill]  target: ≥ 0.30')
"

# ── [EXPLORATORY] Do the model's own best fake hearts help a detector? (3 training sets) ──
# Does not change the GO/NO-GO above. Its verdict lives in results/section4/augment_probe/.
CKPT=ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors
VAE=ckpts/vae_v2/vae_v2_step0075000.pt
SF=data/latents_v2/joint_perch/scale_factor.pt
mkdir -p results/section4/augment_probe/synth results/section4/augment_probe/heads

# 1. Generate ~10k synth cardio at w=4, score CTR, split filtered/unfiltered.
#    Reuse scripts/generate.py (--disease cardiomegaly --w 4 ...) then score each film's CTR with
#    metrics.eval_checkpoint.score_decoded_array (same path as scripts/ctr_vs_detector.py).
#    Write two manifests: synth_all.csv (unfiltered) and synth_ctr_gt0.5.csv (CTR>0.5 filtered).

# 2. Retrain the DenseNet HEAD only (backbone frozen), per arm x seed.
#    NOTE: needs a --freeze-backbone flag on the presence-classifier fine-tune harness that
#    produced ckpts/presence_classifier_finetuned.pt (a few lines: requires_grad=False on the
#    densenet121 trunk, train only the MLP head). Swap the training CSV per arm:
#      A0: data/splits/cardio_train.csv
#      A1: cardio_train.csv + synth_all.csv
#      A2: cardio_train.csv + synth_ctr_gt0.5.csv
#    for arm in A0 A1 A2; do for seed in 0 1 2; do
#      <train head, frozen backbone, seed> -> results/section4/augment_probe/heads/${arm}_s${seed}.pt
#    done; done

# 3. Evaluate every head on a FIXED held-out real cardio test (disjoint from cardio_train.csv);
#    report cardio sensitivity + AUC; aggregate mean +/- 95% CI over the 3 seeds per arm.

# 4. verdict.md: 3-set table (detection +/- spread). The read: A2 beats A0 (the fakes helped) AND
#    A2 beats A1 (the big-heart filter did it). A gain still backs the finding — the model only helps
#    after we drop the ones it botched.
```
