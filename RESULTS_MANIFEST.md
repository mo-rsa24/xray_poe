# Results Manifest — *Compositional Generation of Comorbid Diseases on Chest X-Rays*

Claim→figure/table→script map for the paper. PoE composition of pretrained single-disease LDMs
(cardiomegaly + effusion), validated on NIH ChestX-ray14 two ways: **ecological validity** (presence
classifier scores composed samples vs a real-data reference) and **augmentation** (synthetic comorbid
images lift a held-out classifier AUC). Optimised for SIMPLE but scientifically valid.

**Conventions:** repo root `/home/molef/PhD/Paper3`. Status: ✅ exists · ⚙️ exists-needs-edit ·
🔨 build new · 🔭 future work.

**Comorbid set (NIH ChestX-ray14):** 1,063 images / 679 unique patients (cardio+effusion), out of
112,120 total. Negatives abundant: 1,713 cardio-only, 12,254 effusion-only, 60,361 normal.
Correlation φ(cardio, effusion) = 0.13 (weak-positive).

**Scope decisions (3 days, one machine):**
- Headline joint metric = **both-present rate** (ecological validity); independence-excess demoted to
  nice-to-have (likely null at φ=0.13); C2ST/MMD two-sample test → future work.
- Joint-test arms shipped = **treatment/∅ + treatment/normal** (anchoring); independent-control pair
  → future work (needs two more single-disease LDMs).

---

## 1 — The manifest

### Qualitative

| # | Claim | Output artifact | Script + entrypoint | Input artifacts | Manual edit | Status |
|---|-------|-----------------|---------------------|-----------------|-------------|--------|
| Q1 | VAE faithfully reconstructs real CXRs | `figures/vae_recon_grid.png` (Input \| Recon \| ×5 diff; SSIM/LPIPS/MAE bars) | `scripts/vae_recon_grid.py --ckpt ckpts/vae_step0025000.pt --n 8 -o figures/vae_recon_grid.png` | VAE ckpt `vae_step0025000.pt`; real PNGs | none | ✅ |
| Q2 | Each single-disease LDM samples its pathology | `figures/ldm_single_{cardio,effusion}.png` grids | `scripts/generate.py --ckpt ckpts/ldm/model_step0040000.safetensors --vae-ckpt ckpts/vae/model.pt --disease cardiomegaly --n 200 --w 1.0 --seed 42 -o outputs/single/cardiomegaly` (repeat `effusion`) | UNet ckpt (`model_step0040000` set — **not** `ckpts/ldm/*` decoy), VAE ckpt, `data/latents/scale_factor.pt` | tile chosen samples into grid | ✅ |
| Q3 | PoE composes both diseases in one image | `figures/poe_compose_grid.png` (w-sweep rows) | `scripts/generate.py --compose --w-sweep 0.5 1.0 1.5 2.0 3.0 --anchor null --n 200 --seed 42 -o outputs/compose` | same as Q2; PoE math at `src/inference/cfg.py:174` | pick w, tile | ✅ |
| Q4 | Composed samples localize correctly | `figures/grad_cam_grid.png` (cardio=red, effusion=blue, +bbox) | `scripts/grad_cam.py --dir outputs/compose/w1.0 --n 16 --bbox --cols 4 -o figures/grad_cam_grid.png` | `ckpts/presence_classifier_finetuned.pt`, composed PNGs | **fix effusion-head corner artifact first**, then regen from final LDM | ⚙️ |
| Q5 | Lesion extent via masks/boxes | `figures/medsam_boxes.png` | `scripts/medsam_boxes.py` (**new**) — auto heart box from Q6; manual effusion box | MedSAM weights, composed PNGs | **effusion bbox drawn manually**; heart bbox auto — label each in caption | 🔨 |
| Q6 | Cardiomegaly quantified by CTR | `figures/ctr_panel.png` + CTR values | `scripts/ctr_hybridgnet.py` (**new**) | HybridGNet weights, composed/real PNGs | confirm heart-size instrument (locked decision, PARKING_LOT) | 🔨 |
| Q7 | No train→sample leakage across the pipeline | `figures/leakage_grid.png` — 5 cols: real→VAE-recon→LDM-cardio→LDM-effusion→PoE | `scripts/leakage_grid.py` (**new**, composites Q1–Q3 outputs) | outputs of Q1, Q2, Q3 | **blinded radiologist presence labels** per disease per cell | 🔨 |

### Quantitative

| # | Claim | Output artifact | Script + entrypoint | Input artifacts | Status |
|---|-------|-----------------|---------------------|-----------------|--------|
| N1 | **Ecological validity:** composed both-present rate matches real comorbid (with CIs) | `results/exp5_presence.json` → `tables/presence_rates.tex` | `metrics/presence_classifier.py --mode joint --composed_both outputs/compose/w1.0 --real_both data/nih/<both> --n 1000 -o results/exp5_presence.json` | classifier ckpt; composed + **real comorbid** dirs | ⚙️ (add real-reference + Wilson CI) |
| N2 | Composition beats naive overlay on fidelity | `results/fid.json` → `tables/fid.tex` | `metrics/fid_xrv.py --real data/nih/<both> --gen outputs/compose/w1.0 --baseline outputs/overlay --embed xrv -o results/fid.json` (**new**); overlay via `scripts/naive_overlay.py` (**new**) | XRV DenseNet121 feats; composed + real + overlay sets | 🔨 |
| N3 | Joint structure indistinguishable from real (C2ST/MMD vs floor, 3 arms) | `results/c2st.json` | floor: `eda/floor_power_check.py` (✅); C2ST classifier (**new**) | real-vs-composed sets | 🔭 future work |
| N4 | **Augmentation Δ:** synthetic comorbid lifts held-out AUC | `figures/aug_curve.png` (AUC vs real-N, arms a/b/c) + `results/aug_delta.json` | `scripts/aug_experiment.py` (**new**) | comorbid split, PoE + overlay synthetics, DenseNet-121 | 🔨 |

**N1 detail (headline ecological-validity number).** Three rates, same classifier: π_PoE (composed
both-present), π_real (real comorbid both-present), π_indep (product of composed marginals). Report
**Wilson** CI per rate, **Newcombe** CI for π_PoE − π_real. Lead with "PoE puts both diseases in at a
rate matching real comorbid"; report independence-excess (π − π_indep) but flag *likely null at φ=0.13*.

**N4 detail (the cheap headline experiment — only the small classifier retrains; fits 3 days/one machine).**
- **Task:** binary comorbid-vs-rest; **negatives include single-disease** (forces "both" not "either").
  Money metric = **comorbid-vs-single-disease AUC** (report pooled AUC too).
- **Split:** patient-disjoint on 679 patients → ~200 patients held out as **real-only** test (fixed
  across all rungs/arms), ~480 patients as augmentable pool.
- **Ladder** on comorbid *positives*: {16, 32, 64, 128, ~480 patients}; negatives fixed/plentiful.
- **Arms per rung:** (a) real-only, (b) real + PoE-composed, (c) real + naive-overlay. **(b)−(c) is the thesis.**
- **Synthetics:** fixed absolute count, **capped ≤10× real** at the lowest rung. ≥5 seeds; held-out AUC
  mean ± 95% CI; **DeLong** for paired Δ; **Benjamini-Hochberg** across the ladder.

---

## 2 — What was grafted, and from where

- **Both-present rate as compositional-presence metric (N1).** Composable Diffusion (Liu et al., ECCV
  2022) / T2I-CompBench — external detector scores each sample for each concept. *Strengthened:* bare
  presence number → headline. *Trade-off:* leans on classifier reliability; mitigated by single-disease
  training. *Confidence: high.*
- **Independence-baseline decomposition (N1).** Elementary co-occurrence math, implicit in PoE-eval
  lineage. *Strengthened:* makes the rate a *joint* claim. *Trade-off:* **partly retracted** — at φ=0.13
  with forced composition the excess is near-ceiling and likely underpowered; demoted to nice-to-have.
  *Confidence: high on math, low on its power here.*
- **Wilson/Newcombe CIs (N1).** Standard biostatistics. *Strengthened:* CIs that don't fall off [0,1]
  at saturated rates (replaces current Wald). *Trade-off:* ~10 lines `statsmodels`. *Confidence: high.*
- **Domain-specific FID + KID (N2).** CheXGenBench (arXiv 2505.10496); foundation-CXR paper
  (arXiv 2509.03903) computes FID on a CXR DenseNet. *Strengthened:* fidelity number that responds to
  pathology, not natural-image texture; KID unbiased at small N. *Trade-off:* not comparable to
  ImageNet-FID elsewhere. *Confidence: high (FID embedding), medium (KID specifics).*
- **Naive-overlay baseline (N2).** ScoreMix (arXiv 2506.10226) / Composable Diffusion strawman.
  *Strengthened:* recasts claim as "score-*product* beats score-*sum*/overlay." *Trade-off:* must build
  overlay generator. *Confidence: high.*
- **Learning-curve augmentation protocol (N4).** Sagers et al. (arXiv 2308.12453); long-tail CXR aug
  (PMC11936509). *Strengthened:* answers "why this training size" by sweeping the regime; reporting
  recipe (seeds, CI, two-sample test, BH). *Trade-off:* ladder × seeds × arms runs (cheap here).
  *Confidence: high.*
- **Class-imbalance / rare-comorbidity framing (N4 "why").** PMC11936509; Sagers. *Strengthened:* the
  scarce-set justification is ecological, not contrived — comorbidity *is* rare (1,063/112,120).
  *Trade-off:* none. *Confidence: high.*
- **Auto-heart / manual-effusion box split (Q5/Q6).** Provenance honesty from segmentation-eval
  practice. *Strengthened:* defensible per-box provenance. *Trade-off:* manual boxes don't scale past
  the figure. *Confidence: medium — verify HybridGNet landmark→box.*

## 3 — What was left native (and why borrowing would have hurt)

- **The single-disease-trained presence classifier (N1).** *Not* replaced by T2I-CompBench's VQA/CLIP
  scorer — no reliable CXR VQA for comorbidity exists; your instrument is better in-domain.
- **One fidelity metric, one joint metric.** Rejected the full CheXGenBench/ECS stack — at 1,063
  comorbid samples the quantitative arm is power-limited; a metric kitchen-sink contradicts
  "SIMPLE but valid."
- **C2ST/MMD stays out of the headline.** Your own simplification; reintroducing it would fight the
  φ=0.13 reality where the joint signal is faint by nature.
- **MedSAM auto-masks not posed as effusion ground truth.** The PARKING_LOT deferral is correct — auto
  effusion masks dressed as truth are worse than honest manual boxes.
- **The pre-registered floor (`eda/floor_power_check.py`).** Distribution-free rank-AUC null — kept as-is.

## 4 — Open grafts (decide later)

- **ECS (Embedded Characteristic Score, arXiv 2501.00744)** as a tail-sensitive complement to FID —
  *only if* a reviewer attacks distributional fidelity. *Confidence: medium, newer — verify first.*
- **C2ST/MMD across all three arms vs floor (N3)** — stronger joint-structure test, deferred; floor
  already exists. Needs the independent-control LDM pair to be meaningful. *Confidence: high it's the
  right future test.*
- **Independent-control disease pair (φ≈0)** — requires two *more* single-disease LDMs; out of scope
  for 3 days. Note in-paper that treatment/∅ vs treatment/normal tests *anchoring*, not *correlation
  specificity*.
- **Synthetic:real ratio sweep (N4)** — Sagers found ratio > count in low-data; a small ratio sweep at
  the lowest rung if time allows. *Confidence: high.*

---

## Still open / un-augmented

- **Q5/Q6 instrument decision** (heart-size: Grad-CAM bbox vs HybridGNet CTR vs MedSAM-cut) is locked
  in PARKING_LOT — Q6 can't be pinned to a final script until settled.
- **Q4** requires the effusion-head Grad-CAM corner artifact fixed before regenerating paper figures.

## Build queue (🔨/⚙️ items, dependency order)

1. ⚙️ Q4 — fix effusion-head corner artifact in `scripts/grad_cam.py`, regen from final LDM.
2. ⚙️ N1 — add real-comorbid reference + Wilson/Newcombe CIs to `metrics/presence_classifier.py`.
3. 🔨 N2 — `scripts/naive_overlay.py` + `metrics/fid_xrv.py` (XRV embedding, FID+KID).
4. 🔨 N4 — `scripts/aug_experiment.py` (patient-disjoint split, ladder, arms a/b/c, DeLong+BH).
5. 🔨 Q7 — `scripts/leakage_grid.py` (composites Q1–Q3; blinded labels).
6. 🔨 Q5/Q6 — `scripts/ctr_hybridgnet.py` + `scripts/medsam_boxes.py` (after instrument decision).
