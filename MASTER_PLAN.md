# Paper 3 — PoE Composition of Correlated Chest Diseases

## Mission
Test whether composing two single-disease conditions at generation time via
Product-of-Experts yields a *realistic* both-disease X-ray — and isolate
**correlation** as the cause when it breaks. A correlated treatment pair
(cardiomegaly + effusion) is contrasted against an independent control pair
under the same VAE, LDM, and method, so the only difference is correlation.

## Objectives
1. Acquire, extract, and characterize the dataset — archives unpacked, data types
   and label table understood, distributions and sample images visualized — before
   any modeling. This groundwork also feeds the correlation matrix (Exp1).
2. Establish whether PoE composition reproduces a realistic both-disease appearance.
3. Isolate correlation as the cause of any failure, via a treatment-vs-control design.
4. Determine whether the failure is structural — not fixable by reweighting.
5. Test whether anchoring at health beats anchoring at the mixture.
6. Build and validate the supporting infrastructure (VAE, single-disease LDM,
   feature extractors, floor baseline), each gated by an overfit sanity run before
   any full training, so a broken loop is caught before GPU-hours are spent.

## Goals
1. **Data ready** — archives extracted, every image readable, label table parsed,
   data types and counts inventoried, distribution + sample-image figures produced.
2. **H1 (marginals preserved)** — presence within 5 points of real and two-sample
   ≤ 0.60 for both diseases.
3. **H2 (joint broken, treatment)** — two-sample ≥ 0.65, above the floor's 95% upper bound.
4. **H2-control (joint preserved, control)** — two-sample ≤ 0.55, at the floor.
5. **H3 (structural)** — best reweighting over the grid still leaves ≥ 0.65.
6. **H-anchoring** — report whether the `normal` anchor yields a smaller gap than `∅`
   (null result acceptable).
7. **Overfit sanity** — VAE drives reconstruction error on a tiny fixed batch to near
   zero; LDM overfits a handful of single-disease latents and regenerates them — both
   before the corresponding full train.
8. **Gates cleared** — Exp1 (both a strong and a near-zero pair exist), Exp3 ceiling
   high, Exp4 single-disease FID convincing, Exp5 marginals gate passed.

## Expected Outcome
A controlled, pre-registered result: under one identical pipeline, PoE reproduces
the joint for the independent pair but not the correlated pair (or a clean,
publishable null), with the VAE ceiling check ruling out the codec as the cause
and reweighting ruling out a fixable imbalance. The figure: a (heart size,
blunting) scatter where each axis overlaps but the diagonal coupling differs only
when correlated.

## Operating Constraint — Local Authoring, Dumb Remote Execution
All code is written, debugged, and committed **locally**, then pushed to GitHub.
The rented RunPod GPU is a **dumb executor**: the only actions performed on it are
`git clone` (or `git pull`) and running pre-written `bash` scripts to train the
1 VAE and the 2 single-disease LDMs. No code authoring, no interactive debugging,
no manual file editing, no notebook hacking on the remote box. If a training run
needs a code change, the change is made locally, committed, pushed, and pulled —
never edited in place on RunPod. This keeps every GPU-hour spent on compute, not
development, and makes each remote run reproducible from a single commit SHA.

Implications baked into the plans:

- Each trainable component ships with a committed, self-contained `bash` entrypoint
  (env setup → data fetch → train → checkpoint out) runnable with zero edits.
- Data acquisition/preprocessing that the remote needs is either committed or
  fetched by script at run time — not assumed present on the box.
- The RunPod lifecycle (provision → clone → run → retrieve checkpoints → teardown)
  lives entirely in `compute-budget/plans/runpod-execution/`.

## Definition of Done
1. Dataset acquired and archives extracted; disk layout documented; every image
   verified readable (no corrupt/truncated files).
2. Data inventory complete — image format and dtype, resolution(s), bit depth,
   per-label counts, the both-disease label table, and any splits — written down.
3. EDA figures produced — label-frequency and co-occurrence distributions, plus a
   sample-image grid spanning the relevant conditions.
4. Exp1 correlation matrix computed; treatment pair (strong φ) and control pair
   (φ ≈ 0) both confirmed in *this* data; both-disease N reported per pair (power check).
5. Heart-size + blunting extractor validated to behave identically on real and generated images.
6. VAE overfit sanity passed (near-zero recon on a fixed mini-batch), then VAE trained;
   SSIM/LPIPS passes the reconstruction gate; checkpoint in `ckpts/`.
7. Ceiling check passes (real both-disease images reconstruct faithfully) — or VAE
   fixed and re-run before any composition claim.
8. LDM overfit sanity passed (memorizes a handful of single-disease latents), then
   single-disease LDM trained with CFG dropout (supports both `∅` and `normal`
   anchors); single-disease FID convincing.
9. Exp5 marginals gate cleared for both diseases.
10. Exp6 headline run across all three arms (treatment/`∅`, control/`∅`, treatment/`normal`);
    decision recorded vs the floor.
11. Exp7 reweighting grid run (H3 decision recorded); Exp8 floor baseline (PoE vs overlay) run.
12. Every experiment's outcome filed against the pre-registration table — no post-hoc reinterpretation.

## Sub-Scopes
- ⚠️ plans/data-foundation/ — "acquire, extract, and DICOM/monochrome-normalize into a clean corpus"
- ⚠️ plans/eda/ — "size, imbalance, preprocessing, correlation matrix + visualizations; the gate"
- ⚠️ plans/vae/ — "label-blind codec + ceiling check"
- ⚠️ plans/single-disease-ldm/ — "single-disease experts with dual anchors"
- ⚠️ plans/metrics-extractors/ — "the C2ST/MMD/FID/floor + extractor measurement layer Exp 5–8 depend on"
- ⚠️ plans/composition-experiments/ — "PoE composition + the headline experiments"
- ⚠️ plans/documentation/ — "running record of results + progress across all phases"

## Plans
(none yet — added by plan-day)
