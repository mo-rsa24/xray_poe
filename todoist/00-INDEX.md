# Paper 3 — PoE Composition of Correlated Chest Diseases

Leaf folder for `todoist-publish`. **As published** (2026-06-06) under the custom container
mapping: the Todoist **section** is `🩻 PoE Composition of Correlated Chest Diseases`; each
**scope** (theme below) is a **Task** carrying its MASTER_PLAN block; each **leaf** is a
**subtask** of its scope. Numbers carry global work order. Target project **🧑‍🎓 PhD**
(`6XHxp94hx7x3PMMR`), section `6gpfrxx3Vqx3wHC2`, label `paper3`.

> ⚠️ This custom scope→task / leaf→subtask mapping is **not** one of `todoist-publish`'s
> standard A/B/C mappings, so a re-run won't reproduce it automatically — re-apply the same
> manual choices. Numbering here matches the live tasks for a faithful record.

## 📦 Data Foundation  *(scope task)*
- [ ] [01 · Acquisition + DICOM/Monochrome Decoding](01-acquisition-dicom-monochrome.md)
- [ ] [02 · Integrity Scan + Manifest](02-integrity-manifest.md)

## 🔍 Exploratory Data Analysis  *(scope task · gate)*
- [ ] [03 · Dataset Size & Composition](03-dataset-size-composition.md) — ▶ `/data-inventory`
- [ ] [04 · Class Imbalance](04-imbalance.md) — ▶ `/data-distributions`
- [ ] [05 · Image Properties & Preprocessing](05-image-properties-preprocessing.md) — ▶ `/data-distributions`
- [ ] [06 · Correlation Matrix — Pair Selection & Gate](06-correlation-matrix.md) — custom (no skill)
- [ ] [07 · EDA Visualization Set](07-visualizations.md) — ▶ `/visualize-data-samples` + `/data-distributions`
- [ ] [08 · Preprocessing — Vision Pipeline](08-preprocess-vision.md) — ▶ `/preprocess-vision` ⚠️ inferred/uninstalled
- [ ] [09 · Feature-Engineering Preview](09-feature-engineering-preview.md) — ▶ `/feature-engineering-preview` ⚠️ inferred/uninstalled

## 🗜️ VAE  *(scope task)*
> 🔄 **Re-scoped 2026-06-09:** 5→8 leaves. The VAE scope now builds + tests + profiles
> the codec **locally** (512²→4×128×128, f=4) and hands measured VRAM/throughput to
> `compute-budget`; the real-data train + recon gate + ceiling check moved downstream to
> `runpod-execution`. These 8 use **local VAE-N numbering** (matching the live `VAE 1…8`
> subtasks) under the `10-vae-NN` block, so items 15–32 keep their global order untouched.
- [ ] [VAE 1 · SOTA Architecture Selection](10-vae-01-architecture-selection.md) — ▶ `/augment`
- [ ] [VAE 2 · Architecture Sanity Checks](10-vae-02-architecture-sanity-checks.md) — custom (no skill)
- [ ] [VAE 3 · Local Env Setup + launch.json](10-vae-03-local-env-setup.md) — ▶ `/update-config` ⚠️
- [ ] [VAE 4 · VAE Implementation (testable)](10-vae-04-vae-implementation.md) — custom (no skill)
- [ ] [VAE 5 · Profiling Code + Monitoring Scripts](10-vae-05-profiling-and-monitoring.md) — ▶ `/experiment-planner`
- [ ] [VAE 6 · Budget Calculator](10-vae-06-budget-calculator.md) — custom (no skill)
- [ ] [VAE 7 · Overfit Sanity Run (first pass)](10-vae-07-overfit-sanity-run.md) — custom (no skill)
- [ ] [VAE 8 · Noise Training Run + Hand-off](10-vae-08-noise-training-run.md) — custom (no skill)

## 🎲 Single-Disease LDM  *(scope task)*
- [ ] [15 · LDM Overfit Sanity](15-ldm-overfit-sanity.md)
- [ ] [16 · Latent Prep & Condition Set](16-latent-prep-conditions.md)
- [ ] [17 · Train Conditional LDM (CFG dropout)](17-train-ldm-cfg.md)
- [ ] [18 · Single-Disease FID Gate](18-fid-eval.md)

## 📏 Metrics & Extractors  *(scope task)*
- [ ] [19 · Two-Sample (C2ST) + MMD](19-two-sample-mmd.md)
- [ ] [20 · FID](20-fid.md)
- [ ] [21 · The Floor](21-floor.md)
- [ ] [22 · Heart-Size & Blunting Extractors](22-extractors.md)
- [ ] [23 · Extractor Validation](23-extractor-validation.md)

## 🧪 Composition Experiments  *(scope task · headline)*
- [ ] [24 · PoE Composition](24-poe-implementation.md)
- [ ] [25 · Marginals Check (Exp5 / H1 gate)](25-marginals-check.md)
- [ ] [26 · Joint-Structure Test (Exp6)](26-joint-structure-test.md)
- [ ] [27 · Reweighting Control (Exp7 / H3)](27-reweighting-control.md)
- [ ] [28 · Floor Baseline (Exp8)](28-floor-baseline.md)

## 📝 Documentation  *(scope task · cross-cutting)*
- [ ] [29 · Progress Log](29-progress-log.md)
- [ ] [30 · Figure & Metric Register](30-figure-metric-register.md)
- [ ] [31 · Pre-Registration Outcomes](31-prereg-outcomes.md)
- [ ] [32 · Methods & Results Draft](32-methods-results-draft.md)
