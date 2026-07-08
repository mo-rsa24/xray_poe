# PoE Composition — Probe, Explain, and (Maybe) Improve Co-Presence

## Scope status (created 2026-07-08)
Spun out of the Act-9 co-presence null via the three-augment triage (`1a/1b/1c-augment.md`, triage compile
2026-07-08). **Infra-separate from the closed vae-v2 science.** The *explain-the-null* work (O2/O3) feeds
**Track A's paper** (`plans/03-figure-table-paper-ready.md`); the *improve* work (O4/O5) is **Track-B**
investigation. Model discipline: explain-the-null runs on **arm1 (`gtr6kqpk`, the shipped null, yield 0.125)**;
operator/improve runs on **v4-60k (the vae-v2 live model, yield 2.6%)**; every plan names its model.

## Mission
The PoE operator (additive classifier-free guidance sampled ancestrally) does not compose cardiomegaly and
effusion into one X-ray — a robust, dose-response-flat null. This scope does three things with that null:
(a) **localize its cause** among codec / data / composition operator / sampler / target support, with cheap
instruments run in a fixed order so each result narrows blame before the next is paid for; (b) **turn it into
the paper's lead section**, a pre-registered composition investigation carried by a toy positive control and
paired quantitative + qualitative (overlay) evidence; (c) **test whether any frozen, no-retrain operator
recovers it**, gated on the diagnostics so no combiner is run once the ingredient is shown empty. A clean,
localized null is a full success here; an inference-time recovery is the upside.

## Objectives
1. **O1 — Diagnose (cheap, first).** Localize the bottleneck with four independent probes run in order:
   training co-occurrence base-rate; k-NN coverage/density on the real comorbid manifold; cardio-alone ceiling
   probe; pixels→z→h separability/CAV probe with a permutation null. Each result narrows blame before the next.
2. **O2 — Explain-the-null (paper-critical, arm1).** Build the paper's lead section: a toy 2×2 positive control
   (ancestral-CFG fails / corrected sampler recovers a known product), the qualitative figure pack (PSPNet CTR +
   costophrenic-angle overlays and GradCAM callouts pinned to the CTR×CP scatter clusters, each with a reading
   and a competing reading), and the framing (flat-sweep-is-multiplicative, four-locus subtraction, pre-registered
   two-abstract fork). Feeds `plans/03-figure-table-paper-ready.md`.
3. **O3 — Adjudicate operator vs sampler (paper-critical, v4-60k).** Run a Metropolis/ULA corrected sampler on
   the frozen CXR PoE, read asymmetrically (ε-prediction admits only the unadjusted sampler, or an approximate
   energy `f=−½‖s‖²`), to separate "the additive operator is wrong" from "the ancestral sampler is wrong."
4. **O4 — Improve (Track-B; GATED on O1 ceiling + coverage).** If there is something to combine, try frozen
   no-retrain operators in cost order: clean-unconditional-ε anchor first, then SuperDiff AND, then a best-of-N
   control, scored on co-presence yield-per-NFE on a shared scorecard.
5. **O5 — Utility (GATED on O1 coverage; confound-flagged).** Rejection-sample both-disease composites, augment a
   real comorbid classifier, and report the delta honestly — with the Exp-6 test-set saturation confound stated,
   so a null is not over-read.

## Goals
1. **Bottleneck localized** — base-rate measured; coverage/density reported in an embedding (never pixels);
   cardio-alone ceiling CTR reported; separability probe reports per-stage accuracy + CAV-cosine + channel-mass +
   a permutation-null p-value; blame attributed to exactly one of {codec, data, operator, sampler, support}, or
   the residual named.
2. **Toy control passes** — the 2×2 shows ancestral-CFG collapses to the axes ("A or B") while the corrected
   sampler recovers the known top-right product; the CXR punchline cell has a calibrated reference.
3. **Paper section evidenced** — the CTR×CP scatter carries ≥ 6 overlay callouts (abstain pile + "big heart, no
   fluid" corner), each with a falsifiable reading; the discussion reads as a subtraction, not a menu.
4. **Sampler adjudicated** — corrected-sampler yield on CXR reported vs the ancestral baseline with chain
   diagnostics; the pre-registered fork resolves to Abstract-N (support-limited) or Abstract-M (sampler-fixable).
5. **Improve arms resolved (if gate fires)** — clean-ε / SuperDiff / best-of-N each report yield-per-NFE on the
   shared scorecard (co-presence yield, cardio CTR, effusion-must-hold, artifact guard, NFE); verdict recorded as
   **recoverable-for-free** vs **operator-limited** vs **manifold/support-limited**.
6. **Utility reported (if gate fires)** — comorbid classifier ΔAUC + sensitivity vs starved, with the saturation
   confound stated and the harvest cost (composites kept / total generated) logged.

## Expected Outcome
A pre-registered composition investigation that ends in one of three honest, publishable states: the null is
**support/manifold-limited** (no frozen operator can help — the strongest mechanistic result), **operator- or
sampler-limited** (a corrected sampler or SuperDiff recovers co-presence for free — the upside), or **codec-
limited** (the separability probe demands a VAE re-disentangle, which is parked). Whichever it is, the paper's
lead section is carried by a toy positive control and paired quantitative + qualitative overlay evidence, and the
bottleneck is attributed to exactly one locus rather than left as a menu of maybes.

## Definition of Done
1. **Bottleneck localized** among codec / data / operator / sampler / support, each probe reporting its number
   with a significance/uncertainty band (permutation null, Wilson band, or coverage fraction).
2. **Null explained** — the paper's lead section exists with the toy 2×2 positive control and the qualitative
   figure pack, filed to `plans/03-figure-table-paper-ready.md`; the pre-registered fork is resolved.
3. **Improve arms resolved** — the O4 operators report on the shared scorecard and land a verdict
   (recoverable-for-free / operator-limited / manifold-limited), OR are formally skipped because O1's gate showed
   nothing to combine (that skip is itself recorded as the finding).
4. **Utility reported or gated-out** — O5 reports the comorbid-augmentation delta with the confound stated, OR is
   gated out by near-zero coverage (recorded).
5. **Parked fallbacks stay parked** unless their firing condition trips (concept-algebra, HMC, VAE re-disentangle,
   beam-search — see `PARKING_LOT.md`).

## Sub-Scopes
(none)

## Plans
- ⚠️ **01-diagnose.md** — localize the bottleneck (base-rate, coverage/density, cardio-alone ceiling, separability/CAV probe); the GATE for O4/O5 (O1).
- ⚠️ **02-explain-null-paper.md** — the paper's lead section: toy 2×2 control + qualitative overlay figure pack + framing; feeds Track A (O2, run-regardless).
- ⚠️ **03-adjudicate-sampler.md** — corrected (Metropolis/ULA) sampler on frozen v4-60k; resolves the pre-registered fork (O3, run-regardless).
- ⚠️ **04-improve-operators.md** — clean-ε → SuperDiff AND → best-of-N on yield-per-NFE (O4, GATED on O1 ceiling+coverage).
- ⚠️ **05-comorbid-utility.md** — harvest composites → augment a real comorbid classifier → honest ΔAUC (O5, GATED on O1 coverage; Exp-6 confound flagged).
