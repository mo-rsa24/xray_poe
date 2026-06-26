<div align="center">

# 🩺 Supervisor Reconciliation — Paper Structure vs. Our Experimental Plan

`reconciled: 2026-06-26` · `artifact: EXPERIMENTS.md + MASTER_PLAN.md + plans/*` · `intent: supervisor-meeting "Paper Structure" doc` · `next: /skill-route`

</div>

> Produced by `/reconcile`. This is the hand-off artifact for `/skill-route`: every supervisor item → covered / new / deferred, each new deliverable with its script + checkpoint + acceptance criteria. When every task derived from this file is checked off, we have provably fulfilled the supervisors' expectations with paper-ready figures, trained checkpoints, and figure-emitting code.

---

## 1 · The reconciled paper shape

**What the paper should do.** A sober, mechanistic-exposure paper on the proxy (cardiomegaly + effusion, NIH ChestX-ray14). In order: (1) state what the classifier is and does; (2) show individual single-disease synthetics are faithful enough to classify — *before* any composition; (3) show real comorbidity is hard for the classifier and that synthetic comorbidity, by GradCAM + CTR + downstream metrics, **is not there yet**; (4) test whether augmenting a starved classifier with synthetic data (individual, then compositional) helps, measured honestly against an oracle ceiling. The lead is the **exposure of the comorbidity gap**, not a utility win. The utility result is reported either way — a gain is the upside case, a null is a clean, pre-registered, equally-publishable contribution (Devon's framing, §6 of the supervisor doc).

**What it should not do.**
- Not lead with "PoE beats overlay" — that becomes one result inside the sober frame, not the thesis.
- Not treat FID as a headline gate — appendix sanity check only.
- Not execute the monolithic joint-condition diffusion baseline now — specify + motivate it, defer execution.
- Not report only accuracy or only AUC — report accuracy + sensitivity + specificity **and** AUC-by-stratum.
- Not silently drop entangled / border-filtered films — quantify what was excluded and probe it.
- Not blur the two torchxrayvision models (DenseNet classifier-under-test vs PSPNet segmentation instrument).
- Not add a train-on-synthetic-only (TSTR) arm — out, to avoid over-engineering (revisit only on request).

**Figure principle (load-bearing, carried into `plan-figures`/`design-figure`).** One claim per figure; before/after side-by-side where it shows the retrain effect; readable at a glance; no dense multi-panels beyond the supervisor Fig-1 4×4 (kept tidy — small thumbnails, one saliency map each).

**Where each perspective landed.**
- *What exists* — the seven-experiment gate chain survives almost intact; Exps 1–4 (done) and 5 (retrain, running) carry forward unchanged. What changed is **role and emphasis**, not the experiments.
- *Supervisor intent* — adopted as the lead: 7.11 flow, outcome-independent narrative, sens/spec, individual-before-compositional, quantified cleaning-bias caveat, their Fig 1–3 / Table 1–2 as the deliverable set.
- *My added calls* — oracle **retained** (makes a null interpretable), FID **demoted**, individual-disease validation **elevated to the body-or-no-body gate**, every synthetic figure rendered **before/after** (pre-retrain + rebalanced checkpoint).
- *Synthesis* — same machinery, re-roled under a sober lead; 2 new work items (monolithic = deferred, sens/spec = cheap) + ~5 new figure/table harnesses + a narrative reframe; the rest already covered.

---

## 2 · The classifier (grounded in `scripts/finetune_classifier.py`)

The **classifier under test** is a torchxrayvision **DenseNet-121** (`densenet121-res224-nih`) with `denseblock4 + norm5` unfrozen and a fresh **MLP head → 2 logits `[cardio, effusion]`** (sigmoid, multi-label, `BCEWithLogitsLoss`). Input: grayscale → xrv-normalize → center-crop → **224×224**. Output: two independent presence probabilities + a saved Youden threshold. Trained on `images_normal [0,0]`, `images_cardio_only [1,0]`, `images_effusion_only [0,1]`; **both-disease images held out, never in training** — which is *why* it is expected to struggle on comorbidity (it has never seen the joint case). That weakness is the paper's "we're not there yet" engine, not a bug.

**Two torchxrayvision models, kept separate:** the DenseNet above is *judged*; the **PSPNet segmentation** (`ctr_overlay.py` / `eval_checkpoint.py`) is the *instrument* that judges (CTR/CP geometry). Never blur them in the methods (D1).

**The classification flow (supervisor 7.11, corrected staging):**
- **A — face validity, NO training (Table 1, Fig 1):** evaluate the existing classifier on **real** vs **individual synthetics** (cardio-only, effusion-only). Does a synthetic cardio film read as cardio?
- **B — composition detection:** run the same classifier on **PoE comorbid** synthetics. Can it see both at once?
- **C — augmentation/utility (Table 2, Fig 3):** *now* train — fine-tune the starved classifier with synthetic cardio, then with synthetic PoE comorbid; vs overlay and oracle; **test on real only.**

---

## 3 · Before/after — both checkpoints (the retrain is a visible fix)

Render every synthetic figure on **both** the pre-retrain checkpoint (100k / `model_step0040000`) **and** the rebalanced-cardio checkpoint. This turns the retrain into a before/after that *visualizes the fix*. It also **unblocks the whole synthetic track now**: render on the before-model today, refresh on the after-model when it lands. Same harness, run twice.

**Contamination caveat (label, don't hide):** the pre-retrain model was **test-patient-contaminated** (the reason Exp 5 retrains from scratch).
- **Problem-exposure figures** (CTR scatter, GradCAM-on-synth, individual-disease validation) — before-model is *ideal*: it under-renders cardio *despite* seeing test patients → a stronger "generator can't do cardio yet" statement.
- **Utility numbers** (Table 2 augmentation) — a classifier trained on before-model synthetics + tested on the frozen real set inherits a **leakage asterisk**. Label before-model utility as *"pre-retrain (contaminated) — illustrative of the gap, not clean utility evidence."* The clean utility number comes only from the after-model.

---

## 4 · The gap map (every supervisor item → disposition)

**Critical-path note:** doc/spec/real-data work (D1, D5-real, D8, D9, D10-data, D11) runs now. Synthetic-classification deliverables (D2, D3, D4, D5-synth, D6, D7) run **now on the before-model** and **refresh on the after-model** when the retrain lands.

| ID | Supervisor item | Disposition | Script | Checkpoint | Acceptance criteria |
|---|---|---|---|---|---|
| **D1** | §0 questions | 🆕 doc | new `docs/SYSTEM_DESCRIPTION.md` | none | Classifier I/O stated (224² input; 2 sigmoid logits [cardio,effusion]); full model inventory with reuse/retrain status; what each model outputs; **DenseNet-classifier vs PSPNet-instrument disambiguated**; held-out-both-disease training choice stated. Supervisor can answer all three §0 questions from it. |
| **D2** | 1.1 / 7.2 / 7.7 — individual-disease validation gate | 🔴 new harness | new `scripts/validate_individual_synthetics.py` (wraps `generate.py` + classifier) | before + after | Synth-only cardio & synth-only effusion each run through classifier; accuracy/sens/spec vs real single-disease baseline; explicit pass/fail per disease. **Gate:** if individual synthetics fail, paper reports "diffusion-not-composition." |
| **D3** | Fig 1 — GradCAM 4-col × (2 correct+2 incorrect), real vs synth | 🟡 new panel | extend `scripts/grad_cam.py` → new `scripts/fig1_gradcam_panel.py` | before + after (synth col) | Normal/Effusion/Cardio/Comorbid × {2 correct, 2 incorrect}, real & synth side-by-side; corner-artifact-fixed `grad_cam_utils.py`; saliency on correct anatomy for correct cases. Tidy thumbnails. |
| **D4** | Table 1 — pretrained classifier accuracy real vs synth | 🔴 new | new `metrics/classifier_report.py` (shared metric module) | before + after (synth half) | **Eval-only, no training.** Accuracy + sensitivity + specificity per category (normal/cardio/effusion/comorbid), real vs synth columns, bootstrap CIs. |
| **D5** | Fig 2 — CTR scatter heart-width(Y) vs lung-width(X), real vs synth | 🟡 new plot | new `scripts/fig2_ctr_scatter.py` (consumes `ctr_overlay.py`) | real now; synth before+after | Scatter, real vs synth color-coded, CTR=0.5 iso-line; synth tracks real on single-disease, diverges on comorbid (the "not there yet" visual). |
| **D6** | §4 + Fig 3 — fine-tuned GradCAM | 🟢 reframed | reuse D3 harness on fine-tuned classifier | after + Exp 7 arms | Same panel as Fig 1 on +PoE fine-tuned model; comparison to Fig 1 visible. |
| **D7** | Table 2 — post-fine-tune accuracy/sens/spec | 🟢 reframed | reuse D4 `classifier_report.py` | after + Exp 7 arms | Same metric table after fine-tuning; ΔAUC + Δsens/spec on comorbid stratum, bootstrap CIs; supports the gain or the clean null. |
| **D8** | 7.9 — data-cleaning bias → **STRATIFICATION** (Devon 2026-06-26, narrative pt 3) | 🟡 new check | new `scripts/excluded_set_audit.py` (uses Exp 1 logged-not-deleted entangled set) | none | Counts excluded by `border_visible`/entangled filter (N, % of both-labelled); classifier run on excluded set. **Reframe: this is data *stratification* (selecting the single-PA-diagnosable subset), NOT bias correction.** The label-vs-pixel mismatch is **information asymmetry** — radiologists labelled from AP+PA + longitudinal + clinical history, absent in our single PA image; excluded films carry valid but out-of-modality information, not errors. Report as stratification, not apology. See `docs/SUPERVISOR_NARRATIVE.md` pt 3 (also defends the CTR instrument: real labelled-cardio sits at median CTR≈0.53 by design). |
| **D9** | 7.5 / 7.10 / 7.1 / 7.3 — sens/spec + leakage rigor | 🟢 covered+extend | `classifier_report.py` (D4) + existing `cross_leakage_check.py` / `filter_test_patients.py` | none | Sens/spec everywhere; explicit no-leak statement (patient-disjoint, test-on-real-only) in methods + verified. |
| **D10** | Exp 6 — oracle (retained) | 🟢 leaf plan 06 | existing oracle plan | real comorbid | oracle − starved ΔAUC on both-separable reported; establishes headroom so the null is interpretable. **Role upgraded under sober lead.** |
| **D11** | §5 + §6 + headline reframe + FID demote | ✅ **DONE 2026-06-26** | `EXPERIMENTS.md` §0/§3/§1-gates/§5a/§7, `MASTER_PLAN.md` mission | none | ✅ Thesis → mechanistic-exposure lead; Exp 7 re-roled (null accepted, §5a); FID demoted (Gate 4 + appendix); outcome-independent §5a written; LoRA + deferred-monolithic added to §7; sens/spec added to §3 + Gate 6. |
| **D12** | 1.2 / 1.3 / 7.8 — monolithic joint-condition diffusion baseline | 🔵 **deferred** | new stub plan, not executed | (proxy comorbid data, if run) | Fully-specified deferred task + one motivating paragraph answering "why not monolithic?". Executed only if time allows. |
| **D13** | before/after comparison framing | 🆕 cross-cutting | applies to D2–D7 | before + after | Every synthetic figure rendered as a before/after pair; before-model utility numbers carry the contamination label. |

**Scope.** In: D1–D11, D13. Deferred (specified, not run): D12. Out: silicoTB execution (future-work prose), VAE retrain (Exp 2 ruled reusable), TSTR arm, new gate machinery beyond demoting FID.

**Plan-tree shape for `/skill-route`.** Now-runnable track (D1, D5-real, D8, D9, D11) + synthetic track (D2, D3, D4, D5-synth, D6, D7, D13, runnable on before-model now). Retained/deferred leaves: D10 = existing plan 06; D12 = new deferred stub. Existing leaf plans 05/06/07 get **extended**, not replaced — D2/D3/D4/D5 are new plan(s); D6/D7 fold into plan 07; D10 is plan 06.

**Known unknowns.**
1. Retrain landing date sets the after-model refresh of the synthetic track (before-model results available immediately).
2. If D2 fails post-retrain (individual cardio synthetics still don't classify), the paper pivots fully to "diffusion-quality-not-composition"; §4 results become confirmatory of the null. D11 already accommodates this; figure emphasis shifts.
3. Oracle data volume (D10) bounded by NIH both-labelled count (~1063, ~72% PA-separable); thin headroom weakens the null's interpretability — check N before reading.
