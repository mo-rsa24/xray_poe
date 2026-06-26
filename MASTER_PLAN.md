# Paper 3 — PoE Composition for Pixel-Verifiable Co-Morbid X-Rays

## Mission
Investigate, on a proxy where every claim is pixel-verifiable (cardiomegaly + effusion on NIH ChestX-ray14), whether composing two single-disease diffusion experts via Product-of-Experts can synthesize co-morbid chest X-rays from *no* co-morbid training data that are faithful enough to help a data-starved classifier — **and, where it breaks, expose exactly where.** The paper **leads with mechanistic exposure** (validate individual-disease synthetics → compositional → measure downstream utility against an oracle ceiling), not with a utility win. A gain is the upside case; a clean null — *"composition in latent space does not translate to composition in disease space"* — is an equally informative, pre-registered contribution (supervisor §6). Utility is measured by **accuracy + sensitivity + specificity *and* stratified AUC**; FID is demoted to an appendix sanity check; every synthetic result is reported **before/after** the rebalanced-cardio retrain. The whole structure remains a pre-registered trust chain (Gates 0–6). silicoTB and LoRA composition are named future work — **not** run here; the **monolithic joint-condition baseline IS run** as a comparison that justifies (or unseats) composition (scope set in `docs/SUPERVISOR_RECONCILE.md`).

## Objectives
1. Establish the pixel-verifiable study population — the separable-both subset of NIH where both signs are independently readable (Gate 0).
2. Build clean single-disease experts on deconfounded splits, so composition is the only place the two diseases meet.
3. Show the codec can *hold* and the experts can *render* the separable-both appearance — ruling out the VAE as a scapegoat and quantifying PoE yield (Gate 3).
4. Commit, before spending the most expensive compute, to a reuse-vs-retrain rule for the VAE/LDM checkpoints.
5. Build the oracle ceiling so the headline comparison is interpretable (Gate 6 headroom).
6. Run the headline: PoE- vs overlay- vs starved-augmented classifiers (and the oracle), stratified AUC on the pixel-verifiable set.
7. Hold the protocol discipline throughout — the symmetry rule (gates filter synthetic *and* real), the independence rule (CTR never both teaches and validates), and the pre-registration table (no post-hoc reinterpretation).

## Goals
1. **Population exists** — separable fraction ≥ 30% of NIH both-labelled films; instrument validated ≥ 90% against ~30 hand labels; the separable set frozen as the pixel-verifiable test set with N reported.
2. **Clean experts** — deconfounded splits (`cardio ∧ ¬effusion`, `effusion ∧ ¬cardio`) built and sized; cross-leakage ≤ real single-disease base-rate + margin.
3. **Codec ceiling high** — on separable-both round-trip, |ΔCTR| ≤ 0.02 and CP class preserved ≥ 95%, local LPIPS below the single-disease control.
4. **Yield sufficient** — PoE Gate-3 pass-rate ≥ 30% (or the Exp 5 retrain recovers it from < 10%).
5. **Reuse/retrain decided** — the decision recorded against the pre-committed rule (reuse iff ceiling high **and** yield ≥ 30%), logging which gate forced any retrain.
6. **Oracle headroom** — oracle − starved ΔAUC ≥ 0.05 on both-separable, so the test set has room to show utility.
7. **Headline resolved** — ΔAUC(PoE − overlay) ≥ 0.03 **and** ΔAUC(PoE − starved) ≥ 0.05 on both-separable with disjoint 95% bootstrap CIs and a *lumpy* gain profile — **or** a clean, reported null (PoE ≤ overlay).

## Expected Outcome
A pre-registered, gate-disciplined result on the proxy: a frozen pixel-verifiable test set, two deconfounded single-disease experts, a documented reuse-or-retrain decision, an oracle ceiling, and a four-arm stratified-AUC figure that either shows **+PoE > +overlay > starved concentrated on both-separable (and closer to the oracle than overlay)**, or cleanly refutes it. Every experiment is filed against the pre-registration table, every synthetic image that entered augmentation passed the identical Gate 0–5 filter, and silicoTB is framed honestly as future work whose transfer risk (no CTR-equivalent instrument) is named, not hidden.

## Definition of Done
1. ✅ **Exp 1** — separable/entangled counts over all NIH both-labelled films (N total / PA / AP / separable / entangled + fraction + CI); instrument validated ≥ 90% on ~30 hand labels; separable set frozen, entangled logged-not-deleted.  *(done; instrument validated via researcher visual sign-off in lieu of the 30-film numeric table — see plan Carry-forward deviation note.)*
2. ✅ **Exp 4** — deconfounded single-disease splits constructed with N reported; cross-leakage check on ≥ 1000 expert samples each, at/below base-rate.  *(done; cross-leakage floor-corrected SUPPORT @100k. NB: cardio expert under-renders cardiomegaly [4.2% vs 49.1% real] — a fidelity issue feeding Exp 3/5, not a leakage failure.)*
3. ✅ **Exp 2** — localized VAE recon fidelity at heart-border + CP-angle on the frozen separable set + matched control; |ΔCTR| and CP-flip-rate reported; ceiling verdict recorded (feeds Exp 5).  *(done; boundary-refined ceiling INCONCLUSIVE-but-not-low — VAE faithful [|ΔCTR| 0.011, LPIPS ≈ control], reusable, NOT a retrain trigger; carry abstain margin to Exp 3/7.)*
4. ✅ **Exp 3** — ≥ 2000 PoE samples per setting; Gate-3 yield with CI + CTR×CP scatter; yield verdict recorded (feeds Exp 5).  *(done; N=2000 → yield NULL [floor-corrected 9.3% / confident 9.1% < 10%, raw 13.2%]. PoE renders cardio 25.4% vs 4.2% alone but barely clears CTR=0.5 → fires the Exp 5 cardio retrain.)*
5. **Exp 5** — reuse-vs-retrain decision written down against the pre-committed rule; if retrain fired, the rebalanced-cardio LDM trained on hippo behind the `df` disk guard, W&B-logged.
6. ✅ **Exp 6** — oracle classifier trained on abundant real both-disease (≥ 3 seeds, bootstrap CIs); oracle − starved gap on both-separable reported; headroom verdict recorded.  *(done 2026-06-26; verdict = **NULL**, oracle−starved ΔAUC = −0.012 on both-separable, CIs < 0 → test set SATURATED. Blocks Exp 7 until the test set/stratum is revisited. Detector pointing-game 0.23 FAIL — cardio detector doesn't attend to the heart.)*
7. **Exp 7** — Starved / +Overlay / +PoE / Oracle arms trained (≥ 3–5 seeds), every synthetic image gated through the identical Gate 0–5 filter; AUC-by-stratum figure produced; headline rule evaluated (support / null / inconclusive) and filed.
8. **Protocol** — every experiment's outcome filed against the §5 pre-registration table; no post-hoc reinterpretation; W&B run-naming + gate/checkpoint tags applied throughout.
9. ⚠️ **Supervisor deliverables (D1–D13, `docs/SUPERVISOR_RECONCILE.md`)** — system description answering §0 (D1); the individual-disease body-or-no-body gate evaluated (D2); Fig 1 GradCAM panel (D3), Table 1 classifier report (D4), Fig 2 CTR scatter (D5), Fig 3 fine-tuned GradCAM (D6), Table 2 post-fine-tune (D7) — each a **before/after** pair with before-model utility numbers labelled contaminated (D13); data-cleaning bias quantified (D8); sens/spec + no-leak rigor stated and computed (D9); oracle role-upgraded as the null-interpretability headroom-guard (D10); headline reframe done (D11 ✅); **monolithic joint-condition baseline trained + compared to PoE on both-separable (D12 — promoted from deferred 2026-06-26)**. Plus the reconcile-2 additions: **Exp 7 carries a 5th +Individual-synthetic arm** (composition control; ΔAUC(PoE−Individual) reported, non-gating) and **Table 1 carries the real cardio-alone-vs-cardio+effusion breakdown (7.6)**.

## Sub-Scopes
- ⚠️ plans/rebalanced-cardio-retrain/ — FIRED (Exp 5 = retrain) and **RUNNING** 2026-06-25: from-scratch conditional LDM on border-visible deconfounded latents; config-rebalance + split-wiring ✅, hippo-train live, yield-recovery pending checkpoint.
  - **First signal (≈step 34k):** rebalance is WORKING — gen-cardio CTR median crossed **0.5** for the first time (0.468→0.502), `frac>0.5/cardio` ~0.52, `border_frac/cardio` ~0.50. Still climbing toward target (real 0.535); not plateaued.
  - **Crashes this session were tooling, not the model** — a buggy divergence guard (false-kill) then an eval-hook GPU OOM; the model was healthy at every crash. See EXPERIMENTS.md §6 and [[rebalanced-retrain-operational]].
  - **New apparatus:** an in-training generation-quality dashboard (`metrics/eval_checkpoint.py`) logs live CTR/border/yield/presence curves + overlay panels to the training W&B run.

## Plans
(Authored next by `populate-plans` — seven leaf plans in dependency order `1 → 4 → 2 → 3 → 5 → 6 → 7`.)
- ✅ 01-dynamics-population.md — Exp 1, Gate 0: count separable vs entangled, freeze the test set (separable PA 72%, verdict=support; patient-leakage guard carried to Exp 4 + retrain)
- ✅ 02-deconfounded-splits.md — Exp 4: clean single-disease expert training data (splits built + patient-disjoint + materialized; cross-leakage floor-corrected SUPPORT; cardio under-renders → reuse effusion/retrain cardio)
- ✅ 03-vae-ceiling.md — Exp 2: localized recon fidelity at heart-border + CP-angle (ceiling boundary-refined INCONCLUSIVE-not-low; VAE faithful & reusable; abstain margin → Exp 3/7)
- ✅ 04-poe-yield.md — Exp 3, Gate 3: PoE co-presence pass-rate (yield NULL — floor-corrected 9.3% <10%; cardio channel the bottleneck → fires Exp 5 retrain)
- ⚠️ 05-reuse-vs-retrain.md — Exp 5: the VAE/LDM decision gate
- ✅ 06-oracle-baseline.md — Exp 6, Gate 6: the ceiling on the proxy (D10 role-upgraded — the headroom-guard that makes an Exp 7 null interpretable) — **DONE: NULL headroom (ΔAUC −0.012), test set saturated, blocks Exp 7 pending test-set revisit**
- ⚠️ 07-headline-auc.md — Exp 7, Gate 6: ⭐ overlay-vs-PoE stratified downstream AUC (+D6 Fig 3 fine-tuned GradCAM, +D7 Table 2 post-fine-tune, +D13 before/after framing; **5-arm** Starved/+Individual/+Overlay/+PoE/Oracle, ΔAUC(PoE−Individual) diagnostic)

### Supervisor-reconcile additions (D1–D13 · 2026-06-26, gap map `docs/SUPERVISOR_RECONCILE.md`; dependency order D1→D2→(D3,D4,D5)→D8→(D6,D7 in 07)→D10 in 06→D12 deferred)
- ✅ 08-methods-system-description.md — D1 (+D9 no-leak/sens-spec statement): `docs/SYSTEM_DESCRIPTION.md` — classifier I/O, model inventory, DenseNet-vs-PSPNet disambiguation; answers §0
- ✅ 09-individual-disease-validation.md — **D2, the body-or-no-body GATE**: `scripts/validate_individual_synthetics.py`, runnable NOW on the before-model; gates whether the paper has a synthetic body
- ⚠️ 10-figure1-gradcam-panel.md — D3 (Fig 1): `scripts/fig1_gradcam_panel.py` — real-vs-synth GradCAM, before/after
- ⚠️ 11-tables-classifier-report.md — D4 + D9 (Table 1 + shared metric module): `metrics/classifier_report.py` — accuracy/sens/spec, eval-only, before/after
- ⚠️ 12-figure2-ctr-scatter.md — D5 (Fig 2): `scripts/fig2_ctr_scatter.py` — CTR scatter real-vs-synth, real now / synth before/after
- ⚠️ 13-data-cleaning-bias-audit.md — D8 (7.9): `scripts/excluded_set_audit.py` — quantify the entangled/border-filtered exclusion as a limitation
- ⚠️ 14-monolithic-baseline-deferred.md — D12: **monolithic joint-condition baseline — ACTIVE** (promoted 2026-06-26): build comorbid set → train joint-condition diffusion (hippo) → compare vs +PoE on both-separable (filename keeps legacy `-deferred` slug for ref stability)
