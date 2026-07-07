# Paper 3 — PoE Composition for Pixel-Verifiable Co-Morbid X-Rays

## Scope status (frozen 2026-07-02)
The work is now two parallel tracks.

- **Track A — this paper (arm1).** The cardio expert is **frozen**: run `gtr6kqpk` (`ckpts_arm1_cleancardio/model_step0100000`, arm1 clean-cardio @100k) is the **final** cardio LDM. **No more cardio retraining in this scope — analysis only.** Arm1 converged but plateaued *at* the diagnostic line (cardio CTR 0.499 vs real 0.535; PoE Gate-3 yield 0.125). That plateau **is** the result: the paper **leads with the mechanistic null** — *composition in latent space does not translate to composition in disease space* — and shows exactly where the cardio channel breaks. Exp 7 (downstream AUC) is **demoted to optional/appendix** because Exp 6 came back saturated (oracle−starved ΔAUC −0.012). The monolithic baseline (D12) still trains — it is the only new training this track allows.
- **Track B — improvement initiative (parallel, non-blocking).** `additional_plans/` (vae-v2 + vindr-transfer, spatial-conditioning gated) is one combined effort to mix VinDr deeper and lift arm1's results. It is **separate work, not a dependency of Track A** — the arm1 paper ships on the frozen `gtr6kqpk` regardless. See `additional_plans/INITIATIVE.md`.

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
5. ✅ **Exp 5** — decided. The rule fired **retrain(cardio)**; the retrain ran (arm1 clean-cardio, from scratch, run `gtr6kqpk`, 100k, W&B-logged). Outcome **accepted as final**: arm1 recovered cardio partway (CTR crossed 0.5 at 50k, cross-leakage halved) but **plateaued at the line**, so its yield (0.125) is reported as the mechanistic finding, not a recovery. `ckpts_arm1_cleancardio/model_step0100000` is the released cardio checkpoint. **No further cardio retrain in this scope.** Full verdict: `results/section4/analysis_gtr6kqpk_cardio_retrain.md`.
6. ✅ **Exp 6** — oracle classifier trained on abundant real both-disease (≥ 3 seeds, bootstrap CIs); oracle − starved gap on both-separable reported; headroom verdict recorded.  *(done 2026-06-26; verdict = **NULL**, oracle−starved ΔAUC = −0.012 on both-separable, CIs < 0 → test set SATURATED. Blocks Exp 7 until the test set/stratum is revisited. Detector pointing-game 0.23 FAIL — cardio detector doesn't attend to the heart.)*
7. ⏸️ **Exp 7 — DEMOTED to optional/appendix.** Exp 6 came back saturated (oracle−starved ΔAUC −0.012), so the downstream-AUC headline has no headroom to prove and is **not** the paper's lead. The paper leads with the mechanistic null instead. If run at all, the 5-arm AUC (Starved / +Individual / +Overlay / +PoE / Oracle, arm1 as the synthetic source) is an appendix sanity check, not the result. Original spec retained in `plans/07-headline-auc.md`.
8. **Protocol** — every experiment's outcome filed against the §5 pre-registration table; no post-hoc reinterpretation; W&B run-naming + gate/checkpoint tags applied throughout.
9. ⚠️ **Supervisor deliverables (D1–D13, `docs/SUPERVISOR_RECONCILE.md`)** — system description answering §0 (D1); the individual-disease body-or-no-body gate evaluated (D2); Fig 1 GradCAM panel (D3), Table 1 classifier report (D4), Fig 2 CTR scatter (D5), Fig 3 fine-tuned GradCAM (D6), Table 2 post-fine-tune (D7) — each a **before/after** pair with before-model utility numbers labelled contaminated (D13); data-cleaning bias quantified (D8); sens/spec + no-leak rigor stated and computed (D9); oracle role-upgraded as the null-interpretability headroom-guard (D10); headline reframe done (D11 ✅); **monolithic joint-condition baseline trained + compared to PoE on both-separable (D12 — promoted from deferred 2026-06-26)**. Plus the reconcile-2 additions: **Exp 7 carries a 5th +Individual-synthetic arm** (composition control; ΔAUC(PoE−Individual) reported, non-gating) and **Table 1 carries the real cardio-alone-vs-cardio+effusion breakdown (7.6)**.

## Sub-Scopes
- ✅ plans/complete/rebalanced-cardio-retrain/ — **CLOSED 2026-07-02.** Retrain ran as Arm-1 (`gtr6kqpk`, `ckpts_arm1_cleancardio/model_step0100000`), the released final cardio checkpoint. Plateaued at the line (that plateau is the reported finding). Arms 2+ moved to Track B. Section-4 augmentation arms (aug_indiv / aug_poe, 3 seeds) train on Arm-1 → before/after = 40k-contaminated vs Arm-1.
- ⏭️ additional_plans/ — **Track B, parallel & non-blocking.** vae-v2 + vindr-transfer as one effort to mix VinDr deeper and lift Arm-1's cardio. Separate scope; the paper does not wait on it. Parent: `additional_plans/INITIATIVE.md`.
- ⚠️ plans/repo-hygiene/ — repo navigability + release readiness. Organization only, changes no number. See its master plan.

## Plans

**How to read this:** the runbook `plans/PAPER_ASSEMBLY.md` is the ordered set of steps that produces every paper figure/result/table. Finished plans are archived in `plans/complete/` (index there). What's left is below.

### ✅ Done — archived in `plans/complete/`
Experiments 01–06 (the Gate 0–6 trust chain), methods 08, individual-validation 09, Fig 2 harness 12, and the closed retrain sub-scope. Headlines: separable test set frozen; clean experts built; VAE reusable; PoE yield null → fired the cardio retrain; Arm-1 trained + accepted; oracle saturated → Exp 7 demoted. Full one-liners in `plans/complete/INDEX.md`.

### Remaining — active in `plans/` (renumbered 2026-07-02 in dependency order; run order = `plans/PAPER_ASSEMBLY.md`)
- ⚠️ **01-figure1-gradcam-panel.md** — Fig 1 GradCAM harness. Forward assembly → plan 03.
- ⚠️ **02-tables-classifier-report.md** — Table 1 metric module. One task left: the real cardio-alone-vs-cardio+effusion breakdown (D9 / 7.6).
- ⚠️ **03-figure-table-paper-ready.md** — the figure/table assembly owner (Fig 1/2/3 + Table 1/2, canonical paths, before/after audit, submission sync). Consumes 01/02.
- ⚠️ **04-figure-review.md** — judgement pass over all ~25 figures → verdict register + routed backlog.
- ⚠️ **05-data-cleaning-bias-audit.md** — D8. Quantify the excluded (entangled / border-filtered) set as a limitation.
- ⚠️ **06-monolithic-baseline-deferred.md** — D12 (active). Build comorbid set → train joint-condition baseline → compare vs +PoE.
- ⏸️ **07-headline-auc.md** — Exp 7, **demoted to optional appendix** (no headroom). 5-arm AUC spec retained if run at all.
