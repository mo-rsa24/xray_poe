# External-CXR-Transfer Cardio Lever

> **Dir name note:** this scope lives in `additional_plans/vindr-transfer/` (renamed 2026-06-27 from `padchest-transfer/`); the **primary base is VinDr-CXR**, with PadChest demoted to the headroom escalation. This is **Scope 1** ("is clean VinDr+NIH data enough for a vanilla conditional LDM?"); the conditioning follow-up (heart-mask + haze-map) is the sibling **Scope 2** (`additional_plans/spatial-conditioning/`), gated on this scope failing the yield gate.

> ## ⛔ TRANSFER APPROACH SUPERSEDED (2026-07-01)
> The **two-stage transfer** thesis of this scope (Stage-1 pretrain on VinDr → Stage-2
> low-LR NIH finetune, plans 05/06/08) is **retired**. It never ran (no Stage-1/Stage-2
> checkpoint). The cardio it aimed to inject is now delivered by the **vae-v2
> joint-from-scratch** LDM — one class-conditional model trained on the union of NIH +
> VinDr μ-latents in a single VAE-v2 codec (`additional_plans/vae-v2/`, plans 05–07). The
> downstream Gate-3 / Exp-09 eval moves to `vae-v2/plans/07-downstream-eval.md`.
>
> **What survives:** (1) the VinDr **data-prep** (plans 01–04, 09) — done, and its 6,671
> border-visible films feed the vae-v2 diet (`data/vae_v2_diet/train_manifest.csv`);
> (2) the two **sub-scopes** (opacity-instrument, composition-operators) as standalone,
> model-agnostic eval tooling. Todoist reconciled 2026-07-01: data-prep tasks completed,
> 05/06/08 + the Stage-2 composition subtask deleted, the two sub-scope tools kept as
> standalone tasks in the external-transfer-VinDr section.

## Mission
The from-scratch cardio retrain (`plans/rebalanced-cardio-retrain/`) stalled: with only 854 NIH cardio latents the conditional UNet collapses the lower chest to white mush, so generated CTR/border/yield plateau at the pre-retrain ~9% (corroborated by the Exp-09 classifier gate: synth cardio sens 0.10→0.25, still FAIL). This scope tries the alternative lever — **transfer, not from-scratch**. It pretrains the *same* conditional UNet on an **external, NIH-disjoint cohort** — **VinDr-CXR / VinBigData** (15,000 PA films at native **512²**; **1,952 deconfounded cardio = 2.3× NIH's 854**, plus **10,606 normal** as a general-anatomy prior; effusion 684) — in **our existing VAE latent space**, so it learns chest anatomy + extra cardio NIH lacks; then **plain low-LR finetunes to NIH appearance** on the clean deconfounded latents. It is a **supplementary lever, outside the pre-registered Exp 1–7 trust-chain**, judged head-to-head against the from-scratch lever on the same NIH frozen test. **PadChest (10k cardio, 1 TB) is the headroom escalation** if VinDr's ~1.3k border-visible cardio plateaus.

> **Guardrails (non-negotiable — design invariants, not tasks):** (1) keep the **class-conditional UNet** (no_finding/cardio/effusion + null token) and `cfg_compose` — no text-conditioning; the contribution is the composition. (2) Keep **our VAE** — one latent space for both stages, so Exp-2's codec ceiling carries untouched. (3) The base is trained on the **external cohort only (VinDr / PadChest), never NIH** — any NIH split risks the 235 reserved/frozen test patients. (4) **Final eval only on the NIH frozen separable set** with the NIH-calibrated CTR/CP instrument. VinDr was dropped earlier purely for compute (pre-hippo); that constraint is gone. See [[external-transfer-scope]]; fallbacks in `PARKING_LOT.md`.

## Objectives
1. Acquire and investigate an external cohort (VinDr-512) — confirm it is patient-disjoint from the NIH test and supplies extra cardio + a normal-anatomy prior.
2. Validate our VAE as the shared codec for it (round-trip gate), keeping a single latent space and Exp-2 intact.
3. Pretrain the conditional UNet on the external cohort so it learns lower-chest rendering the from-scratch model never could.
4. Adapt the pretrained model to NIH appearance with minimal forgetting, and re-measure the Gate-3 yield (and the Exp-09 classifier gate) against the from-scratch lever.

## Goals
1. **External base investigated** — VinDr cardio/effusion/normal N reported (deconfounded); patient-disjointness from the NIH 235-reserved + frozen test verified.
2. **Codec confirmed** — VAE round-trip gate passed on ~50 VinDr films (small |ΔCTR|, lower zone preserved); else the foreign-VAE fallback is triggered.
3. **Stage-1 prior** — pretrained checkpoint renders an enlargeable, border-readable heart (cardio CTR distribution shifted toward real; Exp-09 synth-cardio sens rising).
4. **Stage-2 adapted** — NIH-appearance checkpoint with effusion blunt-rate HELD and no new cardio→effusion leakage.
5. **Yield re-measured** — PoE Gate-3 yield on the NIH frozen set (≥2000 samples, CI) + the Exp-09 cardio gate vs the from-scratch ~9.3% / sens 0.25: recovery or an honest recorded null.

## Expected Outcome
A cardio-faithful checkpoint obtained by external transfer whose NIH PoE composites clear the Gate-3 yield and whose synth-only cardio clears the Exp-09 body gate — beating the from-scratch lever and unblocking the Exp-7 +PoE arm — **or** a clean recorded null (transfer did not beat from-scratch), with the cardio shift shown either way. If transfer wins, its checkpoint is released to Exp 6/7. PoE math confirmed untouched throughout.

## Definition of Done
1. ✅ VinDr-512 acquired on hippo (~4 GB Kaggle pull); per-image presence labels derived from the bboxes; deconfounded counts reported (cardio∧¬eff ≈ 1,952 vs NIH 854).
2. ✅ Preprocessing (VinDr is PA-only, already 512²) + border-visible filter applied; N per class reported (6,671 border-visible).
3. ✅ Data investigation: counts, **patient-leakage check vs NIH 235-reserved + frozen test (disjoint)**, distributions + VinDr-vs-NIH CTR-norm gap.
4. ✅ VAE round-trip gate on ~50 VinDr films: |ΔCTR| + lower-zone fidelity; **pass → our VAE is the codec (Exp-2 carries)**.
5. ⛔ SUPERSEDED — Stage-1 pretrain (transfer approach retired; replaced by vae-v2 joint-from-scratch).
6. ⛔ SUPERSEDED — Stage-2 NIH finetune (retired with Stage-1).
7. ⛔ SUPERSEDED — Gate-3 / Exp-09 eval now carried by `vae-v2/plans/07-downstream-eval.md` against the joint LDM.
8. ⛔ SUPERSEDED — head-to-head with from-scratch folds into the vae-v2 downstream verdict; PoE class-conditioning + null-token + `cfg_compose` remain untouched by the pivot.

## Sub-Scopes
- ⚠️ plans/opacity-instrument/ — validated regional lung-opacity instrument (RALE-style); SURVIVES as standalone model-agnostic tooling (cleaning/inspection + any future haze-map conditioning). `metrics/opacity_score.py` not yet built.
- ⚠️ plans/composition-operators/ — alternative AND-operators (MCMC-corrected + SuperDiff-AND) + in-training 3-way grid; eval-only, SURVIVES as standalone tooling applicable to any LDM incl. the vae-v2 one (detached from the retired Stage-2).

## Plans
Original dependency order (transfer): `01 → 09 → {02, 03, 04} → 05 → 06 → 08`.
**05/06/08 are ⛔ SUPERSEDED** (transfer approach → vae-v2 joint-from-scratch); the data-prep chain 01–04, 09 is ✅ done and feeds vae-v2.
- ✅ 01-vindr-acquire.md — Kaggle VinDr-512 pull (~4 GB) + bbox→presence labels (DoD 1) — `data/vindr512/` present, 15k films
- ✅ 09-morbidity-criteria-instrument.md — manual criteria + 2nd-disease pick (`results/vindr_transfer/criteria/second_disease.md`); RALE opacity instrument split out to the surviving sub-scope (DoD 2–3)
- ✅ 02-vae-roundtrip-gate.md — our VAE round-trips VinDr at the border (`results/vindr_transfer/vae_gate/DECISION.md`) (DoD 4)
- ✅ 03-preprocess-projection-filter.md — PA-only @512² + border filter → `border_visible_manifest.csv` (6,671); latents encoded (DoD 2)
- ✅ 04-data-investigation.md — counts + patient-leakage check vs NIH reserved/frozen test + distributions (`results/vindr_transfer/eda/REPORT.md`) (DoD 3)
- ⛔ 05-stage1-pretrain.md — SUPERSEDED by vae-v2 joint-from-scratch; never ran (banner in file)
- ⛔ 06-stage2-nih-finetune.md — SUPERSEDED; never ran (banner in file)
- ⛔ 08-eval-gate3-yield.md — SUPERSEDED; downstream eval → `vae-v2/plans/07-downstream-eval.md` (banner in file)
