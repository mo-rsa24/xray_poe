# Paper 3 — PoE Composition (Todoist staging)

Staged for `/todoist-publish`. One leaf here → one Todoist task in the named section.
Work order = file order. Source of truth stays the plan tree under `plans/`; this is the
derived view. (No Todoist MCP was connected when staged — publish writes it when the
server is available.)

## 🫁 PoE Composition of Single-Disease Experts

Work order = file order (`1 → 4 → 2 → 3 → 5 → 6 → 7` is already baked into the NN numbering).

- [ ] [👯 01 · Dynamics Population](01-dynamics-population.md) — Gate 0: separable-vs-entangled population + frozen test set (the paper's go/no-go)
- [ ] [🧪 02 · Deconfounded Splits](02-deconfounded-splits.md) — Exp 4: clean single-disease expert training data + cross-leakage check
- [ ] [🔁 03 · VAE Ceiling](03-vae-ceiling.md) — Exp 2: localized recon fidelity at heart-border + CP-angle
- [ ] [🎯 04 · PoE Yield](04-poe-yield.md) — Exp 3 / Gate 3: co-presence pass-rate of composed samples
- [ ] [🚦 05 · Reuse-vs-Retrain](05-reuse-vs-retrain.md) — Exp 5: the VAE/LDM decision gate (retrain is a dormant sub-scope)
- [ ] [🧭 06 · Oracle Baseline](06-oracle-baseline.md) — Exp 6 / Gate 6: detector build + oracle/starved ceiling
- [ ] [⭐ 07 · Headline AUC](07-headline-auc.md) — Exp 7 / Gate 6: overlay-vs-PoE stratified downstream AUC (**5-arm** +Individual control; +PoE−+Individual diagnostic)

---

# Supervisor-Reconcile Deliverables (D1–D13 · `docs/SUPERVISOR_RECONCILE.md`)

New + extended leaves from the 2026-06-26 supervisor reconcile — **one section** (additive; do NOT touch the PoE-Composition leaves above). Leaf order below = work order (dependency `D1→D2→(D3,D4,D5)→D8→(D6,D7 after Exp 7 arms)→D10→D12 deferred`). Leaves 15/16 are the additions to plans 07/06 (kept as separate leaves so existing 06/07 leaves are untouched). 14 is a deferred 🔵 stub — do not schedule as active work.

## 🩺 Supervisor-Reconcile Deliverables
- [ ] [🧾 08 · Methods — System Description](08-methods-system-description.md) — D1 (+D9): classifier I/O, model inventory, DenseNet-vs-PSPNet disambiguation; answers §0
- [ ] [🚦 09 · Individual-Disease Validation](09-individual-disease-validation.md) — **D2, the body-or-no-body GATE**; runnable NOW on the before-model
- [ ] [🖼️ 10 · Figure 1 — GradCAM Panel](10-figure1-gradcam-panel.md) — D3: real-vs-synth GradCAM, before/after
- [ ] [📊 11 · Table 1 — Classifier Report](11-tables-classifier-report.md) — D4 + D9: shared acc/sens/spec module, eval-only, before/after
- [ ] [📈 12 · Figure 2 — CTR Scatter](12-figure2-ctr-scatter.md) — D5: CTR scatter real-vs-synth, real now / synth before/after
- [ ] [🧹 13 · Data-Cleaning Bias Audit](13-data-cleaning-bias-audit.md) — D8 (7.9): quantify the entangled/border-filtered exclusion as a limitation
- [ ] [📈 15 · Headline — Fine-Tuned GradCAM + Table 2](15-headline-finetuned-deliverables.md) — D6 + D7 + D13: after Exp 7 arms; before/after-labelled
- [ ] [🧭 16 · Oracle Headroom-Guard](16-oracle-headroom-guard.md) — D10: confirm the role upgrade — the guard that makes an Exp 7 null interpretable
- [ ] [⚙️ 14 · Monolithic Joint-Condition Baseline](14-monolithic-baseline-deferred.md) — D12: **ACTIVE** (promoted 2026-06-26) — train joint-condition diffusion + compare vs +PoE on both-separable
