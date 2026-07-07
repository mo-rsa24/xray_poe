# 16 · 🧭 Oracle — Null-Interpretability Headroom-Guard (D10 role upgrade)

## Reference while you do it
- 📄 Plan: plans/06-oracle-baseline.md (§ "Role upgrade under the sober lead" + the headroom-verdict subtask)
- 🧭 Authority: docs/SUPERVISOR_RECONCILE.md (D10 — oracle retained, role upgraded)
- ℹ️ The oracle/starved *training* stays in leaf 06; this leaf is the role-statement + sens/spec on the verdict.

## Section context (paste into the Todoist section)
**Description:** Confirm the oracle's role upgrade under the sober lead — the oracle−starved headroom is the **guard that makes an Exp 7 null interpretable**. Acceptance is unchanged; what changes is *why* it matters.
**Objective:** Establish that oracle−starved headroom on both-separable makes a PoE-≤-overlay null clean/publishable, not an artifact of a dead test set.
**Goal:** `results/exp06_oracle/verdict.md` states the role as null-interpretability guard, reports sens/spec, notes the N-bound caveat.
**Verify (whole leaf):** `grep -iE "null|interpretable|guard|sensitivity|specificity" results/exp06_oracle/verdict.md`.
**▶ Recommended prompt:** — custom; no skill fits.

## Tasks (one at a time)
- [ ] 🧭 State the headroom verdict as the null-interpretability guard (D10) — record in `verdict.md` that headroom = the guard making an Exp 7 null interpretable (real headroom → a PoE-≤-overlay null is clean/publishable; saturated → uninterpretable, revisit the test set); report sens/spec alongside ΔAUC; note headroom is bounded by NIH both-labelled N (~1063, ~72% PA-separable) — thin headroom weakens the null. **Supervisor (D10, Exp 6 oracle retained):** "oracle − starved ΔAUC on both-separable reported; establishes headroom so the null is interpretable. Role upgraded under sober lead." Script: existing oracle plan (`plans/06`) — headroom-verdict subtask · Checkpoint: real comorbid · **Accept (verbatim):** "oracle − starved ΔAUC on both-separable reported; establishes headroom so the null is interpretable. Role upgraded under sober lead." Verify: `verdict.md` cites the both-separable ΔAUC + CI and the headroom call, AND states the null-interpretability guard role with sens/spec reported.
