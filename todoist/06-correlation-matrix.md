# 06 · Correlation Matrix — Pair Selection & Gate

## Reference while you do it
- 📄 Plan: plans/eda/plans/04-correlation-matrix.md

## Section context (paste into the Todoist section)
**Description:** Compute the 19×19 φ/odds-ratio co-occurrence matrix; select the treatment (strong) and control (≈0) pairs. **Experiment 1 — the project go/no-go gate.**
**Objective:** Pick treatment + control and gate the project — no strong pair AND no near-zero pair ⇒ no controlled comparison.
**Goal:** A φ matrix + locked treatment/control pairs, go/no-go recorded, both-disease N per pair.
**Verify (whole leaf):** `python -m eda.correlation --manifest data/manifest.parquet --out figures/correlation_heatmap.png` → heatmap + top-5/bottom-5 pairs; GATE go if both a strong and a near-zero pair exist.

**▶ Recommended prompt:** — custom/project-specific; no EDA-workflow skill covers φ + pair selection.

## Tasks (one at a time)
- [ ] Compute pairwise φ-coefficient (and odds ratio) over the 19 labels
- [ ] Identify the strongest pair (treatment) and a near-zero pair (control); confirm cardiomegaly+effusion is strong and verify the control is ≈0 in this data
- [ ] Record the go/no-go gate decision; report both-disease N per chosen pair
