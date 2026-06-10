# 30 · Figure & Metric Register

## Reference while you do it
- 📄 Plan: plans/documentation/plans/02-figure-metric-register.md

## Section context (paste into the Todoist section)
**Description:** Collect and caption the key figures and metric values from every scope into one register (source scope → figure/number → caption).
**Objective:** One place to find every result and figure when assembling the paper.
**Goal:** A register listing each figure/metric, its source scope, and a caption.
**Verify (whole leaf):** `test -f documentation/REGISTER.md && grep -c '|' documentation/REGISTER.md` → rows of (scope, artifact, caption).

## Tasks (one at a time)
- [ ] Create the register; add entries as figures/metrics are produced
- [ ] Caption each entry for paper reuse
