# 🔎 Data Investigation (Counts · Leakage · Distributions)

## Background
Before pretraining, two things must be shown: VinDr actually supplies the scarce cardio, and it is **patient-disjoint from the NIH test** (235 reserved + frozen separable set). Leakage here lets test-patient appearance reach the generator and invalidates the pixel-verifiable headline. This is the workflow-5 data pass on VinDr.

## Description
Profile the border-visible VinDr cache: per-class counts, a hard patient-leakage check vs NIH, distribution figures, synthesized into a findings doc (compared against the proposal's Phase-3 EDA) that green-lights Stage-1.

## Purpose
Serves Objective 1 and DoD item 3 — proves the base is useful (enough cardio) and clean (disjoint).

## Goal
A findings doc with counts, a PASS/FAIL disjointness verdict, and distribution figures.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks

- [x] ✅ **Count cardio/effusion/normal in the kept cache**  ✓ verified (normal=4,774 / cardio=1,136 / disease2=761; cardio 1,136 ≫ NIH 854 ✓; counts in `04-distributions.md §1`; `01-inventory.md` standalone not created — tee step superseded by distributions doc which captures the same counts)

- [x] ✅ **Patient-leakage check vs NIH 235-reserved + frozen test**  ✓ verified (`scripts/vindr_leakage_check.py` 10,379 bytes; `results/vindr_transfer/eda/02-integrity.md` verdict: "PASS — disjoint from NIH test"; 16/16 intersection checks = 0 across 8 NIH test-side ID sets on both BV and full manifest)

- [x] ✅ **Class distributions (+ VinDr-vs-NIH CTR gap)**  ✓ verified (`figures/vindr_dists.png` 188 KB 3-panel figure; `results/vindr_transfer/eda/04-distributions.md`: CTR gap +0.010 (1.9%) NIH−VinDr median, MWU p<0.001, documented as Stage-2 monitoring item; implemented via `scripts/vindr_eda_distributions.py`)

- [x] ✅ **Synthesize findings → Stage-1 go/no-go (+ compare vs proposal Phase-3 EDA)**  ✓ verified (`results/vindr_transfer/eda/REPORT.md` 209 lines; cites counts + disjointness + domain-gap + Phase-3 comparison table with agree/diverge column; conclusion: "Stage-1 (VinDr pretrain) is cleared" on all three load-bearing gates)

## Recommended skill
`/data-integrity-check` ✅ — the leakage-vs-NIH check is load-bearing.
   alt: `/data-inventory` → `/data-distributions` → `/eda-synthesize` ✅.

## Engagement Instructions
```bash
$ grep -iE "disjoint|intersection|leak" results/vindr_transfer/eda/02-integrity.md
$ grep -iE "cardio|854|green|phase.?3" results/vindr_transfer/eda/REPORT.md | head
# expect: "disjoint from NIH test" + cardio N ≫ 854 + a Phase-3 comparison + a Stage-1 go/no-go line
```
