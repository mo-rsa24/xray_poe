# 🗂️ Figure & Metric Register

## Description
Collect and caption the key figures and metric values from every scope into one register
(source scope → figure/number → caption).

## Purpose
One place to find every result and figure when assembling the paper.

## Goal
A register listing each figure/metric, its source scope, and a caption.

## Tasks
- [ ] ⚠️ Create the register; add entries as figures/metrics are produced
- [ ] ⚠️ Caption each entry for paper reuse

## Engagement Instructions
```
$ test -f documentation/REGISTER.md && grep -c '|' documentation/REGISTER.md
# expect: rows of (scope, artifact path, caption)
```
