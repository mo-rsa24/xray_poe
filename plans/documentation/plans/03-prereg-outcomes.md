# 🔒 Pre-Registration Outcomes

## Description
Record each experiment's result against the pre-registration table — claim holds / claim
wrong / anchoring / in-between — with no post-hoc reinterpretation.

## Purpose
Keeps the study honest; results can't be narrated after the fact.

## Goal
A filled pre-registration outcome table, one row per experiment.

## Tasks
- [ ] ⚠️ Copy the pre-registration table from EXPERIMENTS.md; fill the observed outcome per experiment
- [ ] ⚠️ Flag any "in-between" result for rerun, not reinterpretation

## Engagement Instructions
```
$ test -f documentation/PREREG.md && grep -A4 -i 'pre-registration' documentation/PREREG.md
# expect: each experiment mapped to one pre-registered outcome
```
