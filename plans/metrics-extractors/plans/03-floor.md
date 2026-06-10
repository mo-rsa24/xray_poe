# 📏 The Floor

## Description
Compute the floor by splitting the real both-disease set in half and measuring the halves
against each other (real-vs-real) with confidence intervals — the smallest gap measurable.

## Purpose
Everything in Exp6/7 is judged against this floor and its 95% upper bound; it is the
reference for "indistinguishable from real."

## Goal
A floor value + 95% bounds for each pair; a power flag if N is small (wide floor).

## Tasks
- [ ] ⚠️ Split real both-disease set in half; compute two-sample + MMD between halves
- [ ] ⚠️ Bootstrap to get 95% bounds; flag if N small

## Engagement Instructions
```
$ python -m metrics.floor --pair cardiomegaly,effusion
# expect: floor ~0.5x with 95% upper bound; N reported with a power flag
```
