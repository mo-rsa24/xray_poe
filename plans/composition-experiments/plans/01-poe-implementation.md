# ➕ PoE Composition

## Description
Implement Product-of-Experts composition by adding the two single-disease score predictions
at each denoising step, with a selectable null anchor (`∅` or `normal`) and per-expert weights.

## Purpose
The method under test; adding scores = multiplying distributions = sampling the product
`p(x|c₁)·p(x|c₂)/p(x)`.

## Goal
A composer that produces both-disease samples for any pair, anchor, and weight setting.

## Tasks
- [ ] ⚠️ Implement score addition with null subtraction; unit-check (score add = product)
- [ ] ⚠️ Expose the null anchor (`∅` / `normal`) and weights (w₁, w₂)
- [ ] ⚠️ Smoke-test on the treatment pair

## Engagement Instructions
```
$ python -m compose.poe --pair cardiomegaly,effusion --null empty --n 8
# expect: 8 composed both-disease samples saved; anchor + weights configurable
```
