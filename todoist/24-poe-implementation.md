# 24 · PoE Composition

## Reference while you do it
- 📄 Plan: plans/composition-experiments/plans/01-poe-implementation.md

## Section context (paste into the Todoist section)
**Description:** Implement Product-of-Experts composition by adding two single-disease score predictions per denoising step, with selectable null anchor (`∅`/`normal`) and per-expert weights.
**Objective:** Build the method under test — adding scores = multiplying distributions = sampling the product.
**Goal:** A composer producing both-disease samples for any pair, anchor, and weights.
**Verify (whole leaf):** `python -m compose.poe --pair cardiomegaly,effusion --null empty --n 8` → 8 composed samples; anchor + weights configurable.

## Tasks (one at a time)
- [ ] Implement score addition with null subtraction; unit-check (score add = product)
- [ ] Expose the null anchor (`∅` / `normal`) and weights (w₁, w₂)
- [ ] Smoke-test on the treatment pair
