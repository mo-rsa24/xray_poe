# 18 · Single-Disease FID Gate

## Reference while you do it
- 📄 Plan: plans/single-disease-ldm/plans/04-fid-eval.md

## Section context (paste into the Todoist section)
**Description:** Generate single-disease samples and measure FID vs real; confirm both `∅` and `normal` nulls are callable.
**Objective:** Guard Exp5/6 — unconvincing single-disease samples make composition meaningless.
**Goal:** FID passing gate, a sample grid saved, both anchors confirmed.
**Verify (whole leaf):** `python -m ldm.evaluate --ckpt ckpts/ldm_*.pt --fid` → FID ≤ gate; figures/ldm_samples.png; both null modes sample.

## Tasks (one at a time)
- [ ] Sample per single-disease condition; compute FID vs real single-disease
- [ ] Save a sample grid; confirm sampling with null=`∅` and null=`normal` both work
