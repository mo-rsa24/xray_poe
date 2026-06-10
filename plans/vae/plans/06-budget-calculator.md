# 🧮 Budget Calculator

## Description
A calculator that turns measured usage (peak VRAM, img/s) plus assumed
steps/epochs, image count, and $/GPU-hr into GPU-hours and dollars — parameterized
so the dataset size and price can be filled in once known.

## Purpose
Converts the profiled *rate* into a cost forecast. It pairs the measured
denominator (img/s) with an assumed numerator (steps) and a pricing input; the
output feeds compute-budget's cost memo. Being explicit about which inputs are
measured vs assumed keeps the forecast honest.

## Goal
`vae.budget` (or a script) that, given img/s + steps/epochs + N + $/hr, prints
GPU-hours and dollars and reproduces a documented worked example.

## Tasks
- [x] ✅ Implement the cost formula: `hours = epochs×N / img_per_s / 3600`; `$ = hours × rate`
- [x] ✅ Parameterize: read measured img/s from the profiling log; take steps/N/$per-hr as args; add a contingency multiplier
- [x] ✅ Flag measured vs assumed inputs in the output
- [x] ✅ Reproduce a worked example and document it

## Recommended skill
custom; no skill fits — a small calculator + doc.

## Engagement Instructions
```
$ python -m vae.budget --img-s 120 --epochs 150 --n 50000 --rate 0.79
# expect: "≈ H GPU-hours, ≈ $C  (measured img/s; assumed epochs, N, rate)"
```
