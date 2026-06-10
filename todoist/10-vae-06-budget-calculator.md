# VAE 6 · Budget Calculator

## Reference while you do it
- 📄 Plan: plans/vae/plans/06-budget-calculator.md

## Section context (paste into the Todoist subtask)
**Description:** A calculator turning measured usage (img/s) + assumed steps/epochs, N, and $/GPU-hr into GPU-hours and dollars — parameterized so dataset size + price fill in later.
**Objective:** Convert the profiled rate into a cost forecast; pair measured img/s with assumed steps + pricing; output feeds compute-budget's cost memo.
**Goal:** `vae.budget` that, given img/s + steps + N + $/hr, prints GPU-hours + $ and reproduces a worked example, flagging measured-vs-assumed inputs.
**Verify (whole leaf):** `python -m vae.budget --img-s 120 --epochs 150 --n 50000 --rate 0.79` → "≈ H GPU-hours, ≈ $C (measured img/s; assumed epochs/N/rate)".

## Tasks (one at a time)
- [ ] Cost formula: hours = epochs×N / img_per_s / 3600; $ = hours × rate
- [ ] Parameterize: read img/s from profiling log; steps/N/$per-hr as args; contingency multiplier
- [ ] Flag measured vs assumed inputs in output
- [ ] Reproduce + document a worked example
