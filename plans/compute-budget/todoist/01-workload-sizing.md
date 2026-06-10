# 01 · Workload Sizing

[⌂ Index](00-INDEX.md) · [next →](02-runpod-pricing-survey.md)

## Reference while you do it
- 📄 Plan: plans/compute-budget/plans/01-workload-sizing.md

## Section context (paste into the Todoist section)
**Description:** Measure/estimate the compute footprint of both trains — the shared VAE and the single-disease LDM — for each candidate input resolution (512²/768²/1024²) into the shared 4×128×128 latent. Output is a sizing table feeding the pricing and cost steps.
**Objective:** Produce the VRAM numbers that select the GPU tier and the throughput/steps that drive cost — without them, pricing has no VRAM target and the estimate has no GPU-hours.
**Goal:** A written sizing table: per resolution × {VAE, LDM} — peak VRAM (GB), throughput (steps/s or img/s), target steps/epochs, and dataset size (image count + on-disk GB).
**Verify (whole leaf):** `cat plans/compute-budget/sizing-table.md` shows rows 512/768/1024 × {VAE,LDM} with columns VRAM, steps/s, steps, GB — plus a profiling log (nvidia-smi peak mem) backing ≥1 VRAM number so estimates aren't pure guesses.
**▶ Recommended prompt:** `/experiment-planner` ✅ — frames each train as an experiment with resource estimates; alt: a manual profiling run for the VRAM numbers.

## Tasks (one at a time)
- [ ] Fix the dataset-size assumptions (image count, batch size) used for sizing, per resolution
- [ ] Estimate/profile VAE peak VRAM + throughput at 512²/768²/1024² → 4×128×128 (short profiling run on an available GPU, or analytic estimate)
- [ ] Estimate/profile single-disease LDM peak VRAM + throughput in the 4×128×128 latent
- [ ] Set target steps/epochs per train (VAE recon gate; LDM convergence) and compute on-disk corpus size per resolution
- [ ] Record the sizing table (resolution × {VAE,LDM} → VRAM, throughput, steps, data size)
