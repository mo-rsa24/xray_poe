# 🧮 Workload Sizing

## Description
Measure/estimate the compute footprint of both trains — the shared VAE and the
single-disease LDM — for each candidate input resolution (512²/768²/1024²) into
the shared 4×128×128 latent. Output is a sizing table feeding the pricing and
cost steps.

## Purpose
GPU selection is driven by peak VRAM; cost is driven by throughput × steps.
Without these numbers the pricing survey has no VRAM target and the cost
estimate has no GPU-hours. Serves Objectives 1–2, Goal 1.

## Goal
A written sizing table: per resolution × {VAE, LDM} — peak VRAM (GB), throughput
(steps/s or img/s), target steps/epochs, and dataset size (image count + on-disk GB).

## Tasks
- [ ] ⚠️ Dataset = NIH ChestX-ray14 (~112,120 images, ~42 GB on disk); pull image count + four-group counts from data-foundation/plans/04 → `data/nih/four_group_counts.md`; fix batch size per resolution <!-- tid:6gqM3rPCXvmFWm32 -->
- [ ] ⚠️ Estimate/profile VAE peak VRAM + throughput at 512²/768²/1024² → 4×128×128 (short profiling run on an available GPU, or analytic estimate)
- [ ] ⚠️ Estimate/profile single-disease LDM peak VRAM + throughput in the 4×128×128 latent
- [ ] ⚠️ Set target steps/epochs per train (VAE recon gate; LDM convergence) and compute on-disk corpus size per resolution
- [ ] ⚠️ Record the sizing table (resolution × {VAE,LDM} → VRAM, throughput, steps, data size)

## Engagement Instructions
```
$ cat plans/compute-budget/sizing-table.md   # rows: 512/768/1024 × {VAE,LDM}; cols: VRAM, steps/s, steps, GB
# plus a profiling log (nvidia-smi peak mem) backing ≥1 VRAM number, so estimates aren't pure guesses
```

## Recommended skill
▶ `/experiment-planner` ✅ — frames each train as an experiment with resource estimates.
   alt: a manual profiling run for the VRAM numbers.
