# 🌫️ Noise-Data Training Run + Hand-off

## Description
Run the training loop for a sustained number of steps on randomly generated noise
tensors shaped exactly like real inputs (1×512×512), to confirm the code runs
end-to-end without OOM/crash and to capture steady-state peak VRAM + throughput.
Then hand the measured usage to the compute-budget scope.

## Purpose
This is the end-to-end code confirmation **and** the source of the real usage
numbers. Noise is sufficient because peak VRAM and throughput are
content-independent; it does **not** give steps-to-convergence (that needs real
data, handled later under runpod-execution).

## Goal
A multi-step noise train that completes without OOM, with peak VRAM + img/s
recorded and copied into `compute-budget/01-workload-sizing`'s sizing table.

## Tasks
- [x] ✅ Generate noise tensors shaped (1,512,512) in the real dtype; stream via the real dataloader path
- [x] ✅ Run the train loop for K steps at the target precision/batch; confirm no OOM/crash
- [x] ✅ Capture peak VRAM + steady-state img/s via the profiler
- [x] ✅ Hand the measured numbers to `plans/compute-budget/plans/01-workload-sizing.md`; note real-data train + recon gate + ceiling check are deferred to `plans/compute-budget/plans/runpod-execution/`

## Recommended skill
custom; no skill fits — runs the loop on synthetic data + records usage.

## Engagement Instructions
```
$ python -m vae.train --data noise --res 512 --steps 2000
# expect: completes without OOM; prints peak VRAM + img/s
# then: numbers copied into compute-budget's sizing table (512 × VAE row)
```
