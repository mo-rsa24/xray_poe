# VAE 8 · Noise Training Run + Hand-off

## Reference while you do it
- 📄 Plan: plans/vae/plans/08-noise-training-run.md

## Section context (paste into the Todoist subtask)
**Description:** Run the training loop for sustained steps on noise tensors shaped exactly like real inputs (1×512×512) to confirm end-to-end run without OOM and capture steady-state peak VRAM + img/s. Then hand measured usage to compute-budget.
**Objective:** End-to-end code confirmation + source of the real usage numbers. Noise suffices (VRAM/throughput content-independent); it does NOT give convergence (real-data, later).
**Goal:** A multi-step noise train completing without OOM, peak VRAM + img/s recorded and copied into compute-budget/01-workload-sizing.
**Verify (whole leaf):** `python -m vae.train --data noise --res 512 --steps 2000` → completes, prints peak VRAM + img/s; numbers in the 512×VAE sizing-table row.

## Tasks (one at a time)
- [ ] Generate noise tensors (1,512,512) in real dtype; stream via real dataloader path
- [ ] Run K steps at target precision/batch; confirm no OOM/crash
- [ ] Capture peak VRAM + steady-state img/s via the profiler
- [ ] Hand numbers to plans/compute-budget/plans/01-workload-sizing.md; note real-data train + recon gate + ceiling check deferred to runpod-execution
