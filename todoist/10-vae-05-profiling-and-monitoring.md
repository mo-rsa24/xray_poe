# VAE 5 · Profiling Code + Monitoring Scripts

## Reference while you do it
- 📄 Plan: plans/vae/plans/05-profiling-and-monitoring.md

## Section context (paste into the Todoist subtask)
**Description:** Profiling code measuring the content-independent numbers — peak VRAM (torch.cuda.max_memory_allocated / nvidia-smi) and steady-state img/s at a given res/batch/precision — plus bash scripts to launch + monitor training from the terminal.
**Objective:** Produce the measured inputs the budget calculator and compute-budget need. Valid on noise because VRAM/throughput depend only on shape/dtype/batch/arch/optimizer.
**Goal:** A profile entrypoint printing peak VRAM + img/s (with a batch-size sweep), plus monitor scripts.
**Verify (whole leaf):** `python -m vae.profile --res 512 --batch 8 --precision bf16` → "peak VRAM X GB | Y img/s" + saved log; `bash scripts/monitor_gpu.sh` tails usage.
**▶ Recommended prompt:** `/experiment-planner` ✅ — frames profiling as a resource experiment.

## Tasks (one at a time)
- [ ] `vae.profile`: warm up, K steady-state steps, report peak VRAM + img/s
- [ ] Batch-size sweep to find max-fit (VRAM-tier selector)
- [ ] `scripts/monitor_gpu.sh` (nvidia-smi loop) + `scripts/train_watch.sh` (launch + tail)
- [ ] Capture a profiling log backing ≥1 measured number
