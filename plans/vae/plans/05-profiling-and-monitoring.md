# 📊 Profiling Code + Monitoring Scripts

## Description
Profiling code that measures the content-independent numbers — peak VRAM
(`torch.cuda.max_memory_allocated` / `nvidia-smi`) and steady-state throughput
(img/s) — at a given resolution/batch/precision, plus bash scripts to launch a
training process and monitor it from the terminal.

## Purpose
These are the measured inputs the budget calculator (06) and the compute-budget
scope need. Profiling is valid on noise tensors because peak VRAM and throughput
depend only on shape, dtype, batch size, architecture, and optimizer — not on data
content.

## Goal
A profile entrypoint that prints peak VRAM + img/s for a config (and sweeps batch
size to find the max that fits), plus bash scripts that launch training and tail
GPU + loss.

## Tasks
- [x] ✅ Implement `vae.profile`: warm up, run K steady-state steps, report peak VRAM + img/s for given res/batch/precision
- [x] ✅ Add a batch-size sweep to find the max that fits (the VRAM-tier selector)
- [x] ✅ Write `scripts/monitor_gpu.sh` (nvidia-smi loop) and `scripts/train_watch.sh` (launch train + tail log)
- [x] ✅ Capture a profiling log (peak mem) backing ≥1 measured number

## Recommended skill
▶ `/experiment-planner` ✅ — frames profiling as a resource experiment.

## Engagement Instructions
```
$ python -m vae.profile --res 512 --batch 8 --precision bf16
# expect: "peak VRAM X.X GB | Y.Y img/s" + a saved profiling log
$ bash scripts/monitor_gpu.sh           # tails nvidia-smi memory + util
```
