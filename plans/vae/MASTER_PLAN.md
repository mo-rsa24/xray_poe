# 🗜️ VAE

## Mission
Design, implement, and locally profile the label-blind VAE codec — compressing
512² grayscale chest X-rays to a **4×128×128 latent (f=4)** — to the point where
the code is tested, runnable here, and its real resource usage is measured. A
SOTA AutoencoderKL-style architecture (ResNet + low-res attention, GroupNorm+SiLU,
KL bottleneck, **no encoder→decoder skips** so the latent is a true bottleneck).
The scope confirms the code works (overfit-sanity + a noise-data train), produces
a cost calculator, and **hands the measured VRAM/throughput to the compute-budget
scope**, which provisions a rented GPU and runs the real-data train. The
reconstruction gate and ceiling check are *designed* here but *execute*
downstream on real data under `compute-budget/runpod-execution`.

## Objectives
1. Select a SOTA VAE architecture for 512² grayscale → 4×128×128, via research /
   a deliberate prompt, and write the decision down.
2. Sanity-check the architecture as a bare module (shapes, params, forward/backward,
   bottleneck contract) before building the training stack.
3. Stand up a reproducible local run/test environment — pinned deps, pytest, and a
   `.vscode/launch.json` — so the code is runnable and breakpoint-debuggable here.
4. Implement the VAE as a tested, modular codebase: encoder/decoder/KL, training
   loop, and the eval code (recon metric + ceiling-check script) that runs unchanged
   on real data later.
5. Write profiling code + terminal monitoring scripts that capture the
   content-independent numbers — peak VRAM and throughput.
6. Build a budget calculator that turns measured usage + assumed steps/N + $/hr
   into GPU-hours and dollars.
7. Confirm the code end-to-end via an overfit-sanity pass and a noise-data
   training run.
8. Hand the measured usage to `compute-budget/01-workload-sizing`; defer the
   real-data train + recon gate + ceiling check to `compute-budget/runpod-execution`.

## Goals
1. An architecture-decision note: blocks, channel schedule, attention placement,
   latent 4×128×128 @ f=4, activation, output nonlinearity, KL weight, param estimate,
   plus the rejected alternatives (f=8, enc→dec skips, DiT) and why.
2. Architecture sanity checks pass: `(B,1,512,512)→(B,4,128,128)→(B,1,512,512)`,
   finite forward/backward, printed param count, no enc→dec skip.
3. Local env reproducible: deps pinned, `pytest` collects + runs, and
   `.vscode/launch.json` launches overfit / train(noise) / profile / pytest.
4. The `vae` package passes unit tests (shape round-trip, finite KL, one-step loss
   decrease, metric smoke test).
5. `vae.profile` reports peak VRAM (`torch.cuda.max_memory_allocated`/`nvidia-smi`)
   + steady-state img/s for a config; a bash script tails GPU + loss.
6. The budget calculator: usage in → GPU-hours + $ out, parameterized by steps/N/$per-hr,
   reproducing a worked example and flagging measured-vs-assumed inputs.
7. Overfit-sanity recon ≈ 0 on a fixed 8-image batch; the noise-data train runs K
   steps without OOM/crash and logs peak VRAM + throughput.
8. Measured VRAM + img/s recorded and copied into `compute-budget`'s workload-sizing table.

## Expected Outcome
A working, tested, profiled VAE codebase in the repo — runnable and debuggable
locally, with a cost calculator and measured resource numbers. So compute-budget
can size the GPU and cost the real train, and runpod-execution can run it on real
data behind the recon + ceiling gates whose code and criteria are defined here.

## Definition of Done
1. Architecture decision written (512²→4×128×128 grayscale; blocks/channels/activations;
   no enc→dec skip; rejected alternatives recorded).
2. Architecture sanity checks logged (shapes, params, forward/backward, latent bottleneck verified).
3. Local env ready: pinned deps, `.vscode/launch.json`, `pytest` green.
4. VAE implemented + unit-tested; encode/decode + SSIM/LPIPS recon metric + ceiling-check
   script present (execution on real data deferred).
5. Profiling code + bash monitoring scripts produce peak VRAM + throughput on demand,
   backed by a profiling log.
6. Budget calculator implemented; a worked example reproduced.
7. Overfit-sanity pass logged (recon ≈ 0); noise-data train completes without OOM, with
   peak VRAM + img/s captured.
8. Measured usage handed to `compute-budget/01-workload-sizing`; recon gate + ceiling
   check (real data) explicitly deferred to `compute-budget/runpod-execution` with their
   criteria recorded here.

## Hand-off
This scope ends at *measured usage numbers + working code*. The real-data train,
the reconstruction gate (SSIM/LPIPS), and the ceiling check (Exp3) execute under
the sibling scope `plans/compute-budget/` (specifically `plans/runpod-execution/`),
which owns running on rented infra. Their eval code and pass/fail criteria are
authored here (plan 04); only their execution is deferred.

## Sub-Scopes
(none)

## Plans
- ✅ 01-architecture-selection.md
- ✅ 02-architecture-sanity-checks.md
- ✅ 03-local-env-setup.md
- ✅ 04-vae-implementation.md
- ✅ 05-profiling-and-monitoring.md
- ✅ 06-budget-calculator.md
- ✅ 07-overfit-sanity-run.md
- ✅ 08-noise-training-run.md
