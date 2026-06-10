# 🎲 LDM Pilot → Bound the Budget (cost early, convergence after the VAE)

**Scope:** `plans/compute-budget/plans/runpod-execution/` · **Status:** ⚠️ pending · **Companion to `00-profile-and-budget` (VAE)**

## Background
Unlike the VAE, the LDM has **two** unknowns, and they resolve at different times:

1. **Per-step cost** (it/s, peak VRAM) — *content-independent*, so it's profilable on
   random 4×128×128 latents the moment the LDM denoiser code exists (exactly like the VAE).
   Today it is only EST in the sizing table (§B). **Step A** kills this EST cheaply.
2. **Steps to "good enough"** (the per-condition FID gate) — *content-dependent*. Nothing in
   the architecture predicts it; you cannot quote it a-priori. **Step B** bounds it with a
   short real-latent pilot + extrapolation.

The LDM is **one conditional model** (CFG dropout; conditions = `normal`/`cardiomegaly`/`effusion`),
not one-per-disease — so this is a single train, with FID gated **per condition**.

> **The likely binding constraint is data, not hours.** `cardiomegaly` is tiny (~hundreds–2k
> latents). A diffusion model on that few examples **overfits / mode-collapses before FID
> converges** — past that knee, more GPU-hours yield a *worse* (memorizing) model, and FID is
> noisy with so few reals. So "how many hours for good enough?" can legitimately answer
> *"hours stop helping at the data ceiling on the small conditions."* Watch per-condition
> overfitting, not just the aggregate loss.

## What is EST vs measured
- **EST until Step A:** LDM it/s + peak VRAM (sizing-table §B: ~300M UNet, ~5 GB @B=1, ~2–6 it/s).
- **EST until Step B:** steps-to-FID-plateau per condition (content-dependent; only knowable from the curve).
- **Already known:** the latent is **resolution-invariant** (4×128×128 regardless of 512/768/1024),
  so there is **one** LDM cost, not three.

## Purpose
Convert both LDM unknowns into measured/bounded numbers before committing the long train —
the same de-risking posture as the VAE, one notch more conservative because convergence is
genuinely unpredictable here. Feeds the cost memo (`compute-budget/03`) and `04-ldm-train`.

## Goal
A recorded LDM `it/s | peak VRAM | max-batch` (Step A) and an extrapolated steps-to-plateau
with a **budget cap** (Step B), so `04-ldm-train` runs gate-bounded against a known ceiling.

---

## Step A — cost profile (content-independent; same rental as the VAE `00`, ~mins)
Requires the LDM denoiser code (the `single-disease-ldm` scope, built + unit-tested locally,
like the VAE was). Profiles on **random latents** — no VAE checkpoint, no corpus needed.

```
# on the profiling pod, on noise latents shaped (B, 4, 128, 128):
$ python -m ldm.profile --latent 4x128x128 --precision bf16 --sweep
# GET THAT → "peak VRAM X.X GB | Y.Y it/s" + max batch that fits the tier
```
Record it/s + peak VRAM + max batch. **This removes the §B EST entirely.**

## Step B — convergence pilot (needs the trained VAE → runs as an early slice of `04`)
The FID slope needs *real* latents, so this runs after `03-vae-train` produces a VAE checkpoint.

```
# encode the LARGEST condition (normal) to latents with the frozen VAE, then a short pilot:
$ bash scripts/train_ldm.sh --config configs/ldm.yaml --max-steps 30000 --eval-fid-every 5000
# GET THAT → FID at 5k,10k,…,30k → the early FID-vs-steps slope
```
**Extrapolate:** diffusion FID falls fast then flattens; after 4–6 eval points the knee is
visible → read off steps-to-plateau (≈ ±30%). Re-run per condition for the small classes, and
**stop the small ones at their overfit knee** (FID stops improving / samples start memorizing),
not at the normal-class step count.

## Step C — budget table (steps as the parameter; it/s from Step A)
`hours = steps / it_s / 3600`; `cost = hours × $/hr × 1.2`.

### Named target: **100k steps** (mid of the ~50k–150k bracket for the larger conditions)
*(parameter — change it once Step B's extrapolation lands; the row is fixed by the measured it/s)*

| measured it/s | GPU-hours | A6000/A40 ($0.44/hr) | A100 ($1.19/hr) | H100 ($2.69/hr) |
|---:|---:|---:|---:|---:|
| 2 | 13.9 h | **$7**  | $20 | $45 |
| 4 |  6.9 h | **$4**  | $10 | $22 |
| 8 |  3.5 h | **$2**  | $5  | $11 |

### Rescale by step target (tier-independent GPU-hours)
| steps | it/s=2 | it/s=4 | it/s=8 |
|---:|---:|---:|---:|
| 30k  |  4.2 h | 2.1 h | 1.0 h |
| 60k  |  8.3 h | 4.2 h | 2.1 h |
| 100k | 13.9 h | 6.9 h | 3.5 h |
| 150k | 20.8 h | 10.4 h | 5.2 h |
| 250k | 34.7 h | 17.4 h | 8.7 h |

*(exact arithmetic; 1.2× contingency on cost. The LDM is cheaper per run than the VAE because
it's counted in ~10⁵ steps, not millions of images — but the real spend = the step count Step B
extrapolates, which can land anywhere in this grid. It is **one** conditional train, so no ×N.)*

## Decision / cap rule (how to budget without a pre-committed hour count)
1. After Step A, the it/s row is fixed → the cost is `f(steps)` only.
2. Set a **GPU-hour cap** from the table (e.g. "≤ 150k steps").
3. Train with **periodic FID + sample-grid checkpoints**; **stop on plateau or cap**, keep the best checkpoint.
4. For `cardiomegaly`/`effusion`: stop at the **per-condition overfit knee** — more steps there buy memorization, not quality.
5. Write the measured it/s + extrapolated steps + the cap into the cost memo / run log.

## Tasks
- [ ] ⚠️ (Step A) `ldm.profile` on noise latents → record it/s + peak VRAM + max batch (kills §B EST)
- [ ] ⚠️ (Step B) after `03-vae-train`: short `normal`-condition pilot with FID-every-5k → record the FID-vs-steps slope
- [ ] ⚠️ Extrapolate steps-to-plateau (per condition); flag the small-class overfit knee
- [ ] ⚠️ Drop measured it/s + step target into the budget table → set a GPU-hour cap
- [ ] ⚠️ Record it/s + extrapolated steps + cap into `runs/pod_provision.md` and the cost memo

## Recommended skill
— custom; no skill fits (short profiling + pilot rental on the locally-built `ldm` loop).
   — alt: `/analyze-run` to read the FID curve once the pilot is logging.

## Engagement Instructions
```
# DO THIS — cost first (noise latents), convergence after the VAE exists
$ python -m ldm.profile --latent 4x128x128 --precision bf16 --sweep        # → it/s, VRAM, max batch
$ bash scripts/train_ldm.sh --max-steps 30000 --eval-fid-every 5000        # → early FID slope

# GET THAT — a measured per-step cost + an extrapolated step budget with a cap
$ cat runs/pod_provision.md   # expect: LDM it/s, peak VRAM, max batch, extrapolated steps, GPU-hour cap
```
