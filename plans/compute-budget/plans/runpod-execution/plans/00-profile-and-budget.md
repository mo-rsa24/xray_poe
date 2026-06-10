# 📐 Profile on the Rented GPU → Lock the Budget (FIRST on-pod step)

**Scope:** `plans/compute-budget/plans/runpod-execution/` · **Status:** ⚠️ pending · **Runs:** first, before the full provision + corpus transfer

## Background
The cost memo (`compute-budget/03`) picks a GPU tier from an **estimated** VAE
throughput — but throughput can't be measured without a GPU, and the local 8 GB
RTX 4070 thrashes at 512² (spillover), so it gave a valid *VRAM* number but no
valid *img/s*. The two estimate sources disagree ~7× (~8 vs ~50 img/s), which is
the difference between a ~$100 and a ~$600 VAE train. This step closes that gap
for ~$0.20: rent the **cheapest 48 GB** box briefly, run the already-built
profiler on noise tensors (no corpus needed), read off the real img/s, and turn
the budget band into a single number **before** transferring 40 GB and committing
to the long train.

## What is already MEASURED (local, RTX 4070 8 GB — see `plans/vae/profiling-notes.md`)
- **VAE = 49.10 M params.** Peak VRAM at 512²: `peak(B) ≈ 0.2 + 10.4·B` GB (≈30 % less
  with `--grad-checkpoint`). ⇒ **48 GB floor; 80 GB ≈ doubles batch.** This part needs
  no re-confirmation — it is hardware-independent.
- Max batch @512²: **48 GB → ~4 (plain) / ~8 (ckpt); 80 GB → ~7 / ~14.**
- **Unknown until this step:** steady-state **img/s** at 512² on a real datacenter GPU.

## Purpose
Replace the estimated throughput with a measured one so the cost is firm before the
expensive steps. Serves the cost memo's GO decision and de-risks every downstream
GPU-hour. This is *cheap insurance*, not a commitment to the full run.

## Goal
A recorded `peak VRAM | img/s | max-batch` triple for the chosen resolution/precision
on a real GPU, plus the **exact** `vae.budget` figure for the named convergence target —
written into the provisioning record so the tier choice is justified by measurement.

---

## Step A — profile (rent the cheapest 48 GB box, ~15 min ≈ $0.11)
The profiler runs unchanged from the `vae` scope; it needs only noise tensors.

```
# on a freshly rented A40/A6000 48 GB pod, repo cloned + deps installed (see 01-provision)
$ python -m vae.profile --res 512 --precision bf16 --sweep
# GET THAT → "peak VRAM X.X GB | Y.Y img/s" + the max batch that fits 48 GB
$ python -m vae.profile --res 512 --batch <max-that-fit> --precision bf16 --grad-checkpoint
# GET THAT → the steady-state img/s at the batch you'll actually train at
```

Record the measured `img/s` (and peak VRAM, max batch) into `runs/pod_provision.md`.

## Step B — turn img/s into $ (the budget table)
`hours = images_seen / img_s / 3600`; `cost = hours × $/hr × 1.2 (contingency)`.
GPU-hours depend **only** on img/s; cost adds the tier rate. Run it directly:

```
$ python -m vae.budget --img-s <measured> --epochs 50 --n 112120 --rate <tier $/hr>
# GET THAT → "≈ H GPU-hours, ≈ $C  (measured img/s; assumed epochs, N, rate)"
```

### Named convergence target: **50 epochs over NIH ChestX-ray14 (112,120 imgs) = 5,606,000 images seen**
*(parameter — change `--epochs`/`--n` to rescale; the row you land on is set by the measured img/s)*

| measured img/s | GPU-hours | A6000/A40 48 GB ($0.44/hr) | A100 80 GB ($1.19/hr) | H100 SXM ($2.69/hr) |
|---:|---:|---:|---:|---:|
| 8  | 194.7 h | **$103** | $278 | $628 |
| 16 |  97.3 h | **$51**  | $139 | $314 |
| 32 |  48.7 h | **$26**  | $69  | $157 |
| 64 |  24.3 h | **$13**  | $35  | $79  |

*(exact, from `vae.budget`, 1.2× contingency. Faster tiers post higher img/s **and** higher $/hr,
so $-per-image is similar across tiers — pick the cheap 48 GB box for cost, A100/H100 for wall-clock.
Once Step A gives the real img/s, only that GPU's actual row is the live number.)*

**Convergence length is the other lever** (when the SSIM/LPIPS recon gate trips). 50 epochs is a
planning figure, not a measurement — cap the run by the gate, and re-cost with the real epoch count
once the recon curve is visible.

## Decision rule (which GPU)
1. After Step A, drop the measured img/s into the table → read cost at each tier.
2. If wall-clock is irrelevant → **cheapest 48 GB (A40/A6000)** wins on $.
3. If the recon-gate run must finish fast, or batch ≥ 8 at 512² is wanted → **A100 80 GB** (best balance), **H100** only if hours matter more than dollars.
4. Write the chosen tier + the measurement that justifies it into the cost memo / provisioning record, then proceed to `02-corpus-transfer`.

## Tasks
- [ ] ⚠️ Rent the cheapest 48 GB pod; clone repo + install pinned deps (minimal `01-provision`)
- [ ] ⚠️ `vae.profile --res 512 --precision bf16 --sweep` → record peak VRAM + img/s + max batch
- [ ] ⚠️ Re-profile at the train batch (`--grad-checkpoint`) → record steady-state img/s
- [ ] ⚠️ Run `vae.budget` with the measured img/s for the named target at each candidate $/hr
- [ ] ⚠️ Pick the tier per the decision rule; write img/s + tier + justification into `runs/pod_provision.md` (and back into `compute-budget/03` cost memo)

## Recommended skill
— custom; no skill fits (short profiling rental + the locally-built `vae.profile`/`vae.budget`).

## Engagement Instructions
```
# DO THIS — cheapest 48 GB pod, then:
$ python -m vae.profile --res 512 --precision bf16 --sweep
$ python -m vae.budget --img-s <measured> --epochs 50 --n 112120 --rate 0.44

# GET THAT — a firm GPU-hours + $ figure, and the table row that matches the real img/s
$ cat runs/pod_provision.md     # expect: measured img/s, peak VRAM, max batch, chosen tier + $ justification
```
