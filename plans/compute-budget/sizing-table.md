# 🧮 Sizing Table — VAE & Single-Disease LDM

`feeds: 02-runpod-pricing-survey, 03-cost-estimate-and-decision-memo` · `method: profiling-protocol.md` · `pinned: 2026-06-09`

> **Status: ANALYTIC ESTIMATES.** Every number below the line is a first-principles
> placeholder to be **replaced** by measured values from `profiling-protocol.md`
> (local 4070 batch-sweep → linear VRAM model → TFLOPS-ratio throughput). Cells are
> tagged `EST` (analytic) until a profiling log backs them, then `MEAS`. **Do not
> cost the real train off the EST numbers** — they exist so the pricing survey has a
> VRAM target and a throughput ballpark, not a commitment.

## Structural facts (these shape the whole table)

- **The LDM is resolution-invariant.** It lives entirely in the fixed **4×128×128**
  latent, so 512/768/1024 give **one** LDM row, not three. Only the **VAE** scales
  with input resolution.
- **Minimum tier is 48 GB** (4090 dropped). So feasibility is not the question —
  **throughput/$ per completed train** is. Every candidate config fits ≥48 GB with
  room; the table ranks them by cost, not by fit.
- **Dataset = NIH ChestX-ray14**, native 1024² grayscale (~42 GB full). Conditions:
  `normal` (~60k "No Finding"), `cardiomegaly`-only (~hundreds–2k), `effusion`-only
  (~3–5k). The **LDM corpus is small and heavily imbalanced** — that bounds steps
  more than raw N does (see §Steps).

---

## A · VAE — peak VRAM & throughput (per input resolution → 4×128×128)

`batch=1, bf16, AdamW, grad-ckpt OFF, bottleneck attention = windowed; EST until profiled`

| Res | f | enc/dec stages | params | peak VRAM @B=1 | act/img (slope `s`) | max batch @48GB | max batch @80GB | img/s 4070 | img/s A100 (η.6) |
|---|---|---|---|---|---|---|---|---|---|
| **512²** | 4 | 2 | **49.1M `MEAS`** | **10.6 GB `MEAS`** | **10.4 GB `MEAS`** | **~4 `MEAS`** | **~7 `MEAS`** | n/a † `MEAS` | ~50–70 `EST` |
| **768²** | 6 | 3 | ~50M `EST` | ~22 GB `EST` | ~23 GB `EST` | ~2 `EST` | ~3 `EST` | ~2 `EST` | ~25–35 `EST` |
| **1024²** | 8 | 3 | ~51M `EST` | ~38 GB `EST` ⚠️ | ~41 GB `EST` | ~1 `EST` | ~2 `EST` | — `EST` | ~14–20 `EST` |

**512² row is MEASURED** (RTX 4070 Laptop **8 GB**, bf16, AdamW, grad-ckpt OFF, **full
mid-block attention — the decided contract**). Slope from b1=10.6 / b2=21.0 GB ⇒ s≈10.4
GB/img, base≈0.2 GB ⇒ max batch ≈ (tier−0.2)/10.4. Source: `plans/vae/profiling-notes.md`,
`logs/vae_profile.log`. 768²/1024² are EST, scaled from the 512² MEAS slope by pixel count
(activations ∝ res²) — re-profile on the rented GPU.

† **No valid 512² img/s locally:** the contract needs 10.6 GB/img and the 8 GB card thrashes
in WSL2 spillover (measured 0.2 img/s @B=1 +ckpt = artifact). A clean local rate exists only
well under 8 GB (~4.0 img/s @ 256² b2 +ckpt). Target-res throughput **must** come from the
rented GPU. Also note: the actual local card is **8 GB**, not the 12 GB assumed elsewhere here.

⚠️ 1024² @B=1 (~38 GB EST) exceeds 48 GB once batched — grad-ckpt mandatory at that res.

**Dominant VAE lever — bottleneck attention, now MEASURED.** Full self-attention at the
128×128 bottleneck (16 384 tokens) is **not** dropped/windowed in the decided architecture
(`with_*_nonlocal_attn=True`); it costs **~4.3 GB/img** of the 10.6 GB at 512² b1 (attention
OFF measures 6.3 GB). **Gradient checkpointing cuts 512² b1 from 10.6 → 7.4 GB (~30%)** and is
the recommended default on the rented GPU to lift batch size. The earlier EST (~3.5 GB,
"attention windowed") understated the VAE footprint ~3×.

---

## B · LDM — peak VRAM & throughput (one row; latent 4×128×128, all resolutions)

`bf16, AdamW, CFG dropout, grad-ckpt OFF; EST until profiled`

| Config | params | peak VRAM @B=1 | act/img (slope `s`) | max batch @48GB | max batch @80GB | it/s 4070 @B=1 | it/s A100 @B=8 (η.5) |
|---|---|---|---|---|---|---|---|
| UNet 128ch, **attn@{32,16}** | ~300M `EST` | ~5 GB `EST` | ~3.0 GB `EST` | ~13 `EST` | ~23 `EST` | ~3 `EST` | ~4–6 `EST` |
| UNet 128ch, **attn@{64,32,16}** | ~320M `EST` | ~9 GB `EST` ⚠️ | ~5.5 GB `EST` | ~7 `EST` | ~12 `EST` | ~1.5 `EST`† | ~2–4 `EST` |

⚠️/† attn@64 doubles the dominant term — the `128²`-latent attention is what makes
this LDM far heavier than a standard `64²`-latent SD UNet. **This is the config the
stale `batch≤8 fits 4070` numbers were wrong about.**

**Dominant LDM lever — attention resolution.** Whether attention runs at the 64×64
latent level is the whole story for section B. Decide it on a recon/quality basis
downstream; size **both** so the cost memo can price the choice.

---

## C · On-disk corpus (per resolution)

| Item | 512² | 768² | 1024² (native) | note |
|---|---|---|---|---|
| Image corpus (8-bit grayscale PNG, ~112k) | ~11 GB | ~24 GB | ~42 GB | linear in pixels |
| Precomputed LDM latents (4×128×128 fp16, ~65k single-disease) | ~8.5 GB | ~8.5 GB | ~8.5 GB | **resolution-invariant** |
| Checkpoints (VAE + LDM, a few snapshots) | ~5–10 GB | ~5–10 GB | ~5–10 GB | size by params |
| **RunPod volume target** | **~30 GB** | **~45 GB** | **~65 GB** | + headroom for the disk guard |

---

## D · Steps, epochs & wall-clock (worked example — replace with measured throughput)

`EST — illustrative only; recompute from MEAS throughput once profiled`

| Train | target steps | batch | img seen | epochs (corpus) | A100 wall-clock `EST` | H100 wall-clock `EST` |
|---|---|---|---|---|---|---|
| **VAE 512²** | ~150k | 32 | 4.8M | ~48 over 100k | ~20–30 h | ~8–12 h |
| **VAE 1024²** | ~150k | 16 | 2.4M | ~24 over 100k | ~70–100 h | ~25–40 h |
| **LDM (attn@{32,16})** | ~200k | 32 | 6.4M | ~100 over 65k | ~14–25 h | ~6–10 h |

> Steps are bounded by **gates, not epochs**: VAE stops at its SSIM/LPIPS recon gate,
> LDM at its single-disease FID gate. The cardiomegaly-only class being tiny (~hundreds–2k)
> risks **overfitting/mode-collapse on that condition long before** the FID gate — flag
> for class-balanced sampling; it changes effective steps more than total N does.

## E · Cost handoff (do NOT finalize here — feeds 02/03)

Cost = `Σ_train (wall-clock_hrs × $/hr_tier)`. This file supplies the **hours** (col D,
once MEAS) and the **VRAM target** (≥48 GB fits everything; 80 GB buys ~1.7× the VAE
batch and shorter VAE wall-clock). The **$/hr per tier** and the tier choice are
`02-runpod-pricing-survey` + `03-cost-estimate-and-decision-memo`'s job — not this file's.

---

## Sizing experiments (provisional EXP schema — engineering, not paper figures)

> These sit in the compute-budget scope, not the root science `EXPERIMENTS.md`. The
> `falsify_condition` is the sizing decision rule (fit / cost), not a scientific claim.

### SIZE-01: VAE resource profile (512/768/1024 → 4×128×128)
- claim_id: `local:` VAE peak-VRAM & throughput are measurable locally at batch=1 and extrapolate to the rented tiers within the protocol's tolerance.
- independent_var: input resolution {512,768,1024} × bottleneck-attention {none, windowed} × grad-ckpt {on,off}
- dependent_var: peak VRAM (GB), img/s, on-disk GB
- ablation_rows: section A rows + attention/grad-ckpt variants (protocol §4)
- metric: peak VRAM (`torch.cuda.max_memory_allocated`) + median img/s over K≥50 steps — must change with resolution if the encoder-stage hypothesis holds
- sample_size: noise data; ≥3 batch points per config for the linear fit; ≥50 timed steps after 10 warm-up
- falsify_condition: **fit/cost rule** — config "fits a tier" if predicted peak VRAM < 0.9×tier at batch≥8; *cheapest tier* = min(wall-clock_hrs × $/hr). Inconclusive if the §5 on-target confirm diverges >15% from prediction → re-fit before costing. Resolution is *rejected as too costly* if its VAE wall-clock × $/hr exceeds the recon gain budget set in 03.
- figures: VRAM-vs-batch fit line (per config); img/s-vs-resolution bar
- compute: profile on local RTX 4070 12GB; real train deferred to runpod-execution; ckpt → `/workspace/ckpts` with the §6 disk guard
- status: ⚠️ pending

### SIZE-02: LDM resource profile (4×128×128 latent, resolution-invariant)
- claim_id: `local:` the single-disease LDM cost is independent of input resolution (one latent shape) and the attn@64 lever is the dominant VRAM term.
- independent_var: attention resolution {@32,16 vs @64,32,16} × grad-ckpt {on,off} × batch
- dependent_var: peak VRAM (GB), it/s, max batch per tier
- ablation_rows: section B rows
- metric: peak VRAM + median it/s — attn@64 must raise peak VRAM materially over attn@32 if it's the dominant term
- sample_size: noise latents + random condition id + sampled timestep; ≥3 batch points; ≥50 timed steps
- falsify_condition: **fit/cost rule** — same as SIZE-01. Additionally: if attn@64 peak VRAM ≈ attn@32 (≤15% diff), the "attention is dominant" assumption is *rejected* → re-examine the architecture's cost driver. If even batch=1 attn@64 OOMs on 48GB (it should not), escalate the tier.
- figures: peak-VRAM bar attn@32 vs attn@64; max-batch-per-tier table
- compute: local 4070 (attn@64 @B=1 may need grad-ckpt or analytic-only); real train deferred to runpod-execution
- status: ⚠️ pending
