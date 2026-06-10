# 💵 Cost Estimate & Decision Memo

**Scope:** `plans/compute-budget/` · **Status:** 🟡 DRAFT — one blank remaining
**Blocking input:** measured **512² VAE img/s** from the runpod `00-profile` step.
Every `⟨measure⟩` cell below is filled by `vae.profile` on a rented GPU; until then the
$ figures are **bands, not commitments** (per plan-03's blocker note).

Inputs: sizing → [sizing-table.md](sizing-table.md) (512² VAE row = **MEAS**); pricing →
[runpod-pricing.md](runpod-pricing.md); profiling step → [runpod-execution/plans/00-profile-and-budget.md](plans/runpod-execution/plans/00-profile-and-budget.md).

---

## 0. What is already measured (no longer estimated)
- **VAE = 49.10 M params.** 512² peak VRAM `≈ 0.2 + 10.4·B` GB (≈30 % less with grad-ckpt).
  ⇒ **48 GB is the floor; 80 GB ≈ doubles batch.** Hardware-independent — locked.
- **Input resolution is pinned to 512²** (latent 4×128×128 @ f=4). 768²/1024² are shown as the
  *cost of reopening*, not live candidates — they require reopening the latent-shape decision.
- **The LDM is resolution-invariant** (one 4×128×128 latent) ⇒ one LDM cost, not three.

## 1. The one open number
| | unit | source | status |
|---|---|---|---|
| 512² VAE throughput | img/s | `vae.profile --res 512 --sweep` on rented GPU | **⟨measure⟩** ← blocks this memo |
| 512² VAE peak VRAM | GB | measured local | ✅ 10.6 (7.4 +ckpt) |
| LDM throughput | it/s | `ldm.profile` (noise latents) or EST | 🟡 EST 2–6 |
| Pricing $/hr | $/GPU-hr | runpod-pricing.md | ✅ A40/A6000 0.44 · A100 1.19 · H100 2.69 |

---

## 2. Resolution × scenario comparison (VAE/LDM broken out)
Target: **VAE ≈ 5.6 M images (50 epochs × 112,120)**; **LDM ≈ 100 k steps** (1.2× contingency in cost).
Cells are `$VAE + $LDM` at the scenario's GPU. The VAE row is a **band over img/s** until `00-profile` fixes it.

### 512² — the live config (pinned)
| Scenario | GPU | VAE $ (img/s 8→32 band) | LDM $ (it/s 2→8 band) | + storage/setup | **subtotal** |
|---|---|---|---|---|---|
| **Cheapest-viable** | A6000/A40 48 GB ($0.44) | $26 – $103 | $2 – $7 | ~$3 | **~$31 – $113** |
| **Balanced** | A100 80 GB ($1.19) | $69 – $278 | $5 – $20 | ~$4 | **~$78 – $302** |
| **Fastest** | H100 SXM ($2.69) | $157 – $628 | $11 – $45 | ~$6 | **~$174 – $679** |

*(once `00-profile` returns the real img/s, only that column's VAE row stays — collapsing each band to one number.)*

### 768² / 1024² — cost of reopening (EST, scaled by pixel count)
| Res | VAE peak VRAM @B=1 (EST) | VAE $ multiplier vs 512² | note |
|---|---|---|---|
| 768² | ~22 GB | ~2.2× | needs ≥48 GB w/ grad-ckpt; ~2× the VAE spend |
| 1024² | ~38 GB | ~4× | grad-ckpt mandatory; ~4× the VAE spend |

LDM cost is **unchanged** across all three (resolution-invariant latent).

## 3. Other line items
- **Storage:** 512² corpus + latents + ckpts ≈ **30 GB** network volume (sizing §C) → ~$2/mo ($0.07/GB·mo); negligible for a multi-day run.
- **Setup / idle overhead:** provisioning + 30 GB corpus transfer + idle between steps ≈ 2–4 GPU-hrs → ~$2–5 at the chosen tier.
- **Contingency:** 1.2× in every cost above, plus assume **one rerun** of the VAE train in the worst case.

---

## 4. Recommendation (provisional — confirm after `00-profile`)
- **Resolution: 512²** — already pinned; 768²/1024² are not justified unless the recon ceiling at 512² disappoints.
- **GPU: A100 80 GB (balanced)** as the default — comfortable batch (≈8–14 @512² w/ grad-ckpt), good throughput, ~$1.19/hr. Drop to **A6000/A40 48 GB** if minimizing $ and wall-clock is irrelevant; **H100** only if hours matter more than dollars.
- **Total pipeline budget (512², A100, incl. contingency + one rerun): ~$80 – $300**, expected to land **low** in that band once img/s is measured (the local-scaled estimate points to the cheaper end).

## 5. Downstream impact of the resolution choice
- **`plans/data-foundation/`** — corpus stays at **stretch-512** preprocessing; no re-pull / re-resize needed (the 512² pipeline is already the plan). 768²/1024² would force a corpus re-export.
- **`plans/vae/`** — `VAEConfig(input_resolution=512)` stays as-is; latent 4×128×128 unchanged. A resolution change would flip `input_resolution` and rescale the latent (192²/256²) at the same f=4.

---

**Decision: train at 512² on A100 80 GB, ~$⟨T⟩ total — GO**
*(PENDING: `⟨T⟩` and final GO/NO-GO confirmed once `runpod-execution/00-profile` returns the measured 512² img/s. Current best estimate ~$80–300; VRAM/tier and resolution are already settled.)*
