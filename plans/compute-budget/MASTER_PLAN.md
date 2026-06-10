# 💰 Compute Budget

## Mission
Scope and cost out renting a RunPod GPU to train the Paper 3 pipeline — the
shared VAE and the single-disease LDM — before any GPU-hours are spent. The
latent target is 4×128×128 (matched between VAE and LDM so PoE composition has
the spatial capacity to work), but the input resolution is unsettled: 512²
(f=4), 768² (f=6), or 1024² (f=8). Size both training workloads for each
resolution, survey RunPod pricing, estimate total GPU-hours and dollars
end-to-end, and recommend a resolution + GPU + total budget. A decision artifact
that picks the resolution and GPU on cost grounds, then executes the chosen
config on RunPod to produce the trained checkpoints.

## Objectives
1. Characterize the VAE training workload for each candidate resolution —
   512²/768²/1024² input into a 4×128×128 latent — in terms of peak VRAM,
   throughput (steps/sec), target steps/epochs, and dataset size. The 16×-larger
   latent and 4–16×-larger inputs likely exceed 12GB, so VRAM sizing is what
   selects the GPU tier.
2. Characterize the single-disease LDM training workload in the same 4×128×128
   latent — peak VRAM, throughput, target steps/epochs — noting diffusion
   training is typically longer than the VAE and is often the larger line item.
3. Survey RunPod's offering — candidate GPU tiers across the VRAM range the
   workloads demand (e.g. 24GB 4090, 48GB A40, 40/80GB A100), spot/community
   vs. secure-cloud $/hr, network-volume $/GB-month, and egress cost.
4. Estimate GPU-hours (throughput × target steps) and dollars (GPU-hours ×
   $/hr) for VAE and LDM separately, per resolution, under a cheapest-viable
   and a fastest scenario.
5. Account for the whole RunPod workflow, not just training time — corpus
   upload and storage (which grows with resolution), pod setup/idle overhead,
   checkpoint download, and a contingency buffer for reruns (e.g. a failed
   overfit-sanity).
6. Recommend a resolution + GPU configuration + total pipeline budget (VAE +
   LDM), with a go/no-go decision — and flag the downstream updates the chosen
   resolution forces on data-foundation (corpus storage resolution) and vae
   (latent/resolution config).
7. Once the config is chosen, provision it on RunPod, transfer the corpus, run
   the VAE and LDM trains, and retrieve the checkpoints — turning the costed
   decision into trained artifacts.

## Goals
1. VAE and LDM workloads sized per resolution — VRAM, throughput, target
   steps/epochs, dataset size — for 512²/768²/1024² at 4×128×128, written down.
2. Pricing table assembled — candidate GPUs spanning the required VRAM, spot vs.
   secure $/hr, plus volume and egress costs, dated.
3. Cost-per-resolution comparison produced, VAE and LDM broken out, with a
   cheapest-viable + fastest scenario each and an explicit contingency margin.
4. A single recommended resolution + GPU + total pipeline $ figure + go/no-go
   decision recorded, with the cross-scope impact on data-foundation/vae noted.

## Expected Outcome
A short cost memo in the repo: a resolution × GPU cost comparison, VAE and LDM
broken out, concluding "train at <res> on GPU X at ~$Y/hr, expect ~Z_vae +
Z_ldm GPU-hours, total ≈ $T including storage, transfer, and contingency," with
the go/no-go decision and the downstream corpus-resolution implication recorded
— so training can begin without budget surprises and data-foundation knows what
resolution to target.

## Definition of Done
1. VAE and LDM workloads characterized for each candidate resolution
   (512²/768²/1024² → 4×128×128): peak VRAM, throughput, target steps/epochs,
   dataset size — written down.
2. RunPod pricing survey captured — candidate GPUs across the required VRAM range,
   spot vs. secure $/hr, network-volume $/GB-mo, egress — dated (prices drift).
3. GPU-hour and dollar estimate derived for VAE and LDM separately, per
   resolution, for cheapest-viable and fastest scenarios.
4. Total cost per scenario includes both trains, storage (resolution-dependent),
   setup/idle overhead, and a contingency buffer for reruns.
5. Recommended resolution + GPU + total pipeline budget + go/no-go decision
   recorded in a cost memo in the repo, including the downstream update required
   on data-foundation (corpus resolution) and vae (latent/resolution config).
6. Chosen RunPod config provisioned; corpus transferred; VAE and LDM trains run
   to their gates; checkpoints retrieved to `ckpts/`; pod torn down.

## Sub-Scopes
- ⚠️ plans/runpod-execution/ — "provision the costed RunPod config, run the VAE + LDM trains, retrieve checkpoints, tear down"

## Plans
- ⚠️ 01-workload-sizing.md
- ⚠️ 02-runpod-pricing-survey.md — *partial:* on-demand $/hr captured (see Artifacts)
- ⚠️ 03-cost-estimate-and-decision-memo.md — **blocked on profiling** (see Artifacts)
- ⚠️ 04-provision-and-run.md

## Artifacts & Status (2026-06-09)

Two artifacts now exist and are folded into the completion picture:

1. **`runpod-pricing.md`** — dated on-demand $/hr snapshot for 7 GPUs + FX ≈ R16.5.
   Partially satisfies plan 02 / DoD 2. **Still missing for 02 to close:**
   spot/community $/hr, network-volume $/GB-mo, egress, availability caveats.
2. **Profiling-blocked decision note** (in 03) — a found third-party cost ranking
   was reviewed and rejected as decision-grade (speed multipliers were invented,
   not measured). This makes the go/no-go a **hard downstream dependency** on
   `vae.profile` (plans/vae/plans/05 → real img/s). No $ figure or DoD 3–5 can
   close until that number exists.

**Remaining tasks, in order:**

- Finish 02: capture spot/community $/hr, volume $/GB-mo, egress, availability.
- Run `vae.profile` (vae scope) → real img/s for VAE; size LDM likewise (01).
- Feed measured img/s into `vae.budget` → GPU-hours + $ per scenario.
- Write `COST-MEMO.md` recommendation + go/no-go (03); then provision (04 /
  runpod-execution). Until profiling lands, treat every $ figure as a placeholder.
