# 📊 Cost Estimate & Decision Memo

## Description
Combine the sizing table and pricing table into per-resolution cost estimates,
then write the recommendation memo.

## Purpose
The deliverable of the costing scope — turns numbers into a resolution + GPU
recommendation, a total pipeline budget, and a go/no-go. Serves Objectives 4–6,
Goals 3–4 / DoD 3, 4, 5.

## Goal
A cost memo in the repo: per-resolution × scenario (cheapest-viable, fastest)
GPU-hours and dollars (VAE/LDM broken out), totals incl. storage/setup/contingency,
a recommended resolution + GPU + budget, the go/no-go, and the downstream-impact
note on data-foundation + vae.

## Tasks
- [ ] ⚠️ Compute GPU-hours = throughput × target steps, for VAE and LDM, per resolution
- [ ] ⚠️ Compute dollars = GPU-hours × $/hr for cheapest-viable and fastest scenarios, per resolution
- [ ] ⚠️ Add storage (resolution-dependent), pod setup/idle overhead, and a contingency buffer (reruns)
- [ ] ⚠️ Build the resolution × scenario comparison table (VAE/LDM broken out)
- [ ] ⚠️ Recommend one resolution + GPU + total pipeline budget; record the go/no-go decision
- [ ] ⚠️ Note the downstream updates the chosen resolution forces on data-foundation (corpus resolution) and vae (latent/resolution config)

## Blocker note (2026-06-09)
A found third-party cost analysis was reviewed and **rejected as decision-grade**:
its GPU speed/cost ranking rests on invented "relative speed" multipliers, not
measured img/s. Two reasoning flaws: linear speed-scaling assumes compute-bound
(a VAE at batch 8–16 on small latents is often I/O/CPU-bound), and 2×A40 was
treated as one 0.9× unit (needs DDP to use both cards). Practical takeaway: the
whole cost spread is ~R200–R1100 — too small to optimize per-rand; pick an
available, low-risk card (A100/H100). **No task here is closed**: every $ figure
stays a placeholder until `vae.profile` (plans/vae/plans/05) yields a real img/s.

## Engagement Instructions
```
$ cat plans/compute-budget/COST-MEMO.md   # ends with: "Decision: train at <res> on <GPU>, ~$T total — GO / NO-GO"
```

## Recommended skill
— custom; no skill fits (arithmetic synthesis + memo writing).
   alt: `/zettelkasten` to polish the memo prose.
