# 03 · Cost Estimate & Decision Memo

[⌂ Index](00-INDEX.md) · [← prev](02-runpod-pricing-survey.md)

## Reference while you do it
- 📄 Plan: plans/compute-budget/plans/03-cost-estimate-and-decision-memo.md

## Section context (paste into the Todoist section)
**Description:** Combine the sizing table and pricing table into per-resolution cost estimates, then write the recommendation memo.
**Objective:** Turn the numbers into a resolution + GPU recommendation, a total pipeline budget, and a go/no-go.
**Goal:** A cost memo: per-resolution × scenario (cheapest-viable, fastest) GPU-hours and dollars (VAE/LDM broken out), totals incl. storage/setup/contingency, a recommended resolution + GPU + budget, the go/no-go, and the downstream-impact note on data-foundation + vae.
**Verify (whole leaf):** `cat plans/compute-budget/COST-MEMO.md` ends with `Decision: train at <res> on <GPU>, ~$T total — GO / NO-GO`, above a filled resolution × scenario comparison table and a downstream-impact section naming data-foundation + vae.
**▶ Recommended prompt:** — custom; no skill fits (arithmetic synthesis + memo writing); alt: `/zettelkasten` to polish the memo prose.

## Tasks (one at a time)
- [ ] Compute GPU-hours = throughput × target steps, for VAE and LDM, per resolution
- [ ] Compute dollars = GPU-hours × $/hr for cheapest-viable and fastest scenarios, per resolution
- [ ] Add storage (resolution-dependent), pod setup/idle overhead, and a contingency buffer (reruns)
- [ ] Build the resolution × scenario comparison table (VAE/LDM broken out)
- [ ] Recommend one resolution + GPU + total pipeline budget; record the go/no-go decision
- [ ] Note the downstream updates the chosen resolution forces on data-foundation (corpus resolution) and vae (latent/resolution config)
