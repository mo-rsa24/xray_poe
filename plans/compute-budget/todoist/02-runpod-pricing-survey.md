# 02 · RunPod Pricing Survey

[⌂ Index](00-INDEX.md) · [← prev](01-workload-sizing.md) · [next →](03-cost-estimate-and-decision-memo.md)

## Reference while you do it
- 📄 Plan: plans/compute-budget/plans/02-runpod-pricing-survey.md

## Section context (paste into the Todoist section)
**Description:** Capture current RunPod GPU and storage pricing across the VRAM range the sizing table demands, so cost can be computed from $/hr.
**Objective:** Turn GPU-hours into dollars by sourcing dated, candidate-fitting prices.
**Goal:** A dated pricing table: candidate GPUs covering the required VRAM, spot/community vs. secure-cloud $/hr, network-volume $/GB-month, egress cost.
**Verify (whole leaf):** `cat plans/compute-budget/runpod-pricing.md` shows GPU × {spot,secure} $/hr + volume $/GB-mo + egress, every price stamped `YYYY-MM-DD`.
**▶ Recommended prompt:** `/deep-research "current RunPod GPU pricing 4090/A40/A100 spot vs secure, network volume, egress"` ✅ — fan-out web search + cited synthesis.

## Tasks (one at a time)
- [ ] From the sizing table's peak VRAM, list the GPU tiers that fit (e.g. 24GB 4090, 48GB A40, 40/80GB A100)
- [ ] Record spot/community and secure-cloud $/hr for each candidate GPU, dated
- [ ] Record network-volume $/GB-month and any egress/bandwidth cost
- [ ] Note availability caveats (spot interruption, region, current stock)
