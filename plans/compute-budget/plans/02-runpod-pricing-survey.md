# 💵 RunPod Pricing Survey

## Description
Capture current RunPod GPU and storage pricing across the VRAM range the sizing
table demands, so cost can be computed from $/hr.

## Purpose
Turns GPU-hours into dollars. Serves Objective 3, Goal 2 / DoD 2.

## Goal
A dated pricing table: candidate GPUs covering the required VRAM, spot/community
vs. secure-cloud $/hr, network-volume $/GB-month, egress cost.

## Tasks
- [ ] ⚠️ From the sizing table's peak VRAM, list the GPU tiers that fit (e.g. 24GB 4090, 48GB A40, 40/80GB A100) <!-- candidate list snapshotted in runpod-pricing.md, but VRAM target not yet fixed (sizing table pending) -->
- [~] ⚠️ Record spot/community and ~~secure-cloud $/hr for each candidate GPU, dated~~ — **on-demand $/hr for 7 GPUs captured 2026-06-09 in runpod-pricing.md**; spot/community still open
- [ ] ⚠️ Record network-volume $/GB-month and any egress/bandwidth cost
- [ ] ⚠️ Note availability caveats (spot interruption, region, current stock)

## Findings so far (2026-06-09)
- On-demand $/hr for 7 GPUs captured in `runpod-pricing.md` (dated); FX ≈ R16.5.
- Spot/community $/hr, volume $/GB-mo, and egress **not** captured — DoD 2 not met.

## Engagement Instructions
```
$ cat plans/compute-budget/runpod-pricing.md   # GPU × {spot,secure} $/hr + volume $/GB-mo + egress, dated YYYY-MM-DD
```

## Recommended skill
▶ `/deep-research "current RunPod GPU pricing 4090/A40/A100 spot vs secure, network volume, egress"` ✅ — fan-out web search + cited synthesis.
