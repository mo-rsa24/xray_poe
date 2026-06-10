# RunPod Pricing Snapshot

**Captured:** 2026-06-09 (source: RunPod pricing screenshots)
**FX assumption:** 1 USD ≈ R16.5 (placeholder — use live USD/ZAR at purchase time)

## On-demand GPU $/hr

| GPU          |   VRAM | USD/hr | ZAR/hr (@16.5) |
| ------------ | -----: | -----: | -------------: |
| 2× A40       |  96 GB |  $0.88 |          R14.5 |
| A100 PCIe    |  80 GB |  $1.39 |          R22.9 |
| A100 SXM     |  80 GB |  $1.49 |          R24.6 |
| MI300X       | 192 GB |  $1.99 |          R32.8 |
| RTX PRO 6000 |  96 GB |  $2.09 |          R34.5 |
| H100 SXM     |  80 GB |  $3.29 |          R54.3 |
| H200 SXM     | 141 GB |  $4.39 |          R72.4 |

## Still open (DoD 2 not yet met)
- **Spot / community $/hr** — not captured; only on-demand above.
- **Network-volume $/GB-month** — not captured.
- **Egress / bandwidth cost** — not captured.
- **Availability caveats** (spot interruption, region, current stock) — not captured.

## Caveat on use
These prices alone do **not** decide a GPU. The cost ranking depends on
throughput (img/s), which is **unmeasured** — it must come from `vae.profile`
(plans/vae/plans/05). Any "best value" claim before profiling is a guess. See
the verification note in 03-cost-estimate-and-decision-memo.
