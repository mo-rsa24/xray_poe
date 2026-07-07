# VAE Round-Trip Gate (Go/No-Go)

**Description:** Encode ~50 VinDr films through our VAE, decode, score |ΔCTR| + lower-zone fidelity — a go/no-go before pretrain compute.
**Objective:** Decide "keep our VAE" vs the foreign-VAE fallback (DoD 4).
**Goal:** a PASS/FAIL verdict recorded.
**Verify:** median |ΔCTR| ≤ ~0.02 → PASS; else FAIL routes to foreign-VAE.

## Tasks
- [x] Sample ~50 border-visible VinDr films — ✅ done (sample_ids.txt 50 lines)
- [x] Round-trip through our VAE + score with the instrument — ✅ done (roundtrip_metrics.csv + 50 pair images; ctr_overlay.py extended with basal/hemi instruments; 1,952-film cardio scan run)
- [x] Record the go/no-go decision — ✅ done (DECISION.md: PASS, median |ΔCTR|=0.009)
