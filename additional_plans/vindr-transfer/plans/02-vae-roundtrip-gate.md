# 🔬 VAE Round-Trip Gate (Go/No-Go)

## Background
The spine keeps **our** VAE so Exp-2's codec ceiling carries untouched and Stage-1/Stage-2 share one latent space. That only holds if our NIH-trained VAE faithfully round-trips *VinDr* films at the heart border and lower zone. Cheap go/no-go before any pretrain compute. Prior evidence is encouraging (effusion's lower zone renders fine; Exp-2 |ΔCTR| 0.011 on NIH) but was never measured on VinDr.

## Description
Encode ~50 VinDr films through `ckpts/vae_step0025000.pt`, decode, score |ΔCTR| + lower-zone fidelity vs the original with the CTR/CP instrument.

## Purpose
Serves Objective 2 and DoD item 4 — the gate deciding "keep our VAE" vs the Foreign-VAE fallback.

## Goal
A pass/fail verdict with the codec decision recorded.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks

- [x] ✅ **Sample ~50 border-visible VinDr films** — ✓ verified (`results/vindr_transfer/vae_gate/sample_ids.txt` 50 lines; 46/50 turned out border_visible=Y in scoring pass)
  - `$ run:`
    ```bash
    $PY -c "
    import pandas as pd
    df=pd.read_csv('data/vindr512/presence_manifest.csv')
    df=df.sample(50, random_state=42)
    df['image_id'].to_csv('results/vindr_transfer/vae_gate/sample_ids.txt', index=False, header=False)
    print('wrote 50 ids')"
    ```
  - `✓ verify:` `wc -l results/vindr_transfer/vae_gate/sample_ids.txt` = 50.

- [x] ✅ **Round-trip through our VAE + score with the instrument** — ✓ verified (`scripts/vindr_vae_gate.py` written + run; `results/vindr_transfer/vae_gate/roundtrip_metrics.csv` 50 rows; `figures/vindr_vae_gate_panel.png` + `results/vindr_transfer/vae_gate/pairs/` 50 individual PSPNet-annotated side-by-side pair images)
  - `⊳ note:` `scripts/ctr_overlay.py` extended in this session with basal opacity, hemidiaphragm elevation, and lung coverage instruments + visual markers (amber basal tint, dome lines); confound audit of all 1,952 cardio∧¬effusion VinDr films run via `scripts/vindr_cardio_scan.py` → `results/vindr_transfer/cardio_scan/`; `border_visible=Y` confirmed as the sole validated diet filter (basal/hemi thresholds deferred to VAE-v2 track).
  - `▶ prompt:` `Write scripts/vindr_vae_gate.py: for each image_id in results/vindr_transfer/vae_gate/sample_ids.txt, load the VinDr PNG from /home/molefe/data/vindr512/train/, encode→decode through ckpts/vae_step0025000.pt (reuse the encode/decode in scripts/vae_recon_pilot.py, fp32 decode to avoid black pixels, scale_factor data/latents/scale_factor.pt), and score original-vs-recon with scripts/ctr_overlay.py (CTR, CP angle, border_visible, peri-cardiac + basal opacity). Write per-film |ΔCTR| / CP-flip / lower-zone fidelity to results/vindr_transfer/vae_gate/roundtrip_metrics.csv and a before/after panel to figures/vindr_vae_gate_panel.png. Then run it.`
  - `✓ verify:` every sampled film has a metric row; the panel renders.

- [x] ✅ **Record the go/no-go decision** — ✓ verified (`results/vindr_transfer/vae_gate/DECISION.md` written; verdict PASS)
  - `⊳ decide:` Read the median |ΔCTR| (`$PY -c "import pandas as pd; print(pd.read_csv('results/vindr_transfer/vae_gate/roundtrip_metrics.csv')['abs_dCTR'].median())"`). **PASS** if ≤ ~0.02 and the lower zone is visibly preserved (consistent with Exp-2's 0.011) → our VAE is the codec, Exp-2 carries. **FAIL** → trigger the parked **Foreign-VAE path (+ Exp-2 re-run)**. Write the verdict to `results/vindr_transfer/vae_gate/DECISION.md`.
  - `✓ verify:` `DECISION.md` states PASS/FAIL + median |ΔCTR| + routed next step.

## Recommended skill
— custom; project codec check (`scripts/vae_recon_pilot.py` + `scripts/ctr_overlay.py`).
   alt: `/analyze-figure figures/vindr_vae_gate_panel.png` ✅.

## Engagement Instructions
```bash
$ column -s, -t results/vindr_transfer/vae_gate/roundtrip_metrics.csv | head
$ grep -E "PASS|FAIL" results/vindr_transfer/vae_gate/DECISION.md
# expect: median |ΔCTR| ≤ ~0.02 for PASS; FAIL routes to the Foreign-VAE fallback (PARKING_LOT.md)
```
