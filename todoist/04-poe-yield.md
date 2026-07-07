# 🎯 04 · PoE Yield

> ✅ **DONE (2026-06-25).** Gate-3 co-presence yield **NULL** — N=2000, floor-corrected **9.3%** / confident **9.1%** < 10% (raw 13.2%). PoE renders cardio 25.4% (vs 4.2% cardio-alone) but barely clears CTR=0.5 → cardio channel is the bottleneck. **→ Fires Exp 5 retrain.** Decision now fully determined: reuse VAE + effusion, RETRAIN cardio. (w_c sweep 3.0/4.0 running as pre-retrain confirmation.)

## Reference while you do it
- 📄 Plan: plans/04-poe-yield.md
- 🧭 Authority: EXPERIMENTS.md (EXP-03, Gate 3) · docs/EVAL_PHILOSOPHY.md (yield)
- 💻 Reuse: scripts/generate.py + src/inference/cfg.py (cfg_compose) · model_step0040000.safetensors · ctr_overlay.py

## Section context (paste into the Todoist section)
**Description:** Compose the two single-disease experts via PoE (`ε_compose = ε_uncond + w_c·(ε_cardio − ε_uncond) + w_e·(ε_effusion − ε_uncond)`, w_c=2.0/w_e=1.0), sample ≥ 2000, and measure the Gate-3 pass-rate: fraction with CTR > 0.5 AND CP blunted at once. This is where cardio under-rendering shows up as a number.
**Objective:** The Gate-3 co-presence certificate / augmentation volume for the +PoE arm (`local:yield`, DoD 4).
**Goal:** Yield (%) + CI per setting + the CTR×CP scatter with the pass region shaded.
**Verify (whole leaf):** `grep -E "support|inconclusive|null" results/exp03_yield/verdict.md` → yield ≥ 30% → support; < 10% → fire Exp 5 retrain.
**▶ Recommended prompt:** — custom; no skill fits.

## Tasks (one at a time)
- [x] 🔌 Wire the PoE composer + Gate-3 yield harness — reuse `generate.py` + `cfg_compose` + `model_step0040000` (NOT the decoy `ckpts/ldm/*`); gate via `ctr_overlay.py`. Verify: runs end-to-end on 10 samples, prints yield + per-sample CTR/CP.
- [x] ✋ Spot-check a ~50-sample batch (MANUAL) — plot CTR×CP scatter + eyeball a few; confirm passes land where both signs are visible. Watch the kill-criterion (yield→0 = stop). Verify: scatter shows the pass region; passes match visible co-presence.
- [x] 📊 Sample ≥ 2000 + compute yield + figure + verdict — yield + CI + shaded scatter; apply 30%/10% rule. Verify: `verdict.md` cites yield + CI and the support/null call.
