# 📈 Utility — do harvested composites help a real comorbid classifier? (GATED on O1 coverage)

## Description
Harvest the both-disease composites the model DOES produce (rejection sampling), add them to a real comorbid
classifier's training set, and report the delta honestly. Model: composites from **the best operator that
cleared O4** (or vanilla PoE-CFG best-of-N if none did). **GATED on O1 coverage:** if coverage is near-zero
there aren't enough real-quality composites to harvest and this is moot (record the gate-out). **Confound
flagged up front:** Exp-6 came back saturated (oracle−starved ΔAUC −0.012), so the test set has little headroom
— a null here is ambiguous (can't separate "fakes don't help" from "saturated test can't show it").

## Purpose
Serves O5 and DoD-4. The upside/utility case named in the paper's mission — NOT the lead (the lead is the null).

## Goal
A comorbid-classifier ΔAUC + sensitivity vs the starved baseline, with the saturation confound stated and the
harvest cost (composites kept / total generated) logged — or a recorded gate-out if coverage was near-zero.

## Tasks
- [ ] ⚠️ **GATE CHECK** — confirm O1 coverage is non-trivial. If near-zero, STOP and record the gate-out as the
  O5 result (no harvestable pool); do not spend the ~40k generations.
- [ ] ⚠️ **Harvest both-disease composites** — rejection-sample via the PSPNet-CTR verifier (keep CTR>0.5 AND CP
  blunted); log kept/total (≈2.6% yield → ~40k generations for ~1k composites). Save the pool + a manifest.
  - [ ] ⚠️ Remedy: if harvesting ~1k is too slow at 2.6%, use the best O4 operator's higher-yield pool, or cap at
    the largest pool the budget allows and report the reduced N (don't silently undershoot).
- [ ] ⚠️ **Augment + train the comorbid classifier** — starved baseline vs starved+harvested-composites, ≥3
  seeds, on the leakage-held-out real comorbid test; report AUC + sensitivity with bootstrap CIs.
  - [ ] ⚠️ Remedy: reuse `scripts/train_detector.py` (its leakage-controlled split + augment hook); if the
    comorbid test stratum is the saturated one, ALSO report on a non-saturated stratum if one exists.
- [ ] ⚠️ **Report honestly** — ΔAUC/sens vs starved WITH the Exp-6 saturation confound stated in the same
  breath; a flat/negative delta is reported, not over-read as "fakes are useless."

## Engagement Instructions
```bash
# harvest: rejection-sample composites, PSPNet-CTR verifier keeps both-disease → pool + manifest
python scripts/gate3_yield.py ... --out results/poe_composition/harvest/   # + keep-both selection
# train: starved vs starved+harvested, 3 seeds, leakage-held-out comorbid test
python scripts/train_detector.py --augment <harvested-manifest> --seed {0,1,2} ...
```
Verify: the GATE decision is recorded; kept/total logged; a 2-arm (± harvested) AUC+sens table with 3-seed CIs;
the Exp-6 saturation confound is stated alongside the delta; a null is reported as ambiguous, not conclusive.

## Recommended skill
— custom; `scripts/gate3_yield.py` (harvest) + `scripts/train_detector.py` (augmented comorbid classifier).
