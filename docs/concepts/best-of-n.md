---
name: Best-of-N
kind: method
grounded: scripts/cardio_enhance_diag.py
confidence: verified
---
# Best-of-N (rejection sampling toward a bigger heart)

**Gist.** Draw N candidate images for one slot, keep the one with the biggest heart (highest CTR),
throw the rest away — a free inference-time way to bias toward severe cardiomegaly.

**Explainer.** Selection is by the *independent geometric* CTR ([[pspnet]]), **not** by the detector
— so when we then read the [[presence-classifier]] on the kept set, the number is honest (we didn't
select on the thing we're measuring). On the v4-60k model, `w4 best-of-8` gives CTR median 0.610 and
detector sensitivity **0.875**, which *beats* real cardio (0.806) — both gates cleared with zero
retraining. Caveat: best-of-8 CTR is high **by construction** (we keep the max-CTR draw), which is
why the detector read (not the CTR) is the real utility check. Cost: 8× generation. Cheaper
alternative that still clears the 0.70 gate: plain hotter guidance `w6` at 1× cost ([[cfg-guidance]]).

**Deep-dive.** `cardio_enhance_diag.py c` (reachability probe); lever comparison
`cardio_enhance_compare.py` → `compare_levers.csv`.

Related: [[cfg-guidance]], [[pspnet]], [[poe-composition]]
