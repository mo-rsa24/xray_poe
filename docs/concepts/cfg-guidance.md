---
name: CFG-guidance
kind: method
grounded: scripts/cardio_enhance_diag.py
confidence: verified
---
# Classifier-free guidance strength `w`

**Gist.** A sampling knob: higher `w` pushes the generated image harder toward the conditioning
(here, toward "cardio"), which makes the drawn heart bigger.

**Explainer.** Classifier-free guidance forms the sampled noise as
`ε = ε_uncond + w·(ε_cond − ε_uncond)` — at `w=1` you follow the conditional; raising `w` amplifies
the disease direction. The **anchor** is the reference pushed away from: `null` (blank) or `normal`
(a healthy chest, so the push is "healthy → sick"). On the v4-60k model, sweeping `w` on cardio
lifts CTR monotonically: 0.477 (w1) → 0.538 (w4), reaching the real reference; `w6` reaches CTR
0.561 and clears the detector gate (sens 0.750) at 1× cost. The `normal` anchor is a **dead lever**
here — it raises geometric CTR but *lowers* detector sensitivity, so it is not used. Guidance is the
"free inference-time fix" the Act-8 checks pointed to.

**Deep-dive.** Sweep: `cardio_enhance_diag.py a`; per-arm comparison `cardio_enhance_compare.py` →
`results/section4/cardio_enhance/compare_levers.csv`.

Related: [[poe-composition]], [[best-of-n]], [[ctr]]
