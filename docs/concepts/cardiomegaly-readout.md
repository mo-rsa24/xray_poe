---
name: Cardiomegaly readout
kind: epistemics
grounded: scripts/validate_individual_synthetics.py
confidence: verified
---
# How we know a generated image "has cardiomegaly" (and what a label even means)

**Gist.** Two independent judges must agree — the geometric CTR ([[pspnet]]) and the appearance
classifier ([[presence-classifier]]) — and even then the *label* is a noisy radiologist read, not
ground-truth anatomy.

**Explainer.** (1) Geometric judge: PSPNet measures CTR; > 0.5 = enlarged. (2) Appearance judge:
the DenseNet cardio sigmoid. We only trust "this synthetic is cardiomegaly" when both agree, which
is why Exp09 uses both and reports where they diverge. But an NIH "Cardiomegaly" label is a
radiologist's read (or NLP-mined from a report), inter-rater noisy, and on AP films the heart is
magnified so CTR over-reads. So the honest claim is "it matches the real-cardio CTR distribution
(median 0.536) and reads positive to a classifier calibrated on real cardio" — *not* "it is
definitively diseased". This epistemic caveat is why the paper leads with a mechanistic null rather
than a clinical claim.

**Deep-dive.** `validate_individual_synthetics.py` (Exp09 body-or-no-body gate).

Related: [[pspnet]], [[presence-classifier]], [[metrics]]
