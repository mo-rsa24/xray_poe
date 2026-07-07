---
name: CTR
kind: reference
grounded: metrics/eval_checkpoint.py:114-151
confidence: verified
---
# CTR (cardiothoracic ratio)

**Gist.** Heart width ÷ chest width, measured off the X-ray by the segmenter; > 0.5 is the classic
cardiomegaly sign. Real cardio median ≈ **0.536**.

**Explainer.** Computed geometrically by [[pspnet]] (widest heart span ÷ widest thorax span), not by
a classifier. It's the project's primary *geometric* evidence for heart enlargement, reported as a
median plus the share above 0.5 / 0.53 / 0.55. Related basal sign: **CP** (costophrenic angle)
blunting → effusion.

**Deep-dive.** `eval_checkpoint.py:114-151` (`score_decoded_array`), `ctr_overlay.py` (overlay).

Related: [[pspnet]], [[cfg-guidance]], [[best-of-n]]
