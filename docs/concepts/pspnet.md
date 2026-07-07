---
name: PSPNet
kind: instrument
grounded: metrics/eval_checkpoint.py:114-151
confidence: verified
---
# PSPNet (the CTR/CP segmenter) — the "judge"

**Gist.** A pretrained anatomical segmenter that outlines heart and lungs so we can measure the
cardiothoracic ratio geometrically — it *is not* a classifier, it's the independent geometric judge.

**Explainer.** torchxrayvision's ChestX-Det PSPNet segments heart + left/right lung from an X-ray.
CTR = widest horizontal heart span ÷ widest thorax span, drawn as two horizontal rules on the
overlay. CTR > 0.5 is the classic cardiomegaly sign; real cardio sits at a median of **0.536**. It
is validated on real films, so a disagreement between it and the [[presence-classifier]] is
informative (that two-judge agreement is how we call a synthetic image cardiomegaly — see
[[cardiomegaly-readout]]).

**Deep-dive.** In-memory scorer `score_decoded_array` in `eval_checkpoint.py:114-151`; overlay
drawing in `ctr_overlay.py`. CP = costophrenic angle (basal corner; blunting = effusion sign).

Related: [[ctr]], [[grad-cam]], [[cardiomegaly-readout]]
