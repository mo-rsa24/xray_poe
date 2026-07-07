---
name: Presence classifier
kind: instrument
grounded: metrics/presence_classifier.py:112
confidence: verified
---
# Presence classifier — the "judged"

**Gist.** A DenseNet-121 with two *independent* sigmoid heads [cardio, effusion] that reads an
image and says how strongly each disease is present — it's the thing being *judged*, not the
judge.

**Explainer.** torchxrayvision DenseNet-121 (`densenet121-res224-nih` weights) + a small MLP head
(Linear→ReLU→Dropout→Linear), 224×224 grayscale, xrv-normalized. Backbone mostly frozen; only the
last block (`denseblock4` + `norm5`) is fine-tuned. The output is **two independent sigmoids**, not
a softmax: cardiomegaly and effusion are *not* mutually exclusive (a patient can have both, either,
or neither), so each head is an independent Bernoulli decision (sigmoid + binary cross-entropy). A
softmax would force the two to compete for one probability mass — wrong for comorbidity, and it's
why this classifier can even represent a both-disease image. **Both-disease images are held out of
its training on purpose**, so it's honest about co-presence rather than tuned to it.

**Deep-dive.** Sigmoid: `presence_classifier.py:112` (`torch.sigmoid(model(batch))`). Preprocess:
`:75-82`. Youden-optimal thresholds stored in the checkpoint: `:178-179`. Bootstrap CI: `:155-161`.

Related: [[metrics]], [[cardiomegaly-readout]], [[grad-cam]]
