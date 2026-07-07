# Concept KB — index

Atomic, non-ordered notes for the instruments and methods the timeline references. Each has three
depths: **Gist** (one line, inlined in the spine) → **Explainer** (plain-English) → **Deep-dive**
(code-verified, `file:line`). The spine cites by name, e.g. `→ see [PSPNet]`. These are the split-out
successors of the old Methods appendix (M1–M5) plus the new levers.

| Note | What it is |
|------|-----------|
| [presence-classifier](presence-classifier.md) | DenseNet two-sigmoid classifier (the "judged") — ex-M1 |
| [pspnet](pspnet.md) | anatomical CTR/CP segmenter (the "judge") — ex-M2 |
| [grad-cam](grad-cam.md) | effusion localisation heat-map + bbox — ex-M2 |
| [poe-composition](poe-composition.md) | ε-space CFG composition, the manifold question — ex-M3 |
| [cardiomegaly-readout](cardiomegaly-readout.md) | how we call a synthetic "cardiomegaly"; what a label means — ex-M4 |
| [metrics](metrics.md) | sens/spec/FPR/AUC/ROC, Youden, bootstrap CI, strata — ex-M5 |
| [ctr](ctr.md) | cardiothoracic ratio definition |
| [cfg-guidance](cfg-guidance.md) | guidance strength `w` lever |
| [best-of-n](best-of-n.md) | rejection-sampling lever |
