---
name: PoE composition
kind: method
grounded: scripts/generate.py:181-182
confidence: verified
---
# Product-of-experts composition (is it sound in a continuous latent?)

**Gist.** The both-disease image is made by combining two disease experts *in noise (ε) space* via
classifier-free guidance — **not** by interpolating between two points in the latent.

**Explainer.** One UNet predicts the noise under each conditioning; at every denoising step the
guidance vectors are added:

```
ε = ε_anchor + w·(ε_cardio − ε_anchor) + w·(ε_effusion − ε_anchor)
```

This is the composable-diffusion / product-of-experts-of-scores formulation (Liu et al.): each
expert contributes a score (gradient of a log-density), and adding scores multiplies the
distributions. The sampler follows this *combined score field* back through the reverse-diffusion
trajectory. So the "off-manifold holes" risk lives not on a straight line in latent space but in
the *trajectory* under the summed field: when the experts disagree, the path can enter a region the
data never populated and the decoded image renders only one disease — which is exactly the NULL
co-presence yield we measured ([G1]/Exp03). The fix is a stronger, better-trained cardio score
(Acts 2, 6) + higher guidance ([[cfg-guidance]]), not different geometry. A continuous KL-regularized
latent is required because diffusion needs a smooth space to add/remove Gaussian noise and define
scores (a discrete/VQ latent wouldn't support the ε arithmetic); we train on μ to dodge posterior
collapse ([G5b]).

**Deep-dive.** `generate.py:181-182,282`. Entanglement: a 7:1 effusion:cardio imbalance skewed
capacity toward effusion; generated cardio *leaked* to the effusion head ([G6a]) → rebalancing is a
disentanglement move.

Related: [[cfg-guidance]], [[best-of-n]], [[ctr]]
