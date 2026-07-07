---
name: Grad-CAM
kind: instrument
grounded: metrics/grad_cam_utils.py:105-137
confidence: verified
---
# Grad-CAM (effusion localisation)

**Gist.** A heat-map over the [[presence-classifier]] showing *where in the image the network
looked* to say "effusion" — drawn as a heat overlay plus a bounding box (the "circles").

**Explainer.** Take the effusion logit, back-propagate to the DenseNet's final feature map
(`model.features`, the 7×7 post-`norm5` map), global-average-pool those gradients into per-channel
weights, take the weighted sum of the activation channels, ReLU (keep only evidence *for* the
class), upsample to 512², normalise to [0,1]. The bright region falls on the basal / costophrenic
zone where fluid collects. A bounding box at 50% of the map's peak gives the box.

**Deep-dive.** `grad_cam_utils.py:105-137` (the CAM), `:142-148` (bbox). Border-artifact fix
(`:13-26,125-136`): zero the top row before upsampling (apex is neither cardio- nor effusion-region,
so it's anatomically safe) and normalise by the 99th percentile, so no single hot cell anchors the
scale.

Related: [[presence-classifier]], [[pspnet]]
