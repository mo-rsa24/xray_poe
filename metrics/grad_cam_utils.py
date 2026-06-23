"""Shared Grad-CAM hook over the fine-tuned 2-head DenseNet-121 presence classifier.

Single source of truth for the class-activation machinery used by both the
visualisation (`scripts/grad_cam.py`) and the quantitative extractors
(`metrics/extractors.py`) — same weights, same preprocessing, same CAM, so the
heart-size / blunting scalars and the overlay figure are read off identical maps.

The CAM target layer is `model.features` (DenseNet output, post-norm5 — the same
spatial map as the final denseblock4 conv). Gradients of a head logit w.r.t. that
feature map are global-average-pooled into channel weights, applied to the
activations, ReLU'd, upsampled to 512², and normalised to [0,1].

Border-artifact fix (plan 04, task 4)
    The effusion head produced a spurious hot band along the *top* border — in
    effusion-absent (normal) images the outer ring averaged 1.5× the interior and
    the top-right corner was the global max, which anchored the min-max
    normalisation and bled blue into the apices of every overlay. Measured cause:
    padding/edge activations concentrated in the top row of the 7×7 feature map.
    Fix: zero the top `suppress_top_rows` rows of the low-resolution CAM before
    upsampling (the apical band, where neither cardiomegaly — central — nor
    effusion — basal — localises, so removing it is anatomically safe for both
    heads), and normalise by a high percentile rather than the max so no single
    hot cell sets the scale. After the fix the normal border/interior ratio drops
    to ~0.9 while the effusion-vs-normal basal (costophrenic) separation is
    preserved. Zeroing the *full* ring instead over-suppresses and destroys the
    genuine basal effusion signal, so only the top is removed.

Public API:
    load_model(ckpt_path, device)        -> (model, cardio_idx, effusion_idx)
    preprocess(path)                     -> (model_in (1,1,224,224), disp (512,512))
    GradCAM(model, device).cam(x, head)  -> (512,512) [0,1] CAM
    bbox_from_cam(cam, frac=0.5)         -> (x0,y0,x1,y1) | None
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# torch / torchxrayvision / PresenceClassifier are imported lazily inside the
# functions that need them, so the numpy-only helpers (bbox_from_cam) and the
# constants stay import-light and CPU-importable without a torch install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_CKPT_PATH = "ckpts/presence_classifier_finetuned.pt"
_DISPLAY = 512            # overlay / CAM resolution


# ── model ────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str = _CKPT_PATH, device: str = "cpu"):
    """Load the fine-tuned PresenceClassifier; returns (model, cardio_idx, effusion_idx)."""
    import torch

    from scripts.finetune_classifier import PresenceClassifier

    ckpt = torch.load(ckpt_path, map_location=device)
    model = PresenceClassifier(backbone_weights=ckpt["backbone"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, int(ckpt["cardio_idx"]), int(ckpt["effusion_idx"])


# ── preprocessing (mirrors metrics/presence_classifier._preprocess) ───────────────

def preprocess(path: str | Path):
    """Return (model_input (1,1,224,224) float tensor, display (512,512) [0,1] array)."""
    import torch
    import torchxrayvision as xrv
    from PIL import Image

    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)                          # (H, W)
    arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)    # (1, H, W) in [-1024, 1024]
    arr = xrv.datasets.XRayCenterCrop()(arr)                       # (1, S, S)
    arr = xrv.datasets.XRayResizer(224)(arr)                       # (1, 224, 224)
    arr = arr.copy()
    model_in = torch.from_numpy(arr).unsqueeze(0)                  # (1, 1, 224, 224)

    # display image: same FOV, min-max to [0,1], upsampled to 512
    disp = arr[0]
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
    disp = np.array(
        Image.fromarray((disp * 255).astype(np.uint8)).resize((_DISPLAY, _DISPLAY), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return model_in, disp


# ── Grad-CAM ───────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._act = {}
        model.features.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inp, out):
        if out.requires_grad:          # skip under torch.no_grad() (e.g. a plain probs forward)
            out.retain_grad()
        self._act["v"] = out

    def cam(self, x, head_idx: int,
            suppress_top_rows: int = 1, pct: float = 99.0) -> np.ndarray:
        """Return a (512,512) [0,1] CAM for one head.

        suppress_top_rows  rows of the low-res feature CAM to zero before upsampling
                           (the apical border-artifact band; 0 disables the fix).
        pct                normalise by this percentile rather than the max, so a
                           single hot cell cannot anchor the scale (100 = plain max).
        """
        import torch.nn.functional as F

        x = x.to(self.device)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                       # fires the forward hook
        act = self._act["v"]                         # (1, C, h, w)
        logits[0, head_idx].backward()
        grad = act.grad                              # (1, C, h, w)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))      # (1, 1, h, w)

        # border-artifact fix: drop the apical rows of the low-res map (see module doc)
        if suppress_top_rows > 0:
            cam[..., :suppress_top_rows, :] = 0.0

        cam = F.interpolate(cam, size=(_DISPLAY, _DISPLAY), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam.detach().cpu().numpy()

        # percentile normalisation (robust to a single hot cell); falls back to max
        hi = np.percentile(cam, pct) if pct < 100 else cam.max()
        cam = np.clip(cam / (hi + 1e-8), 0.0, 1.0)
        return cam


# ── bbox ────────────────────────────────────────────────────────────────────────

def bbox_from_cam(cam: np.ndarray, frac: float = 0.5):
    """Bounding box (x0,y0,x1,y1) of the region where cam >= frac*max, or None."""
    mask = cam >= frac * cam.max()
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
