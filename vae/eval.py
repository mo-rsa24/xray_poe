"""Reconstruction-quality evaluation + the ceiling-check skeleton.

Authored here, **executed on real data later** under
``plans/compute-budget/plans/runpod-execution/``. This scope only proves the code
computes; it never runs it on real CXR.

  * ``recon_metrics`` — SSIM + LPIPS between input and reconstruction. LPIPS is used
    only as an *eval metric* (plain LPIPS is acceptable here per §5); it is never a
    training loss. SSIM via MONAI, LPIPS via the ``lpips`` package.
  * ``ceiling_check`` — averages the recon metrics over a loader and writes a JSON
    report. The "ceiling" is the best reconstruction the frozen codec can achieve —
    the upper bound on everything the downstream LDM can reproduce (Exp3 in
    EXPERIMENTS.md). Real-data execution deferred.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from monai.metrics import SSIMMetric

# LPIPS expects 3-channel [-1,1] input; we replicate the single grayscale channel.
_lpips_model = None


def _get_lpips(net: str = "alex"):
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net=net, verbose=False)
    return _lpips_model


@torch.no_grad()
def recon_metrics(x: torch.Tensor, recon: torch.Tensor) -> dict[str, float]:
    """SSIM (↑ better) and LPIPS (↓ better) for an input/recon batch in [-1,1]."""
    ssim = SSIMMetric(spatial_dims=2, data_range=2.0)  # range of [-1,1] is 2.0
    ssim_val = ssim(recon, x).mean().item()

    lp = _get_lpips()
    x3, r3 = x.repeat(1, 3, 1, 1), recon.repeat(1, 3, 1, 1)
    lpips_val = lp(x3.clamp(-1, 1), r3.clamp(-1, 1)).mean().item()
    return {"ssim": ssim_val, "lpips": lpips_val}


@torch.no_grad()
def ceiling_check(model, loader, device: str = "cpu", report_path: str | None = None) -> dict:
    """Average recon metrics over a loader = the codec reconstruction ceiling.

    DEFERRED execution: on real data this runs under runpod-execution as the Exp3
    ceiling gate. Here it just demonstrates the path works on whatever loader given.
    """
    model.eval()
    n, ssim_sum, lpips_sum = 0, 0.0, 0.0
    for batch in loader:
        x = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
        recon = model.reconstruct(x)   # posterior mean — the canonical reconstruction
        m = recon_metrics(x, recon)
        b = x.shape[0]
        ssim_sum += m["ssim"] * b
        lpips_sum += m["lpips"] * b
        n += b
    report = {"n": n, "ssim": ssim_sum / max(n, 1), "lpips": lpips_sum / max(n, 1)}
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2))
    return report
