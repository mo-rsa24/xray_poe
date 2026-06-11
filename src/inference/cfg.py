"""CFG sampling helpers — DDIM denoising for single-class and compositional generation.

Null-token decision (matches plan 06 / src/models/ldm_unet.py):
    Index `null_token_idx` (default 3) is a *learned* embedding, not a zero vector.
    This is the unconditional ε(z, ∅) path trained via CFG dropout.

Anchor modes:
    'null'   — ε_uncond uses null token (index 3).  The model was explicitly trained
               to produce this direction via CFG dropout; it is the canonical ∅ anchor.
    'normal' — ε_uncond uses class 0 (no-finding).  No dedicated dropout training for
               this direction; class 0 may carry pathology-absent signal but is NOT the
               same as ε(z, ∅).  Both modes are logged from step 5k to capture whether
               the two anchors diverge in practice (EXPERIMENTS.md §6).

cfg_compose formula (PoE, per-step):
    ε_composed = ε_anchor + w*(ε_cardio − ε_anchor) + w*(ε_effusion − ε_anchor)

where ε_anchor is either the null token or the no-finding prediction, chosen by `anchor`.
"""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler

from src.models.ldm_unet import LDMUNet

# Label indices — must match src/data/real_cxr_dataset.py and ldm_unet.py
_NF_IDX = 0
_CARDIO_IDX = 1
_EFFUSION_IDX = 2


@torch.no_grad()
def cfg_single(
    unet: LDMUNet,
    noise: torch.Tensor,
    label_idx: int,
    w: float,
    ddim_scheduler: DDIMScheduler,
    steps: int = 50,
    anchor: str = "null",
    null_token_idx: int = 3,
) -> torch.Tensor:
    """DDIM CFG denoising for a single class label.

    Parameters
    ----------
    unet:
        LDMUNet instance (handles class_embed internally).
    noise:
        Starting latent noise of shape (n, 4, 128, 128).
    label_idx:
        Class to condition on (0=no_finding, 1=cardiomegaly, 2=effusion).
    w:
        CFG guidance weight.  w=0 → unconditional; w=1 → conditional only;
        w>1 → amplified guidance.
    ddim_scheduler:
        DDIMScheduler with set_timesteps already configured or to be set here.
    steps:
        Number of DDIM denoising steps.
    anchor:
        'null'   — ε_uncond from null token (index null_token_idx).
        'normal' — ε_uncond from no-finding label (index 0).
    null_token_idx:
        Index of the learned null/unconditional embedding (default 3).

    Returns
    -------
    z_0 : (n, 4, 128, 128) denoised latent.  Decode in the caller.
    """
    ddim_scheduler.set_timesteps(steps)
    device = noise.device
    n = noise.shape[0]

    uncond_idx = null_token_idx if anchor == "null" else _NF_IDX
    label_cond = torch.full((n,), label_idx, device=device, dtype=torch.long)
    label_uncond = torch.full((n,), uncond_idx, device=device, dtype=torch.long)

    was_training = unet.training
    unet.eval()

    z = noise.clone()
    for t in ddim_scheduler.timesteps:
        t_batch = t.expand(n).to(device)

        eps_cond = unet(z, t_batch, label_cond)

        if w == 1.0 and anchor == "null":
            # no guidance needed (pure conditional)
            eps = eps_cond
        else:
            eps_uncond = unet(z, t_batch, label_uncond)
            eps = eps_uncond + w * (eps_cond - eps_uncond)

        z = ddim_scheduler.step(eps, t, z).prev_sample

    if was_training:
        unet.train()

    return z


@torch.no_grad()
def cfg_compose(
    unet: LDMUNet,
    noise: torch.Tensor,
    w: float,
    ddim_scheduler: DDIMScheduler,
    steps: int = 50,
    anchor: str = "null",
    null_token_idx: int = 3,
    cardio_idx: int = _CARDIO_IDX,
    effusion_idx: int = _EFFUSION_IDX,
    nf_idx: int = _NF_IDX,
) -> torch.Tensor:
    """PoE compositional denoising — generates co-morbid (cardio + effusion) latents.

    Formula applied at each DDIM step:
        ε_composed = ε_anchor + w*(ε_cardio − ε_anchor) + w*(ε_effusion − ε_anchor)

    anchor='null'   — ε_anchor = ε(z, ∅)   using null token index.
    anchor='normal' — ε_anchor = ε(z, class_0)  using no-finding label.

    Parameters
    ----------
    unet:
        LDMUNet instance.
    noise:
        Starting latent noise of shape (n, 4, 128, 128).
    w:
        Per-disease guidance weight (same for both cardio and effusion).
    ddim_scheduler:
        DDIMScheduler; set_timesteps is called here.
    steps:
        Number of DDIM denoising steps.
    anchor:
        'null' or 'normal' — see module docstring.
    null_token_idx:
        Learned null token index (default 3).

    Returns
    -------
    z_0 : (n, 4, 128, 128) denoised latent.  Decode in the caller.
    """
    ddim_scheduler.set_timesteps(steps)
    device = noise.device
    n = noise.shape[0]

    anchor_idx = null_token_idx if anchor == "null" else nf_idx
    label_anchor = torch.full((n,), anchor_idx, device=device, dtype=torch.long)
    label_cardio = torch.full((n,), cardio_idx, device=device, dtype=torch.long)
    label_effusion = torch.full((n,), effusion_idx, device=device, dtype=torch.long)

    was_training = unet.training
    unet.eval()

    z = noise.clone()
    for t in ddim_scheduler.timesteps:
        t_batch = t.expand(n).to(device)

        eps_anchor = unet(z, t_batch, label_anchor)
        eps_cardio = unet(z, t_batch, label_cardio)
        eps_effusion = unet(z, t_batch, label_effusion)

        eps = eps_anchor + w * (eps_cardio - eps_anchor) + w * (eps_effusion - eps_anchor)

        z = ddim_scheduler.step(eps, t, z).prev_sample

    if was_training:
        unet.train()

    return z
