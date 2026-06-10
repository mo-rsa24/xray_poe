"""Reconstruction + KL losses for the VAE codec.

Per the architecture decision (§1, §5):

    total = recon + kl_weight · KL
    recon = α · (1 - MS-SSIM) + (1-α) · L1      with α = 0.84  (Zhao et al. 2017)
    KL    = 0.5 · mean(μ² + σ² - log σ² - 1)

There is **no adversarial term and no RGB-LPIPS** in training — both rejected (§5).
A RadImageNet-LPIPS perceptual term is the only sanctioned lever (config.perceptual_weight,
OFF by default); it is not wired here and must never fall back to RGB VGG/AlexNet.

MS-SSIM operates on [0,1] images. Training data is normalized to [-1,1] (SD/LDM
convention, no tanh on the decoder), so we map both sides to [0,1] for the MS-SSIM
term only. L1 and KL are computed on the native [-1,1] tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

from .config import DEFAULT_CONFIG, VAEConfig


def kl_divergence(z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
    """KL(N(μ,σ²) || N(0,1)), averaged over the batch (summed over latent dims)."""
    # 0.5 * sum(μ² + σ² - log σ² - 1) per sample, then mean over batch.
    per_sample = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + 1e-12) - 1,
        dim=[1, 2, 3],
    )
    return per_sample.mean()


def _to_unit(x: torch.Tensor) -> torch.Tensor:
    """Map [-1,1] → [0,1] (clamped) for the MS-SSIM term."""
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def ms_ssim_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - MS-SSIM on [0,1] images. Higher MS-SSIM ⇒ lower loss."""
    return 1.0 - ms_ssim(_to_unit(recon), _to_unit(target), data_range=1.0)


def reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor, config: VAEConfig = DEFAULT_CONFIG
) -> torch.Tensor:
    msssim = ms_ssim_loss(recon, target)
    l1 = F.l1_loss(recon, target)
    return config.ms_ssim_weight * msssim + config.l1_weight * l1


@dataclass
class LossTerms:
    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    z_mu: torch.Tensor,
    z_sigma: torch.Tensor,
    config: VAEConfig = DEFAULT_CONFIG,
) -> LossTerms:
    recon_l = reconstruction_loss(recon, target, config)
    kl = kl_divergence(z_mu, z_sigma)
    total = recon_l + config.kl_weight * kl
    return LossTerms(total=total, recon=recon_l, kl=kl)
