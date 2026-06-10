"""Label-blind VAE codec — 512² grayscale CXR → 4×128×128 latent (f=4).

Implements the contract in ``plans/vae/architecture-decision.md``: a kl-f4-class
MONAI ``AutoencoderKL`` (mid-block attention only, no enc→dec skips, KL bottleneck)
trained with ``0.84·MS-SSIM + 0.16·L1 + 1e-6·KL`` — no adversarial term.
"""

from __future__ import annotations

from .config import DEFAULT_CONFIG, VAEConfig
from .model import VAE
from .losses import LossTerms, kl_divergence, reconstruction_loss, vae_loss

__all__ = [
    "VAE",
    "VAEConfig",
    "DEFAULT_CONFIG",
    "LossTerms",
    "vae_loss",
    "kl_divergence",
    "reconstruction_loss",
]
