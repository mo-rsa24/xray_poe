"""The VAE codec — a thin, contract-stable wrapper over MONAI ``AutoencoderKL``.

Why a wrapper instead of using MONAI directly: MONAI's ``encode`` returns the
``(z_mu, z_sigma)`` tuple, but the rest of this package (sanity, eval, the
launch.json engagement lines) expects ``encode(x)`` to hand back a single sampled
latent of shape ``(B, 4, 128, 128)``. This class reconciles that and gives every
caller one stable API:

    encode(x)         -> z          sampled latent (B,4,128,128)
    encode_moments(x) -> (mu, sigma)  for the KL term
    decode(z)         -> recon      (B,1,512,512), raw (no tanh)
    forward(x)        -> (recon, mu, sigma)

The architecture itself is entirely MONAI's — no hand-rolled blocks — so the
"no encoder→decoder skip" bottleneck contract is structural (AutoencoderKL has no
skips) rather than something we maintain.
"""

from __future__ import annotations

import torch
from torch import nn

from monai.networks.nets import AutoencoderKL

from .config import DEFAULT_CONFIG, VAEConfig


class VAE(nn.Module):
    def __init__(self, config: VAEConfig | None = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        self.net = AutoencoderKL(**self.config.monai_kwargs())

    # --- core API -----------------------------------------------------------
    def encode_moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior parameters (z_mu, z_sigma), each (B, z, H/f, W/f)."""
        return self.net.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sampled latent z ~ N(mu, sigma²), shape (B, z, H/f, W/f)."""
        z_mu, z_sigma = self.net.encode(x)
        return self.net.sampling(z_mu, z_sigma)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent back to image space (raw, no output nonlinearity)."""
        return self.net.decode(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction via the posterior **mean** — decode(μ).

        This is the canonical reconstruction (the posterior mode), used for eval,
        the recon/ceiling metrics, and the overfit figure. It is NOT the sampled
        training path: with the contract's near-zero KL weight (1e-6) the sampled
        latent z = μ + σ·ε carries σ-noise that is irrelevant to whether the codec
        *can* represent an image. Reconstruction quality is a property of μ.
        """
        z_mu, _ = self.net.encode(x)
        return self.net.decode(z_mu)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, z_mu, z_sigma) — the training signal."""
        return self.net(x)

    # --- introspection helpers ---------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        """Param count per top-level submodule of the MONAI net (encoder/decoder/…)."""
        out: dict[str, int] = {}
        for name, module in self.net.named_children():
            out[name] = sum(p.numel() for p in module.parameters())
        return out
