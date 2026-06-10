"""Unit tests for the VAE codec (plan 04).

Kept small and CPU-runnable. Shape/KL tests use a tiny 64² input (no MS-SSIM, which
needs ≥~160 px). The loss-decrease and metric tests use 256² so MS-SSIM/SSIM apply,
and run on CUDA when available.
"""

from __future__ import annotations

import pytest
import torch

from vae import VAE, VAEConfig, vae_loss
from vae.losses import kl_divergence, reconstruction_loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _small_model(res: int) -> tuple[VAE, VAEConfig]:
    cfg = VAEConfig(input_resolution=res, use_checkpoint=(DEVICE == "cuda"))
    return VAE(cfg).to(DEVICE), cfg


def test_shape_round_trip():
    model, cfg = _small_model(64)
    x = torch.randn(2, 1, 64, 64, device=DEVICE)
    z = model.encode(x)
    assert tuple(z.shape) == (2, 4, 16, 16)          # 64 / f=4
    recon = model.decode(z)
    assert tuple(recon.shape) == (2, 1, 64, 64)


def test_kl_finite_and_nonnegative():
    mu = torch.randn(3, 4, 8, 8)
    sigma = torch.rand(3, 4, 8, 8) + 1e-3
    kl = kl_divergence(mu, sigma)
    assert torch.isfinite(kl)
    # KL(N(0,1)||N(0,1)) == 0
    z = torch.zeros(3, 4, 8, 8)
    assert torch.isclose(kl_divergence(z, torch.ones_like(z)), torch.tensor(0.0), atol=1e-5)


def test_forward_returns_triplet():
    model, _ = _small_model(64)
    x = torch.randn(1, 1, 64, 64, device=DEVICE)
    out = model(x)
    assert len(out) == 3
    recon, mu, sigma = out
    assert tuple(recon.shape) == (1, 1, 64, 64)
    assert tuple(mu.shape) == tuple(sigma.shape) == (1, 4, 16, 16)


def test_loss_decreases_on_overfit():
    # The VAE samples z stochastically each forward, so a 1-step delta is below the
    # sampling noise. Overfit a *fixed* batch for a few dozen steps instead; the
    # reconstruction term must drop clearly.
    torch.manual_seed(0)
    model, cfg = _small_model(256)
    x = torch.rand(2, 1, 256, 256, device=DEVICE) * 2 - 1   # structured (not pure noise)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    @torch.no_grad()
    def recon_avg(n=4):
        model.eval()
        vals = []
        for _ in range(n):
            recon, mu, sigma = model(x)
            vals.append(vae_loss(recon.float(), x, mu, sigma, cfg).recon.item())
        model.train()
        return sum(vals) / len(vals)

    before = recon_avg()
    model.train()
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        recon, mu, sigma = model(x)
        loss = vae_loss(recon.float(), x, mu, sigma, cfg)
        loss.total.backward()
        opt.step()
    after = recon_avg()
    assert after < before, f"recon did not decrease: {before:.4f} -> {after:.4f}"


def test_recon_metric_smoke():
    from vae.eval import recon_metrics

    x = torch.rand(1, 1, 256, 256) * 2 - 1     # [-1,1]
    recon = x.clone()
    m = recon_metrics(x, recon)
    assert "ssim" in m and "lpips" in m
    assert m["ssim"] > 0.99 and m["lpips"] < 0.05   # identical → near-perfect


def test_reconstruction_loss_zero_on_identity():
    x = torch.rand(1, 1, 256, 256) * 2 - 1
    loss = reconstruction_loss(x, x)
    assert loss.item() < 1e-4
