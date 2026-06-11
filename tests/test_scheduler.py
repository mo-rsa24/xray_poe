"""Unit tests for DDPMScheduler noise corruption — plan 07, task 1."""

from __future__ import annotations

import pytest
import torch
from diffusers import DDPMScheduler


@pytest.fixture(scope="module")
def scheduler() -> DDPMScheduler:
    return DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")


def test_add_noise_shape(scheduler):
    B, C, H, W = 4, 4, 128, 128
    z = torch.randn(B, C, H, W)
    noise = torch.randn_like(z)
    t = torch.randint(0, 1000, (B,))

    z_noisy = scheduler.add_noise(z, noise, t)

    assert z_noisy.shape == z.shape, f"shape mismatch: {z_noisy.shape} != {z.shape}"


def test_add_noise_finite(scheduler):
    z = torch.randn(2, 4, 128, 128)
    noise = torch.randn_like(z)
    t = torch.tensor([0, 999])

    z_noisy = scheduler.add_noise(z, noise, t)

    assert torch.isfinite(z_noisy).all(), "NaN/Inf in z_noisy"


def test_add_noise_at_t0_close_to_z(scheduler):
    # At t=0 (minimal noise), z_noisy should be close to z (not dominated by noise)
    z = torch.randn(2, 4, 128, 128)
    noise = torch.randn_like(z)
    t = torch.zeros(2, dtype=torch.long)

    z_noisy = scheduler.add_noise(z, noise, t)
    # z_noisy = sqrt(alpha_bar_0)*z + sqrt(1-alpha_bar_0)*noise
    # alpha_bar_0 ≈ 1, so z_noisy ≈ z
    assert torch.allclose(z_noisy, z, atol=0.1), "t=0 noisy latent too far from original"


def test_add_noise_at_tmax_dominated_by_noise(scheduler):
    # At t=999 (maximum noise), z_noisy should be close to pure noise
    z = torch.zeros(2, 4, 128, 128)   # zero signal to isolate noise contribution
    noise = torch.ones_like(z)
    t = torch.full((2,), 999, dtype=torch.long)

    z_noisy = scheduler.add_noise(z, noise, t)
    # At t=999, alpha_bar ≈ 0, so z_noisy ≈ noise = 1.0
    assert z_noisy.mean().abs() > 0.5, "t=999 noisy latent not dominated by noise"


def test_scheduler_config():
    s = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    assert s.config.num_train_timesteps == 1000
    assert s.config.beta_schedule == "linear"
