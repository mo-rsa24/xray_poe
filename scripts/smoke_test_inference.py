"""Smoke test for cfg_single and cfg_compose — both anchor modes.

Runs on CPU with a tiny dummy UNet (no MONAI required in this file).
On the pod, pass --ckpt to test against the real overfit checkpoint.

Usage:
    python scripts/smoke_test_inference.py             # dummy unet, CPU
    python scripts/smoke_test_inference.py \\
        --ckpt ckpts/ldm/model_step0000200.safetensors \\
        --vae-ckpt ckpts/vae/best.pt                   # real checkpoint
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from diffusers import DDIMScheduler
from src.inference.cfg import cfg_single, cfg_compose


# ---------------------------------------------------------------------------
# Minimal stub LDMUNet — no MONAI; passes shape contract
# ---------------------------------------------------------------------------

class _StubClassEmbed(nn.Embedding):
    pass


class _StubLDMUNet(nn.Module):
    def __init__(self, null_token_idx: int = 3):
        super().__init__()
        self.null_token_idx = null_token_idx
        self.class_embed = _StubClassEmbed(null_token_idx + 1, 512)

    def forward(self, z_t, timesteps, labels):
        # Return slightly label-dependent noise so cond ≠ uncond
        out = torch.zeros_like(z_t)
        for i, lbl in enumerate(labels):
            out[i] = z_t[i] * (1.0 + 0.01 * lbl.float())
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler(steps: int = 10) -> DDIMScheduler:
    s = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    s.set_timesteps(steps)
    return s


def _check_finite(t: torch.Tensor, name: str) -> bool:
    ok = torch.isfinite(t).all().item()
    print(f"  {name:<35} {'✓' if ok else '✗ FAIL (NaN/Inf)'}")
    return bool(ok)


def _check_shape(t: torch.Tensor, expected: tuple, name: str) -> bool:
    ok = tuple(t.shape) == expected
    print(f"  {name:<35} shape {tuple(t.shape)}  {'✓' if ok else f'✗ FAIL expected {expected}'}")
    return ok


def _check_different(a: torch.Tensor, b: torch.Tensor, name: str) -> bool:
    diff = (a - b).abs().mean().item()
    ok = diff > 1e-6
    print(f"  {name:<35} mean |a−b| = {diff:.6f}  {'✓' if ok else '✗ FAIL (outputs identical)'}")
    return ok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cfg_single_shapes(unet, steps: int = 10) -> bool:
    print("\ntest_cfg_single — shape and finiteness")
    noise = torch.randn(2, 4, 128, 128)
    scheduler = _make_scheduler(steps)
    z0 = cfg_single(unet, noise, label_idx=1, w=1.0, ddim_scheduler=scheduler,
                    steps=steps, anchor="null")
    ok = True
    ok &= _check_shape(z0, (2, 4, 128, 128), "z0 shape")
    ok &= _check_finite(z0, "z0 finite")
    return ok


def test_cfg_single_anchor_modes_differ(unet, steps: int = 10) -> bool:
    print("\ntest_cfg_single — null vs normal anchor differ")
    noise = torch.randn(2, 4, 128, 128)
    z0_null = cfg_single(unet, noise.clone(), label_idx=1, w=2.0,
                         ddim_scheduler=_make_scheduler(steps), steps=steps, anchor="null")
    z0_norm = cfg_single(unet, noise.clone(), label_idx=1, w=2.0,
                         ddim_scheduler=_make_scheduler(steps), steps=steps, anchor="normal")
    ok = True
    ok &= _check_finite(z0_null, "z0_null finite")
    ok &= _check_finite(z0_norm, "z0_normal finite")
    ok &= _check_different(z0_null, z0_norm, "null vs normal differ")
    return ok


def test_cfg_compose_shapes(unet, steps: int = 10) -> bool:
    print("\ntest_cfg_compose — shape and finiteness")
    noise = torch.randn(2, 4, 128, 128)
    scheduler = _make_scheduler(steps)
    z0 = cfg_compose(unet, noise, w=1.0, ddim_scheduler=scheduler, steps=steps, anchor="null")
    ok = True
    ok &= _check_shape(z0, (2, 4, 128, 128), "z0 shape")
    ok &= _check_finite(z0, "z0 finite")
    return ok


def test_cfg_compose_anchor_modes(unet, steps: int = 10) -> bool:
    print("\ntest_cfg_compose — null vs normal anchor")
    noise = torch.randn(4, 4, 128, 128)
    z0_null = cfg_compose(unet, noise.clone(), w=1.5,
                          ddim_scheduler=_make_scheduler(steps), steps=steps, anchor="null")
    z0_norm = cfg_compose(unet, noise.clone(), w=1.5,
                          ddim_scheduler=_make_scheduler(steps), steps=steps, anchor="normal")
    ok = True
    ok &= _check_finite(z0_null, "compose null finite")
    ok &= _check_finite(z0_norm, "compose normal finite")
    ok &= _check_different(z0_null, z0_norm, "null vs normal differ")
    return ok


def test_cfg_single_w0_equals_uncond(unet, steps: int = 10) -> bool:
    """w=0 should give purely unconditional output (same as label=null directly)."""
    print("\ntest_cfg_single — w=0 equals uncond")
    noise = torch.randn(2, 4, 128, 128)
    z0_w0 = cfg_single(unet, noise.clone(), label_idx=1, w=0.0,
                       ddim_scheduler=_make_scheduler(steps), steps=steps, anchor="null")
    z0_uncond = cfg_single(unet, noise.clone(), label_idx=unet.null_token_idx, w=0.0,
                           ddim_scheduler=_make_scheduler(steps), steps=steps, anchor="null")
    diff = (z0_w0 - z0_uncond).abs().mean().item()
    ok = diff < 1e-5
    print(f"  w=0 vs uncond label mean |diff|  {diff:.2e}  {'✓' if ok else '✗ FAIL'}")
    return ok


def test_reproducibility(unet, steps: int = 10) -> bool:
    """Same noise + same label must give identical output."""
    print("\ntest_cfg_single — reproducibility")
    noise = torch.randn(2, 4, 128, 128)
    a = cfg_single(unet, noise.clone(), label_idx=2, w=1.0,
                   ddim_scheduler=_make_scheduler(steps), steps=steps)
    b = cfg_single(unet, noise.clone(), label_idx=2, w=1.0,
                   ddim_scheduler=_make_scheduler(steps), steps=steps)
    diff = (a - b).abs().max().item()
    ok = diff < 1e-6
    print(f"  max |run1 − run2|               {diff:.2e}  {'✓' if ok else '✗ FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Real-checkpoint path
# ---------------------------------------------------------------------------

def _run_real_ckpt(args) -> bool:
    from scripts.generate import _load_unet, _load_vae, _decode_latents, _noise_batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Real checkpoint] {args.ckpt}  device={device}")

    unet = _load_unet(Path(args.ckpt), device)
    vae = _load_vae(Path(args.vae_ckpt), device)
    scale_factor = 1.0
    if args.scale_factor and Path(args.scale_factor).exists():
        scale_factor = torch.load(args.scale_factor, map_location="cpu").item()

    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    null_token_idx = unet.null_token_idx
    ok = True

    for anchor in ("null", "normal"):
        for disease, label_idx in [("cardiomegaly", 1), ("effusion", 2)]:
            noise = _noise_batch(2, 42, device)
            z0 = cfg_single(unet, noise, label_idx, w=1.0, ddim_scheduler=ddim,
                            steps=50, anchor=anchor, null_token_idx=null_token_idx)
            ok &= _check_shape(z0, (2, 4, 128, 128), f"cfg_single {disease} anchor={anchor}")
            ok &= _check_finite(z0, f"  finite")

    # compose, both anchors at w=1.0
    for anchor in ("null", "normal"):
        noise = _noise_batch(2, 42, device)
        z0 = cfg_compose(unet, noise, w=1.0, ddim_scheduler=ddim,
                         steps=50, anchor=anchor, null_token_idx=null_token_idx)
        ok &= _check_shape(z0, (2, 4, 128, 128), f"cfg_compose anchor={anchor}")
        ok &= _check_finite(z0, "  finite")

    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="Real safetensors checkpoint (optional)")
    p.add_argument("--vae-ckpt", default=None)
    p.add_argument("--scale-factor", default="data/latents/scale_factor.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    print("smoke_test_inference.py")
    print("=" * 48)

    unet = _StubLDMUNet(null_token_idx=3)
    failures = []

    for fn in [
        test_cfg_single_shapes,
        test_cfg_single_anchor_modes_differ,
        test_cfg_compose_shapes,
        test_cfg_compose_anchor_modes,
        test_cfg_single_w0_equals_uncond,
        test_reproducibility,
    ]:
        if not fn(unet):
            failures.append(fn.__name__)

    if args.ckpt and args.vae_ckpt:
        if not _run_real_ckpt(args):
            failures.append("real_checkpoint")

    print("\n" + "=" * 48)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print("smoke_test_inference: FAILED")
        sys.exit(1)
    else:
        print("smoke_test_inference: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
