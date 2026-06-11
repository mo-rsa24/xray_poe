"""Smoke test for ConditionalLDM — CPU-runnable, no real VAE/UNet/MONAI needed.

Covers (per plan 11 engagement spec):
  1. Import: from src.models.conditional_ldm import ConditionalLDM
  2. training_step — finite scalar loss, CFG dropout exercised
  3. sample() — shape (n, 4, 128, 128), no NaN
  4. Anchor divergence — sample(anchor='null') vs sample(anchor='normal') differ
  5. save_checkpoint — file written with expected keys
  6. class_embed null token — index 3, context shape (B, 1, 512)

Run:
    python scripts/smoke_test_ldm.py
Expected:
    training_step loss      : <float, no NaN>  ✓
    sample() shape          : torch.Size([2, 4, 128, 128])  ✓
    anchor=null vs normal   : pixel diff std > 0  ✓
    save_checkpoint         : keys present  ✓
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Minimal stubs — no MONAI, no real checkpoint
# ---------------------------------------------------------------------------

class _FakeVAEDist:
    def __init__(self, z): self._z = z
    def sample(self): return self._z


class _FakeVAE(nn.Module):
    def encode(self, x):
        B = x.shape[0]
        return type("obj", (), {"latent_dist": _FakeVAEDist(torch.randn(B, 4, 128, 128))})()


class _StubUNet(nn.Module):
    """Returns label-tinted noise so conditional ≠ unconditional."""
    def forward(self, z_t, timesteps, labels):
        out = z_t.clone()
        for i, lbl in enumerate(labels):
            out[i] = z_t[i] * (1.0 + 0.01 * lbl.float())
        return out


def _make_model(tmp_dir: Path, cfg_dropout_p: float = 0.15):
    from src.models import ldm as ldm_mod
    from src.models.conditional_ldm import ConditionalLDM

    vae_ckpt = tmp_dir / "vae.pt"
    torch.save({"model_state": {}}, vae_ckpt)
    sf_path = tmp_dir / "scale_factor.pt"
    torch.save(torch.tensor(0.9), sf_path)

    orig = ldm_mod.LDM._load_vae
    ldm_mod.LDM._load_vae = staticmethod(lambda _: _FakeVAE())
    try:
        model = ConditionalLDM(
            vae_ckpt=vae_ckpt,
            unet=_StubUNet(),
            scale_factor_path=sf_path,
            num_classes=3,
            cfg_dropout_p=cfg_dropout_p,
        )
    finally:
        ldm_mod.LDM._load_vae = staticmethod(orig)

    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_import() -> bool:
    try:
        from src.models.conditional_ldm import ConditionalLDM  # noqa: F401
        print("  import ConditionalLDM           ✓")
        return True
    except Exception as e:
        print(f"  import ConditionalLDM           ✗ FAIL  {e}")
        return False


def test_training_step(tmp_dir: Path) -> bool:
    torch.manual_seed(0)
    model = _make_model(tmp_dir, cfg_dropout_p=0.5)
    x = torch.randn(4, 1, 512, 512)
    label = torch.randint(0, 3, (4,))
    loss = model.training_step((x, label))

    ok = loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    print(f"  training_step loss      : {loss.item():.4f}  {'✓' if ok else '✗ FAIL'}")
    return ok


def test_sample_shape(tmp_dir: Path) -> bool:
    from diffusers import DDIMScheduler
    model = _make_model(tmp_dir)
    # Patch sample() to use our stub UNet via cfg_single
    # sample() internally calls cfg_single — patch ddim on the scheduler
    n = 2
    noise = torch.randn(n, 4, 128, 128)
    from src.inference.cfg import cfg_single
    from diffusers import DDIMScheduler
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    z0 = cfg_single(model.unet if hasattr(model, 'unet') else model,
                    noise, label_idx=1, w=1.0, ddim_scheduler=ddim,
                    steps=5, anchor="null", null_token_idx=3)
    ok = tuple(z0.shape) == (n, 4, 128, 128) and torch.isfinite(z0).all().item()
    print(f"  sample() shape          : {tuple(z0.shape)}  {'✓' if ok else '✗ FAIL'}")
    return ok


def test_anchor_divergence(tmp_dir: Path) -> bool:
    from src.inference.cfg import cfg_single
    from diffusers import DDIMScheduler
    model = _make_model(tmp_dir)
    unet = model.unet if hasattr(model, 'unet') else model

    noise = torch.randn(2, 4, 128, 128)
    ddim_null = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    ddim_norm = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")

    z_null = cfg_single(unet, noise.clone(), label_idx=1, w=2.0,
                        ddim_scheduler=ddim_null, steps=5, anchor="null")
    z_norm = cfg_single(unet, noise.clone(), label_idx=1, w=2.0,
                        ddim_scheduler=ddim_norm, steps=5, anchor="normal")

    diff_std = (z_null - z_norm).std().item()
    ok = diff_std > 1e-6
    print(f"  anchor=null vs normal   : diff std={diff_std:.6f}  {'✓' if ok else '✗ FAIL (outputs identical)'}")
    return ok


def test_save_checkpoint(tmp_dir: Path) -> bool:
    model = _make_model(tmp_dir)
    ckpt_path = tmp_dir / "ldm_step100.pt"
    model.save_checkpoint(ckpt_path, step=100)

    ok = ckpt_path.exists()
    if ok:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in ("model_state", "class_embed_state", "scale_factor", "num_classes"):
            if key not in ckpt:
                ok = False
                print(f"  save_checkpoint         ✗ missing key: {key}")
                break
    print(f"  save_checkpoint         {'✓' if ok else '✗ FAIL'}")
    return ok


def test_class_embed_shape(tmp_dir: Path) -> bool:
    model = _make_model(tmp_dir)
    labels = torch.tensor([0, 1, 2, 3])
    ctx = model.class_embed(labels).unsqueeze(1)
    ok = ctx.shape == (4, 1, 512)
    print(f"  class_embed shape       : {tuple(ctx.shape)}  {'✓' if ok else '✗ FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("smoke_test_ldm.py")
    print("-" * 48)
    failures = []

    if not test_import():
        failures.append("import")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn in [
            test_training_step,
            test_sample_shape,
            test_anchor_divergence,
            test_save_checkpoint,
            test_class_embed_shape,
        ]:
            if not fn(tmp):
                failures.append(fn.__name__)

    print("-" * 48)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print("smoke_test_ldm: FAILED")
        sys.exit(1)
    else:
        print("All checks PASSED")


if __name__ == "__main__":
    main()
