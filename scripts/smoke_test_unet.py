"""Forward-pass smoke test for LDMUNet.

Checks:
  1. Output shape == input latent shape (B, 4, 128, 128)
  2. No NaN or Inf in output
  3. Null-token (label index 3) produces a valid output
  4. Peak VRAM < 6 000 MB (GPU only)
  5. Param count printed

Usage:
    python scripts/smoke_test_unet.py           # uses CUDA if available
    python scripts/smoke_test_unet.py --cpu     # force CPU
    python scripts/smoke_test_unet.py --bf16    # test bf16 cast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.ldm_unet import build_unet

LATENT_SHAPE = (4, 4, 128, 128)   # (B, C, H, W)
LABEL_CLASSES = 3                  # valid labels 0/1/2; 3 = null token


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    p.add_argument("--bf16", action="store_true", help="Cast model and inputs to bfloat16")
    p.add_argument("--model-channels", type=int, default=128)
    p.add_argument("--wandb-project", default=None,
                   help="If set, log param count and VRAM to W&B run ldm_smoke_<date>")
    return p.parse_args()


def peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"Device : {device}  |  dtype: {dtype}")
    print(f"Building LDMUNet  model_channels={args.model_channels} ...")

    model = build_unet(num_classes=LABEL_CLASSES, model_channels=args.model_channels)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Param count       : {n_params / 1e6:.2f} M")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    failures = []

    # --- test 1: normal labels -----------------------------------------------
    torch.manual_seed(0)
    z = torch.randn(*LATENT_SHAPE, device=device, dtype=dtype)
    t = torch.randint(0, 1000, (LATENT_SHAPE[0],), device=device)
    labels = torch.randint(0, LABEL_CLASSES, (LATENT_SHAPE[0],), device=device)

    with torch.no_grad():
        out = model(z, t, labels)

    shape_ok = tuple(out.shape) == LATENT_SHAPE
    nan_inf_ok = torch.isfinite(out).all().item()

    print(f"\nTest 1 — normal labels {labels.tolist()}")
    print(f"  Output shape      : {tuple(out.shape)}  {'✓' if shape_ok else '✗ FAIL'}")
    print(f"  NaN/Inf in output : {not nan_inf_ok}  {'✓' if nan_inf_ok else '✗ FAIL'}")

    if not shape_ok:
        failures.append(f"output shape {tuple(out.shape)} != {LATENT_SHAPE}")
    if not nan_inf_ok:
        failures.append("NaN/Inf in output (normal labels)")

    # --- test 2: null token (index 3) ----------------------------------------
    null_labels = torch.full((LATENT_SHAPE[0],), model.null_token_idx, device=device)
    with torch.no_grad():
        out_null = model(z, t, null_labels)

    null_ok = torch.isfinite(out_null).all().item()
    print(f"\nTest 2 — null token (index {model.null_token_idx})")
    print(f"  NaN/Inf in output : {not null_ok}  {'✓' if null_ok else '✗ FAIL'}")
    if not null_ok:
        failures.append("NaN/Inf in output (null token)")

    # --- test 3: VRAM ---------------------------------------------------------
    vram_mb = peak_vram_mb(device)
    vram_ok = vram_mb < 6000 or device.type == "cpu"
    if device.type == "cuda":
        print(f"\nTest 3 — peak VRAM")
        print(f"  Peak VRAM (MB)    : {vram_mb:.0f}  {'✓' if vram_ok else '✗ FAIL (>6000 MB)'}")
        if not vram_ok:
            failures.append(f"peak VRAM {vram_mb:.0f} MB > 6000 MB")
    else:
        vram_mb = 0.0

    # --- W&B logging (optional) -----------------------------------------------
    if args.wandb_project:
        try:
            import wandb
            from datetime import date
            run = wandb.init(
                project=args.wandb_project,
                name=f"ldm_smoke_{date.today().isoformat()}",
                config={"model_channels": args.model_channels, "dtype": str(dtype)},
            )
            wandb.log({
                "config/unet_params": n_params,
                "config/vram_smoke_mb": vram_mb,
            })
            wandb.finish()
            print(f"\nW&B logged to project '{args.wandb_project}'")
        except Exception as e:
            print(f"\n[warn] W&B logging failed: {e}")

    # --- summary --------------------------------------------------------------
    print("\n" + "-" * 42)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print("smoke_test_unet: FAILED")
        sys.exit(1)
    else:
        print("smoke_test_unet: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
