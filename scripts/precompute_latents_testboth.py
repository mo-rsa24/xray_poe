"""Encode VAE latents for the held-out both-disease group (Cardiomegaly ∧ Effusion).

These images were intentionally excluded from the train/val latent cache
(precompute_latents_fast.py excludes both-class to keep the label space clean).
This script encodes only the both-group images into a separate test cache at
data/latents/test_both/ — consumed by the product-of-experts evaluation.

Usage (RunPod, after extracting both-group PNGs):
    python scripts/precompute_latents_testboth.py \\
        --csv       data/nih/Data_Entry_2017.csv \\
        --image-dir data/nih/images \\
        --vae-ckpt  ckpts/vae_step0025000.pt \\
        --out-dir   data/latents/test_both \\
        --batch-size 24 --num-workers 8 --write-workers 8
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.precompute_latents import _load_image
from scripts.precompute_latents_fast import encode_split


def _collect_both(csv_path: str, image_dir: str) -> list[tuple[str, int]]:
    """Return [(abs_path, label=3), ...] for Cardiomegaly ∧ Effusion images."""
    img_dir = Path(image_dir)
    samples: list[tuple[str, int]] = []
    missing = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("View Position", "").strip() not in ("PA", "AP"):
                continue
            labels_set = set(row["Finding Labels"].split("|"))
            if "Cardiomegaly" not in labels_set or "Effusion" not in labels_set:
                continue
            p = img_dir / row["Image Index"].strip()
            if not p.exists() or p.stat().st_size == 0:
                missing += 1
                continue
            samples.append((str(p), 3))  # label 3 = both
    if missing:
        print(f"WARNING: {missing} both-group images not found in {image_dir}", flush=True)
    print(f"Found {len(samples)} both-group images with latents to encode", flush=True)
    return samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--out-dir", default="data/latents/test_both")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--write-workers", type=int, default=8)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--compile", dest="compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--compile-mode", default="default")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    from vae.model import VAE
    from vae.config import DEFAULT_CONFIG as vae_cfg

    vae = VAE(vae_cfg).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt.get("model", ckpt.get("model_state", ckpt)))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"Loaded VAE from {args.vae_ckpt}", flush=True)

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    if args.compile:
        vae.encode = torch.compile(vae.encode, mode=args.compile_mode)
        print(f"Compiled VAE encode (mode={args.compile_mode})", flush=True)

    samples = _collect_both(args.csv, args.image_dir)
    if not samples:
        print("No both-group images found — did you extract them first?", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    encode_split(samples, vae, out_dir, device, "test_both", args)
    print(f"\nBoth-group latents → {out_dir}  ({len(list(out_dir.glob('*.pt')))} files)")


if __name__ == "__main__":
    main()
