"""Rung 1 — VAE reconstruction baseline for the pilot ladder.

Reconstructs the SAME 100 both-group images that Rung 0.5 (presence_classifier
--mode joint) sampled, so the real→recon presence delta is paired. Uses the
canonical deterministic recon path VAE.reconstruct(x) = decode(mu); the scale
factor cancels in an encode→decode round-trip, so it is not applied here.

    python scripts/vae_recon_pilot.py \
        --images data/nih/images_both --vae-ckpt ckpts/vae_step0025000.pt \
        --n 100 --out outputs/vae_recon
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae.model import VAE
from vae.config import DEFAULT_CONFIG as vae_cfg


def _same_sample_as_joint(image_dir: Path, n: int, seed: int = 42) -> list[Path]:
    """Reproduce metrics.presence_classifier._gather_paths(dir, n, seed) exactly."""
    paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if n is not None and n < len(paths):
        paths = random.Random(seed).sample(paths, n)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="VAE reconstruction baseline (Rung 1)")
    p.add_argument("--images", default="data/nih/images_both")
    p.add_argument("--vae-ckpt", default="ckpts/vae_step0025000.pt")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/vae_recon")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    device = torch.device(args.device)
    model = VAE(vae_cfg).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt.get("model_state", ckpt)))
    model.eval()

    paths = _same_sample_as_joint(Path(args.images), args.n, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reconstructing {len(paths)} images from {args.images} → {out}")

    saved = 0
    for start in range(0, len(paths), args.batch_size):
        chunk = paths[start : start + args.batch_size]
        arrs = []
        for p_ in chunk:
            arr = np.array(Image.open(p_).convert("L").resize((512, 512)), dtype=np.float32)
            arrs.append(arr / 127.5 - 1.0)
        x = torch.from_numpy(np.stack(arrs)).unsqueeze(1).to(device)  # (B,1,512,512)
        with torch.no_grad():
            r = model.reconstruct(x).clamp(-1, 1)
        for i in range(r.shape[0]):
            img = ((r[i, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(out / f"recon_{saved:04d}.png")
            saved += 1
    print(f"Saved {saved} reconstructions to {out}")


if __name__ == "__main__":
    main()
