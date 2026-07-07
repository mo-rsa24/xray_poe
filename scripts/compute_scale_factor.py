"""Compute and save the VAE latent scale_factor.

scale_factor = 1.0 / std(latents) over a 512-image sample.

The scale_factor is saved to data/latents/scale_factor.pt as a scalar tensor.
It is loaded by LDM.__init__ as a non-trainable buffer so every run uses the
same normalisation without recomputing it.

Usage (on pod, after latent cache is built):
    python scripts/compute_scale_factor.py \
        --latent-dir data/latents \
        --n-samples 512 \
        --out data/latents/scale_factor.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dir", default="data/latents",
                   help="Directory containing .pt latent tensors")
    p.add_argument("--n-samples", type=int, default=512,
                   help="Number of latent files to sample for std estimation")
    p.add_argument("--out", default="data/latents/scale_factor.pt",
                   help="Output path for the scale_factor tensor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-channel", action="store_true",
                   help="Compute a PER-CHANNEL (4,1,1) scale_factor = 1/std_c so every latent "
                        "channel is normalized to unit variance (isotropic — matches the diffusion "
                        "N(0,1) prior). REQUIRED for the posterior-collapsed VAE-v2 μ latents, whose "
                        "per-channel std is ~[0.05,0.36,0.09,0.12] (7.6x anisotropy) — a single global "
                        "scalar leaves the schedule mismatched per channel and washes out generations.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    latent_dir = Path(args.latent_dir)
    out_path = Path(args.out)

    pt_files = sorted(latent_dir.glob("*.pt"))
    # exclude scale_factor.pt itself if it already exists
    pt_files = [f for f in pt_files if f.name != "scale_factor.pt"]

    if not pt_files:
        print(f"ERROR: no .pt files found in {latent_dir}", file=sys.stderr)
        sys.exit(1)

    rng = torch.Generator().manual_seed(args.seed)
    n = min(args.n_samples, len(pt_files))
    indices = torch.randperm(len(pt_files), generator=rng)[:n].tolist()
    sampled = [pt_files[i] for i in indices]

    print(f"Sampling {n} latents from {latent_dir} ...")
    tensors = []
    skipped = []
    for p in sampled:
        try:
            z = torch.load(p, map_location="cpu")
        except Exception as exc:  # noqa: BLE001 — truncated/0-byte .pt; skip, don't crash
            skipped.append((p, exc))
            continue
        # Support both raw tensors and dicts with a 'latent' key
        if isinstance(z, dict):
            z = z.get("latent", z.get("z"))
        # per-channel mode keeps the (C,H,W) shape; global mode flattens
        tensors.append(z.float() if args.per_channel else z.float().flatten())

    if skipped:
        print(f"WARNING: skipped {len(skipped)} unreadable latent file(s):",
              file=sys.stderr)
        for p, exc in skipped[:10]:
            print(f"  {p}: {exc}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more", file=sys.stderr)

    if not tensors:
        print("ERROR: no readable latents in the sample", file=sys.stderr)
        sys.exit(1)

    if args.per_channel:
        # Stack (N,C,H,W) → per-channel std over (N,H,W) → (C,1,1) scale = 1/std_c.
        Z = torch.stack(tensors)                      # (N,4,H,W)
        std_c = Z.std(dim=(0, 2, 3))                  # (4,)
        scale_factor = (1.0 / std_c).view(-1, 1, 1)   # (4,1,1) — broadcasts over (B,4,H,W)
        print(f"per-channel std : {[round(float(v),4) for v in std_c]}")
        print(f"per-channel sf  : {[round(float(v),3) for v in (1.0/std_c)]}")
        # sanity: after ×sf each channel is unit variance; global stays ~1
        chk = (Z * scale_factor).std(dim=(0, 2, 3))
        print(f"post-norm per-ch std (should be ~1.0): {[round(float(v),3) for v in chk]}")
        assert bool((std_c > 1e-4).all()), "a channel has ~zero variance — cannot normalize"
        assert bool((1.0 / std_c < 100).all()), "per-channel sf > 100 — check latents"
    else:
        all_latents = torch.cat(tensors)
        std = all_latents.std()
        scale_factor = torch.tensor(1.0 / std.item())
        print(f"Latent std  : {std.item():.6f}")
        print(f"scale_factor: {scale_factor.item():.6f}")
        # sampled z~N(μ,σ)→std~1→sf~1; μ (encode_moments) collapsed VAE-v2→std~0.19→sf~5.3.
        assert 0.5 < scale_factor.item() < 10.0, (
            f"scale_factor {scale_factor.item():.4f} out of expected range (0.5, 10.0) "
            "— check that latents are VAE latents (sampled or μ), not raw pixel tensors"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scale_factor, out_path)
    print(f"Saved → {out_path}  (shape {tuple(scale_factor.shape)})")


if __name__ == "__main__":
    main()
