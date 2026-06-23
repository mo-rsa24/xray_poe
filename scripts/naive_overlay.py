"""Naive pixel-overlay baseline for the N2 "composition beats overlay" claim.

The paper's headline composition result (N2) is a *contrast*: our PoE compositional
generation should produce both-disease (cardiomegaly+effusion) chest X-rays that are
MORE realistic than the obvious dumb alternative — just taking a cardiomegaly image
and an effusion image and blending them on top of each other. This script builds that
dumb alternative so domain-FID can score it.

Why the overlay is a fair-but-naive baseline:
    Two different patients' CXRs have different ribcages, heart positions and lung
    fields. Blending them in pixel space yields an image that *technically* contains
    both pathologies but shows ghosted/duplicated anatomy — visibly unreal. PoE
    composition instead generates ONE coherent body that carries both findings. So
    FID(real-both, overlay)  ≫  FID(real-both, PoE-compose) is the evidence that the
    compositional mechanism — not merely "put both diseases in the picture" — is what
    buys realism. The overlay is the number the composition has to beat.

Inputs are two directories of single-disease images. Point them at:
    real single-disease sets   data/nih/images_cardio_only , images_effusion_only
        → "naive overlay of real CXRs"
    OR the SAME LDM's single-disease generations
        → the stronger, generator-controlled baseline: same model, single-disease
          mode, combined in pixels instead of in noise space (isolates the PoE
          mechanism as the only difference from the compose set).

Blend modes (CXR pathology shows as added radio-opacity = brighter pixels):
    mean    alpha·A + (1-alpha)·B    canonical alpha-overlay (default, alpha=0.5)
    max     max(A, B)                "lighten" — keep the denser/brighter pixel
    screen  1-(1-A)(1-B)             combine bright features without over-saturating

Pairing is a seeded shuffle of each set, zipped — every overlay mixes a different
cardio image with a different effusion image, up to min(len_a, len_b, n) pairs.

No torch — PIL + numpy only, so it runs anywhere and is unit-testable.

CLI:
    # naive overlay of real single-disease CXRs → the N2 baseline set
    python scripts/naive_overlay.py \\
        --a data/nih/images_cardio_only --b data/nih/images_effusion_only \\
        --n 500 --out outputs/overlay

    # then score it (higher FID than the compose set = composition wins):
    python -m metrics.fid --embed xrv --a data/nih/images --b outputs/overlay \\
        --n 500 --out results/overlay_fid.json
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics.fid import _gather_paths  # noqa: E402  (same seeded sampler the FID harness uses)


def load_gray(path: str | Path, res: int) -> np.ndarray:
    """PNG/JPG → (res, res) float32 in [0, 1], grayscale (LANCZOS to a common size)."""
    img = Image.open(path).convert("L")
    if img.size != (res, res):
        img = img.resize((res, res), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def blend(a: np.ndarray, b: np.ndarray, mode: str, alpha: float) -> np.ndarray:
    """Combine two [0,1] grayscale images into one 'both-disease' overlay."""
    if mode == "mean":
        return alpha * a + (1.0 - alpha) * b
    if mode == "max":
        return np.maximum(a, b)
    if mode == "screen":
        return 1.0 - (1.0 - a) * (1.0 - b)
    raise ValueError(f"unknown blend mode: {mode!r} (mean|max|screen)")


def make_overlays(
    paths_a: list[Path],
    paths_b: list[Path],
    out_dir: Path,
    mode: str,
    alpha: float,
    res: int,
    seed: int,
) -> int:
    """Seeded-shuffle pair the two sets, blend each pair, write overlay_NNNNN.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    a = list(paths_a)
    b = list(paths_b)
    rng.shuffle(a)
    rng.shuffle(b)
    n_pairs = min(len(a), len(b))

    for i in range(n_pairs):
        img = blend(load_gray(a[i], res), load_gray(b[i], res), mode, alpha)
        arr = (img.clip(0.0, 1.0) * 255).round().astype(np.uint8)
        Image.fromarray(arr, "L").save(out_dir / f"overlay_{i:05d}.png")
        if (i + 1) % 100 == 0 or i + 1 == n_pairs:
            print(f"  wrote {i + 1}/{n_pairs} overlays")
    return n_pairs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Naive pixel-overlay baseline for the N2 composition-vs-overlay claim.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--a", "--cardio", dest="a", default="data/nih/images_cardio_only",
                   help="Dir of disease-A (cardiomegaly) single-disease images")
    p.add_argument("--b", "--effusion", dest="b", default="data/nih/images_effusion_only",
                   help="Dir of disease-B (effusion) single-disease images")
    p.add_argument("--n", type=int, default=None,
                   help="Cap images sampled from each set before pairing (default: all)")
    p.add_argument("--blend", choices=["mean", "max", "screen"], default="mean",
                   help="Pixel combine rule (default: mean alpha-overlay)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight on A for --blend mean (default 0.5)")
    p.add_argument("--res", type=int, default=512, help="Common resolution for the blend")
    p.add_argument("--seed", type=int, default=42, help="Pairing/sampling seed")
    p.add_argument("--out", default="outputs/overlay", help="Output dir for overlays")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    a_dir = Path(args.a) if Path(args.a).is_absolute() else root / args.a
    b_dir = Path(args.b) if Path(args.b).is_absolute() else root / args.b
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out

    paths_a = _gather_paths(a_dir, args.n, seed=args.seed)
    paths_b = _gather_paths(b_dir, args.n, seed=args.seed)
    print(f"Overlaying {len(paths_a)} × {len(paths_b)} single-disease images "
          f"(blend={args.blend}{f', alpha={args.alpha}' if args.blend == 'mean' else ''}) ...")

    n = make_overlays(paths_a, paths_b, out_dir, args.blend, args.alpha, args.res, args.seed)
    print(f"\nWrote {n} overlay images → {out_dir}")
    print(f"Next: python -m metrics.fid --embed xrv --a data/nih/images --b {out_dir} "
          f"--n {n} --out results/overlay_fid.json")


if __name__ == "__main__":
    main()
