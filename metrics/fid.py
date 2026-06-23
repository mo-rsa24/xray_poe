"""Fréchet / kernel distances for "how realistic is this image set?".

Two complementary realism distances, both reported on a shared feature space:

  FID (Fréchet Inception Distance)
      The Fréchet (Wasserstein-2) distance between two Gaussians fitted to the
      feature activations of set A (real) and set B (generated). Lower = more
      realistic; 0 only when the two feature means and covariances match exactly.
      This is the Exp4 LDM realism gate and the Exp8 floor baseline.
      Heusel et al., "GANs Trained by a Two Time-Scale Update Rule" (NeurIPS 2017).

  KID (Kernel Inception Distance)
      An *unbiased* MMD² with a degree-3 polynomial kernel. Unlike FID — whose
      Gaussian fit is badly biased at small N — KID's estimator has no sample-size
      bias, so it is the number to trust when each set has only a few hundred
      images, and its across-subset spread is an honest noise band.
      Bińkowski et al., "Demystifying MMD GANs" (ICLR 2018).

Two embeddings, switched by `--embed`:

  inception   ImageNet InceptionV3 pool3 (2048-d) — the *canonical* FID feature
              space. Responds to natural-image texture; reported for comparability
              with the wider generative-models literature.
  xrv         TorchXRayVision NIH DenseNet-121 penultimate layer (1024-d) — the
              *domain* FID/KID. The distance then responds to chest-pathology
              structure rather than natural-image texture, which is what we
              actually care about for composed CXRs (CheXGenBench, arXiv 2505.10496;
              foundation-CXR FID, arXiv 2509.03903). Shares the c2st/presence
              preprocessing and feature extractor.

The numeric core (`frechet_distance`, `fid_from_features`, `kid_poly`) operates on
feature arrays (N, D) and depends only on numpy + scipy, so it is CPU-runnable and
unit-testable without torch or any checkpoint. Image → feature extraction
(`extract_features`) lazily imports torch.

The real-vs-real run is the *noise floor*: split one real set in half and compute
FID/KID across the split. No generated-vs-real number is meaningful unless it sits
clearly above that floor — at small N the FID floor is large and positive purely
from the Gaussian-fit bias, which is exactly why KID (and its across-subset std) is
reported alongside.

CLI:
    # sanity checks (no images, no torch needed):
    python -m metrics.fid --selftest

    # identical sets  → FID ~0
    python -m metrics.fid --a real/ --b real/

    # obviously different → FID large
    python -m metrics.fid --a real/ --b noise/

    # domain-FID + KID (Sprint S4): embedding that responds to pathology
    python -m metrics.fid --a real_both/ --b gen_both/ --embed xrv --kid \\
        --out results/exp6_fid.json

    # real-vs-real floor (split a real set in half before trusting any FID):
    python -m metrics.fid --a real_both/ --b real_both_split/ --embed xrv --kid

    # operate directly on precomputed feature arrays (.npy of shape (N, D)):
    python -m metrics.fid --features_a a.npy --features_b b.npy

Public API:
    frechet_distance(mu1, sigma1, mu2, sigma2)   -> float
    fid_from_features(feats_a, feats_b)          -> dict (fid, n_a, n_b)
    kid_poly(feats_a, feats_b, ...)              -> dict (kid, kid_std, subsets)
    extract_features(paths, embed=..., ...)      -> np.ndarray (N, D)   [needs torch]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


# ── FID (Fréchet distance between two Gaussians) ──────────────────────────────────

def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Fréchet distance between N(mu1, sigma1) and N(mu2, sigma2).

    FID = ||mu1 - mu2||² + tr(sigma1 + sigma2 - 2·(sigma1·sigma2)^½).

    The matrix square root of the (generally non-symmetric) product sigma1·sigma2 is
    taken via scipy's `sqrtm`; a small `eps·I` is added to each covariance if the root
    is numerically non-finite, and any tiny imaginary residue is discarded. Same
    stabilisation as the reference `pytorch-fid` implementation.
    """
    from scipy import linalg

    def _sqrtm(m: np.ndarray) -> np.ndarray:
        # Older scipy returns (sqrt, errest); newer scipy (>=1.18) drops the tuple and
        # deprecates the `disp` kwarg. Call plainly and accept either return shape.
        out = linalg.sqrtm(m)
        return out[0] if isinstance(out, tuple) else out

    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))
    if mu1.shape != mu2.shape:
        raise ValueError(f"mean vectors differ in length: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"covariances differ in shape: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2

    covmean = _sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        # product was singular: nudge both covariances onto the PD cone and retry
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        # sqrtm of a PSD product is real up to rounding; keep the real part
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def _mean_cov(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feature mean (D,) and covariance (D, D) — the sufficient statistics for FID."""
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError("features must be 2-D (N, D)")
    if len(feats) < 2:
        raise ValueError(f"need >= 2 samples to estimate a covariance (got {len(feats)})")
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, np.atleast_2d(sigma)


def fid_from_features(feats_a: np.ndarray, feats_b: np.ndarray) -> dict:
    """FID between two feature sets (N, D).

    Returns:
        fid          Fréchet distance (>= 0 up to numerical noise)
        n_a, n_b     sample sizes
        dim          feature dimensionality
    """
    feats_a = np.asarray(feats_a, dtype=np.float64)
    feats_b = np.asarray(feats_b, dtype=np.float64)
    if feats_a.shape[1] != feats_b.shape[1]:
        raise ValueError(f"feature dim mismatch: {feats_a.shape[1]} vs {feats_b.shape[1]}")

    mu_a, sig_a = _mean_cov(feats_a)
    mu_b, sig_b = _mean_cov(feats_b)
    fid = frechet_distance(mu_a, sig_a, mu_b, sig_b)
    # tiny negative values are pure floating-point noise around a true 0 → clamp
    fid = max(fid, 0.0)
    return {
        "fid": fid,
        "n_a": int(len(feats_a)),
        "n_b": int(len(feats_b)),
        "dim": int(feats_a.shape[1]),
    }


# ── KID (unbiased polynomial-kernel MMD², small-N honest) ─────────────────────────

def _poly_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3,
                 gamma: float | None = None, coef0: float = 1.0) -> np.ndarray:
    """Polynomial kernel k(x, y) = (gamma·⟨x, y⟩ + coef0)^degree.

    gamma defaults to 1/D (the Bińkowski et al. choice), which keeps the inner
    product O(1) regardless of feature dimensionality."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * (X @ Y.T) + coef0) ** degree


def _mmd2_poly_unbiased(X: np.ndarray, Y: np.ndarray, degree: int, coef0: float) -> float:
    """Unbiased MMD² estimate with the polynomial kernel (diagonal removed)."""
    m, n = X.shape[0], Y.shape[0]
    Kxx = _poly_kernel(X, X, degree=degree, coef0=coef0)
    Kyy = _poly_kernel(Y, Y, degree=degree, coef0=coef0)
    Kxy = _poly_kernel(X, Y, degree=degree, coef0=coef0)
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sum_xy = Kxy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def kid_poly(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    n_subsets: int = 100,
    subset_size: int = 1000,
    degree: int = 3,
    coef0: float = 1.0,
    seed: int = 0,
) -> dict:
    """Kernel Inception Distance: unbiased polynomial-kernel MMD², averaged over
    random subsets.

    The MMD² estimator is sample-size-unbiased (unlike FID's Gaussian fit), so KID
    is the realism number to trust at small N. It is computed on `n_subsets` random
    draws of `subset_size` (capped at the available count) from each set; the mean is
    the KID and the across-subset std is an honest small-N noise band.

    Returns:
        kid          mean unbiased MMD² across subsets
        kid_std      across-subset std (the noise band)
        kid_subsets  per-subset MMD² values
        subset_size  the effective subset size used
        n_subsets    number of subsets drawn
        n_a, n_b     sample sizes
    """
    feats_a = np.asarray(feats_a, dtype=np.float64)
    feats_b = np.asarray(feats_b, dtype=np.float64)
    if feats_a.shape[1] != feats_b.shape[1]:
        raise ValueError(f"feature dim mismatch: {feats_a.shape[1]} vs {feats_b.shape[1]}")

    rng = np.random.default_rng(seed)
    m = min(subset_size, len(feats_a), len(feats_b))
    if m < 2:
        raise ValueError(f"need >= 2 samples per subset (got effective size {m})")

    vals: list[float] = []
    for _ in range(n_subsets):
        ia = rng.choice(len(feats_a), m, replace=False)
        ib = rng.choice(len(feats_b), m, replace=False)
        vals.append(_mmd2_poly_unbiased(feats_a[ia], feats_b[ib], degree, coef0))

    vals_arr = np.asarray(vals, dtype=np.float64)
    return {
        "kid": float(vals_arr.mean()),
        "kid_std": float(vals_arr.std()),
        "kid_subsets": [float(v) for v in vals_arr],
        "subset_size": int(m),
        "n_subsets": int(n_subsets),
        "n_a": int(len(feats_a)),
        "n_b": int(len(feats_b)),
    }


# ── image → feature extraction (lazy torch) ───────────────────────────────────────

def _gather_paths(dir_path: str | Path, n: int | None, seed: int = 42) -> list[Path]:
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise RuntimeError(
            f"Directory does not exist: {dir_path}  "
            f"(the spec's `real/`, `noise/`, `real_both/` are placeholders — "
            f"point --a/--b at actual image dirs, e.g. data/nih/images_both)"
        )
    paths = sorted(dir_path.glob("*.png")) + sorted(dir_path.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No PNG/JPG images in {dir_path}")
    if n is not None and n < len(paths):
        rng = random.Random(seed)
        paths = sorted(rng.sample(paths, n))
    return paths


def _extract_inception(paths: Sequence[str | Path], device: str, batch_size: int) -> np.ndarray:
    """Canonical FID features: ImageNet InceptionV3 pool3 (2048-d).

    Grayscale CXRs are replicated to 3 channels, resized to 299², and ImageNet-
    normalised. The classifier head is replaced with identity so the 2048-d pooled
    activation is returned.
    """
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    model.fc = torch.nn.Identity()                 # expose the 2048-d pool3 vector
    model = model.to(device).eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def load(p: str | Path) -> "torch.Tensor":
        img = Image.open(p).convert("RGB")
        t = TF.to_tensor(img)                      # (3, H, W) in [0, 1]
        t = TF.resize(t, [299, 299], antialias=True)
        return TF.normalize(t, mean, std)

    feats: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        chunk = list(paths[i : i + batch_size])
        tensors: list[torch.Tensor] = []
        for p in chunk:
            try:
                tensors.append(load(p))
            except Exception as exc:
                print(f"  [warn] skipping {Path(p).name}: {exc}")
                tensors.append(torch.zeros(3, 299, 299))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            f = model(batch).cpu().numpy()         # (B, 2048)
        feats.append(f)
    if not feats:
        return np.empty((0, 2048), dtype=np.float32)
    return np.concatenate(feats, axis=0).astype(np.float32)


def extract_features(
    paths: Sequence[str | Path],
    embed: str = "inception",
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Image → feature array (N, D) for FID/KID.

    embed="inception"  ImageNet InceptionV3 pool3 (2048-d) — canonical FID space.
    embed="xrv"        NIH DenseNet-121 penultimate (1024-d) — domain FID space,
                       reusing the c2st/presence extractor (xrv preprocessing).

    Lazily imports torch / torchvision / torchxrayvision so the numeric core stays
    import-light.
    """
    if embed == "xrv":
        from metrics.c2st import extract_features as xrv_features  # 1024-d xrv DenseNet
        return xrv_features(paths, device=device, batch_size=batch_size)
    if embed == "inception":
        return _extract_inception(paths, device=device, batch_size=batch_size)
    raise ValueError(f"unknown embedding: {embed!r} (inception|xrv)")


# ── self-test (sanity checks; no images, no torch) ────────────────────────────────

def _selftest(seed: int = 0) -> bool:
    """Identical sets → FID ~0 / KID ~0; shifted sets → FID large / KID large."""
    rng = np.random.default_rng(seed)
    d, n = 32, 2000
    base = rng.standard_normal((n, d))
    same = rng.standard_normal((n, d))                 # same distribution
    shifted = rng.standard_normal((n, d)) + 3.0        # obviously different (mean shift)

    print("── sanity check: IDENTICAL distribution (expect FID ~0, KID ~0)")
    f_same = fid_from_features(base, same)
    k_same = kid_poly(base, same, n_subsets=50, subset_size=500, seed=seed)
    print(f"   FID = {f_same['fid']:.3f}   (D={f_same['dim']})")
    print(f"   KID = {k_same['kid']:+.4f} ± {k_same['kid_std']:.4f}")

    print("\n── sanity check: SHIFTED distribution (expect FID large, KID large)")
    f_diff = fid_from_features(base, shifted)
    k_diff = kid_poly(base, shifted, n_subsets=50, subset_size=500, seed=seed)
    print(f"   FID = {f_diff['fid']:.3f}")
    print(f"   KID = {k_diff['kid']:+.4f} ± {k_diff['kid_std']:.4f}")

    ok = (
        f_same["fid"] < 1.0 and abs(k_same["kid"]) < 0.01
        and f_diff["fid"] > 50.0 and k_diff["kid"] > 0.5
    )
    print(f"\n{'SELFTEST PASS ✓' if ok else 'SELFTEST FAIL ✗'}")
    return ok


# ── CLI ───────────────────────────────────────────────────────────────────────────

def _load_features(args: argparse.Namespace, which: str) -> np.ndarray:
    """Resolve set A or B into a feature array, from .npy or an image directory."""
    npy = getattr(args, f"features_{which}")
    img = getattr(args, which)
    if npy is not None:
        feats = np.load(npy)
        print(f"  set {which.upper()}: loaded features {feats.shape} from {npy}")
        return feats
    if img is not None:
        paths = _gather_paths(img, args.n)
        print(f"  set {which.upper()}: extracting {args.embed} features "
              f"from {len(paths)} images in {img} ...")
        return extract_features(paths, embed=args.embed, device=args.device,
                                batch_size=args.batch_size)
    raise SystemExit(f"provide --{which} <image_dir> or --features_{which} <file.npy>")


def main() -> None:
    p = argparse.ArgumentParser(
        description="FID + KID realism distances on an Inception or XRV feature space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--a", default=None, help="Image dir for set A (e.g. real)")
    p.add_argument("--b", default=None, help="Image dir for set B (e.g. generated / noise)")
    p.add_argument("--features_a", default=None, help="Precomputed feature .npy for A (N, D)")
    p.add_argument("--features_b", default=None, help="Precomputed feature .npy for B (N, D)")
    p.add_argument("--embed", choices=["inception", "xrv"], default="inception",
                   help="Feature space: inception (canonical FID) or xrv (domain FID/KID)")
    p.add_argument("--n", type=int, default=None, help="Cap images per set")
    p.add_argument("--kid", action="store_true",
                   help="Also compute KID (unbiased MMD², trustworthy at small N)")
    p.add_argument("--kid_subsets", type=int, default=100, help="KID random subsets")
    p.add_argument("--kid_subset_size", type=int, default=1000, help="KID samples per subset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    p.add_argument("--out", default=None, help="Write results JSON here")
    p.add_argument("--selftest", action="store_true",
                   help="Run the identical/shifted sanity checks (no images, no torch)")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest(args.seed) else 1)

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    print("Loading feature sets ...")
    feats_a = _load_features(args, "a")
    feats_b = _load_features(args, "b")
    print(f"  A: {feats_a.shape}   B: {feats_b.shape}   (embed={args.embed})\n")

    out: dict = {
        "set_a": args.features_a or args.a,
        "set_b": args.features_b or args.b,
        "embed": args.embed,
    }

    print("── FID (Fréchet distance)")
    f = fid_from_features(feats_a, feats_b)
    out["fid"] = f
    print(f"   FID = {f['fid']:.3f}   (D={f['dim']}, n_a={f['n_a']}, n_b={f['n_b']})")
    print("   [floor] a real-vs-real split is the noise floor — only trust FID above it")

    if args.kid:
        print("\n── KID (kernel distance, unbiased at small N)")
        k = kid_poly(feats_a, feats_b, n_subsets=args.kid_subsets,
                     subset_size=args.kid_subset_size, seed=args.seed)
        out["kid"] = k
        print(f"   KID = {k['kid']:+.4f} ± {k['kid_std']:.4f}"
              f"   ({k['n_subsets']} subsets of {k['subset_size']})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
