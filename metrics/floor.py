"""The floor — the smallest measurable gap, from splitting a real set in half.

Split the real both-disease set into two disjoint halves and measure them against
each other. Both halves are samples from the *same* distribution, so every metric
sits at its no-difference floor: C2ST-AUC ≈ 0.5, MMD ≈ 0 (p large), FID/KID ≈ 0.
The 95% upper bound on that floor is the bar a real treatment-vs-real comparison
(Exp6/7) must clear to count as a resolvable distribution gap — nothing below the
floor is distinguishable from sampling noise.

Reuses the implemented metric cores (no new math):
    C2ST-AUC + MMD   metrics.c2st   (NIH DenseNet-121 1024-d feature space)
    FID + KID        metrics.fid    (domain-xrv embedding, same feature space)

Two 95%-bound sources, by metric:
    C2ST-AUC   the analytic real-vs-real null upper bound (distribution-free,
               from eda.floor_power_check) — a same-distribution pair sits below it
               95% of the time; the point AUC should land at/under it.
    MMD/FID/KID  a nonparametric bootstrap: resample each half with replacement,
               recompute, take the [2.5, 97.5] percentile band. The 97.5%ile is the
               floor's upper bound.

Power flag: when each half is smaller than --min-n the floor is wide and weakly
estimated — any treatment number near it is unreliable. Flagged, not silently passed.

CLI:
    # the treatment pair (cardiomegaly,effusion → data/nih/images, the both set)
    python -m metrics.floor --pair cardiomegaly,effusion --out results/floor.json

    # numeric-core sanity (no images, no torch): identical halves → floor at ~0.5 / ~0
    python -m metrics.floor --selftest

    # operate on a precomputed feature .npy (N, D) instead of extracting
    python -m metrics.floor --features data/latents/both_xrv.npy

Public API:
    floor_from_features(feats, ...)  -> dict (c2st, mmd, fid, kid, n, power_flag)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.c2st import c2st_auc, mmd_rbf  # noqa: E402
from metrics.fid import _mmd2_poly_unbiased, fid_from_features, kid_poly  # noqa: E402

# Pairs map to the real image set used as the floor reference. Today there is one
# treatment pair; the both-disease set is data/nih/images (data/nih/four_group_counts.md).
_PAIR_DIRS = {
    "cardiomegaly,effusion": "data/nih/images",
}
_MIN_N = 200          # halves smaller than this → wide floor, power-flagged


# ── deterministic split ───────────────────────────────────────────────────────────

def split_halves(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint even/odd split of a feature array — reproducible, no shared rows."""
    feats = np.asarray(feats)
    return feats[0::2], feats[1::2]


# ── split-bootstrap CI for a feature-only metric ──────────────────────────────────

def _split_bootstrap_ci(feats: np.ndarray, a: np.ndarray, b: np.ndarray,
                        fn, n_boot: int, seed: int) -> dict:
    """[2.5, 97.5] percentile CI of fn over repeated random *disjoint* half-splits.

    The floor is "the metric between two random halves of the real set", so its
    sampling distribution comes from re-drawing the split — NOT from resampling one
    split with replacement, which would inject duplicate rows and bias FID upward
    (its point estimate would fall outside the interval). Point estimate is fn on the
    canonical even/odd split; the band is over fresh random splits.
    """
    rng = np.random.default_rng(seed)
    point = fn(a, b)
    half = len(feats) // 2
    vals = np.empty(n_boot)
    for i in range(n_boot):
        perm = rng.permutation(len(feats))
        vals[i] = fn(feats[perm[:half]], feats[perm[half:2 * half]])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)], "upper95": float(hi)}


# ── floor ─────────────────────────────────────────────────────────────────────────

def floor_from_features(
    feats: np.ndarray,
    n_boot: int = 200,
    kid_subset_size: int = 200,
    min_n: int = _MIN_N,
    seed: int = 0,
) -> dict:
    """Compute the real-vs-real floor from one feature set, split into halves.

    Returns C2ST (with analytic null upper bound), MMD, FID and KID — each at its
    no-difference floor with a 95% upper bound — plus a power flag.
    """
    a, b = split_halves(feats)
    n_half = min(len(a), len(b))

    # C2ST-AUC + analytic null upper bound (the AUC floor is ~0.5, must not clear it)
    c2st = c2st_auc(a, b, seed=seed)
    # MMD: point + permutation p-value, plus a split-bootstrap CI on MMD² (fixed gamma)
    mmd = mmd_rbf(a, b, seed=seed)
    mmd_boot = _split_bootstrap_ci(feats, a, b, lambda x, y: _mmd2_rbf_point(x, y, mmd["gamma"]),
                                   n_boot=n_boot, seed=seed)
    # FID + KID: feature-only → split-bootstrap over fresh random halves
    fid_boot = _split_bootstrap_ci(feats, a, b, lambda x, y: fid_from_features(x, y)["fid"],
                                   n_boot=n_boot, seed=seed)
    kid_point = kid_poly(a, b, subset_size=kid_subset_size, seed=seed)
    kid_boot = _split_bootstrap_ci(feats, a, b, lambda x, y: _mmd2_poly_unbiased(x, y, 3, 1.0),
                                   n_boot=n_boot, seed=seed)

    power_flag = bool(n_half < min_n or c2st["clears_floor"])

    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "n_half": int(n_half),
        "c2st": {
            "auc": c2st["auc"],
            "auc_std": c2st["auc_std"],
            "floor95_analytic": c2st["floor95"],   # analytic null 95% upper bound
            "clears_floor": c2st["clears_floor"],   # should be False for real-vs-real
        },
        "mmd": {
            "mmd2": mmd["mmd2"],
            "p_value": mmd["p_value"],              # should be > 0.05 (indistinguishable)
            "ci95": mmd_boot["ci95"],
            "upper95": mmd_boot["upper95"],
        },
        "fid": {
            "fid": fid_boot["point"],
            "ci95": fid_boot["ci95"],
            "upper95": fid_boot["upper95"],
        },
        "kid": {
            "kid": kid_point["kid"],
            "kid_std": kid_point["kid_std"],
            "ci95": kid_boot["ci95"],
            "upper95": kid_boot["upper95"],
        },
        "min_n": int(min_n),
        "power_flag": power_flag,
        "n_boot": int(n_boot),
    }


def _mmd2_rbf_point(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """RBF MMD² at a fixed gamma (no permutation) — for the bootstrap inner loop."""
    from sklearn.metrics.pairwise import rbf_kernel
    m, n = len(x), len(y)
    Kxx = rbf_kernel(x, x, gamma=gamma)
    Kyy = rbf_kernel(y, y, gamma=gamma)
    Kxy = rbf_kernel(x, y, gamma=gamma)
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(sum_xx + sum_yy - 2 * Kxy.mean())


# ── self-test (no images, no torch) ───────────────────────────────────────────────

def _selftest(seed: int = 0) -> bool:
    """Identical-distribution set → AUC ~0.5 (within floor), FID ~0, MMD p large."""
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((4000, 32))    # one set; halves are same distribution
    f = floor_from_features(feats, n_boot=100, kid_subset_size=500, seed=seed)
    print(f"── floor on identical halves (n_half={f['n_half']})")
    print(f"   C2ST AUC = {f['c2st']['auc']:.3f}  (floor95={f['c2st']['floor95_analytic']:.3f}, "
          f"clears={f['c2st']['clears_floor']})")
    print(f"   MMD²     = {f['mmd']['mmd2']:+.4f}  p={f['mmd']['p_value']:.3f}  "
          f"upper95={f['mmd']['upper95']:+.4f}")
    print(f"   FID      = {f['fid']['fid']:.3f}  upper95={f['fid']['upper95']:.3f}")
    print(f"   KID      = {f['kid']['kid']:+.4f}  upper95={f['kid']['upper95']:+.4f}")
    print(f"   power_flag = {f['power_flag']}")
    ok = (
        not f["c2st"]["clears_floor"] and f["mmd"]["p_value"] > 0.05
        and f["fid"]["fid"] < 1.0 and abs(f["kid"]["kid"]) < 0.01
    )
    print(f"\n{'SELFTEST PASS ✓' if ok else 'SELFTEST FAIL ✗'}")
    return ok


# ── CLI ───────────────────────────────────────────────────────────────────────────

def _load_features(args: argparse.Namespace) -> np.ndarray:
    if args.features is not None:
        feats = np.load(args.features)
        print(f"  loaded features {feats.shape} from {args.features}")
        return feats

    pair = ",".join(sorted(s.strip() for s in args.pair.split(",")))
    img_dir = args.dir or _PAIR_DIRS.get(pair)
    if img_dir is None:
        raise SystemExit(f"unknown pair {args.pair!r}; known: {list(_PAIR_DIRS)} (or pass --dir)")

    from metrics.c2st import _gather_paths, extract_features
    paths = _gather_paths(img_dir, args.n)
    print(f"  pair {pair}: extracting xrv features from {len(paths)} images in {img_dir} ...")
    return extract_features(paths, device=args.device, batch_size=args.batch_size)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Real-vs-real floor (split a real set in half) with 95% bounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pair", default="cardiomegaly,effusion",
                   help="Disease pair → the real both-disease set (default cardiomegaly,effusion)")
    p.add_argument("--dir", default=None, help="Override the image dir for the pair")
    p.add_argument("--features", default=None, help="Precomputed feature .npy (N, D)")
    p.add_argument("--n", type=int, default=None, help="Cap images")
    p.add_argument("--n_boot", type=int, default=200, help="Bootstrap resamples")
    p.add_argument("--kid_subset_size", type=int, default=200, help="KID samples per subset")
    p.add_argument("--min_n", type=int, default=_MIN_N, help="Power-flag threshold per half")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    p.add_argument("--out", default=None, help="Write results JSON here")
    p.add_argument("--selftest", action="store_true",
                   help="Run the identical-halves sanity check (no images, no torch)")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest(args.seed) else 1)

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    print("Computing the floor ...")
    feats = _load_features(args)
    f = floor_from_features(feats, n_boot=args.n_boot, kid_subset_size=args.kid_subset_size,
                            min_n=args.min_n, seed=args.seed)

    print(f"\n── FLOOR (real-vs-real, halves of {f['n_a'] + f['n_b']} → {f['n_half']}/half)")
    c = f["c2st"]
    print(f"   C2ST AUC = {c['auc']:.3f} ± {c['auc_std']:.3f}   "
          f"floor95(analytic) = {c['floor95_analytic']:.3f}   "
          f"→ {'CLEARS (unexpected!)' if c['clears_floor'] else 'within floor (as expected)'}")
    m = f["mmd"]
    print(f"   MMD²     = {m['mmd2']:+.4f}   p={m['p_value']:.3f}   "
          f"95% upper = {m['upper95']:+.4f}")
    fd = f["fid"]
    print(f"   FID      = {fd['fid']:.3f}   95% CI [{fd['ci95'][0]:.3f}, {fd['ci95'][1]:.3f}]")
    k = f["kid"]
    print(f"   KID      = {k['kid']:+.4f} ± {k['kid_std']:.4f}   "
          f"95% upper = {k['upper95']:+.4f}")
    print(f"   N/half = {f['n_half']}   power_flag = {f['power_flag']}"
          f"{'  ⚠️ wide floor — small N' if f['power_flag'] else ''}")

    if args.out:
        out = {"pair": args.pair, **f}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nResults → {args.out}")


if __name__ == "__main__":
    main()
