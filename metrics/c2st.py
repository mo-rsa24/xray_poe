"""Two-sample tests for "are these two image sets the same distribution?".

Two complementary metrics, both reported on a shared feature space:

  C2ST (Classifier Two-Sample Test)
      Train a classifier to separate set A (real) from set B (generated) and report
      its held-out ROC-AUC, cross-validated so the score is never read off training
      data. AUC ≈ 0.50 → the two sets are indistinguishable (good for composition);
      AUC ≈ 1.0 → trivially separable (bad). This is the Exp5/6/7 composition metric.
      Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests" (ICLR 2017).

  MMD (Maximum Mean Discrepancy)
      A kernel distance between the two feature distributions. The unbiased MMD²
      estimate is ≈ 0 for samples from the same distribution and grows with the gap;
      a permutation test turns it into a p-value. Gretton et al., JMLR 2012.

The numeric core (`c2st_auc`, `mmd_rbf`) operates on feature arrays (N, D) and depends
only on numpy + scikit-learn, so it is CPU-runnable and unit-testable without torch or
any checkpoint. Image → feature extraction (`extract_features`) lazily imports
torchxrayvision and uses the pretrained NIH DenseNet-121 penultimate layer (1024-d) as
the feature space, sharing the presence-classifier preprocessing.

The reported C2ST-AUC is compared against the analytic real-vs-real null floor from
`eda.floor_power_check` (the smallest AUC a same-distribution pair can be expected to
sit below 95% of the time at this sample size): a treatment AUC must clear that floor
to count as a real distribution gap.

CLI:
    # sanity checks (no images, no torch needed):
    python -m metrics.c2st --selftest

    # identical sets  → AUC ~0.5,  MMD ~0
    python -m metrics.c2st --a real_a/ --b real_a/

    # obviously different → AUC ~1.0, MMD large
    python -m metrics.c2st --a real/ --b noise/

    # treatment-only C2ST vs floor (Sprint S4):
    python -m metrics.c2st --a data/nih/.../both --b outputs/compose/w1p0/ \\
        --out results/exp6_c2st.json

    # operate directly on precomputed feature arrays (.npy of shape (N, D)):
    python -m metrics.c2st --features_a a.npy --features_b b.npy

Public API:
    c2st_auc(feats_a, feats_b, ...)        -> dict (auc, auc_std, per-fold, n, floor)
    mmd_rbf(feats_a, feats_b, ...)         -> dict (mmd2, p_value, gamma)
    extract_features(paths, ...)           -> np.ndarray (N, 1024)   [needs torch+xrv]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# Analytic real-vs-real null floor for the C2ST-AUC (labels-only, distribution-free).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eda.floor_power_check import floor_upper_analytic  # noqa: E402


# ── C2ST ────────────────────────────────────────────────────────────────────────

def _make_classifier(kind: str, seed: int):
    """Build a fresh sklearn classifier. logreg is the standard, well-calibrated
    C2ST baseline; rf/mlp capture nonlinear gaps at the cost of more overfit margin."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        )
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                          early_stopping=True, random_state=seed),
        )
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown classifier kind: {kind!r} (logreg|mlp|rf)")


def c2st_auc(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    clf: str = "logreg",
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Classifier two-sample test: held-out ROC-AUC separating A from B.

    A is labelled 0 (e.g. real), B labelled 1 (e.g. generated). A stratified k-fold
    classifier is trained; out-of-fold predicted probabilities are pooled into a single
    AUC, and per-fold AUCs give a spread. AUC is folded to [0.5, 1] in expectation
    (a same-distribution pair sits near 0.5; the sign of separation is not meaningful).

    Returns:
        auc          pooled out-of-fold ROC-AUC
        auc_std      std of per-fold AUCs
        auc_folds    list of per-fold AUCs
        n_a, n_b     sample sizes
        floor95      analytic real-vs-real null 95% upper bound at this size
        clears_floor auc > floor95 (a real, resolvable distribution gap)
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    feats_a = np.asarray(feats_a, dtype=np.float64)
    feats_b = np.asarray(feats_b, dtype=np.float64)
    if feats_a.ndim != 2 or feats_b.ndim != 2:
        raise ValueError("feats_a, feats_b must be 2-D (N, D)")
    if feats_a.shape[1] != feats_b.shape[1]:
        raise ValueError(f"feature dim mismatch: {feats_a.shape[1]} vs {feats_b.shape[1]}")

    # Overlap guard: rows present in BOTH sets are not evidence of a distributional
    # difference — included with contradictory labels they only inject anti-correlated
    # leakage that *inflates* AUC (the degenerate `real_a/ real_a/` sanity case drives
    # AUC → 1). Drop the exact-duplicate overlap; for the real treatment comparison
    # (disjoint real vs generated) this is a no-op.
    n_a0, n_b0 = len(feats_a), len(feats_b)
    a_keys = [r.tobytes() for r in feats_a]
    b_keys = [r.tobytes() for r in feats_b]
    overlap = set(a_keys) & set(b_keys)
    n_overlap = sum(k in overlap for k in a_keys) + sum(k in overlap for k in b_keys)
    if overlap:
        feats_a = feats_a[[k not in overlap for k in a_keys]]
        feats_b = feats_b[[k not in overlap for k in b_keys]]

    # If the sets are identical / fully overlapping, nothing distinguishes them → 0.5.
    if min(len(feats_a), len(feats_b)) < 2:
        return {
            "auc": 0.5, "auc_std": 0.0, "auc_folds": [],
            "n_a": n_a0, "n_b": n_b0, "n_overlap": n_overlap, "n_folds": 0,
            "clf": clf, "floor95": float(floor_upper_analytic(max(n_a0, 1))),
            "clears_floor": False, "degenerate": True,
        }

    X = np.vstack([feats_a, feats_b])
    y = np.concatenate([np.zeros(len(feats_a)), np.ones(len(feats_b))]).astype(int)

    n_eff = min(len(feats_a), len(feats_b))
    k = max(2, min(n_splits, n_eff))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    oof = np.full(len(y), np.nan)
    fold_aucs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        model = _make_classifier(clf, seed)
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = prob
        # per-fold AUC needs both classes present in the test fold (guaranteed by stratify)
        fold_aucs.append(float(roc_auc_score(y[test_idx], prob)))

    auc = float(roc_auc_score(y, oof))
    # fold to [0.5, 1]: separability is symmetric, only the magnitude of the gap matters
    auc = max(auc, 1.0 - auc)
    fold_aucs = [max(a, 1.0 - a) for a in fold_aucs]

    # floor: per-side m is the smaller sample (the limiting half)
    floor95 = floor_upper_analytic(n_eff)

    return {
        "auc": auc,
        "auc_std": float(np.std(fold_aucs)),
        "auc_folds": fold_aucs,
        "n_a": int(len(feats_a)),
        "n_b": int(len(feats_b)),
        "n_overlap": n_overlap,
        "n_folds": k,
        "clf": clf,
        "floor95": float(floor95),
        "clears_floor": bool(auc > floor95),
    }


# ── MMD ───────────────────────────────────────────────────────────────────────

def _median_bandwidth(X: np.ndarray, Y: np.ndarray, max_n: int = 1000,
                      rng: np.random.Generator | None = None) -> float:
    """RBF gamma from the median pairwise-squared-distance heuristic over a pooled
    subsample. Returns gamma = 1 / (2 * median_sq_dist)."""
    from sklearn.metrics import pairwise_distances

    rng = rng or np.random.default_rng(0)
    pooled = np.vstack([X, Y])
    if len(pooled) > max_n:
        pooled = pooled[rng.choice(len(pooled), max_n, replace=False)]
    d2 = pairwise_distances(pooled, metric="sqeuclidean")
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    if med <= 0:
        return 1.0
    return float(1.0 / (2.0 * med))


def _mmd2_unbiased(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    """Unbiased MMD² estimate (Gretton et al. 2012, eq. with diagonal removed)."""
    m = Kxx.shape[0]
    n = Kyy.shape[0]
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sum_xy = Kxy.mean()
    return float(sum_xx + sum_yy - 2 * sum_xy)


def mmd_rbf(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    gamma: float | None = None,
    n_perm: int = 200,
    seed: int = 0,
) -> dict:
    """RBF-kernel MMD² between two feature sets, with a permutation-test p-value.

    gamma defaults to the median-distance heuristic. The permutation test shuffles
    the A/B assignment `n_perm` times to build the null of MMD² under "same
    distribution"; p = P(MMD²_perm >= MMD²_observed).

    Returns mmd2 (unbiased, can be slightly negative under H0), gamma, p_value.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    feats_a = np.asarray(feats_a, dtype=np.float64)
    feats_b = np.asarray(feats_b, dtype=np.float64)
    if feats_a.shape[1] != feats_b.shape[1]:
        raise ValueError(f"feature dim mismatch: {feats_a.shape[1]} vs {feats_b.shape[1]}")

    rng = np.random.default_rng(seed)
    if gamma is None:
        gamma = _median_bandwidth(feats_a, feats_b, rng=rng)

    m, n = len(feats_a), len(feats_b)
    pooled = np.vstack([feats_a, feats_b])
    K = rbf_kernel(pooled, pooled, gamma=gamma)        # (m+n, m+n) full Gram matrix

    def split_mmd2(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        Kxx = K[np.ix_(idx_a, idx_a)]
        Kyy = K[np.ix_(idx_b, idx_b)]
        Kxy = K[np.ix_(idx_a, idx_b)]
        return _mmd2_unbiased(Kxx, Kyy, Kxy)

    idx = np.arange(m + n)
    mmd2_obs = split_mmd2(idx[:m], idx[m:])

    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(m + n)
        mmd2_p = split_mmd2(perm[:m], perm[m:])
        if mmd2_p >= mmd2_obs:
            ge += 1
    p_value = (ge + 1) / (n_perm + 1)                  # +1 smoothing (never exactly 0)

    return {
        "mmd2": mmd2_obs,
        "gamma": float(gamma),
        "p_value": float(p_value),
        "n_perm": n_perm,
        "n_a": int(m),
        "n_b": int(n),
    }


# ── image → feature extraction (lazy torch + torchxrayvision) ────────────────────

def _gather_paths(dir_path: str | Path, n: int | None, seed: int = 42) -> list[Path]:
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise RuntimeError(
            f"Directory does not exist: {dir_path}  "
            f"(the spec's `real_a/`, `noise/`, `real/` are placeholders — "
            f"point --a/--b at actual image dirs, e.g. data/nih/images_cardio_only)"
        )
    paths = sorted(dir_path.glob("*.png")) + sorted(dir_path.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No PNG/JPG images in {dir_path}")
    if n is not None and n < len(paths):
        rng = random.Random(seed)
        paths = sorted(rng.sample(paths, n))
    return paths


def extract_features(
    paths: Sequence[str | Path],
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Pooled 1024-d features from the pretrained NIH DenseNet-121 penultimate layer.

    Shares the presence-classifier preprocessing (xrv normalise → centre-crop → 224).
    Lazily imports torch / torchxrayvision so the numeric core stays import-light.
    Returns (N, 1024) float32.
    """
    import torch
    import torchxrayvision as xrv

    from metrics.presence_classifier import _preprocess  # shared preprocessing

    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()

    feats: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        chunk = list(paths[i : i + batch_size])
        tensors: list[torch.Tensor] = []
        for p in chunk:
            try:
                tensors.append(_preprocess(p))
            except Exception as exc:
                print(f"  [warn] skipping {Path(p).name}: {exc}")
                tensors.append(torch.zeros(1, 224, 224))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            f = model.features2(batch).cpu().numpy()   # (B, 1024)
        feats.append(f)
    if not feats:
        return np.empty((0, 1024), dtype=np.float32)
    return np.concatenate(feats, axis=0).astype(np.float32)


# ── self-test (sanity checks; no images, no torch) ───────────────────────────────

def _selftest(seed: int = 0) -> bool:
    """Identical sets → AUC ~0.5 / MMD ~0; shifted sets → AUC ~1.0 / MMD large."""
    rng = np.random.default_rng(seed)
    d, n = 32, 400
    base = rng.standard_normal((n, d))
    same = rng.standard_normal((n, d))                 # same distribution
    shifted = rng.standard_normal((n, d)) + 3.0        # obviously different (mean shift)

    print("── sanity check: IDENTICAL distribution (expect AUC ~0.5, MMD ~0, p large)")
    c_same = c2st_auc(base, same, seed=seed)
    m_same = mmd_rbf(base, same, seed=seed)
    print(f"   C2ST AUC = {c_same['auc']:.3f} ± {c_same['auc_std']:.3f}"
          f"   (floor95={c_same['floor95']:.3f}, clears={c_same['clears_floor']})")
    print(f"   MMD²     = {m_same['mmd2']:+.4f}   p={m_same['p_value']:.3f}")

    print("\n── sanity check: SHIFTED distribution (expect AUC ~1.0, MMD large, p small)")
    c_diff = c2st_auc(base, shifted, seed=seed)
    m_diff = mmd_rbf(base, shifted, seed=seed)
    print(f"   C2ST AUC = {c_diff['auc']:.3f} ± {c_diff['auc_std']:.3f}"
          f"   (floor95={c_diff['floor95']:.3f}, clears={c_diff['clears_floor']})")
    print(f"   MMD²     = {m_diff['mmd2']:+.4f}   p={m_diff['p_value']:.3f}")

    ok = (
        c_same["auc"] < 0.60 and not c_same["clears_floor"] and m_same["p_value"] > 0.05
        and c_diff["auc"] > 0.95 and c_diff["clears_floor"] and m_diff["p_value"] < 0.05
    )
    print(f"\n{'SELFTEST PASS ✓' if ok else 'SELFTEST FAIL ✗'}")
    return ok


# ── CLI ──────────────────────────────────────────────────────────────────────────

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
        print(f"  set {which.upper()}: extracting features from {len(paths)} images in {img} ...")
        feats = extract_features(paths, device=args.device, batch_size=args.batch_size)
        return feats
    raise SystemExit(f"provide --{which} <image_dir> or --features_{which} <file.npy>")


def main() -> None:
    p = argparse.ArgumentParser(
        description="C2ST + MMD two-sample tests on a feature space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--a", default=None, help="Image dir for set A (e.g. real)")
    p.add_argument("--b", default=None, help="Image dir for set B (e.g. generated)")
    p.add_argument("--features_a", default=None, help="Precomputed feature .npy for A (N, D)")
    p.add_argument("--features_b", default=None, help="Precomputed feature .npy for B (N, D)")
    p.add_argument("--n", type=int, default=None, help="Cap images per set")
    p.add_argument("--clf", choices=["logreg", "mlp", "rf"], default="logreg",
                   help="C2ST classifier (default logreg = standard C2ST)")
    p.add_argument("--n_splits", type=int, default=5, help="C2ST cross-val folds")
    p.add_argument("--n_perm", type=int, default=200, help="MMD permutation-test reps")
    p.add_argument("--no_mmd", action="store_true", help="Skip MMD (C2ST only)")
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
    print(f"  A: {feats_a.shape}   B: {feats_b.shape}\n")

    out: dict = {
        "set_a": args.features_a or args.a,
        "set_b": args.features_b or args.b,
    }

    print("── C2ST (classifier two-sample test)")
    c = c2st_auc(feats_a, feats_b, clf=args.clf, n_splits=args.n_splits, seed=args.seed)
    out["c2st"] = c
    if c.get("degenerate"):
        print(f"   AUC = 0.500 (degenerate: sets fully overlap, {c['n_overlap']} shared rows)")
    else:
        if c["n_overlap"]:
            print(f"   [note] dropped {c['n_overlap']} rows shared between A and B before training")
        print(f"   AUC = {c['auc']:.3f} ± {c['auc_std']:.3f}   ({c['clf']}, {c['n_folds']} folds)")
        print(f"   floor95 (real-vs-real null) = {c['floor95']:.3f}"
              f"   → {'CLEARS floor (real gap)' if c['clears_floor'] else 'within floor (indistinguishable)'}")

    if not args.no_mmd:
        print("\n── MMD (maximum mean discrepancy, RBF)")
        m = mmd_rbf(feats_a, feats_b, n_perm=args.n_perm, seed=args.seed)
        out["mmd"] = m
        print(f"   MMD² = {m['mmd2']:+.4f}   gamma={m['gamma']:.4g}   p={m['p_value']:.3f}"
              f"  ({'distinguishable' if m['p_value'] < 0.05 else 'not distinguishable'})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
