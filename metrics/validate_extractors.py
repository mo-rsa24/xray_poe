"""Validate that the heart-size & blunting extractors read real and generated images
the same way (plan 06-05) — the check the Exp5 marginals gate depends on.

If an extractor responds differently to generated images than to real ones, every
joint number in Exp6 is confounded: a gap would be the extractor's bias, not the
model's. So we run the extractors on *matched* single-disease sets (real cardio vs
generated cardio, real effusion vs generated effusion) and require their output
distributions to AGREE.

Agreement metric — a two-sample C2ST on the *extractor outputs* themselves (the 2-D
[heart_size, blunting] vectors), not on image features: train a classifier to tell
real-extractor-outputs from generated ones and read its held-out AUC. AUC ≈ 0.5 →
indistinguishable → the extractor is unbiased across the real/generated boundary;
AUC above the threshold (default 0.60) → the extractor sees the domains differently
and is not safe for Exp6. Reported alongside per-scalar means and a KS test, and the
analytic real-vs-real null floor (`clears_floor` should be False under agreement).

The numeric core (`compare_extractor_dists`) takes (N, 2) extractor-output arrays and
needs only numpy + sklearn + scipy, so it is CPU-runnable and unit-testable without
torch. Image → extractor outputs goes through `metrics.extractors.extract_batch`
(lazy torch). The figure overlays the real vs generated distributions per scalar.

CLI:
    # numeric-core sanity (no images, no torch)
    python -m metrics.validate_extractors --selftest

    # real vs generated cardio (heart_size is the disease-relevant scalar here)
    python -m metrics.validate_extractors \\
        --real data/nih/images_cardio_only --gen outputs/single/cardiomegaly \\
        --out results/extractor_validation_cardio.json

Public API:
    compare_extractor_dists(real_hb, gen_hb, threshold=0.60) -> dict
    make_figure(real_hb, gen_hb, result, out_path)           -> None
    validate(real_dir, gen_dir, ...)                          -> dict   [needs torch]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.c2st import c2st_auc  # noqa: E402

_SCALARS = ("heart_size", "blunting")


# ── numeric core: compare two (N, 2) extractor-output sets ────────────────────────

def _scalar_stats(real: np.ndarray, gen: np.ndarray) -> dict:
    """Per-scalar means, KS two-sample statistic, and 1-D C2ST-AUC."""
    from scipy.stats import ks_2samp

    real = real[np.isfinite(real)]
    gen = gen[np.isfinite(gen)]
    ks = ks_2samp(real, gen)
    c = c2st_auc(real.reshape(-1, 1), gen.reshape(-1, 1))
    return {
        "real_mean": float(real.mean()), "real_std": float(real.std()),
        "gen_mean": float(gen.mean()), "gen_std": float(gen.std()),
        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        "auc": c["auc"], "floor95": c["floor95"], "clears_floor": c["clears_floor"],
    }


def compare_extractor_dists(
    real_hb: np.ndarray,
    gen_hb: np.ndarray,
    threshold: float = 0.60,
    seed: int = 0,
) -> dict:
    """Two-sample agreement of extractor outputs on real vs generated.

    real_hb, gen_hb: (N, 2) arrays, columns [heart_size, blunting]. Returns the joint
    2-D C2ST-AUC (the headline agreement number), per-scalar stats, and an `agree`
    verdict (AUC <= threshold and within the real-vs-real null floor).
    """
    real_hb = np.asarray(real_hb, dtype=np.float64)
    gen_hb = np.asarray(gen_hb, dtype=np.float64)
    if real_hb.ndim != 2 or real_hb.shape[1] != 2:
        raise ValueError("real_hb must be (N, 2) [heart_size, blunting]")
    if gen_hb.ndim != 2 or gen_hb.shape[1] != 2:
        raise ValueError("gen_hb must be (N, 2) [heart_size, blunting]")

    # drop rows with non-finite extractor outputs
    real_hb = real_hb[np.isfinite(real_hb).all(axis=1)]
    gen_hb = gen_hb[np.isfinite(gen_hb).all(axis=1)]

    joint = c2st_auc(real_hb, gen_hb, seed=seed)
    per = {s: _scalar_stats(real_hb[:, i], gen_hb[:, i]) for i, s in enumerate(_SCALARS)}

    agree = bool(joint["auc"] <= threshold and not joint["clears_floor"])
    return {
        "auc_2d": joint["auc"],
        "auc_2d_std": joint["auc_std"],
        "floor95": joint["floor95"],
        "clears_floor": joint["clears_floor"],
        "threshold": threshold,
        "agree": agree,
        "n_real": int(len(real_hb)),
        "n_gen": int(len(gen_hb)),
        "per_scalar": per,
    }


# ── figure ─────────────────────────────────────────────────────────────────────────

def make_figure(real_hb: np.ndarray, gen_hb: np.ndarray, result: dict, out_path: str | Path) -> None:
    """Overlaid real vs generated histograms for each scalar; AUC + verdict in the title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_hb = np.asarray(real_hb, dtype=np.float64)
    gen_hb = np.asarray(gen_hb, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (ax, scalar) in enumerate(zip(axes, _SCALARS)):
        r = real_hb[:, i][np.isfinite(real_hb[:, i])]
        g = gen_hb[:, i][np.isfinite(gen_hb[:, i])]
        lo, hi = float(min(r.min(), g.min())), float(max(r.max(), g.max()))
        bins = np.linspace(lo, hi, 30)
        ax.hist(r, bins=bins, alpha=0.55, label=f"real (μ={r.mean():.3f})", color="#2b7", density=True)
        ax.hist(g, bins=bins, alpha=0.55, label=f"gen (μ={g.mean():.3f})", color="#a3a", density=True)
        ps = result["per_scalar"][scalar]
        ax.set_title(f"{scalar}\n1-D AUC={ps['auc']:.3f}  KS={ps['ks_stat']:.3f} (p={ps['ks_p']:.2f})")
        ax.set_xlabel(scalar)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    verdict = "AGREE ✓" if result["agree"] else "DISAGREE ✗"
    fig.suptitle(
        f"Extractor validation (real vs generated)   "
        f"joint 2-D AUC={result['auc_2d']:.3f} (≤{result['threshold']:.2f}? {verdict}; "
        f"floor95={result['floor95']:.3f})   n_real={result['n_real']} n_gen={result['n_gen']}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ── image → extractor outputs (lazy torch) ────────────────────────────────────────

def _extract_hb(dir_path: str | Path, n: int | None, device: str | None) -> np.ndarray:
    """Run the extractors over a directory → (N, 2) [heart_size, blunting] array."""
    from metrics.extractors import _get_engine, extract_batch

    d = Path(dir_path)
    if not d.is_dir():
        raise SystemExit(f"directory does not exist: {d}")
    paths = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no PNG/JPG images in {d}")
    if n:
        paths = paths[:n]
    eng = _get_engine(device=device)
    out = extract_batch(paths, engine=eng, progress=True)
    return np.column_stack([out["heart_size"], out["blunting"]])


def validate(
    real_dir: str | Path,
    gen_dir: str | Path,
    n: int | None = None,
    threshold: float = 0.60,
    fig_path: str | Path = "figures/extractor_validation.png",
    device: str | None = None,
    seed: int = 0,
) -> dict:
    """Full pipeline: extract on real & generated dirs, compare, save the figure."""
    print(f"Extracting real ({real_dir}) ...")
    real_hb = _extract_hb(real_dir, n, device)
    print(f"Extracting generated ({gen_dir}) ...")
    gen_hb = _extract_hb(gen_dir, n, device)

    result = compare_extractor_dists(real_hb, gen_hb, threshold=threshold, seed=seed)
    make_figure(real_hb, gen_hb, result, fig_path)
    result["figure"] = str(fig_path)
    result["real_dir"] = str(real_dir)
    result["gen_dir"] = str(gen_dir)
    return result


# ── self-test (no images, no torch) ───────────────────────────────────────────────

def _selftest(seed: int = 0) -> bool:
    """Same-distribution extractor outputs → AUC ~0.5 / agree; shifted → AUC high / disagree."""
    rng = np.random.default_rng(seed)
    n = 400
    real = np.column_stack([rng.normal(0.55, 0.15, n), rng.normal(0.06, 0.05, n)])
    gen_same = np.column_stack([rng.normal(0.55, 0.15, n), rng.normal(0.06, 0.05, n)])
    gen_shift = np.column_stack([rng.normal(0.75, 0.15, n), rng.normal(0.20, 0.05, n)])

    a = compare_extractor_dists(real, gen_same, seed=seed)
    b = compare_extractor_dists(real, gen_shift, seed=seed)
    print(f"── matched (expect AUC ~0.5, agree):   AUC={a['auc_2d']:.3f}  agree={a['agree']}")
    print(f"── shifted (expect AUC high, disagree): AUC={b['auc_2d']:.3f}  agree={b['agree']}")
    ok = a["agree"] and a["auc_2d"] <= 0.60 and not b["agree"] and b["auc_2d"] > 0.60
    print(f"\n{'SELFTEST PASS ✓' if ok else 'SELFTEST FAIL ✗'}")
    return ok


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate extractor agreement on real vs generated images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--real", default=None, help="Real single-disease image dir")
    p.add_argument("--gen", default=None, help="Generated single-disease image dir")
    p.add_argument("--n", type=int, default=None, help="Cap images per set")
    p.add_argument("--threshold", type=float, default=0.60, help="Max agreeing 2-D AUC")
    p.add_argument("--fig", default="figures/extractor_validation.png", help="Output figure path")
    p.add_argument("--out", default=None, help="Write results JSON here")
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true",
                   help="Run the matched/shifted sanity check (no images, no torch)")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest(args.seed) else 1)

    if not args.real or not args.gen:
        p.error("provide --real <dir> and --gen <dir> (or --selftest)")

    result = validate(args.real, args.gen, n=args.n, threshold=args.threshold,
                      fig_path=args.fig, device=args.device, seed=args.seed)

    print(f"\n── extractor validation (real vs generated)")
    print(f"   joint 2-D AUC = {result['auc_2d']:.3f} ± {result['auc_2d_std']:.3f}"
          f"   (threshold {result['threshold']:.2f}, floor95 {result['floor95']:.3f})")
    for s in _SCALARS:
        ps = result["per_scalar"][s]
        print(f"   {s:11s}: real μ={ps['real_mean']:.3f}  gen μ={ps['gen_mean']:.3f}"
              f"  1-D AUC={ps['auc']:.3f}  KS p={ps['ks_p']:.3f}")
    print(f"   → {'AGREE ✓ (extractor unbiased across real/gen)' if result['agree'] else 'DISAGREE ✗ (extractor sees domains differently)'}")
    print(f"   figure → {result['figure']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"   results → {args.out}")


if __name__ == "__main__":
    main()
