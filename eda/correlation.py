#!/usr/bin/env python3
"""
Exp1 — Pairwise label correlation matrix (NIH ChestX-ray14).

Computes the full N_labels × N_labels phi-coefficient matrix from the label CSV alone
(zero GPU-hours), then:
  1. Prints the top-K and bottom-K pairs by |phi|.
  2. Reports the treatment pair (Cardiomegaly × Effusion) and flags the best
     control candidate (pair with phi ≈ 0).
  3. Saves a labelled heatmap PNG with the treatment and control pairs ringed.
  4. Prints a go/no-go gate verdict.

Decision rule (from EXPERIMENTS.md Exp1):
  ✅ go   — treatment phi significantly positive AND a near-zero control pair exists.
  ❌ pivot — treatment phi ≤ PHI_FLOOR or no control pair within PHI_CTRL_BAND of 0.

Usage:
  python eda/correlation.py \\
      --csv data/nih/Data_Entry_2017.csv \\
      --out eda/out/correlation_heatmap.png \\
      --treatment Cardiomegaly Effusion \\
      --topk 10

  # Quickly inspect on any CSV that has 'Finding Labels' (pipe-separated):
  python eda/correlation.py --csv data/nih/Data_Entry_2017.csv --no-plot

Deps: numpy, pandas, matplotlib, seaborn.
  pip install numpy pandas matplotlib seaborn
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# Gate thresholds (tunable via CLI)
PHI_FLOOR = 0.05       # treatment phi must exceed this to call "go"
PHI_CTRL_BAND = 0.05   # control phi must be within ±this of 0

NORMAL_LABEL = "No Finding"

# NIH ChestX-ray14 canonical 14 pathology labels (order used in the heatmap)
NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_binary_matrix(
    csv_path: str,
    label_col: str = "Finding Labels",
    sep: str = "|",
    labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (bool matrix N×L, label list L) from a wide-format label CSV."""
    if pd is None:
        sys.exit("pandas required: pip install pandas")
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        sys.exit(f"column '{label_col}' not found; columns: {list(df.columns)}")

    sets = df[label_col].fillna("").map(
        lambda s: {x.strip() for x in str(s).split(sep) if x.strip()}
    )

    if labels is None:
        # Derive from data, drop normal label, sort
        all_labels: set[str] = set()
        for s in sets:
            all_labels |= s
        all_labels.discard(NORMAL_LABEL)
        labels = sorted(all_labels)

    mat = np.column_stack([sets.map(lambda s, l=l: l in s).to_numpy() for l in labels])
    return mat.astype(bool), labels


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class PairStat:
    a: str
    b: str
    n11: int; n10: int; n01: int; n00: int
    phi: float
    odds_ratio: float; or_lo: float; or_hi: float
    chi2: float; p: float


def pair_stat(a_vec: np.ndarray, b_vec: np.ndarray, a_name: str, b_name: str) -> PairStat:
    n11 = int(np.sum(a_vec & b_vec))
    n10 = int(np.sum(a_vec & ~b_vec))
    n01 = int(np.sum(~a_vec & b_vec))
    n00 = int(np.sum(~a_vec & ~b_vec))
    N = n11 + n10 + n01 + n00

    r1, r0 = n11 + n10, n01 + n00
    c1, c0 = n11 + n01, n10 + n00
    denom = math.sqrt(r1 * r0 * c1 * c0) if (r1 and r0 and c1 and c0) else float("nan")
    phi = (n11 * n00 - n10 * n01) / denom if (denom and not math.isnan(denom)) else float("nan")

    # Haldane-Anscombe 0.5 correction if any zero cell
    a_, b_, c_, d_ = n11, n10, n01, n00
    if 0 in (a_, b_, c_, d_):
        a_, b_, c_, d_ = a_ + 0.5, b_ + 0.5, c_ + 0.5, d_ + 0.5
    odds = (a_ * d_) / (b_ * c_)
    se = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    or_lo = math.exp(math.log(odds) - 1.96 * se)
    or_hi = math.exp(math.log(odds) + 1.96 * se)

    chi2 = (N * (abs(n11 * n00 - n10 * n01) - N / 2) ** 2) / (r1 * r0 * c1 * c0) if denom else float("nan")
    chi2 = max(chi2, 0.0)
    p = math.erfc(math.sqrt(chi2 / 2)) if not math.isnan(chi2) else float("nan")

    return PairStat(a_name, b_name, n11, n10, n01, n00, phi, odds, or_lo, or_hi, chi2, p)


def build_phi_matrix(mat: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[list[PairStat]]]:
    """Return (L×L phi matrix, L×L PairStat grid) — upper triangle filled; diagonal = nan."""
    L = len(labels)
    phi_mat = np.full((L, L), float("nan"))
    stats: list[list[PairStat | None]] = [[None] * L for _ in range(L)]

    for i in range(L):
        phi_mat[i, i] = float("nan")
        for j in range(i + 1, L):
            ps = pair_stat(mat[:, i], mat[:, j], labels[i], labels[j])
            phi_mat[i, j] = ps.phi
            phi_mat[j, i] = ps.phi
            stats[i][j] = ps
            stats[j][i] = ps

    return phi_mat, stats  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def top_pairs(phi_mat: np.ndarray, labels: list[str], k: int) -> list[tuple[float, int, int]]:
    """Return k pairs sorted by descending |phi| (upper triangle only)."""
    L = len(labels)
    pairs = [
        (phi_mat[i, j], i, j)
        for i in range(L) for j in range(i + 1, L)
        if not math.isnan(phi_mat[i, j])
    ]
    pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    return pairs[:k]


def bottom_pairs(phi_mat: np.ndarray, labels: list[str], k: int) -> list[tuple[float, int, int]]:
    """Return k pairs sorted by ascending |phi| (nearest to 0)."""
    L = len(labels)
    pairs = [
        (phi_mat[i, j], i, j)
        for i in range(L) for j in range(i + 1, L)
        if not math.isnan(phi_mat[i, j])
    ]
    pairs.sort(key=lambda x: abs(x[0]))
    return pairs[:k]


def render_report(
    phi_mat: np.ndarray,
    stats_grid: list[list[PairStat]],
    labels: list[str],
    treatment: tuple[str, str],
    k: int,
    phi_floor: float,
    ctrl_band: float,
) -> str:
    L = [""]
    w = L.append

    w("# Exp1 — Pairwise label correlation (NIH ChestX-ray14)\n")
    w(f"Labels: {len(labels)}  |  Pairs: {len(labels) * (len(labels) - 1) // 2}\n")

    # Treatment pair
    ta, tb = treatment
    try:
        ti, tj = labels.index(ta), labels.index(tb)
    except ValueError:
        sys.exit(f"treatment label not found in dataset: {ta} or {tb}")

    ps = stats_grid[ti][tj]
    w("## Treatment pair\n")
    w(f"  **{ta} × {tb}**")
    w(f"  phi = {ps.phi:+.4f}  |  OR = {ps.odds_ratio:.2f} [{ps.or_lo:.2f}, {ps.or_hi:.2f}]  |  p = {ps.p:.2e}")
    w(f"  n(A∧B) = {ps.n11:,}  n(A only) = {ps.n10:,}  n(B only) = {ps.n01:,}  n(neither) = {ps.n00:,}")
    w("")

    # Gate verdict
    go = (not math.isnan(ps.phi)) and (ps.phi > phi_floor)
    w("## Go / No-Go gate\n")
    if go:
        w(f"  ✅ **GO** — treatment phi={ps.phi:+.4f} > threshold {phi_floor}")
    else:
        w(f"  ❌ **PIVOT** — treatment phi={ps.phi:+.4f} ≤ threshold {phi_floor}")
        w("     Consider a stronger treatment pair or revisit framing.")
    w("")

    # Top-K
    w(f"## Top-{k} pairs by |phi|\n")
    w("| rank | pair | phi | OR [95% CI] | n(A∧B) |")
    w("|------|------|-----|-------------|--------|")
    for rank, (phi, i, j) in enumerate(top_pairs(phi_mat, labels, k), 1):
        ps2 = stats_grid[i][j]
        marker = " ← treatment" if (labels[i], labels[j]) in {(ta, tb), (tb, ta)} else ""
        w(f"| {rank} | {labels[i]} × {labels[j]}{marker} | {phi:+.4f} | "
          f"{ps2.odds_ratio:.2f} [{ps2.or_lo:.2f}, {ps2.or_hi:.2f}] | {ps2.n11:,} |")
    w("")

    # Bottom-K (control candidates)
    w(f"## Bottom-{k} pairs by |phi|  (control candidates)\n")
    w("| rank | pair | phi | OR [95% CI] | n(A∧B) |")
    w("|------|------|-----|-------------|--------|")
    best_ctrl: tuple[float, int, int] | None = None
    for rank, (phi, i, j) in enumerate(bottom_pairs(phi_mat, labels, k), 1):
        ps2 = stats_grid[i][j]
        in_band = abs(phi) <= ctrl_band
        tag = " ← best control" if (rank == 1 and in_band) else (" (near-zero)" if in_band else "")
        if rank == 1 and in_band:
            best_ctrl = (phi, i, j)
        w(f"| {rank} | {labels[i]} × {labels[j]}{tag} | {phi:+.4f} | "
          f"{ps2.odds_ratio:.2f} [{ps2.or_lo:.2f}, {ps2.or_hi:.2f}] | {ps2.n11:,} |")
    w("")

    if best_ctrl is not None:
        _, ci, cj = best_ctrl
        w(f"  ✅ Control pair: **{labels[ci]} × {labels[cj]}**  phi={phi_mat[ci, cj]:+.4f}")
    else:
        w(f"  ⚠️  No pair within ±{ctrl_band} of 0 — controlled comparison weakens.")
    w("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(
    phi_mat: np.ndarray,
    labels: list[str],
    out_path: str,
    treatment: tuple[str, str],
    control: tuple[str, str] | None = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import seaborn as sns
    except ImportError:
        print("[warn] matplotlib/seaborn not available — skipping heatmap", file=sys.stderr)
        return

    L = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, L * 0.65), max(7, L * 0.6)))

    mask = np.eye(L, dtype=bool)  # mask diagonal (nan)
    sns.heatmap(
        phi_mat, mask=mask,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.3,
        linewidths=0.3, linecolor="white",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "phi coefficient", "shrink": 0.7},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_title("NIH ChestX-ray14 — pairwise phi-coefficient matrix\n"
                 "(Exp1: pair selection gate)", fontsize=11, pad=12)

    def ring_pair(name_a: str, name_b: str, color: str, lw: float = 2.0) -> None:
        if name_a not in labels or name_b not in labels:
            return
        i, j = labels.index(name_a), labels.index(name_b)
        for (r, c) in [(i, j), (j, i)]:
            ax.add_patch(mpatches.Rectangle(
                (c, r), 1, 1,
                fill=False, edgecolor=color, linewidth=lw, zorder=5,
            ))

    ring_pair(*treatment, color="lime", lw=2.5)
    if control is not None:
        ring_pair(*control, color="cyan", lw=2.0)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor="lime", linewidth=2, label="treatment pair"),
    ]
    if control is not None:
        legend_handles.append(
            mpatches.Patch(facecolor="none", edgecolor="cyan", linewidth=2, label="control pair")
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.8, bbox_to_anchor=(1.0, 1.12))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", "--csv", dest="manifest",
                    default="data/nih/Data_Entry_2017.csv",
                    help="NIH wide-format label CSV (default: data/nih/Data_Entry_2017.csv)")
    ap.add_argument("--label-col", default="Finding Labels",
                    help="column name for the pipe-separated labels")
    ap.add_argument("--treatment", nargs=2, default=["Cardiomegaly", "Effusion"],
                    metavar=("A", "B"), help="treatment pair labels (default: Cardiomegaly Effusion)")
    ap.add_argument("--control", nargs=2, default=None, metavar=("A", "B"),
                    help="explicit control pair; auto-selected from bottom-1 if omitted")
    ap.add_argument("--topk", type=int, default=10, help="pairs to show in top/bottom tables")
    ap.add_argument("--phi-floor", type=float, default=PHI_FLOOR,
                    help=f"go/no-go gate: treatment phi must exceed this (default {PHI_FLOOR})")
    ap.add_argument("--ctrl-band", type=float, default=PHI_CTRL_BAND,
                    help=f"control pair must have |phi| ≤ this (default {PHI_CTRL_BAND})")
    ap.add_argument("--out", default="eda/out/correlation_heatmap.png",
                    help="output heatmap PNG path")
    ap.add_argument("--no-plot", action="store_true", help="skip heatmap, print report only")
    ap.add_argument("--report-out", default=None, help="also write the text report to this file")
    args = ap.parse_args()

    print(f"[load] {args.manifest}", file=sys.stderr)
    mat, labels = load_binary_matrix(args.manifest, label_col=args.label_col)
    print(f"[info] {mat.shape[0]:,} images × {mat.shape[1]} labels", file=sys.stderr)

    print("[compute] phi matrix ...", file=sys.stderr)
    phi_mat, stats_grid = build_phi_matrix(mat, labels)

    treatment = tuple(args.treatment)  # type: ignore[arg-type]

    report = render_report(
        phi_mat, stats_grid, labels,
        treatment=treatment,
        k=args.topk,
        phi_floor=args.phi_floor,
        ctrl_band=args.ctrl_band,
    )
    print(report)

    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w") as f:
            f.write(report + "\n")
        print(f"[saved] {args.report_out}", file=sys.stderr)

    if not args.no_plot:
        # Auto-detect best control if not supplied
        control: tuple[str, str] | None = tuple(args.control) if args.control else None  # type: ignore[assignment]
        if control is None:
            bp = bottom_pairs(phi_mat, labels, 1)
            if bp:
                _, ci, cj = bp[0]
                control = (labels[ci], labels[cj])

        plot_heatmap(phi_mat, labels, args.out, treatment=treatment, control=control)


if __name__ == "__main__":
    main()
