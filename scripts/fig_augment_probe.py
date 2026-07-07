"""Figure for plan-08's augment-probe (EXPLORATORY): does adding fake cardio help a
frozen-backbone detector? Answer is a NULL, and the figure is built to SHOW the flatness
honestly, not to dress it up.

Two panels — cardio AUC (threshold-free) and sensitivity at 0.90 specificity — each with the
three training arms A0 (real only) / A1 (real+all fake) / A2 (real+big-heart fake) on the x.
One thin line per seed connects an arm's three values; a bold line marks the per-arm mean.
Both synthetic sources are shown (w6 solid, w4 dashed) to prove the null is robust to the
generation setting. Y-axis pinned 0.0-1.0 — never zoomed — so seed noise cannot masquerade
as a real effect (a zoomed axis would turn ±0.005 into a fake-looking win).

Deliberate deviation from the plan's original caption: it assumed an A2>A1>A0 "small win" and
0.365 / 0.80 reference marks. Those marks are a DIFFERENT quantity (detector sensitivity on
synthetic cardio, Exp-09) than what this probe measures (a retrained detector's performance on
REAL held-out cardio), so mixing them onto this axis would mislead. The real-only baseline is
arm A0 itself; we annotate that instead.

Usage: python scripts/fig_augment_probe.py
Out:   results/section4/augment_probe/fig_augment_probe.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results/section4/augment_probe"
ARMS = ["A0_real", "A1_real+all", "A2_real+big"]
ARM_LABELS = ["A0\nreal only", "A1\nreal + all fake", "A2\nreal + big-heart fake"]
METRICS = [("auc", "cardio AUC  (threshold-free)"),
           ("sens@spec0.9", "cardio sensitivity @ 0.90 specificity")]
SOURCES = [("w6", "-", "#1f77b4"), ("w4", "--", "#ff7f0e")]


def main() -> None:
    runs = {s: pd.read_csv(OUT / f"probe_runs_{s}.csv") for s, _, _ in SOURCES}
    x = np.arange(len(ARMS))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, (met, mlabel) in zip(axes, METRICS):
        for src, ls, col in SOURCES:
            df = runs[src]
            for seed in sorted(df["seed"].unique()):
                ys = [df[(df.arm == a) & (df.seed == seed)][met].values[0] for a in ARMS]
                ax.plot(x, ys, ls, color=col, alpha=0.28, lw=1.0, zorder=2)
            means = [df[df.arm == a][met].mean() for a in ARMS]
            ax.plot(x, means, ls, color=col, lw=2.6, marker="o", ms=8,
                    label=f"{src} mean (3 seeds)", zorder=4)
        # real-only baseline (A0 mean, pooled sources) as the honest reference
        base = np.mean([runs[s][runs[s].arm == "A0_real"][met].mean() for s, _, _ in SOURCES])
        ax.axhline(base, color="grey", ls=":", lw=1.2, zorder=1)
        ax.text(2.02, base, f" real-only\n baseline {base:.3f}", va="center",
                fontsize=8, color="grey")
        ax.set_xticks(x); ax.set_xticklabels(ARM_LABELS, fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(mlabel, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.margins(x=0.12)
    axes[0].set_ylabel("score  (axis pinned 0–1, never zoomed)")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle("Augment-probe (exploratory): fake cardio does NOT help the frozen-backbone detector",
                 fontsize=12.5, y=0.99)
    fig.text(0.5, 0.03,
             "NULL: A2 does not beat A0 and A2 does not beat A1 — every arm sits inside every other arm's "
             "seed spread, for both the w6 and w4 synthetic sources.\nFlatness across arms AND source points at "
             "a frozen-backbone ceiling, not fake-heart quality. Backs the 'model under-draws cardio' finding; "
             "does not overturn the paper's story.",
             ha="center", fontsize=8.2, wrap=True)
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    out = OUT / "fig_augment_probe.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
