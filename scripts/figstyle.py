"""
figstyle.py — the single source of truth for the paper's figure look.

Every figure producer imports this and calls `figstyle.use()` once at the top,
then pulls colours from `COLORS` / `color(key)` and draws reference lines and
CI'd bars through the helpers here. This is what makes the set look like one set
(locked in plans/FIGURE_SPEC.md, Part 4).

Usage
-----
    import figstyle                     # from scripts/
    figstyle.use()
    fig, ax = plt.subplots(figsize=figstyle.SIZE_1COL)
    ax.bar(x, y, color=figstyle.color("poe"))
    figstyle.auc_axis(ax)              # honesty: AUC y starts at 0.5
    figstyle.chance_line(ax)

From metrics/ (or any sibling dir):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    import figstyle

Honesty constraints enforced here (do not bypass):
  - AUC/accuracy axes start at 0.5 (chance), never at the lowest bar.
  - `bar_with_ci` refuses to draw M5/M6 comparison bars without a CI.
  - `provisional(ax, ...)` stamps a visible "not-yet-fair" banner on any figure
    whose data is single-seed / before-model / dead-run (FIGURE_SPEC Part 6).
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Locked colour scheme (FIGURE_SPEC Part 4). One meaning per colour.
# --------------------------------------------------------------------------- #
# Palette = Okabe–Ito, colour-blind-safe (safe for red-green and blue-yellow
# colour blindness). One fixed colour per meaning; never reuse across meanings.
COLORS = {
    # image source — real, or which model made the fake
    "real":        "#000000",   # real / ground truth (black)
    "synth_before":"#D55E00",   # fake, old 40k model (NOT trustworthy) — vermillion
    "synth_after": "#0072B2",   # fake, new Arm-1 model — blue
    "poe":         "#0072B2",   # +PoE / our method (same as new-model, by design)
    # training recipes (the augmentation arms)
    "starved":     "#666666",   # no help — dark grey floor
    "individual":  "#E69F00",   # add single-disease fakes — orange
    "overlay":     "#B0B0B0",   # add pasted-together fakes — light grey (recessive)
    "oracle":      "#009E73",   # best-case ceiling (real both-disease) — bluish green
    "monolithic":  "#CC79A7",   # one model on both — reddish purple
    # disease groups (for grouping bars by category)
    "normal":      "#999999",   # grey
    "cardio":      "#E69F00",   # orange
    "effusion":    "#56B4E9",   # sky blue
    "both":        "#CC79A7",   # reddish purple
    # segmentation outline colours — ONLY for the anatomy pictures,
    # never as data-bar colours (they carry a different meaning).
    "seg_cardio":  "#D55E00",   # heart / red-line
    "seg_effusion":"#0072B2",   # effusion / CP region
    "seg_lung":    "#009E73",   # lung field / blue-line
    # neutral furniture
    "reference":   "#666666",   # dashed reference lines (0.5, 30%, chance)
    "grid":        "#e6e6e6",
    "annot":       "#444444",
}

# fixed colour per disease group, in reading order
DISEASE_ORDER = ["normal", "cardio", "effusion", "both"]

# canonical draw order + labels for the 5-arm augmentation figure (M5)
ARM_ORDER  = ["starved", "individual", "overlay", "poe", "oracle"]
ARM_LABELS = {"starved": "Starved", "individual": "+Individual",
              "overlay": "+Overlay", "poe": "+PoE", "oracle": "Oracle"}

# figure sizes (inches) at the paper's column geometry
SIZE_1COL = (3.4, 2.6)   # single column ~4:3
SIZE_2COL = (7.0, 2.8)   # full width, side-by-side panels
SIZE_GRID = (7.0, 3.6)   # 4-col Grad-CAM grid etc.

DPI = 300


def color(key: str) -> str:
    """Colour by semantic key. Raises on unknown keys so typos can't sneak in an off-palette colour."""
    try:
        return COLORS[key]
    except KeyError as e:
        raise KeyError(f"figstyle: unknown colour key {key!r}; known: {sorted(COLORS)}") from e


# --------------------------------------------------------------------------- #
# rcParams — apply once per process.
# --------------------------------------------------------------------------- #
def use() -> None:
    """Install the locked rcParams. Call once at the top of every producer."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",

        # text sized to stay readable in print at 300 DPI
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # clean spines: drop top+right, keep a thin left+bottom
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,

        # light horizontal gridlines only
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,

        "legend.frameon": False,
        "legend.loc": "upper right",

        "lines.linewidth": 1.6,
        "lines.markersize": 4,
    })


# --------------------------------------------------------------------------- #
# Honesty helpers — reference lines and axis floors.
# --------------------------------------------------------------------------- #
def chance_line(ax, y: float = 0.5, label: str = "chance") -> None:
    """Dashed chance line for AUC/accuracy panels."""
    ax.axhline(y, ls="--", lw=0.9, color=COLORS["reference"], zorder=1)
    ax.annotate(label, xy=(0.005, y), xycoords=("axes fraction", "data"),
                va="bottom", ha="left", fontsize=7, color=COLORS["annot"])


def threshold_line(ax, y: float, label: str, axis: str = "y") -> None:
    """Generic reference line, e.g. the 30% yield bar."""
    (ax.axhline if axis == "y" else ax.axvline)(
        y, ls="--", lw=0.9, color=COLORS["reference"], zorder=1)
    ax.annotate(label, xy=(0.99, y) if axis == "y" else (y, 0.99),
                xycoords=("axes fraction", "data") if axis == "y" else ("data", "axes fraction"),
                va="bottom", ha="right", fontsize=7, color=COLORS["annot"])


def ctr_iso_ray(ax, ctr: float = 0.5, label: str = "CTR = 0.5") -> None:
    """
    For Figure 2 (heart-width y vs lung-width x): equal-CTR films lie on the ray
    y = ctr * x from the origin. Draw it across the current x-range.
    """
    x0, x1 = ax.get_xlim()
    ax.plot([0, x1], [0, ctr * x1], ls="--", lw=0.9, color=COLORS["reference"], zorder=1)
    ax.annotate(label, xy=(x1, ctr * x1), fontsize=7, color=COLORS["annot"],
                va="bottom", ha="right")


def auc_axis(ax, lo: float = 0.5, hi: float = 1.0) -> None:
    """AUC/accuracy y-axis MUST start at chance, not at the lowest bar."""
    ax.set_ylim(lo, hi)
    ax.set_ylabel("AUC")


def roc_axis(ax) -> None:
    """Square ROC axes with the diagonal chance line."""
    ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color=COLORS["reference"], zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_aspect("equal")


# --------------------------------------------------------------------------- #
# Bootstrap CI + CI'd bars (the M5/M6 convention).
# --------------------------------------------------------------------------- #
def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """
    Percentile bootstrap CI of the mean. Returns (mean, lo, hi).
    `values` are per-seed (or per-sample) scalars.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(values.mean()), float(lo), float(hi))


def bar_with_ci(ax, x, mean, ci_lo, ci_hi, *, key: str, label: str | None = None,
                require_ci: bool = True, width: float = 0.7):
    """
    One CI'd bar. For M5/M6 comparison figures `require_ci=True` refuses to draw
    a bar whose CI is missing/degenerate — a bar without a CI is not a result
    (FIGURE_SPEC Part 6). Set require_ci=False only for descriptive (non-inferential) bars.
    """
    if require_ci and (ci_lo is None or ci_hi is None or not np.isfinite(ci_lo) or ci_lo == ci_hi):
        raise ValueError(
            f"bar_with_ci: '{label or key}' has no CI. Comparison bars need a bootstrap CI "
            f"(seed-match + ≥3 seeds). If this is single-seed, mark it provisional and use "
            f"require_ci=False knowingly.")
    ax.bar(x, mean, width=width, color=color(key), label=label, zorder=2)
    if ci_lo is not None and np.isfinite(ci_lo):
        ax.errorbar(x, mean, yerr=[[mean - ci_lo], [ci_hi - mean]],
                    fmt="none", ecolor=COLORS["annot"], elinewidth=1.0, capsize=3, zorder=3)


# --------------------------------------------------------------------------- #
# "not-yet-fair" banner (single-seed / before-model / dead-run data).
# --------------------------------------------------------------------------- #
def provisional(ax, reason: str = "single-seed — illustrative, not a clean result") -> None:
    """Stamp a visible banner so a provisional figure can't be mistaken for final."""
    ax.text(0.5, 1.02, f"⚠ {reason}", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.5, color=COLORS["synth_before"], style="italic")


def contamination_caption(fig, before_label: str = "before = pre-train 40k (contaminated)",
                          after_label: str = "after = Arm-1 gtr6kqpk") -> None:
    """The D13 before/after provenance line for any synthetic figure."""
    fig.text(0.5, -0.02, f"{before_label}   ·   {after_label}", ha="center",
             va="top", fontsize=7, color=COLORS["annot"])


__all__ = [
    "COLORS", "ARM_ORDER", "ARM_LABELS", "SIZE_1COL", "SIZE_2COL", "SIZE_GRID", "DPI",
    "color", "use", "chance_line", "threshold_line", "ctr_iso_ray", "auc_axis",
    "roc_axis", "bootstrap_ci", "bar_with_ci", "provisional", "contamination_caption",
]
