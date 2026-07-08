"""Build the vae-v2 timeline exhibit pack (charts). CPU-only, grounded in verified CSVs.

Charts pair each headline number with a clean quantitative figure (axis 0-1 where it's a rate,
titled, labelled). Qualitative overlay proof-images are produced separately (GPU pass).
Reads only committed result CSVs so every number is reproducible.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/vae_v2/exhibits")
OUT.mkdir(parents=True, exist_ok=True)


def chart_ctr_hist():
    """Cardio under-draws: gen CTR 0.482 vs real 0.536 (ctr_vs_detector_v4/per_sample.csv)."""
    d = pd.read_csv("results/section4/ctr_vs_detector_v4/per_sample.csv")
    synth = pd.to_numeric(d[d.set == "synth"].ctr, errors="coerce").dropna()
    real = pd.to_numeric(d[d.set == "real"].ctr, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.3, 0.75, 30)
    ax.hist(real, bins=bins, alpha=0.55, label=f"real (median {real.median():.3f})", color="#2c7fb8")
    ax.hist(synth, bins=bins, alpha=0.55, label=f"generated (median {synth.median():.3f})", color="#d95f0e")
    ax.axvline(real.median(), color="#2c7fb8", ls="--", lw=1.5)
    ax.axvline(synth.median(), color="#d95f0e", ls="--", lw=1.5)
    ax.axvline(0.5, color="grey", ls=":", lw=1, label="CTR 0.50 (enlarged-heart line)")
    ax.set_xlabel("cardiothoracic ratio (CTR = heart width / thorax width)")
    ax.set_ylabel("number of films (n=200 each)")
    ax.set_title("Cardio under-draws: generated hearts are smaller than real\n(v4-60k, PSPNet-measured CTR)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "chart_ctr_hist.png", dpi=130); plt.close(fig)
    return f"CTR gen {synth.median():.3f} vs real {real.median():.3f}"


def chart_blunt_curve():
    """Effusion sanity: blunt-rate vs guidance (wsweep_summary.csv)."""
    d = pd.read_csv("results/vae_v2/ldm_sanity/v4_60k_wsweep/wsweep_summary.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d.w, d.blunt_rate, "o-", color="#d95f0e", lw=2, label="blunt-rate (CP > 80°)")
    for x, y in zip(d.w, d.blunt_rate):
        ax.annotate(f"{y*100:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.axhline(0.80, color="grey", ls="--", lw=1.2, label="pass mark 0.80")
    ax.set_ylim(0, 1); ax.set_xlabel("guidance weight w (how hard the model pushes toward effusion)")
    ax.set_ylabel("share of films with a blunted costophrenic angle")
    ax.set_title("Effusion blunting is guidance-responsive but never reaches 80%\n(v4-60k effusion-only, n=200, same seed)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "chart_blunt_curve.png", dpi=130); plt.close(fig)
    return f"blunt-rate {d.blunt_rate.iloc[0]*100:.0f}%@w1 -> {d.blunt_rate.iloc[-1]*100:.0f}%@w7"


def chart_detector_bar():
    """best-of-8 lifts cardio detector sensitivity above real. Verified levers only.

    Sources (downstream_eval/verdict.md, reconciled): default cardio sens 0.365,
    best-of-8 0.875, real 0.806. w4/w6 per-lever detector-sens are NOT in a clean CSV
    (guidance sweep tracks PoE yield, not single-disease detector sens) -> omitted,
    not fabricated. Flagged on the chart.
    """
    levers = ["default\n(w=1)", "best-of-8\n(w=4)", "real"]
    sens = [0.365, 0.875, 0.806]
    colors = ["#d95f0e", "#31a354", "#2c7fb8"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(levers, sens, color=colors)
    for b, s in zip(bars, sens):
        ax.annotate(f"{s:.3f}", (b.get_x() + b.get_width()/2, s), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.axhline(0.80, color="grey", ls="--", lw=1.2, label="pass mark 0.80")
    ax.set_ylim(0, 1); ax.set_ylabel("synth-cardio detector sensitivity")
    ax.set_title("best-of-8 rejection sampling lifts cardio above real, no retrain\n(v4-60k; w4/w6 per-lever sens not in a clean CSV — omitted)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "chart_detector_bar.png", dpi=130); plt.close(fig)
    return "detector: default 0.365 -> best-of-8 0.875 > real 0.806"


if __name__ == "__main__":
    for fn in (chart_ctr_hist, chart_blunt_curve, chart_detector_bar):
        print(f"[{fn.__name__}]", fn())
    # reuse existing figures by reference (copy into the pack for a single self-contained dir)
    import shutil
    for src in ["results/vae_v2/downstream_eval/guidance_sweep.png",
                "results/vae_v2/downstream_eval/exp03/ctr_cp_scatter.png",
                "results/section4/augment_probe/fig_augment_probe.png"]:
        p = Path(src)
        if p.exists():
            shutil.copy(p, OUT / f"reuse_{p.name}")
            print("reused", p.name)
    print("\nwrote charts to", OUT)
