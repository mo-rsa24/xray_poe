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
    """Full per-lever cardio detector sensitivity (cardio_enhance/compare_levers.csv, n=160/arm).

    Every free inference lever's detector sensitivity is in this CSV — guidance ramps the score
    (w1 0.35 -> w4 0.63 -> w6 0.75) and best-of-8 (w4) tops out at 0.875, above real 0.806.
    """
    d = pd.read_csv("results/section4/cardio_enhance/compare_levers.csv").set_index("arm")
    order = [("w1_null", "w=1"), ("w4_null", "w=4"), ("w6_null", "w=6"),
             ("w4_bo8", "best-of-8\n(w=4)"), ("REAL_cardio", "real")]
    labels = [l for _, l in order]
    sens = [float(d.loc[a, "detector_sens"]) for a, _ in order]
    colors = ["#fdae6b", "#f16913", "#d94801", "#31a354", "#2c7fb8"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(labels, sens, color=colors)
    for b, s in zip(bars, sens):
        ax.annotate(f"{s:.3f}", (b.get_x() + b.get_width()/2, s), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.axhline(0.80, color="grey", ls="--", lw=1.2, label="pass mark 0.80")
    ax.set_ylim(0, 1); ax.set_ylabel("synth-cardio detector sensitivity")
    ax.set_title("Cardio detector sensitivity per free inference lever\n(v4-60k, n=160/arm; guidance ramps it, best-of-8 clears real)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "chart_detector_bar.png", dpi=130); plt.close(fig)
    return f"detector: w1 {sens[0]:.2f} -> w4 {sens[1]:.2f} -> w6 {sens[2]:.2f} -> best-of-8 {sens[3]:.3f} > real {sens[4]:.3f}"


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
