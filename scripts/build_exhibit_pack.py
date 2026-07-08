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


def chart_vae_codec_compare():
    """VAE v1 vs v2 on the four reconstruction gates (eval_gate/metrics_agg.csv, n=854).

    LPIPS + ΔCTR are lower-is-better; SSIM is higher-is-better -> two panels so the axes stay
    honest. v1 (pixel-MSE only) fails both LPIPS gates; v2 (+LPIPS +PatchGAN) passes all four.
    """
    d = pd.read_csv("results/vae_v2/eval_gate/metrics_agg.csv").set_index("checkpoint")
    v1, v2 = d.loc["vae_step0025000"], d.loc["vae_v2_step0075000"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [3, 1]})
    # panel L: lower-is-better (ΔCTR, LPIPS full, LPIPS cardiac) with threshold lines
    lb = [("ΔCTR", "delta_ctr_median", 0.01), ("LPIPS\nfull", "lpips_full", 0.10),
          ("LPIPS\ncardiac", "lpips_cardiac_crop", 0.08)]
    x = np.arange(len(lb)); w = 0.36
    axL.bar(x - w/2, [v1[k] for _, k, _ in lb], w, label="v1 (pixel-MSE)", color="#d95f0e")
    axL.bar(x + w/2, [v2[k] for _, k, _ in lb], w, label="v2 (+LPIPS +PatchGAN)", color="#31a354")
    for xi, (_, k, thr) in enumerate(lb):
        axL.plot([xi - 0.5, xi + 0.5], [thr, thr], color="grey", ls="--", lw=1.2)
        axL.annotate(f"{v1[k]:.3f}", (xi - w/2, v1[k]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        axL.annotate(f"{v2[k]:.3f}", (xi + w/2, v2[k]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    axL.set_xticks(x); axL.set_xticklabels([n for n, _, _ in lb])
    axL.set_ylabel("metric value (dashed = pass threshold)")
    axL.set_title("Lower is better — v1 fails both LPIPS gates, v2 clears all")
    axL.legend(fontsize=8)
    # panel R: higher-is-better (SSIM)
    axR.bar(["v1", "v2"], [v1["lower_zone_ssim"], v2["lower_zone_ssim"]], color=["#d95f0e", "#31a354"], width=0.6)
    axR.plot([-0.5, 1.5], [0.88, 0.88], color="grey", ls="--", lw=1.2, label="pass ≥0.88")
    for xi, val in enumerate([v1["lower_zone_ssim"], v2["lower_zone_ssim"]]):
        axR.annotate(f"{val:.3f}", (xi, val), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    axR.set_ylim(0.85, 1.0); axR.set_title("Higher is better\nlower-zone SSIM"); axR.legend(fontsize=8)
    fig.suptitle("VAE codec: v1 → v2 reconstruction gates (held-out n=854)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "chart_vae_codec_compare.png", dpi=130); plt.close(fig)
    return f"LPIPS full {v1['lpips_full']:.3f}->{v2['lpips_full']:.3f}, SSIM {v1['lower_zone_ssim']:.3f}->{v2['lower_zone_ssim']:.3f}"


def chart_ldm_versions():
    """LDM v1->v4 (VAE-v2 mu-latent) cardio-CTR ladder + why we stopped at v4-60k.

    Numbers are grounded in plan 06 / plan 09 / tracking.md (no single CSV holds them; the
    per-run verdicts live in prose). Run IDs use plan 06's authoritative mapping, NOT the
    DECISION_TIMELINE version table, which has wrong IDs (flagged for reconcile).
    Sources: 06-ldm-retrain.md:56-83, 09-collapse-prevention.md:4-101, tracking.md per-run.
    """
    vers = [
        ("v1\n6wj35ips", 0.508, "#d95f0e", "effusion COLLAPSED\n(0.38→0.003)\nmix .25/.40/.35"),
        ("v2\numqnrs0o", 0.385, "#e6550d", "cardio KILLED\n(share→.34, 77% leak)\n+EMA +min-SNR"),
        ("v3\nyrcj8qgl", 0.457, "#fd8d3c", "both active but\npeak .521@40k→.457@100k\nwashed-out"),
        ("v4\n4vg2h5i4", 0.482, "#31a354", "all classes render\nper-channel whitening\nstop @60k · WINNER"),
    ]
    real = 0.536
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(vers))
    bars = ax.bar(x, [c for _, c, _, _ in vers], color=[col for _, _, col, _ in vers], width=0.62)
    ax.axhline(real, color="#2c7fb8", ls="--", lw=1.4, label=f"real cardio CTR {real}")
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="CTR 0.50 (enlarged line)")
    # v3 peak marker (peaked then decayed)
    ax.plot(2, 0.521, "v", color="#d94801", ms=10)
    ax.annotate("peak 0.521@40k", (2, 0.521), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#d94801")
    for xi, (_, c, _, note) in enumerate(vers):
        ax.annotate(f"{c:.3f}", (xi, c), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, fontweight="bold")
        ax.annotate(note, (xi, 0.02), ha="center", va="bottom", fontsize=7.2, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([v for v, _, _, _ in vers])
    ax.set_ylim(0, 0.60); ax.set_ylabel("cardio CTR (final, PSPNet-measured)")
    ax.set_title("LDM v1→v4 in VAE-v2 μ-latent space: the class-rebalance + whitening ladder\n"
                 "why we kept retraining (each fixed the last) and stopped at v4-60k (cardio peaks then decays)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "chart_ldm_versions.png", dpi=130); plt.close(fig)
    return f"v1 .508 -> v2 .385 -> v3 .457(peak .521) -> v4 .482 (real {real})"


if __name__ == "__main__":
    for fn in (chart_ctr_hist, chart_blunt_curve, chart_detector_bar,
               chart_vae_codec_compare, chart_ldm_versions):
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
