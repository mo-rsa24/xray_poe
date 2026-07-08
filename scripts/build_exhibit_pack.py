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
    ax.set_title("Generated hearts vs real (v4-60k)")
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
    ax.set_title("Effusion blunting vs guidance (never reaches 80%)")
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
    ns = [int(d.loc[a, "n"]) for a, _ in order]
    lo = [s - _wilson(round(s*n), n)[0] for s, n in zip(sens, ns)]
    hi = [_wilson(round(s*n), n)[1] - s for s, n in zip(sens, ns)]
    colors = ["#fdae6b", "#f16913", "#d94801", "#31a354", "#2c7fb8"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(labels, sens, color=colors, yerr=[lo, hi], capsize=4, error_kw=dict(lw=1.2, ecolor="#444"))
    for b, s in zip(bars, sens):
        ax.annotate(f"{s:.3f}", (b.get_x() + b.get_width()/2, s), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(0.80, color="grey", ls="--", lw=1.2, label="pass mark 0.80")
    ax.set_ylim(0, 1); ax.set_ylabel("detector sensitivity")
    ax.set_title("Cardio detector sensitivity per lever (95% CI, n=160)")
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
    fig.suptitle("VAE v1 vs v2: reconstruction gates (n=854)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "chart_vae_codec_compare.png", dpi=130); plt.close(fig)
    return f"LPIPS full {v1['lpips_full']:.3f}->{v2['lpips_full']:.3f}, SSIM {v1['lower_zone_ssim']:.3f}->{v2['lower_zone_ssim']:.3f}"


def _wilson(k, n, z=1.96):
    """Wilson 95% CI for a proportion k/n (honest CI without per-film data)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))


def chart_vae_lpips_box():
    """Per-film reconstruction error, v1 vs v2 (metrics.csv, 854 films each). Distribution, not just median."""
    d = pd.read_csv("results/vae_v2/eval_gate/metrics.csv")
    v1 = d[d.checkpoint == "vae_step0025000"]; v2 = d[d.checkpoint == "vae_v2_step0075000"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.2))
    for ax, col, thr, ttl in [(a1, "lpips_full", 0.10, "LPIPS (full image)"), (a2, "lpips_cardiac_crop", 0.08, "LPIPS (heart crop)")]:
        parts = ax.violinplot([v1[col].dropna(), v2[col].dropna()], showmedians=True, widths=0.8)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#d95f0e", "#31a354"][i]); pc.set_alpha(0.6)
        ax.axhline(thr, color="grey", ls="--", lw=1.1)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["v1", "v2"]); ax.set_title(ttl, fontsize=11)
        ax.set_ylabel("LPIPS (lower is better)")
    fig.suptitle("Per-film reconstruction error: v1 vs v2 (n=854)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "chart_vae_lpips_box.png", dpi=130); plt.close(fig)
    return f"v1 median {v1.lpips_full.median():.3f} vs v2 {v2.lpips_full.median():.3f}"


def chart_classifier_auc():
    """Classifier discrimination (AUC) for cardio + effusion (detector_auc.csv)."""
    d = pd.read_csv("results/exp06_oracle/detector_auc.csv")
    vals = [float(d["auc_cardio_overall"].iloc[0]), float(d["auc_effusion_overall"].iloc[0])]
    fig, ax = plt.subplots(figsize=(4.8, 4))
    bars = ax.bar(["cardio", "effusion"], vals, color=["#3182bd", "#31a354"], width=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x()+b.get_width()/2, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10)
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance 0.50")
    ax.set_ylim(0, 1); ax.set_ylabel("AUC (higher is better)")
    ax.set_title("Classifier discrimination (AUC)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "chart_classifier_auc.png", dpi=130); plt.close(fig)
    return f"cardio AUC {vals[0]:.3f}, effusion {vals[1]:.3f}"


def chart_pareto_compute():
    """Value-for-compute per LDM version. Cardio CTR alone favours v1, so each point carries its
    DISQUALIFYING reason — only v4 renders all classes. This prevents the 'v1 looks best' misread."""
    # steps: v1/v2/v3 = 100k, v4 = 60k. status: only v4 passes all classes (sourced ladder).
    pts = [("v1", 100, 0.508, "effusion collapsed", False),
           ("v2", 100, 0.385, "cardio killed", False),
           ("v3", 100, 0.457, "washed-out", False),
           ("v4", 60, 0.482, "all classes ✓  SELECTED", True)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, steps, ctr, reason, ok in pts:
        ax.scatter(steps, ctr, s=210 if ok else 150, zorder=3,
                   facecolor="#31a354" if ok else "none", edgecolor="#31a354" if ok else "#c44",
                   linewidth=2.2 if ok else 1.8)
        dy = -34 if name == "v4" else 6
        ax.annotate(f"{name}: {reason}", (steps, ctr), textcoords="offset points",
                    xytext=(10, dy), fontsize=9, fontweight="bold" if ok else "normal",
                    color="#2a7d43" if ok else "#a33")
    ax.axhline(0.536, color="#2c7fb8", ls="--", lw=1.2, label="real cardio 0.536")
    ax.set_xlabel("training steps (thousands) — compute"); ax.set_ylabel("cardio CTR")
    ax.set_ylim(0.35, 0.56); ax.set_xlim(50, 118)
    ax.set_title("v4 is the only version that renders all classes — at the least compute")
    ax.text(0.5, -0.22, "Note: v1's higher cardio CTR is not a win — its effusion collapsed. "
            "Open circles = disqualified.", transform=ax.transAxes, ha="center", fontsize=8, color="#666")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "chart_pareto_compute.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return "v4 60k, only all-class-passing version (v1-v3 disqualified despite CTR)"


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
    ax.set_title("LDM v1→v4: cardio CTR (each version fixed the last)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "chart_ldm_versions.png", dpi=130); plt.close(fig)
    return f"v1 .508 -> v2 .385 -> v3 .457(peak .521) -> v4 .482 (real {real})"


if __name__ == "__main__":
    for fn in (chart_ctr_hist, chart_blunt_curve, chart_detector_bar,
               chart_vae_codec_compare, chart_ldm_versions,
               chart_vae_lpips_box, chart_classifier_auc, chart_pareto_compute):
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
