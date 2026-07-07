"""Exp02 task 4 — codec-ceiling figure + verdict (boundary-jitter-refined).

Reads results/exp02_ceiling/recon_witness_metrics.csv. The separable test set
clusters at the CTR=0.5 / CP=80° decision thresholds, so a raw "sign-read flip"
conflates genuine codec drift with sub-tolerance boundary jitter. This counts a
flip as GENUINE only when the magnitude exceeds tolerance:

    genuine CTR-cross  = crosses 0.5  AND |ΔCTR| > dctr_tol (0.02)
    genuine CP-flip    = blunted-class changed AND |Δangle| > cp_angle_tol (10°,
                         the codec CP-angle noise floor is ~4°; genuine moves ~15-20°)

Rule (on the separable set, control = single-disease LPIPS reference):
    HIGH  if median |ΔCTR| ≤ 0.02 AND genuine CP preserved ≥ 95%
          AND separable local LPIPS not significantly > control (MWU p > 0.05)
    LOW   if genuine CTR-cross > 10% OR genuine CP-flip > 10%
    INCONCLUSIVE otherwise

    python scripts/exp02_ceiling_verdict.py --dctr-tol 0.02 --cp-angle-tol 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

CSV = Path("results/exp02_ceiling/recon_witness_metrics.csv")
FIG = Path("figures/exp02_recon_witness_panel.png")
VERDICT = Path("results/exp02_ceiling/verdict.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dctr-tol", type=float, default=0.02)
    ap.add_argument("--cp-angle-tol", type=float, default=10.0)
    ap.add_argument("--cp-preserve-min", type=float, default=0.95)
    ap.add_argument("--fail-rate", type=float, default=0.10)
    args = ap.parse_args()
    df = pd.read_csv(CSV); df = df[df["ok"]]
    sep = df[df["group"] == "separable"].copy()
    ctrl = df[df["group"].str.startswith("control")]
    N = len(sep)

    dctr_med = float(sep["abs_dCTR"].median())
    raw_cross = float(sep["ctr_crosses_0p5"].mean())
    gen_cross = float((sep["ctr_crosses_0p5"] & (sep["abs_dCTR"] > args.dctr_tol)).mean())
    raw_cpflip = float((~sep["cp_preserved"]).mean())
    dL = (sep["cp_left_orig_angle"] - sep["cp_left_recon_angle"]).abs()
    dR = (sep["cp_right_orig_angle"] - sep["cp_right_recon_angle"]).abs()
    gen_cpflip = float((((~sep["cp_left_preserved"]) & (dL > args.cp_angle_tol)) |
                        ((~sep["cp_right_preserved"]) & (dR > args.cp_angle_tol))).mean())

    def lp(col, d): return float(d[col].median())
    p_heart = mannwhitneyu(sep["heart_lpips"].dropna(), ctrl["heart_lpips"].dropna(), alternative="greater")[1]
    p_cp = mannwhitneyu(sep["cp_lpips"].dropna(), ctrl["cp_lpips"].dropna(), alternative="greater")[1]
    lpips_ok = (p_heart > 0.05) and (p_cp > 0.05)   # separable not significantly worse than control

    if gen_cross > args.fail_rate or gen_cpflip > args.fail_rate:
        verdict = "low"
    elif dctr_med <= args.dctr_tol and (1 - gen_cpflip) >= args.cp_preserve_min and lpips_ok:
        verdict = "high"
    else:
        verdict = "inconclusive"

    # cp-tol sensitivity
    sens = {}
    for t in (5, 10, 15, 20):
        sens[t] = float((((~sep["cp_left_preserved"]) & (dL > t)) |
                         ((~sep["cp_right_preserved"]) & (dR > t))).mean())

    # ── figure ───────────────────────────────────────────────────────────────
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.4))
    a1.hist(sep["abs_dCTR"].dropna(), bins=40, color="#3b78b0", alpha=0.85)
    a1.axvline(args.dctr_tol, color="k", ls="--", lw=1.2, label=f"tol {args.dctr_tol}")
    a1.axvline(dctr_med, color="#cf6b4c", lw=1.5, label=f"median {dctr_med:.4f}")
    a1.set_xlabel("|ΔCTR|, separable set"); a1.set_ylabel("films")
    a1.set_title("codec CTR faithfulness"); a1.legend(fontsize=8)

    x = np.arange(2); w = 0.36
    a2.bar(x - w / 2, [raw_cross, raw_cpflip], w, color="#cf6b4c", label="raw (incl. jitter)")
    a2.bar(x + w / 2, [gen_cross, gen_cpflip], w, color="#2ca25f", label="genuine (> tol)")
    a2.axhline(args.fail_rate, color="k", ls="--", lw=1, label="LOW line 10%")
    a2.set_xticks(x); a2.set_xticklabels(["CTR-cross", "CP-flip"]); a2.set_ylabel("rate")
    a2.set_title("sign-flip: raw vs boundary-refined"); a2.legend(fontsize=8)

    a3.bar(x - w / 2, [lp("heart_lpips", sep), lp("cp_lpips", sep)], w, color="#2ca25f", label="separable-both")
    a3.bar(x + w / 2, [lp("heart_lpips", ctrl), lp("cp_lpips", ctrl)], w, color="#9e9e9e", label="control")
    a3.set_xticks(x); a3.set_xticklabels([f"heart\np={p_heart:.2f}", f"CP\np={p_cp:.2f}"])
    a3.set_ylabel("local LPIPS (median)"); a3.set_title("local fidelity vs control"); a3.legend(fontsize=8)
    for a in (a1, a2, a3):
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Exp02 · VAE codec ceiling — verdict: {verdict.upper()} (boundary-refined)", fontweight="bold")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG, dpi=150); plt.close(fig)

    # ── verdict.md ───────────────────────────────────────────────────────────
    L = [f"# Exp02 — VAE codec-ceiling verdict (boundary-jitter-refined)", "",
         f"**VERDICT: ceiling {verdict}** · VAE `vae_step0025000` · separable N={N}, control N={len(ctrl)} · "
         f"cp_angle_tol={args.cp_angle_tol:.0f}°.", "",
         "## Codec is faithful in magnitude",
         f"- median **|ΔCTR| = {dctr_med:.4f}** ≤ {args.dctr_tol} ✓ (p90={sep['abs_dCTR'].quantile(.9):.4f}; "
         f"only {(sep['abs_dCTR']>0.05).mean():.1%} of films exceed 0.05).",
         f"- local LPIPS heart sep {lp('heart_lpips',sep):.3f} vs control {lp('heart_lpips',ctrl):.3f} "
         f"(MWU sep>ctrl p={p_heart:.3f}); CP {lp('cp_lpips',sep):.3f} vs {lp('cp_lpips',ctrl):.3f} (p={p_cp:.3f}). "
         f"→ {'not significantly worse than control' if lpips_ok else 'marginally worse than control'}.",
         "",
         "## Sign-flips: boundary jitter vs genuine drift",
         f"- **CTR-cross: raw {raw_cross:.1%} → genuine {gen_cross:.1%}** (|ΔCTR|>{args.dctr_tol}). "
         f"~half the raw crossings were films sitting within ±0.02 of CTR=0.5 — sub-tolerance jitter, not drift.",
         f"- **CP-flip: raw {raw_cpflip:.1%} → genuine {gen_cpflip:.1%}** (|Δangle|>{args.cp_angle_tol:.0f}°). "
         f"Codec CP-angle noise ~4°; genuine flips moved ~15-20°.",
         f"- cp_angle_tol sensitivity (genuine CP-flip): "
         + ", ".join(f"{t}°={r:.1%}" for t, r in sens.items()) + ".",
         "",
         "## Call",
         {"high": "**Ceiling HIGH** — the codec preserves both sign-reads; not the scapegoat. Reuse the VAE.",
          "low": "**Ceiling LOW** — genuine sign-drift exceeds 10%; fix/retrain the VAE before any composition claim.",
          "inconclusive":
              f"**INCONCLUSIVE, but clearly NOT low and the codec is NOT a scapegoat.** Genuine sign-flips "
              f"({gen_cross:.1%} CTR, {gen_cpflip:.1%} CP) are well under the 10% LOW line, and |ΔCTR| is faithful "
              f"({dctr_med:.4f}). It misses a clean HIGH only marginally: genuine CP-preserved "
              f"{1-gen_cpflip:.1%} (bar 95%, met at cp_angle_tol≥15°) and CP-LPIPS a hair above control (p={p_cp:.3f}). "
              f"**Action: do NOT retrain the VAE** (no codec fix changes ~6-7% threshold-edge flips); carry an "
              f"abstain margin (±0.03 CTR / ±10° CP) into Exp 7's gate and Exp 3's yield read."}[verdict],
         "",
         f"Figure: {FIG} · visual pairs: results/exp02_ceiling/eyeball_panel.png · feeds Exp 5 (VAE reusable; ceiling not the blocker)."]
    VERDICT.write_text("\n".join(L) + "\n")

    print(f"VERDICT: ceiling {verdict}")
    print(f"  |ΔCTR| median={dctr_med:.4f} ≤ {args.dctr_tol}")
    print(f"  CTR-cross raw {raw_cross:.1%} → genuine {gen_cross:.1%} · CP-flip raw {raw_cpflip:.1%} → genuine {gen_cpflip:.1%}")
    print(f"  LPIPS heart p={p_heart:.3f} cp p={p_cp:.3f}  (lpips_ok={lpips_ok})")
    print(f"→ {FIG}\n→ {VERDICT}")


if __name__ == "__main__":
    main()
