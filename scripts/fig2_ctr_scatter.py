"""Fig 2 — CTR scatter: heart-width vs lung-width, real vs synth (D5).

The "we're not there yet at comorbidities" visual. For real and synthetic films,
measure heart width and thorax (lung) width with the classifier-INDEPENDENT PSPNet
instrument (scripts/ctr_overlay.py), then scatter heart-width (Y) against thorax-width
(X), colour-coded real vs synth, with the CTR=0.5 iso-line drawn. A faithful generator
puts its synth cloud on top of the real cloud; the "not there yet" signature is the
synth comorbid points drifting OFF the real comorbid locus (cardio under-rendered →
hearts too small → points sit below the CTR=0.5 line).

Real films come from the Exp 6 test manifest (per category). Synth films come from
results/synth_sets/<tag>/{cardio,effusion,comorbid}. Each before/after tag is a panel.

GPU-light (PSPNet inference only) but still GPU — run when the GPU is free of training.

    python scripts/fig2_ctr_scatter.py --synth-tags before after --n-real 120 --n-synth 120
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.ctr_overlay as C  # noqa: E402

MANIFEST = ROOT / "data/splits/exp06_manifests/test.csv"
SYNTH = ROOT / "results/synth_sets"
STRATUM2CAT = {"normal": "normal", "cardio_only": "cardio",
               "effusion_only": "effusion", "both_separable": "comorbid"}
CONDS = ["cardio", "effusion", "comorbid"]
CAT_COLOR = {"normal": "#888888", "cardio": "#d62728", "effusion": "#1f77b4", "comorbid": "#9467bd"}


def _measure(paths, seg, idx, device):
    """Run the PSPNet instrument → per-image (heart_w, thorax_w, ctr)."""
    rows = []
    for p in paths:
        try:
            m = C.ctr_for_image(seg, idx, Path(p), device)
            if m.get("ok") and m["thorax_w"] > 0:
                rows.append((m["heart_w"], m["thorax_w"], m["ctr"]))
        except Exception:
            continue
    return np.array(rows) if rows else np.empty((0, 3))


def _real_paths(n, seed=42):
    df = pd.read_csv(MANIFEST)
    df["category"] = df["stratum"].map(STRATUM2CAT)
    out = {}
    for cat in CONDS:
        sub = df[df["category"] == cat]
        if len(sub) > n:
            sub = sub.sample(n, random_state=seed)
        out[cat] = [str(ROOT / p) for p in sub["path"]]
    return out


def _synth_paths(tag, n):
    out = {}
    for cond in CONDS:
        d = SYNTH / tag / cond
        ps = sorted(d.glob("*.png"))[:n]
        out[cond] = [str(p) for p in ps]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-tags", nargs="+", default=["before", "after"])
    ap.add_argument("--n-real", type=int, default=120)
    ap.add_argument("--n-synth", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="figures/fig2_ctr_scatter_beforeafter.png")
    args = ap.parse_args()
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seg, idx = C._load_seg(device)

    real = _real_paths(args.n_real)
    real_m = {cat: _measure(real[cat], seg, idx, device) for cat in CONDS}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = args.synth_tags
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), squeeze=False)
    axes = axes[0]

    rows_csv = []
    for ax, tag in zip(axes, panels):
        synth = _synth_paths(tag, args.n_synth)
        synth_m = {cond: _measure(synth[cond], seg, idx, device) for cond in CONDS}
        for cat in CONDS:
            rm, sm = real_m[cat], synth_m[cat]
            if len(rm):
                ax.scatter(rm[:, 1], rm[:, 0], s=18, c=CAT_COLOR[cat], marker="o",
                           alpha=0.55, label=f"real {cat}", edgecolors="none")
            if len(sm):
                ax.scatter(sm[:, 1], sm[:, 0], s=22, facecolors="none",
                           edgecolors=CAT_COLOR[cat], marker="^", alpha=0.8, label=f"synth {cat}")
            for src, mm in (("real", rm), ("synth", sm)):
                for hw, tw, ctr in mm:
                    rows_csv.append({"panel": tag, "category": cat, "source": src,
                                     "heart_w": hw, "thorax_w": tw, "ctr": ctr})
        # CTR=0.5 iso-line (heart = 0.5 * thorax)
        xs = np.array(ax.get_xlim())
        ax.plot(xs, 0.5 * xs, ls="--", c="k", lw=1, label="CTR=0.5")
        ax.set_xlabel("thorax (lung) width  [px]"); ax.set_ylabel("heart width  [px]")
        ax.set_title(f"{tag} model")
        ax.legend(fontsize=6, ncol=2, loc="upper left")
    fig.suptitle("Fig 2 — CTR scatter (heart vs thorax width), real ● vs synth △", fontsize=11)
    fig.tight_layout()
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    pd.DataFrame(rows_csv).to_csv(out.with_suffix(".csv"), index=False)
    print(f"-> {out.relative_to(ROOT)}\n-> {out.with_suffix('.csv').relative_to(ROOT)}")
    # quick numeric summary: median CTR per category/source/panel
    summ = pd.DataFrame(rows_csv).groupby(["panel", "source", "category"])["ctr"].median().round(3)
    print(summ.to_string())


if __name__ == "__main__":
    main()
