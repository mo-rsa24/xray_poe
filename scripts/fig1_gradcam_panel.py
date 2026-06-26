"""Fig 1 — GradCAM panel (3x3, disease-only), real vs synthetic (D3).

Columns: Pleural Effusion · Cardiomegaly · Comorbidity   (Normal dropped — its
specificity story is carried by Table 1).
Rows:    Correct · Partial · Failure, scored on PREDICTION × LOCALIZATION:
    Correct  = predicted right AND CAM in the right place (PSPNet mask)
    Failure  = predicted wrong AND CAM in the wrong place
    Partial  = one axis off; for Comorbidity = one disease caught, one missed
Overlay: red = cardiomegaly CAM, blue = effusion CAM, both on the film (no boxes).
"Right place" = CAM mass inside the PSPNet anatomy mask (heart for red, lung-base
for blue) ≥ T. The real-vs-synth mirror lives across panels: real Comorbid-Partial
fires red / misses blue (anatomical masking); synth fires blue / misses red
(the generator never rendered the heart).

Saves one panel per source (real / before / after). LaTeX uses real + best synth.
GPU (GradCAM backward + PSPNet) — run when the GPU has room.

    python scripts/fig1_gradcam_panel.py --sources real before after
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metrics.grad_cam_utils import GradCAM, load_model, preprocess  # noqa: E402
import scripts.ctr_overlay as C  # noqa: E402

MANIFEST = ROOT / "data/splits/exp06_manifests/test.csv"
SYNTH = ROOT / "results/synth_sets"
# column key → (header, stratum for real, target (cardio,effusion), synth cond dir)
COLS = [("effusion", "Pleural Effusion", (0, 1), "effusion_only", "effusion"),
        ("cardio",   "Cardiomegaly",     (1, 0), "cardio_only",   "cardio"),
        ("comorbid", "Comorbidity",      (1, 1), "both_separable", "comorbid")]
ROWS = ["Correct", "Partial", "Failure"]


def _level(pc, pe, tc, te, red_loc, blue_loc, thr_c, thr_e):
    fc, fe = pc >= thr_c, pe >= thr_e
    hit = int(tc and fc) + int(te and fe)          # target heads that fired
    ntar = tc + te
    spurious = (fc and not tc) or (fe and not te)
    loc_ok = True
    if tc and fc:
        loc_ok = loc_ok and red_loc
    if te and fe:
        loc_ok = loc_ok and blue_loc
    if hit == ntar and not spurious and loc_ok:
        return "Correct"
    if hit == 0:
        return "Failure"
    return "Partial"


def _overlay_dual(disp, cam_c, cam_e, alpha=0.5):
    gray = np.stack([disp, disp, disp], axis=-1)
    heat = np.stack([cam_c, np.zeros_like(cam_c), cam_e], axis=-1)   # red + blue
    m = np.maximum(cam_c, cam_e)[..., None]
    return np.clip(gray * (1.0 - alpha * m) + alpha * heat, 0.0, 1.0)


def _enrich(cam, mask):
    """Enrichment ratio = mean CAM inside the region / mean CAM over the whole image.
    Scale-invariant: >1 means the CAM concentrates in the region more than chance."""
    if mask is None or mask.shape != cam.shape or mask.sum() == 0 or cam.sum() == 0:
        return 0.0
    return float(cam[mask].mean() / (cam.mean() + 1e-9))


def _localize(cam_c, cam_e, masks, E):
    """red_loc / blue_loc via ENRICHMENT ratio ≥ E (red↔heart, blue↔lung-base).
    Replaces the old absolute-mass fraction, which was scale-dependent — the blue
    (effusion) CAM mass in the thin costophrenic band never cleared a heart-sized
    threshold, so Correct-effusion / Correct-comorbid were impossible by construction."""
    if not masks:
        return False, False
    heart = masks.get(C.HEART)
    lung = None
    for k in (C.LLUNG, C.RLUNG):
        m = masks.get(k)
        lung = m if lung is None else (lung | m)
    base = None
    if lung is not None:
        rows = np.where(lung.any(axis=1))[0]
        base = np.zeros_like(lung)
        if len(rows):
            mid = rows[0] + int((rows[-1] - rows[0]) * 0.55)   # lower ~45% = costophrenic band
            base[mid:] = lung[mid:]
    return _enrich(cam_c, heart) >= E, _enrich(cam_e, base) >= E


def _candidates(source, col):
    _, _, _, stratum, cond = col
    if source == "real":
        df = pd.read_csv(MANIFEST)
        return [str(ROOT / p) for p in df[df["stratum"] == stratum]["path"]]
    return [str(p) for p in sorted((SYNTH / source / cond).glob("*.png"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["real", "before", "after"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--scan", type=int, default=40, help="max films scanned per column")
    ap.add_argument("--E", type=float, default=1.5, help="CAM enrichment ratio = 'right place' (>1 = concentrated)")
    ap.add_argument("--ckpt", default="ckpts/presence_classifier_finetuned.pt")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ci, ei = load_model(args.ckpt, device)
    ck = torch.load(args.ckpt, map_location="cpu")
    thr_c, thr_e = ck["youden_threshold_cardio"], ck["youden_threshold_effusion"]
    cam_engine = GradCAM(model, device)
    seg, idx = C._load_seg(device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for source in args.sources:
        # grid[col_key][row] = (overlay_img, pc, pe) or None
        grid = {col[0]: {r: None for r in ROWS} for col in COLS}
        counts = {col[0]: 0 for col in COLS}
        for col in COLS:
            key, _, (tc, te), _, _ = col
            cells = grid[key]
            for p in _candidates(source, col)[: args.scan]:
                if all(cells[r] is not None for r in ROWS):
                    break
                try:
                    x, disp = preprocess(p)
                    with torch.no_grad():
                        prob = torch.sigmoid(model(x.to(device)))[0].cpu().numpy()
                    pc, pe = float(prob[ci]), float(prob[ei])
                    cam_c = cam_engine.cam(x, 0)
                    cam_e = cam_engine.cam(x, 1)
                    m = C.ctr_for_image(seg, idx, Path(p), device, with_masks=True)
                    red_loc, blue_loc = _localize(cam_c, cam_e, m.get("_masks"), args.E)
                except Exception:
                    continue
                counts[key] += 1
                lvl = _level(pc, pe, tc, te, red_loc, blue_loc, thr_c, thr_e)
                if cells[lvl] is None:
                    cells[lvl] = (_overlay_dual(disp, cam_c, cam_e), pc, pe)

        # ── render 3x3 ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 3, figsize=(9.6, 10.0))
        for r, row in enumerate(ROWS):
            for c, col in enumerate(COLS):
                ax = axes[r][c]; ax.set_xticks([]); ax.set_yticks([])
                cell = grid[col[0]][row]
                if cell is None:
                    ax.set_facecolor("#1b1d24")
                    ax.text(0.5, 0.5, f"0 / {counts[col[0]]}\nno example",
                            ha="center", va="center", color="#9aa", fontsize=9, transform=ax.transAxes)
                else:
                    img, pc, pe = cell
                    ax.imshow(img)
                    ax.text(0.03, 0.04, f"c={pc:.2f}  e={pe:.2f}", transform=ax.transAxes,
                            fontsize=7, color="white",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6))
                if r == 0:
                    ax.set_title(col[1], fontsize=12, fontweight="bold", pad=8)
                if c == 0:
                    ax.set_ylabel(row, fontsize=12, fontweight="bold", labelpad=10)
        # legend chips
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color="#d6403f", label="cardiomegaly"),
                            Patch(color="#3f6fd6", label="effusion")],
                   loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005))
        name = {"real": "real images", "before": "synthetic · pre-train 40k",
                "after": "synthetic · rebalanced"}.get(source, source)
        fig.suptitle(f"Grad-CAM — {name}", fontsize=14, fontweight="bold", y=0.985)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        out = Path(ROOT / args.out_dir) / f"fig1_gradcam_{source}{args.label}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140); plt.close(fig)
        filled = sum(grid[col[0]][r] is not None for col in COLS for r in ROWS)
        print(f"-> {out.relative_to(ROOT)}  ({filled}/9 cells filled)")


if __name__ == "__main__":
    main()
