"""Leaf 07 — cleanliness-scorer bake-off: score A vs score B over the cardio pool.

Two candidate "is this cardio film basal-clean?" scorers:
  A (PSPNet/instrument): reuse ctr_overlay — clean = cardiac border visible AND sharp
    costophrenic angles AND low peri-cardiac opacity.  keep_A = border_visible AND neither CP blunted.
  B (instrument-light, basal pixels): structure-to-haze ratio in the bottom ~30% lung band —
    Laplacian energy (visible anatomy) / mean brightness (whiteout).  Higher = clearer.

Saves a **ctr_overlay-style sanity overlay per film** (heart/lung segmentation, CTR line, CP
wedges, opac/border text — exactly like the calibration screenshot) so every cleanliness call is
eyeball-checkable, plus results/clean_cardio/scores.csv.

GPU (PSPNet).  Run on a free GPU, or --device cpu (slow).  --limit keeps the bake-off cheap.

    python scripts/basal_clean_score.py --limit 150            # bake-off sample (leaf 07)
    python scripts/basal_clean_score.py --limit 100000         # whole pool   (leaf 08)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.ctr_overlay import _load_seg, ctr_for_image, _overlay, LLUNG, RLUNG


def basal_clarity(gray: np.ndarray, masks: dict | None):
    """Structure-to-haze ratio in the bottom ~30% band of the lungs (the basal/CP region)."""
    if not masks:
        return float("nan"), float("nan"), float("nan")
    lungs = masks.get(LLUNG)
    r = masks.get(RLUNG)
    lungs = (lungs | r) if (lungs is not None and r is not None) else (lungs if lungs is not None else r)
    if lungs is None or not lungs.any():
        return float("nan"), float("nan"), float("nan")
    rows = np.where(lungs.any(axis=1))[0]
    y0, y1 = int(rows.min()), int(rows.max())
    cut = int(y1 - 0.30 * (y1 - y0))
    band = np.zeros_like(lungs)
    band[cut:y1 + 1, :] = True
    basal = lungs & band
    if basal.sum() < 50:
        return float("nan"), float("nan"), float("nan")
    g = gray.astype(np.float32)
    lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1))
    sharp = float(lap[basal].var())          # high = visible anatomy/structure
    bright = float(g[basal].mean())          # high = whiteout/haze
    return round(sharp / (bright + 1e-6), 4), round(sharp, 2), round(bright, 2)


def iter_pool(nih_csv, vindr_csv, limit, seed):
    rows = []
    if nih_csv and Path(nih_csv).exists():
        d = pd.read_csv(nih_csv)
        col = "path" if "path" in d.columns else d.columns[-1]
        d = d.sample(min(limit, len(d)), random_state=seed)
        rows += [("nih", Path(str(p)).stem, str(p)) for p in d[col]]
    if vindr_csv and Path(vindr_csv).exists():
        d = pd.read_csv(vindr_csv)
        d = d.sample(min(limit, len(d)), random_state=seed)
        rows += [("vindr", r.image_id, r.img_path) for r in d.itertuples()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-csv", default="data/splits/cardio_not_effusion.csv")
    ap.add_argument("--vindr-csv", default="data/vindr512/cardio_manifest.csv")
    ap.add_argument("--limit", type=int, default=150, help="films per source (bake-off sample)")
    ap.add_argument("--out", default="results/clean_cardio/scores.csv")
    ap.add_argument("--overlay-dir", default="results/clean_cardio/overlays")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctr-min", type=float, default=0.5,
                    help="size gate: keep cardio only if CTR > this (genuinely enlarged)")
    a = ap.parse_args()

    seg, idx = _load_seg(a.device)
    Path(a.overlay_dir).mkdir(parents=True, exist_ok=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pool = iter_pool(a.nih_csv, a.vindr_csv, a.limit, a.seed)
    print(f"[basal-clean] scoring {len(pool)} films on {a.device} …")

    out = []
    for i, (src, iid, p) in enumerate(pool):
        path = Path(p)
        if not path.exists():
            continue
        m = ctr_for_image(seg, idx, path, a.device, with_masks=True)
        gray = np.asarray(Image.open(path).convert("L").resize((512, 512)), dtype=np.float32)
        b_ratio, b_sharp, b_bright = basal_clarity(gray, m.get("_masks"))
        cl, cr = m.get("cp_left", {}) or {}, m.get("cp_right", {}) or {}
        blunt = bool(cl.get("blunted")) or bool(cr.get("blunted"))
        opac = m.get("peri_cardiac_opacity", float("nan"))
        keep_A = bool(m.get("border_visible")) and not blunt
        a_clean = round(1.0 - opac, 3) if isinstance(opac, (int, float)) and np.isfinite(opac) else float("nan")
        ctr_v = m.get("ctr")
        size_ok = bool(isinstance(ctr_v, (int, float)) and np.isfinite(ctr_v) and ctr_v > a.ctr_min)
        ov = Path(a.overlay_dir) / f"{src}_{iid}.png"
        try:
            _overlay(path, m, tint=True).save(ov)
        except Exception:
            ov = ""
        out.append(dict(source=src, image_id=iid, img_path=str(path),
                        ctr=m.get("ctr"), opac=opac, border=m.get("border_visible"),
                        cp_l=cl.get("angle_deg"), cp_r=cr.get("angle_deg"), cp_blunt=blunt,
                        A_clean=a_clean, keep_A=keep_A,
                        B_ratio=b_ratio, B_sharp=b_sharp, B_bright=b_bright,
                        size_ok=size_ok, keep_clean_A=bool(keep_A and size_ok),
                        overlay=str(ov)))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pool)}")

    df = pd.DataFrame(out)
    df.to_csv(a.out, index=False)
    kr = df["keep_A"].mean() if len(df) else float("nan")
    bm = df["B_ratio"].median() if len(df) else float("nan")
    print(f"[basal-clean] {len(df)} scored → {a.out}; overlays → {a.overlay_dir}")
    print(f"  keep_A rate: {kr:.2f}  |  B_ratio median: {bm:.3f}  |  CTR>{a.ctr_min}: {df['size_ok'].mean():.2f}")
    if len(df):
        bysrc = df.groupby("source").agg(n=("image_id", "size"),
                                         basal_clean=("keep_A", "sum"),
                                         enlarged=("size_ok", "sum"),
                                         clean_AND_enlarged=("keep_clean_A", "sum"))
        print("  basal-clean (A) AND CTR>0.5, by source:")
        print(bysrc.to_string())


if __name__ == "__main__":
    main()
