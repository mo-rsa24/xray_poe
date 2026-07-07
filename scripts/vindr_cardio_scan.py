"""Scan all VinDr cardio∧¬effusion films for confounders (leaf-03 pre-filter audit).

Runs PSPNet on every deconfounded cardio film in data/vindr512/presence_manifest.csv,
collects CTR / peri-cardiac opacity / basal opacity / hemidiaphragm elevation /
lung coverage, and saves:

  results/vindr_transfer/cardio_scan/scan_metrics.csv   — full per-film table
  results/vindr_transfer/cardio_scan/worst_opacity/     — top-N films by peri-cardiac opacity
  results/vindr_transfer/cardio_scan/worst_basal/       — top-N films by basal opacity
  results/vindr_transfer/cardio_scan/elevated_hemi/     — films with elevated hemidiaphragm

Usage:
    python scripts/vindr_cardio_scan.py [--n-worst 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.ctr_overlay as C

DEFAULT_MANIFEST = "data/vindr512/presence_manifest.csv"
DEFAULT_VINDR    = "/home/molefe/data/vindr512/vinbigdata/train"
DEFAULT_OUT      = "results/vindr_transfer/cardio_scan"


def scan_all(manifest_path: str, vindr_dir: Path, device: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    targets = df[(df["cardio"] == 1) & (df["effusion"] == 0)]["image_id"].tolist()
    print(f"Scanning {len(targets)} cardio∧¬effusion films …")

    seg, idx = C._load_seg(device)
    rows = []
    for i, iid in enumerate(targets):
        path = vindr_dir / f"{iid}.png"
        if not path.exists():
            print(f"  MISSING {iid}")
            continue
        m = C.ctr_for_image(seg, idx, path, device, with_masks=False)
        rows.append({
            "image_id":           iid,
            "ctr":                m.get("ctr"),
            "heart_conf":         m.get("heart_conf"),
            "peri_cardiac_opacity": m.get("peri_cardiac_opacity"),
            "border_visible":     m.get("border_visible"),
            "basal_opacity":      m.get("basal_opacity"),
            "lung_coverage":      m.get("lung_coverage"),
            "global_whiteout":    m.get("global_whiteout"),
            "hemi_frac_L":        m.get("hemi_frac_L"),
            "hemi_frac_R":        m.get("hemi_frac_R"),
            "hemi_elevated_L":    m.get("hemi_elevated_L"),
            "hemi_elevated_R":    m.get("hemi_elevated_R"),
            "cp_angle_L":         (m.get("cp_left") or {}).get("angle_deg"),
            "cp_blunted_L":       (m.get("cp_left") or {}).get("blunted"),
            "cp_angle_R":         (m.get("cp_right") or {}).get("angle_deg"),
            "cp_blunted_R":       (m.get("cp_right") or {}).get("blunted"),
            "seg_ok":             m.get("ok"),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(targets)}")

    return pd.DataFrame(rows)


def save_overlays(ids: list[str], vindr_dir: Path, seg, idx, device: str,
                  out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Rendering {len(ids)} overlays → {out_dir}/")
    for rank, iid in enumerate(ids, 1):
        path = vindr_dir / f"{iid}.png"
        if not path.exists():
            continue
        m = C.ctr_for_image(seg, idx, path, device, with_masks=True)
        ov = C._overlay(path, m, tint=True)
        ov.save(out_dir / f"{rank:02d}_{iid[:12]}.png")


def print_summary(df: pd.DataFrame) -> None:
    ok = df[df["seg_ok"] == True]
    print(f"\n{'='*60}")
    print(f"  VinDr cardio∧¬effusion confound audit  (N={len(df)}, seg ok={len(ok)})")
    print(f"{'='*60}")
    print(f"  CTR distribution:")
    print(f"    median={ok['ctr'].median():.3f}  frac>0.5={(ok['ctr']>0.5).mean():.1%}")
    print(f"  border_visible=Y:  {ok['border_visible'].mean():.1%} of seg-ok films")
    print(f"  global_whiteout:   {df['global_whiteout'].sum()} films")
    print(f"  basal_opacity ≥ {C.TAU_BASAL_OP}:  "
          f"{(ok['basal_opacity'] >= C.TAU_BASAL_OP).sum()} films")
    print(f"  hemi elevated (L or R):  "
          f"{(ok['hemi_elevated_L'] | ok['hemi_elevated_R']).sum()} films")
    print(f"  CP blunted (L or R):  "
          f"{(ok['cp_blunted_L'] | ok['cp_blunted_R']).sum()} films")
    print(f"{'='*60}")

    # diet impact: border filter
    border_ok = ok[ok["border_visible"] == True]
    print(f"\n  Diet after border_visible filter: {len(border_ok)}/{len(df)} "
          f"({len(border_ok)/len(df):.1%})")
    hemi_any  = (ok["hemi_elevated_L"] | ok["hemi_elevated_R"])
    basal_any = ok["basal_opacity"] >= C.TAU_BASAL_OP
    print(f"  Films with elevated hemi AND border=Y:  "
          f"{(hemi_any & (ok['border_visible']==True)).sum()}")
    print(f"  Films with high basal AND border=Y:     "
          f"{(basal_any & (ok['border_visible']==True)).sum()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",  default=DEFAULT_MANIFEST)
    ap.add_argument("--vindr-dir", default=DEFAULT_VINDR)
    ap.add_argument("--out",       default=DEFAULT_OUT)
    ap.add_argument("--n-worst",   type=int, default=30,
                    help="how many worst-case overlays to save per category")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    vindr_dir = Path(args.vindr_dir)
    out_root  = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── pass 1: metrics for all films ─────────────────────────────────────────
    df = scan_all(args.manifest, vindr_dir, args.device)
    csv_path = out_root / "scan_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics → {csv_path}")

    print_summary(df)
    ok = df[df["seg_ok"] == True]

    # ── pass 2: overlay images for worst-case categories ──────────────────────
    print(f"\nRendering worst-case overlays (top {args.n_worst} per category) …")
    seg, idx = C._load_seg(args.device)

    # worst peri-cardiac opacity (whiteout / silhouette sign)
    top_opac = (ok.dropna(subset=["peri_cardiac_opacity"])
                  .nlargest(args.n_worst, "peri_cardiac_opacity")["image_id"].tolist())
    save_overlays(top_opac, vindr_dir, seg, idx, args.device,
                  out_root / "worst_opacity", "opacity")

    # worst basal opacity (effusion base / consolidation)
    top_basal = (ok.dropna(subset=["basal_opacity"])
                   .nlargest(args.n_worst, "basal_opacity")["image_id"].tolist())
    save_overlays(top_basal, vindr_dir, seg, idx, args.device,
                  out_root / "worst_basal", "basal")

    # elevated hemidiaphragm
    hemi_ids = (ok[(ok["hemi_elevated_L"] | ok["hemi_elevated_R"])]
                  .sort_values("hemi_frac_L")["image_id"].tolist()[: args.n_worst])
    save_overlays(hemi_ids, vindr_dir, seg, idx, args.device,
                  out_root / "elevated_hemi", "hemi")

    print(f"\nDone.  All outputs in {out_root}/")


if __name__ == "__main__":
    main()
