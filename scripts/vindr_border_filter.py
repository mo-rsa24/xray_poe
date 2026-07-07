"""Border-visibility filter for VinDr-512 films.

Runs PSPNet (_border_visible / _peri_cardiac_opacity from scripts/ctr_overlay.py)
on the ORIGINAL VinDr PNG (never a recon or std copy) and keeps only films where
the cardiac border is visible.

Class mapping is read from:
    results/vindr_transfer/criteria/second_disease.md

The file must contain a line of the form:
    SECOND_DISEASE: <tag>

where <tag> is one of:
    effusion            → maj_effusion=1
    thickening          → maj_thickening=1
    effusion+thickening → maj_effusion=1 OR maj_thickening=1

Label scheme:
    0  normal   — maj_cardio=0, second_disease=0, normal=1
    1  cardio   — maj_cardio=1, second_disease=0
    2  disease2 — second_disease=1, maj_cardio=0

Comorbid films (cardio + disease2) are excluded.

Output: data/latents/vindr/border_visible_manifest.csv
    image_id, class, class_name, projection, border_visible,
    peri_cardiac_opacity, heart_conf, seg_ok

Usage:
    python scripts/vindr_border_filter.py [--device cuda] [--normal-cap 5000]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.ctr_overlay as C

CRITERIA_FILE   = Path("results/vindr_transfer/criteria/second_disease.md")
PRESENCE        = Path("data/vindr512/presence_manifest.csv")
PROJ_SPLIT      = Path("data/latents/vindr/projection_split.csv")
VINDR_TRAIN     = Path("data/vindr512/vinbigdata/train")
OUT_MANIFEST    = Path("data/latents/vindr/border_visible_manifest.csv")

NORMAL_CAP = 5000   # normals to score (border filter is cheap per-image but 10k is slow)


# ---------------------------------------------------------------------------
# Read second-disease decision
# ---------------------------------------------------------------------------

def _read_second_disease() -> tuple[str, list[str]]:
    """Return (tag, [maj_col, ...]) parsed from criteria file."""
    text = CRITERIA_FILE.read_text()
    m = re.search(r"SECOND_DISEASE:\s*(\S+)", text)
    if not m:
        raise ValueError(f"No 'SECOND_DISEASE: <tag>' line found in {CRITERIA_FILE}")
    tag = m.group(1).strip().lower()

    if tag == "effusion":
        cols = ["maj_effusion"]
    elif tag == "thickening":
        cols = ["maj_thickening"]
    elif tag in ("effusion+thickening", "thickening+effusion"):
        cols = ["maj_effusion", "maj_thickening"]
    else:
        raise ValueError(f"Unknown SECOND_DISEASE tag: {tag!r}")
    return tag, cols


# ---------------------------------------------------------------------------
# Build label table from presence_manifest
# ---------------------------------------------------------------------------

def _build_label_df(
    presence: pd.DataFrame,
    second_disease_cols: list[str],
    normal_cap: int,
) -> pd.DataFrame:
    """Return df with columns [image_id, class, class_name]."""
    df = presence.copy()

    # Boolean flags (majority-vote)
    df["is_cardio"]   = df["maj_cardio"].astype(bool)
    df["is_disease2"] = df[[c for c in second_disease_cols]].any(axis=1)
    df["is_normal"]   = (
        ~df["is_cardio"] & ~df["is_disease2"] &
        (df["cardio"] == 0) & (df["effusion"] == 0) & (df["thickening"] == 0)
    )

    # Exclude comorbid
    comorbid = df["is_cardio"] & df["is_disease2"]

    rows = []

    # Class 1: cardio only
    for iid in df.loc[df["is_cardio"] & ~comorbid, "image_id"]:
        rows.append({"image_id": iid, "class": 1, "class_name": "cardio"})

    # Class 2: disease2 only
    for iid in df.loc[df["is_disease2"] & ~comorbid, "image_id"]:
        rows.append({"image_id": iid, "class": 2, "class_name": "disease2"})

    # Class 0: normal (capped)
    normals = df.loc[df["is_normal"], "image_id"].tolist()
    if normal_cap and len(normals) > normal_cap:
        import random
        normals = random.Random(42).sample(normals, normal_cap)
    for iid in normals:
        rows.append({"image_id": iid, "class": 0, "class_name": "normal"})

    out = pd.DataFrame(rows)
    print(f"  Label table: {(out['class']==0).sum()} normal  "
          f"{(out['class']==1).sum()} cardio  "
          f"{(out['class']==2).sum()} disease2  "
          f"({comorbid.sum()} comorbid excluded)")
    return out


# ---------------------------------------------------------------------------
# Run border filter
# ---------------------------------------------------------------------------

def _run_filter(
    label_df: pd.DataFrame,
    vindr_dir: Path,
    device: str,
) -> pd.DataFrame:
    seg, idx = C._load_seg(device)
    rows = []

    for _, row in tqdm(label_df.iterrows(), total=len(label_df), unit="img"):
        iid   = row["image_id"]
        path  = vindr_dir / f"{iid}.png"
        if not path.exists():
            continue
        m = C.ctr_for_image(seg, idx, path, device, with_masks=False)
        rows.append({
            "image_id":             iid,
            "class":                int(row["class"]),
            "class_name":           row["class_name"],
            "border_visible":       bool(m.get("border_visible", False)),
            "peri_cardiac_opacity": m.get("peri_cardiac_opacity"),
            "heart_conf":           m.get("heart_conf"),
            "seg_ok":               bool(m.get("ok", False)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--normal-cap", type=int, default=NORMAL_CAP,
                   help="Max normal images to score (0 = all)")
    p.add_argument("--criteria",   default=str(CRITERIA_FILE))
    p.add_argument("--presence",   default=str(PRESENCE))
    p.add_argument("--vindr-dir",  default=str(VINDR_TRAIN))
    p.add_argument("--out",        default=str(OUT_MANIFEST))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global CRITERIA_FILE
    CRITERIA_FILE = Path(args.criteria)

    # ── Read second-disease decision ────────────────────────────────────────
    tag, sd_cols = _read_second_disease()
    print(f"Second disease: {tag}  (columns: {sd_cols})")

    # ── Build label table ───────────────────────────────────────────────────
    presence = pd.read_csv(args.presence)
    label_df = _build_label_df(presence, sd_cols, args.normal_cap)

    # ── Run PSPNet border filter on original PNGs ───────────────────────────
    vindr_dir = Path(args.vindr_dir)
    print(f"\nScoring {len(label_df)} images on {args.device} …")
    result_df = _run_filter(label_df, vindr_dir, args.device)

    # ── Join projection tag ─────────────────────────────────────────────────
    proj = pd.read_csv(PROJ_SPLIT)
    result_df = result_df.merge(proj, on="image_id", how="left")
    result_df["projection"] = result_df["projection"].fillna("unknown")

    # ── Filter border_visible ───────────────────────────────────────────────
    kept = result_df[result_df["border_visible"] == True].copy()

    # ── Write output ────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept[["image_id", "class", "class_name", "projection",
          "border_visible", "peri_cardiac_opacity", "heart_conf", "seg_ok"]].to_csv(
        out_path, index=False
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  VinDr border-visible filter  (SECOND_DISEASE={tag})")
    print(f"{'='*55}")
    for cls, name in [(0, "normal"), (1, "cardio"), (2, "disease2")]:
        total = (result_df["class"] == cls).sum()
        keep  = (kept["class"] == cls).sum()
        pct   = keep / total * 100 if total else 0
        print(f"  class {cls} ({name:8s}):  {keep:5d} / {total:5d} kept  ({pct:.1f}%)")
    print(f"  {'TOTAL':12s}:  {len(kept):5d} / {len(result_df):5d}")
    print(f"{'='*55}")
    print(f"\nWrote → {out_path}")


if __name__ == "__main__":
    main()
