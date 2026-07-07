"""Build per-image disease-presence manifest from VinDr-CXR train.csv.

Flags are ANY-radiologist (union across all 3 radiologists per image).
Optional majority-of-3 flags are also computed (maj_cardio, maj_effusion,
maj_thickening) — leaf 03 can tighten with --majority if needed.

Outputs data/vindr512/presence_manifest.csv:
    image_id, cardio, effusion, thickening, normal,
    n_rads_cardio, n_rads_effusion, n_rads_thickening,
    maj_cardio, maj_effusion, maj_thickening

The class integer {normal:0, cardio:1, <2nd>:2} is NOT baked in here —
leaf 03 assigns it after leaf 09 picks effusion vs thickening.

Usage:
    python scripts/vindr_presence_labels.py [--vindr PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CARDIO = "Cardiomegaly"
EFFUSION = "Pleural effusion"
THICKENING = "Pleural thickening"
NO_FINDING = "No finding"

MAJORITY = 2  # ≥ 2 of 3 radiologists


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vindr", default="/home/molefe/data/vindr512/vinbigdata",
                    help="Root dir containing train.csv")
    ap.add_argument("--out", default="data/vindr512/presence_manifest.csv")
    args = ap.parse_args()

    vindr = Path(args.vindr)
    df = pd.read_csv(vindr / "train.csv")

    # ── per-image, per-disease radiologist counts ─────────────────────────────
    def rad_count(cls: str) -> pd.Series:
        return (df[df["class_name"] == cls]
                .groupby("image_id")["rad_id"]
                .nunique()
                .rename(f"n_rads"))

    n_cardio = rad_count(CARDIO).rename("n_rads_cardio")
    n_effusion = rad_count(EFFUSION).rename("n_rads_effusion")
    n_thickening = rad_count(THICKENING).rename("n_rads_thickening")

    # all image ids (include images that only have "No finding")
    all_ids = df["image_id"].unique()
    manifest = pd.DataFrame({"image_id": all_ids}).set_index("image_id")
    manifest = manifest.join(n_cardio).join(n_effusion).join(n_thickening)
    manifest = manifest.fillna(0).astype(int)

    # ── any-radiologist presence flags (union) ────────────────────────────────
    manifest["cardio"] = (manifest["n_rads_cardio"] >= 1).astype(int)
    manifest["effusion"] = (manifest["n_rads_effusion"] >= 1).astype(int)
    manifest["thickening"] = (manifest["n_rads_thickening"] >= 1).astype(int)

    # ── majority-of-3 flags ───────────────────────────────────────────────────
    manifest["maj_cardio"] = (manifest["n_rads_cardio"] >= MAJORITY).astype(int)
    manifest["maj_effusion"] = (manifest["n_rads_effusion"] >= MAJORITY).astype(int)
    manifest["maj_thickening"] = (manifest["n_rads_thickening"] >= MAJORITY).astype(int)

    # ── normal: image has ONLY "No finding" annotations ──────────────────────
    has_finding = df[df["class_name"] != NO_FINDING]["image_id"].unique()
    manifest["normal"] = (~manifest.index.isin(has_finding)).astype(int)

    # ── write ─────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    col_order = [
        "cardio", "effusion", "thickening", "normal",
        "n_rads_cardio", "n_rads_effusion", "n_rads_thickening",
        "maj_cardio", "maj_effusion", "maj_thickening",
    ]
    manifest[col_order].reset_index().to_csv(out, index=False)
    print(f"Wrote {len(manifest):,} images → {out}")

    # ── deconfounded counts for BOTH candidate 2nd-disease choices ────────────
    c = manifest
    n_total = len(c)

    print(f"\n{'='*54}")
    print(f"  VinDr presence counts  (any-rad union, N={n_total:,})")
    print(f"{'='*54}")
    print(f"  normal (pure No-finding):          {c['normal'].sum():>6,}")
    print(f"  cardio (any):                      {c['cardio'].sum():>6,}")
    print(f"  effusion (any):                    {c['effusion'].sum():>6,}")
    print(f"  thickening (any):                  {c['thickening'].sum():>6,}")

    print(f"\n--- Pair A: cardio vs effusion ---")
    ca_ne = ((c['cardio'] == 1) & (c['effusion'] == 0)).sum()
    ea_nc = ((c['effusion'] == 1) & (c['cardio'] == 0)).sum()
    both_ae = ((c['cardio'] == 1) & (c['effusion'] == 1)).sum()
    print(f"  cardio ∧ ¬effusion:                {ca_ne:>6,}   ← deconf cardio (Arm-1 diet)")
    print(f"  effusion ∧ ¬cardio:                {ea_nc:>6,}   ← deconf effusion")
    print(f"  cardio ∧ effusion (co-present):    {both_ae:>6,}")

    print(f"\n--- Pair B: cardio vs thickening ---")
    ca_nt = ((c['cardio'] == 1) & (c['thickening'] == 0)).sum()
    ta_nc = ((c['thickening'] == 1) & (c['cardio'] == 0)).sum()
    both_at = ((c['cardio'] == 1) & (c['thickening'] == 1)).sum()
    print(f"  cardio ∧ ¬thickening:              {ca_nt:>6,}")
    print(f"  thickening ∧ ¬cardio:              {ta_nc:>6,}")
    print(f"  cardio ∧ thickening (co-present):  {both_at:>6,}")

    print(f"\n--- Majority-of-3 (≥2 rads) ---")
    print(f"  maj_cardio ∧ ¬maj_effusion:        "
          f"{((c['maj_cardio']==1)&(c['maj_effusion']==0)).sum():>6,}")
    print(f"  maj_effusion ∧ ¬maj_cardio:        "
          f"{((c['maj_effusion']==1)&(c['maj_cardio']==0)).sum():>6,}")
    print(f"  maj_cardio ∧ ¬maj_thickening:      "
          f"{((c['maj_cardio']==1)&(c['maj_thickening']==0)).sum():>6,}")
    print(f"  maj_thickening ∧ ¬maj_cardio:      "
          f"{((c['maj_thickening']==1)&(c['maj_cardio']==0)).sum():>6,}")
    print(f"{'='*54}")

    # NIH cross-reference reminder
    print(f"\n  NIH cardio training diet: 854 films")
    print(f"  VinDr cardio∧¬effusion (any-rad):  {ca_ne:,}  ({ca_ne/854:.1f}×)")
    print(f"  VinDr cardio∧¬thickening (any-rad):{ca_nt:,}  ({ca_nt/854:.1f}×)")


if __name__ == "__main__":
    main()
