"""Sort the cleanliness-scored overlays into keep/drop folders for eyeballing.

Symlinks each film's ctr_overlay sanity image into folders by verdict, so you can browse
whether the kept set is mostly clean and whether there's enough for augmentation:

  results/clean_cardio/sorted/
    A_keep/     ← scorer A: basal-clean AND CTR>0.5
    B_keep/     ← scorer B: basal-clean AND CTR>0.5
    both_keep/  ← kept by both
    A_only/     ← A keeps, B drops   (disagreement)
    B_only/     ← B keeps, A drops   (disagreement)
    dropped/    ← neither keeps

keep_B uses a threshold matched to scorer A's cleanliness rate (compare WHICH films, not the rate).

    python scripts/sort_clean_overlays.py            # symlinks (re-run after the full-pool score)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/clean_cardio/scores.csv")
    ap.add_argument("--out-dir", default="results/clean_cardio/sorted")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    a = ap.parse_args()

    d = pd.read_csv(a.scores)
    d = d.dropna(subset=["B_ratio"]).reset_index(drop=True)
    # tolerate older scores.csv that predate the size gate — derive it from ctr
    if "size_ok" not in d.columns:
        d["size_ok"] = pd.to_numeric(d["ctr"], errors="coerce") > 0.5

    # scorer B cleanliness verdict at a rate matched to scorer A's
    kA_rate = float(d["keep_A"].mean())
    thr_B = d["B_ratio"].quantile(1.0 - kA_rate)
    d["keep_B"] = d["B_ratio"] >= thr_B
    d["keep_clean_A"] = d["keep_A"].astype(bool) & d["size_ok"].astype(bool)
    d["keep_clean_B"] = d["keep_B"].astype(bool) & d["size_ok"].astype(bool)

    folders = {
        "A_keep":    d["keep_clean_A"],
        "B_keep":    d["keep_clean_B"],
        "both_keep": d["keep_clean_A"] & d["keep_clean_B"],
        "A_only":    d["keep_clean_A"] & ~d["keep_clean_B"],
        "B_only":    d["keep_clean_B"] & ~d["keep_clean_A"],
        "dropped":   ~d["keep_clean_A"] & ~d["keep_clean_B"],
    }

    out = Path(a.out_dir)
    print(f"scored: {len(d)}  |  scorer-A clean-rate {kA_rate:.2f}  |  B threshold ratio ≥ {thr_B:.3f}\n")
    for name, mask in folders.items():
        fdir = out / name
        # clean out old links so re-runs are fresh
        if fdir.exists():
            for f in fdir.iterdir():
                f.unlink()
        fdir.mkdir(parents=True, exist_ok=True)
        sub = d[mask]
        for _, r in sub.iterrows():
            ov = r.get("overlay")
            if not ov or not Path(str(ov)).exists():
                continue
            link = fdir / f"{r['source']}_{r['image_id']}.png"
            src = Path(str(ov)).resolve()
            if a.copy:
                import shutil
                shutil.copy(src, link)
            else:
                os.symlink(src, link)
        bysrc = sub["source"].value_counts().to_dict()
        print(f"  {name:10s} {mask.sum():4d}   {bysrc}")

    keep_tot = int(folders["A_keep"].sum())
    print(f"\n  → A_keep = {keep_tot} films in this sample (floor 854 on the FULL pool).")
    print(f"  browse: {out}/A_keep  ·  {out}/B_keep  ·  {out}/both_keep")


if __name__ == "__main__":
    main()
