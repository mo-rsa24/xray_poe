"""Task 2 — pull the both-disease (Cardiomegaly ∧ Effusion) film list, split PA/AP.

The population for Gate-0: every NIH film whose image-level labels contain BOTH
Cardiomegaly and Effusion. Source of truth is Data_Entry_2017.csv; cross-checked
against partition.parquet's `both` group. Writes results/exp01_dynamics/both_film_list.csv
with a resolved on-disk path per film and the PA/AP view split.

    python scripts/build_both_film_list.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGES = Path("data/nih/images")        # merged dir holding all 112,120 films
OUT = Path("results/exp01_dynamics/both_film_list.csv")


def main() -> None:
    df = pd.read_csv("data/nih/Data_Entry_2017.csv")
    df.columns = [c.strip() for c in df.columns]
    labels = df["Finding Labels"].str.split("|")
    both = df[labels.apply(lambda s: "Cardiomegaly" in s and "Effusion" in s)].copy()

    # cross-check against the frozen partition `both` group
    part = pd.read_parquet("data/nih/partition.parquet")
    part_both = set(part[part["group"] == "both"]["Image Index"])
    csv_both = set(both["Image Index"])
    assert csv_both == part_both, (
        f"both-set mismatch: csv\\partition={len(csv_both - part_both)}, "
        f"partition\\csv={len(part_both - csv_both)}")

    both["path"] = both["Image Index"].apply(lambda n: str(IMAGES / n))
    missing = [p for p in both["path"] if not Path(p).exists()]
    assert not missing, f"{len(missing)} indices do not resolve on disk, e.g. {missing[:3]}"

    cols = ["Image Index", "Patient ID", "View Position", "Finding Labels", "path"]
    both = both[cols].sort_values("Image Index").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    both.to_csv(OUT, index=False)

    vc = both["View Position"].value_counts()
    print(f"both-disease films: N={len(both)}  PA={vc.get('PA',0)}  AP={vc.get('AP',0)}")
    print(f"all indices resolve under {IMAGES}/  ✓")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
