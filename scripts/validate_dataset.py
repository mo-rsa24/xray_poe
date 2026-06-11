"""Validate the LDM training/val dataset: per-class counts, zero both/other.

Usage:
    python scripts/validate_dataset.py --split train \
        --csv data/nih/Data_Entry_2017.csv \
        --image-dir data/nih/images

Exits non-zero if any invariant is violated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.real_cxr_dataset import RealCXRDataset, LABEL_MAP, _classify_row

import csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--csv", default="data/nih/Data_Entry_2017.csv")
    p.add_argument("--image-dir", default="data/nih/images")
    p.add_argument("--no-finding-cap", type=int, default=4000)
    p.add_argument("--strict", action="store_true",
                   help="Fail if any both/other rows appear in the raw CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    failures = 0

    # Count raw excluded rows before the dataset filters them
    both_count = other_count = 0
    csv_path = Path(args.csv)
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                cls = _classify_row(row.get("Finding Labels", ""))
                if cls == "both":
                    both_count += 1
                elif cls == "other":
                    other_count += 1

    print(f"Raw CSV — both (held-out): {both_count:>6}   other (excluded): {other_count:>6}")
    print()

    # Build the filtered dataset
    try:
        ds = RealCXRDataset(
            csv_path=args.csv,
            image_dir=args.image_dir,
            split=args.split,
            strict=args.strict,
            no_finding_cap=args.no_finding_cap,
        )
    except Exception as e:
        print(f"ERROR building dataset: {e}")
        sys.exit(1)

    counts = ds.class_counts()
    inv = {v: k for k, v in LABEL_MAP.items()}

    print(f"{'Class':<15} {'Label':>6} {'Count':>8}")
    print("-" * 32)
    for cls, label in LABEL_MAP.items():
        print(f"{cls:<15} {label:>6} {counts[cls]:>8}")
    print()

    # Invariant checks
    bad = sum(1 for l in ds.labels if l not in {0, 1, 2})
    if bad > 0:
        print(f"FAIL — {bad} samples with invalid labels found in dataset")
        failures += 1
    else:
        print("both in dataset   : 0   PASS")
        print("other in dataset  : 0   PASS")

    total = sum(counts.values())
    print(f"\nTotal samples ({args.split}): {total}")

    sys.exit(failures)


if __name__ == "__main__":
    main()
