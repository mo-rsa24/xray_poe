"""
Extract single-disease validation images from data.zip.

Produces three directories:
  data/nih/images_cardio_only/   — Cardiomegaly ∧ ¬Effusion
  data/nih/images_effusion_only/ — Effusion ∧ ¬Cardiomegaly
  data/nih/images_normal/        — No Finding (only label)

Usage:
  python scripts/extract_single_disease.py
  python scripts/extract_single_disease.py --n 200 --zip data.zip --csv data/nih/Data_Entry_2017.csv
"""

import argparse
import os
import random
import zipfile

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Extract single-disease validation images from data.zip")
    p.add_argument("--zip",  default="data.zip",                     help="Path to data.zip")
    p.add_argument("--csv",  default="data/nih/Data_Entry_2017.csv", help="NIH manifest CSV")
    p.add_argument("--n",    default=200, type=int,                  help="Max images per class")
    p.add_argument("--seed", default=42,  type=int,                  help="Random seed for sampling")
    p.add_argument("--out",  default="data/nih",                     help="Parent output directory")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Reading {args.csv} ...")
    d = pd.read_csv(args.csv)
    d["labels"] = d["Finding Labels"].apply(lambda s: set(s.split("|")))

    cardio_pool   = d[d["labels"].apply(lambda l: "Cardiomegaly" in l and "Effusion" not in l)]["Image Index"].tolist()
    effusion_pool = d[d["labels"].apply(lambda l: "Effusion" in l and "Cardiomegaly" not in l)]["Image Index"].tolist()
    normal_pool   = d[d["labels"].apply(lambda l: l == {"No Finding"})]["Image Index"].tolist()

    rng = random.Random(args.seed)
    cardio_set   = set(rng.sample(cardio_pool,   min(args.n, len(cardio_pool))))
    effusion_set = set(rng.sample(effusion_pool, min(args.n, len(effusion_pool))))
    normal_set   = set(rng.sample(normal_pool,   min(args.n, len(normal_pool))))

    print(f"Cardiomegaly-only pool: {len(cardio_pool):,}  → sampling {len(cardio_set)}")
    print(f"Effusion-only pool:     {len(effusion_pool):,} → sampling {len(effusion_set)}")
    print(f"Normal pool:            {len(normal_pool):,}  → sampling {len(normal_set)}")

    out_cardio   = os.path.join(args.out, "images_cardio_only")
    out_effusion = os.path.join(args.out, "images_effusion_only")
    out_normal   = os.path.join(args.out, "images_normal")
    for path in [out_cardio, out_effusion, out_normal]:
        os.makedirs(path, exist_ok=True)

    counts = {"cardio": 0, "effusion": 0, "normal": 0}
    total_target = len(cardio_set) + len(effusion_set) + len(normal_set)

    print(f"\nScanning {args.zip} (target: {total_target} images) ...")
    with zipfile.ZipFile(args.zip) as zf:
        entries = zf.infolist()
        for i, info in enumerate(entries):
            fname = os.path.basename(info.filename)
            if fname in cardio_set:
                with open(os.path.join(out_cardio, fname), "wb") as f:
                    f.write(zf.read(info.filename))
                counts["cardio"] += 1
            elif fname in effusion_set:
                with open(os.path.join(out_effusion, fname), "wb") as f:
                    f.write(zf.read(info.filename))
                counts["effusion"] += 1
            elif fname in normal_set:
                with open(os.path.join(out_normal, fname), "wb") as f:
                    f.write(zf.read(info.filename))
                counts["normal"] += 1

            found = sum(counts.values())
            if (i + 1) % 5000 == 0:
                print(f"  scanned {i+1:,}/{len(entries):,} entries — found {found}/{total_target}")

            if found == total_target:
                print(f"  all {total_target} found — stopping early at entry {i+1:,}")
                break

    print(f"\nExtracted:")
    print(f"  {counts['cardio']:>3}  → {out_cardio}/")
    print(f"  {counts['effusion']:>3}  → {out_effusion}/")
    print(f"  {counts['normal']:>3}  → {out_normal}/")

    missing = total_target - sum(counts.values())
    if missing:
        print(f"\n  WARNING: {missing} images not found in zip (zip may be a partial extract)")


if __name__ == "__main__":
    main()
