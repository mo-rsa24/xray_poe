#!/usr/bin/env python3
"""Partition NIH ChestX-ray14 into four cardiomegaly×effusion groups.

Groups:
  normal       — "No Finding" only
  cardio_only  — Cardiomegaly ∧ ¬Effusion  (¬ = not mentioned)
  effusion_only— Effusion ∧ ¬Cardiomegaly
  both         — Cardiomegaly ∧ Effusion

Outputs:
  --out    partition.parquet  (Image Index, group)
  --counts four_group_counts.md
"""
import argparse
import os

import pandas as pd


def assign_group(labels):
    has_c = "Cardiomegaly" in labels
    has_e = "Effusion" in labels
    if labels == {"No Finding"}:
        return "normal"
    if has_c and has_e:
        return "both"
    if has_c:
        return "cardio_only"
    if has_e:
        return "effusion_only"
    return None  # other findings only — excluded from the four groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Path to Data_Entry_2017.csv")
    p.add_argument("--latent-dir", default=None,
                   help="If given, restrict to images whose stem has a .pt here")
    p.add_argument("--out", required=True, help="Output parquet path")
    p.add_argument("--counts", required=True, help="Output markdown counts path")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["labels"] = df["Finding Labels"].apply(lambda s: set(s.split("|")))
    df["group"] = df["labels"].apply(assign_group)
    df = df[df["group"].notna()].copy()

    if args.latent_dir:
        available = {
            os.path.splitext(f)[0]
            for f in os.listdir(args.latent_dir)
            if f.endswith(".pt")
        }
        df["stem"] = df["Image Index"].apply(lambda x: os.path.splitext(x)[0])
        before = len(df)
        df = df[df["stem"].isin(available)].copy()
        print(f"Restricted to latent-dir: {len(df):,} / {before:,} four-group images have latents")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df[["Image Index", "group"]].to_parquet(args.out, index=False)

    counts = df.groupby("group").size()
    total = int(counts.sum())

    latent_note = (
        f"Restricted to images with precomputed latents in `{args.latent_dir}`."
        if args.latent_dir
        else "Full CSV — not restricted to precomputed latents."
    )
    lines = [
        "# NIH ChestX-ray14 — four-group partition (cardiomegaly × effusion)",
        "",
        f"Source: `Data_Entry_2017.csv`. {latent_note}",
        "",
        "## Counts",
        "| group | images |",
        "|---|---|",
    ]
    for g in ["normal", "cardio_only", "effusion_only", "both"]:
        n = int(counts.get(g, 0))
        flag = "  ⚠️ thin (<500)" if n < 500 else ""
        lines.append(f"| {g} | {n:,}{flag} |")
    lines += [
        "",
        f"Total four-group images: {total:,}",
        "",
        "## Caveat",
        "¬ means **not mentioned** in the label string — NOT radiologist-confirmed absent.",
        "NIH ChestX-ray14 uses dirty negatives; purity is limited by label noise.",
        "",
        "## Partition index",
        f"Written to `{args.out}` — columns: `Image Index`, `group`.",
    ]

    md = "\n".join(lines) + "\n"
    with open(args.counts, "w") as f:
        f.write(md)

    print(md)
    print(f"Partition index → {args.out}")


if __name__ == "__main__":
    main()
