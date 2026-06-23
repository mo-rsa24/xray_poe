"""Materialize the fine-tune train/val split to disk under data/splits/.

The presence classifier (scripts/finetune_classifier.NIHPresenceDataset) splits
train/val *in memory* (sorted *.png → random.Random(seed).shuffle → first val_frac as
val). Nothing was written to disk, so the documented validate paths
data/splits/val/<class>/ did not exist. This writes that split out as symlinks,
reproducing the in-memory logic EXACTLY, so:

    data/splits/{train,val}/{normal,cardiomegaly,effusion}/

becomes real and the validate command works verbatim on the true held-out set (no
training-data leakage). Symlinks by default (no copy); pass --copy for real files.

Class-dir names match the validate CLI (cardiomegaly/effusion/normal), mapped from the
source NIH group folders.

Usage:
    python scripts/materialize_split.py                  # default data/nih → data/splits
    python scripts/materialize_split.py --val_frac 0.2 --seed 42
    micromamba run -n jaxstack python -m metrics.presence_classifier --mode validate \\
        --real_cardio data/splits/val/cardiomegaly --real_effusion data/splits/val/effusion \\
        --real_normal data/splits/val/normal
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

# source NIH group folder → split class-dir name (matches the fine-tune GROUPS + CLI)
_GROUPS = {
    "images_normal": "normal",
    "images_cardio_only": "cardiomegaly",
    "images_effusion_only": "effusion",
}


def materialize(root: Path, out: Path, val_frac: float, seed: int, copy: bool) -> None:
    for folder, cls in _GROUPS.items():
        paths = sorted(Path(root, folder).glob("*.png"))     # match NIHPresenceDataset
        if not paths:
            raise SystemExit(f"No PNGs in {root}/{folder}")
        rng = random.Random(seed)
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_frac))
        splits = {"val": paths[:n_val], "train": paths[n_val:]}

        for split, chosen in splits.items():
            dst_dir = out / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in chosen:
                dst = dst_dir / p.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if copy:
                    shutil.copy2(p, dst)
                else:
                    dst.symlink_to(p.resolve())
        print(f"  {cls:12s}: {len(splits['val']):>4} val  +  {len(splits['train']):>4} train"
              f"  (from {folder})")


def main() -> None:
    p = argparse.ArgumentParser(description="Materialize the fine-tune train/val split to disk")
    p.add_argument("--root", type=Path, default=Path("data/nih"), help="Source NIH dir")
    p.add_argument("--out", type=Path, default=Path("data/splits"), help="Output split dir")
    p.add_argument("--val_frac", type=float, default=0.2, help="Match fine-tune: 0.2")
    p.add_argument("--seed", type=int, default=42, help="Match fine-tune: 42")
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = p.parse_args()

    kind = "copied" if args.copy else "symlinked"
    print(f"Materializing split ({kind}) {args.root} → {args.out} "
          f"(val_frac={args.val_frac}, seed={args.seed}):")
    materialize(args.root, args.out, args.val_frac, args.seed, args.copy)
    print(f"Done. Held-out val at {args.out}/val/<class>/")


if __name__ == "__main__":
    main()
