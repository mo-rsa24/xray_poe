"""Deterministically split an image directory into two disjoint halves.

The real-vs-real noise floor (plan 03-floor / metrics.fid --selftest) needs two
DISJOINT halves of a real set — pointing `metrics.fid --a X --b X` at the same dir
gives the degenerate identical-set case, not a floor. This makes the split by
sorting filenames and dealing them even→A, odd→B, so the two halves are reproducible
and never share an image.

Symlinks by default (no copy); pass --copy to materialise real files instead.

Usage:
    python scripts/split_dir.py data/nih/images /tmp/rb_a /tmp/rb_b
    micromamba run -n jaxstack python -m metrics.fid --a /tmp/rb_a --b /tmp/rb_b --embed xrv --kid

This is the by-hand version of what metrics/floor.py (plan 03-floor.md) automates
with bootstrap 95% bounds and a small-N power flag.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def split_dir(src: Path, out_a: Path, out_b: Path, copy: bool = False) -> tuple[int, int]:
    if not src.is_dir():
        raise SystemExit(f"source dir does not exist: {src}")
    paths = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
    paths = sorted(paths)
    if not paths:
        raise SystemExit(f"no PNG/JPG images in {src}")

    for out in (out_a, out_b):
        out.mkdir(parents=True, exist_ok=True)

    n_a = n_b = 0
    for i, p in enumerate(paths):
        dst_dir = out_a if i % 2 == 0 else out_b
        dst = dst_dir / p.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy:
            shutil.copy2(p, dst)
        else:
            dst.symlink_to(p.resolve())
        if i % 2 == 0:
            n_a += 1
        else:
            n_b += 1
    return n_a, n_b


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic disjoint even/odd split of an image dir")
    p.add_argument("src", type=Path, help="Source image directory")
    p.add_argument("out_a", type=Path, help="Output dir for even-indexed half")
    p.add_argument("out_b", type=Path, help="Output dir for odd-indexed half")
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = p.parse_args()

    n_a, n_b = split_dir(args.src, args.out_a, args.out_b, copy=args.copy)
    kind = "copied" if args.copy else "symlinked"
    print(f"{kind} disjoint split of {args.src}:")
    print(f"  A: {n_a:>5} → {args.out_a}")
    print(f"  B: {n_b:>5} → {args.out_b}")


if __name__ == "__main__":
    main()
