"""Exp04 task 4 — cross-leakage check harness.

Reuses the ctr_overlay PSPNet instrument to ask: does the cardio-conditioned
generator render EFFUSION (a blunted costophrenic corner), and does the
effusion-conditioned generator render CARDIO (CTR > 0.5)? Each is the *other*
disease leaking into a single-disease expert.

Presence definitions (per generated/real film, from the instrument):
    cardio_present   = border_visible AND CTR > 0.5
    effusion_present = either costophrenic recess blunted

Scores four sets and writes one tidy per-image CSV:
    cardio_gen    — generated --disease cardiomegaly  (self=cardio, cross=effusion LEAK)
    effusion_gen  — generated --disease effusion       (self=effusion, cross=cardio LEAK)
    cardio_real   — real cardio∧¬effusion films         (effusion base-rate reference)
    effusion_real — real effusion∧¬cardio films         (cardio   base-rate reference)

Generate the two _gen sets first with scripts/generate.py (see the leaf commands),
then point this at their output dirs:

    python scripts/cross_leakage_check.py \\
        --cardio-gen  outputs/exp04_leakage/cardiomegaly \\
        --effusion-gen outputs/exp04_leakage/effusion \\
        --n-real 1000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.ctr_overlay as C

SPLITS = Path("data/splits")
OUT = Path("results/exp04_leakage/cross_presence.csv")
CTR_CARDIO = 0.5


def _presence(m: dict) -> tuple[bool, bool]:
    cardio = bool(m.get("border_visible") and np.isfinite(m.get("ctr", float("nan")))
                  and m["ctr"] > CTR_CARDIO)
    cl, cr = m.get("cp_left", {}), m.get("cp_right", {})
    effusion = bool(cl.get("blunted") or cr.get("blunted"))
    return cardio, effusion


def _imgs(d: Path) -> list[Path]:
    return sorted([*d.glob("*.png"), *d.glob("*.jpg")])


def _real(csv: str, n: int, seed: int) -> list[Path]:
    paths = pd.read_csv(SPLITS / csv)["path"].tolist()
    return [Path(p) for p in random.Random(seed).sample(paths, min(n, len(paths)))]


def _score(seg, idx, name: str, paths: list[Path], dev: str) -> list[dict]:
    rows = []
    for p in paths:
        m = C.ctr_for_image(seg, idx, p, dev)
        cardio, effusion = _presence(m)
        rows.append({"set": name, "path": str(p),
                     "ctr": round(m["ctr"], 3) if np.isfinite(m["ctr"]) else np.nan,
                     "border_visible": m.get("border_visible"),
                     "cp_blunted": effusion, "cardio_present": cardio,
                     "effusion_present": effusion})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cardio-gen", type=Path, help="dir of --disease cardiomegaly PNGs")
    ap.add_argument("--effusion-gen", type=Path, help="dir of --disease effusion PNGs")
    ap.add_argument("--floor-gen", type=Path, help="dir of --disease no_finding PNGs "
                    "(generated-image instrument floor)")
    ap.add_argument("--n-real", type=int, default=1000, help="real reference films per side (0=skip)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seg, idx = C._load_seg(args.device)
    sets: list[tuple[str, list[Path]]] = []
    if args.cardio_gen:
        sets.append(("cardio_gen", _imgs(args.cardio_gen)))
    if args.effusion_gen:
        sets.append(("effusion_gen", _imgs(args.effusion_gen)))
    if args.floor_gen:
        sets.append(("no_finding_floor", _imgs(args.floor_gen)))
    if args.n_real:
        sets.append(("cardio_real", _real("cardio_not_effusion.csv", args.n_real, args.seed)))
        sets.append(("effusion_real", _real("effusion_not_cardio.csv", args.n_real, args.seed)))

    rows = []
    for name, paths in sets:
        if not paths:
            print(f"  [skip] {name}: no images"); continue
        rows += _score(seg, idx, name, paths, args.device)
        print(f"  scored {name}: {len(paths)} films")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n=== presence rates ===")
    for name in df["set"].unique():
        d = df[df["set"] == name]
        print(f"{name:14s} n={len(d):4d}  cardio_present={d['cardio_present'].mean():.1%}  "
              f"effusion_present={d['effusion_present'].mean():.1%}")
    print("\nLEAK reads: cardio_gen→effusion_present (effusion leak) · "
          "effusion_gen→cardio_present (cardio leak)")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
