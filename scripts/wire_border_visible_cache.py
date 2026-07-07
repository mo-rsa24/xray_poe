"""Rebalanced-retrain — re-wire the latent cache to BORDER-VISIBLE films only.

The deconfounded splits were filtered by disease LABEL, but ~59% of effusion films
(and ~34% of cardio) have an obscured cardiac border (whiteout). Training the
effusion conditional on border-obscuring films makes PoE compositions untestable
(the composed CTR can't be read). This keeps only films whose cardiac border is
visible (ctr_overlay border_visible=True) — so the training distribution matches
the SEPARABLE test distribution and every composition is testable.

Scores border_visible once (cached to results/exp05_decision/split_border_visible.csv),
then rebuilds data/latents/rebalanced/{train,val} from the visible, patient-clean,
with-latent films. Cardio + effusion + normal, all label-baked, patient-disjoint.

    python scripts/wire_border_visible_cache.py            # full (~15 min: scores ~17k films once)
    python scripts/wire_border_visible_cache.py --score-limit 20   # smoke
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.ctr_overlay as C

TRAIN = Path("data/latents/train")
OUT = Path("data/latents/rebalanced")
SPLITS = Path("data/splits")
CACHE = Path("results/exp05_decision/split_border_visible.csv")
NORMAL_CAP = 4000
NORMAL_CANDIDATES = 6000     # score this many normal; keep the first NORMAL_CAP visible
VAL_FRAC = 0.03
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-limit", type=int, default=0, help="cap films scored per class (smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    present = {p.stem for p in TRAIN.glob("*.pt")}
    de = pd.read_csv("data/nih/Data_Entry_2017.csv"); de.columns = [c.strip() for c in de.columns]
    i2p = dict(zip(de["Image Index"], de["Patient ID"]))
    res = {int(x) for x in (SPLITS / "test_patient_ids.txt").read_text().split()}

    def manifest_stems(csv):
        d = pd.read_csv(SPLITS / csv)
        return [Path(n).stem for n in d["Image Index"] if Path(n).stem in present]

    cardio = manifest_stems("cardio_not_effusion.csv")
    effusion = manifest_stems("effusion_not_cardio.csv")
    part = pd.read_parquet("data/nih/partition.parquet")
    normal = [Path(n).stem for n in part[part["group"] == "normal"]["Image Index"]
              if Path(n).stem in present and i2p.get(Path(n).stem + ".png") not in res]
    normal = random.Random(SEED).sample(normal, min(NORMAL_CANDIDATES, len(normal)))
    cand = {1: cardio, 2: effusion, 0: normal}
    if args.score_limit:
        cand = {k: v[:args.score_limit] for k, v in cand.items()}

    # ── score border_visible (cached) ───────────────────────────────────────
    if CACHE.exists():
        bv = pd.read_csv(CACHE)
        print(f"reusing cached border_visible reads: {len(bv)} films ({CACHE})")
    else:
        seg, idx = C._load_seg(args.device)
        rows = []
        for lab, stems in cand.items():
            for j, s in enumerate(stems):
                m = C.ctr_for_image(seg, idx, Path("data/nih/images") / f"{s}.png", args.device)
                rows.append({"stem": s, "label": lab, "border_visible": bool(m["border_visible"])})
                if (j + 1) % 500 == 0:
                    print(f"  label {lab}: scored {j + 1}/{len(stems)}")
        bv = pd.DataFrame(rows)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        bv.to_csv(CACHE, index=False)
        print(f"border_visible reads → {CACHE}")

    # ── report visibility per class ─────────────────────────────────────────
    print("\n=== border-visible fraction (label 0=normal,1=cardio,2=effusion) ===")
    for lab in (0, 1, 2):
        d = bv[bv["label"] == lab]
        if len(d):
            print(f"  label {lab}: {d['border_visible'].mean():.1%} visible  ({int(d['border_visible'].sum())}/{len(d)})")

    # ── select visible, cap normal, build cache ─────────────────────────────
    sel = {}
    for lab in (1, 2):
        sel[lab] = bv[(bv["label"] == lab) & bv["border_visible"]]["stem"].tolist()
    nvis = bv[(bv["label"] == 0) & bv["border_visible"]]["stem"].tolist()
    sel[0] = nvis[:NORMAL_CAP]

    for sp in ("train", "val"):
        (OUT / sp).mkdir(parents=True, exist_ok=True)
        for old in (OUT / sp).glob("*.pt"):
            old.unlink()
    counts: Counter = Counter()
    for lab, stems in sel.items():
        assert not any(i2p.get(s + ".png") in res for s in stems), f"label {lab} leaks a test patient"
        st = list(stems); random.Random(SEED).shuffle(st)
        k = max(1, int(len(st) * VAL_FRAC))
        for sp, part_sel in (("val", st[:k]), ("train", st[k:])):
            for s in part_sel:
                (OUT / sp / f"{s}.pt").symlink_to((TRAIN / f"{s}.pt").resolve())
            counts[(lab, sp)] = len(part_sel)

    import shutil
    sf = Path("data/latents/scale_factor.pt")
    if sf.exists():
        shutil.copy(sf, OUT / "scale_factor.pt")

    print("\n=== border-visible rebalanced cache → data/latents/rebalanced/ ===")
    for lab, name in [(0, "normal"), (1, "cardio"), (2, "effusion")]:
        print(f"  label {lab} {name:9s}: train {counts[(lab,'train')]:5d}  val {counts[(lab,'val')]:4d}")
    print(f"  TOTAL train {sum(counts[(l,'train')] for l in (0,1,2))} · patient-disjoint ✓ · border-visible only ✓")


if __name__ == "__main__":
    main()
