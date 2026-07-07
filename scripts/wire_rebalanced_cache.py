"""Rebalanced-retrain sub-scope task 02 — wire the deconfounded, patient-disjoint
latent cache the rebalanced LDM trains on.

LatentDataset globs an ENTIRE split dir and reads the label baked into each .pt —
it has no manifest/filter. So we assemble a clean cache by symlinking the SELECTED
latents into data/latents/rebalanced/{train,val}/:
  • cardio (label 1)   = Exp-4 cardio_not_effusion_latents.txt   (deconfounded, patient-filtered)
  • effusion (label 2) = Exp-4 effusion_not_cardio_latents.txt   (deconfounded, patient-filtered)
  • normal (label 0)   = partition 'normal', patient-clean, latent-present, capped (diversity)
Every stem is asserted patient-disjoint from data/splits/test_patient_ids.txt, and the
baked labels are spot-checked. Select-don't-re-encode (symlinks, no VAE).

    python scripts/wire_rebalanced_cache.py
"""
from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

TRAIN = Path("data/latents/train")
OUT = Path("data/latents/rebalanced")
SPLITS = Path("data/splits")
NORMAL_CAP = 4000          # matches the prior no_finding_cap; class_mult sets the 25% share
VAL_FRAC = 0.03
SEED = 42


def main() -> None:
    present = {p.stem for p in TRAIN.glob("*.pt")}
    de = pd.read_csv("data/nih/Data_Entry_2017.csv"); de.columns = [c.strip() for c in de.columns]
    i2p = dict(zip(de["Image Index"], de["Patient ID"]))
    res = {int(x) for x in (SPLITS / "test_patient_ids.txt").read_text().split()}

    def stems(man):
        return [Path(p).stem for p in (SPLITS / man).read_text().split() if p]

    cardio = [s for s in stems("cardio_not_effusion_latents.txt") if s in present]
    effusion = [s for s in stems("effusion_not_cardio_latents.txt") if s in present]
    part = pd.read_parquet("data/nih/partition.parquet")
    normal_all = [Path(n).stem for n in part[part["group"] == "normal"]["Image Index"]]
    normal = [s for s in normal_all if s in present and i2p.get(s + ".png") not in res]
    normal = random.Random(SEED).sample(normal, min(NORMAL_CAP, len(normal)))

    groups = {1: cardio, 2: effusion, 0: normal}

    # ── verify: patient-disjoint + baked labels correct (spot-check) ─────────
    for lab, ss in groups.items():
        leak = sum(1 for s in ss if i2p.get(s + ".png") in res)
        assert leak == 0, f"label {lab} leaks {leak} test patients"
        for s in random.Random(1).sample(ss, min(15, len(ss))):
            got = int(torch.load(TRAIN / f"{s}.pt", map_location="cpu", weights_only=True)["label"])
            assert got == lab, f"{s}: baked label {got} != expected {lab}"

    # ── assemble train/val by symlink ────────────────────────────────────────
    for sp in ("train", "val"):
        (OUT / sp).mkdir(parents=True, exist_ok=True)
        for old in (OUT / sp).glob("*.pt"):
            old.unlink()
    counts: Counter = Counter()
    for lab, ss in groups.items():
        st = list(ss); random.Random(SEED).shuffle(st)
        k = max(1, int(len(st) * VAL_FRAC))
        for sp, sel in (("val", st[:k]), ("train", st[k:])):
            for s in sel:
                (OUT / sp / f"{s}.pt").symlink_to((TRAIN / f"{s}.pt").resolve())
            counts[(lab, sp)] = len(sel)

    sf = Path("data/latents/scale_factor.pt")
    if sf.exists():
        shutil.copy(sf, OUT / "scale_factor.pt")

    print("rebalanced latent cache → data/latents/rebalanced/")
    for lab, name in [(0, "normal"), (1, "cardio"), (2, "effusion")]:
        print(f"  label {lab} {name:9s}: train {counts[(lab,'train')]:5d}  val {counts[(lab,'val')]:4d}")
    tot_tr = sum(counts[(l, 'train')] for l in (0, 1, 2))
    print(f"  TOTAL train {tot_tr}  · patient-disjoint from test ✓ · labels spot-checked ✓ · scale_factor copied")
    print("  (class_mult in configs/ldm_rebalanced.yaml sets the 25/40/35 SAMPLED mix; raw counts don't.)")


if __name__ == "__main__":
    main()
