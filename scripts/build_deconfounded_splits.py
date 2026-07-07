"""Exp04 task 1 — build the deconfounded single-disease split file lists.

Strict ∧ ¬other predicate (same logic as scripts/extract_single_disease.py:39-40,
which samples images; this builds the FULL training-split CSVs over the corpus):

    cardio ∧ ¬effusion = "Cardiomegaly" ∈ labels AND "Effusion" ∉ labels
    effusion ∧ ¬cardio = "Effusion" ∈ labels AND "Cardiomegaly" ∉ labels

Other comorbidities (Mass, Nodule, ...) are allowed — the only thing excluded is
the *other* of the two diseases under study, so each expert never sees it. These
counts are smaller than naive "cardio-any" — that shrinkage is the cost of clean
experts. The real single-disease label base-rates are printed for the downstream
margin reference (the cross-leakage check compares against them).

NOTE: patient-leakage subtraction (test_patient_ids.txt) is the NEXT task, not here.

    python scripts/build_deconfounded_splits.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

IMAGES = Path("data/nih/images")
OUT = Path("data/splits")
CARDIO_OUT = OUT / "cardio_not_effusion.csv"
EFFUSION_OUT = OUT / "effusion_not_cardio.csv"
COLS = ["Image Index", "Patient ID", "View Position", "Finding Labels", "path"]


def main() -> None:
    df = pd.read_csv("data/nih/Data_Entry_2017.csv")
    df.columns = [c.strip() for c in df.columns]
    labels = df["Finding Labels"].str.split("|").apply(set)

    has_cardio = labels.apply(lambda l: "Cardiomegaly" in l)
    has_effusion = labels.apply(lambda l: "Effusion" in l)

    cardio = df[has_cardio & ~has_effusion].copy()
    effusion = df[has_effusion & ~has_cardio].copy()

    for sub in (cardio, effusion):
        sub["path"] = sub["Image Index"].apply(lambda n: str(IMAGES / n))

    # ── verify (the task's acceptance checks) ────────────────────────────────
    c_idx, e_idx = set(cardio["Image Index"]), set(effusion["Image Index"])
    assert not (c_idx & e_idx), f"split label sets overlap by {len(c_idx & e_idx)} films"
    assert not cardio["Finding Labels"].str.contains("Effusion").any(), \
        "an Effusion label leaked into cardio_not_effusion"
    assert not effusion["Finding Labels"].str.contains("Cardiomegaly").any(), \
        "a Cardiomegaly label leaked into effusion_not_cardio"
    miss = [p for p in [*cardio["path"], *effusion["path"]] if not Path(p).exists()]
    assert not miss, f"{len(miss)} indices do not resolve on disk, e.g. {miss[:3]}"

    OUT.mkdir(parents=True, exist_ok=True)
    cardio[COLS].sort_values("Image Index").to_csv(CARDIO_OUT, index=False)
    effusion[COLS].sort_values("Image Index").to_csv(EFFUSION_OUT, index=False)

    # ── report ───────────────────────────────────────────────────────────────
    n = len(df)
    print(f"cardio∧¬effusion N={len(cardio)}   → {CARDIO_OUT}")
    print(f"effusion∧¬cardio N={len(effusion)} → {EFFUSION_OUT}")
    print(f"label sets disjoint ✓ · no Effusion in cardio split ✓ · all paths resolve ✓")
    print("\n-- base-rate reference (real label rates over all NIH films) --")
    print(f"  Cardiomegaly-any : {int(has_cardio.sum()):6d} ({has_cardio.mean():.2%})")
    print(f"  Effusion-any     : {int(has_effusion.sum()):6d} ({has_effusion.mean():.2%})")
    print(f"  cost of cleanliness: cardio∧¬effusion keeps "
          f"{len(cardio)}/{int(has_cardio.sum())} = {len(cardio)/has_cardio.sum():.1%} of cardio-any")


if __name__ == "__main__":
    main()
