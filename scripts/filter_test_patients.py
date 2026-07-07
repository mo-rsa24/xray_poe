"""Exp04 task 2 — patient-leakage guard: drop reserved test patients from the
deconfounded splits, in place.

NIH carries multiple films per patient, so an image-level split leaks: a patient
in the frozen separable test set must not also appear in expert training data.
This drops any split film whose `Patient ID` ∈ data/splits/test_patient_ids.txt
(the 317 reserved in Exp 1) and rewrites the split CSVs in place. Idempotent —
re-running on already-filtered splits drops 0.

    python scripts/filter_test_patients.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SPLITS = Path("data/splits")
RESERVED = SPLITS / "test_patient_ids.txt"
TARGETS = ["cardio_not_effusion.csv", "effusion_not_cardio.csv"]
REPORT = Path("results/exp04_leakage/patient_disjoint_check.md")


def main() -> None:
    reserved = {int(x) for x in RESERVED.read_text().split()}
    lines = ["# Exp04 — patient-disjointness check", "",
             f"Reserved test patients (from Exp 1 `test_patient_ids.txt`): **{len(reserved)}**", "",
             "| split | films before | films dropped | patients dropped | films after |",
             "|---|---|---|---|---|"]
    total_dropped = 0
    for name in TARGETS:
        path = SPLITS / name
        df = pd.read_csv(path)
        before = len(df)
        leak_mask = df["Patient ID"].isin(reserved)
        pats_dropped = df.loc[leak_mask, "Patient ID"].nunique()
        kept = df[~leak_mask].copy()
        kept.to_csv(path, index=False)               # in place
        dropped = before - len(kept)
        total_dropped += dropped
        # verify: no reserved patient survives
        assert not set(kept["Patient ID"]) & reserved, f"{name} still shares test patients"
        print(f"dropped {dropped} films sharing a test patient from {name}; "
              f"splits now patient-disjoint ({len(kept)} remain)")
        lines.append(f"| {name} | {before} | {dropped} | {pats_dropped} | {len(kept)} |")

    print(f"\ntotal dropped across both splits: {total_dropped} films")
    lines += ["",
              f"**Total films dropped:** {total_dropped}.",
              "**Verify:** intersection of either split's `Patient ID`s with "
              "`test_patient_ids.txt` is empty ✓ (asserted in `filter_test_patients.py`).",
              "",
              "Splits rewritten in place; raw (pre-filter) lists are regenerable via "
              "`scripts/build_deconfounded_splits.py`. This filter is idempotent.",
              "",
              "Note: this guards the *file lists*. The existing `data/latents/train` "
              "still physically contains these patients (it predates the hold-out) and "
              "must be re-materialized from these filtered lists before any retrain."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"→ {REPORT}")


if __name__ == "__main__":
    main()
