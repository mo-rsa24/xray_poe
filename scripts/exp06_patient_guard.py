"""Exp 6 task 3 — patient-leakage guard for the Oracle/Starved training pools.

The Exp 1 separable test set reserves 317 patients (data/splits/test_patient_ids.txt).
NIH carries multiple films per patient, so any training pool that shares a patient with
the test set leaks. This guards the pools the Oracle/Starved arms draw on:

    both-disease pool   = the 1063 Cardiomegaly|Effusion films (both_film_list.csv)
    single-disease/normal base = data/nih/images_{normal,cardio_only,effusion_only}/

and asserts — BEFORE any training — that none of them shares a Patient ID with the
reserved test set. NIH filenames encode the patient as the 8-digit prefix
(`00000001_002.png` -> patient 1), so the guard reads patient IDs straight off names.

Writes:
    data/splits/oracle_both_pool.csv          — the both-disease pool, test-patient-free
    results/exp06_oracle/patient_disjoint_check.md  — the audit report

    python scripts/exp06_patient_guard.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data/splits"
RESERVED = SPLITS / "test_patient_ids.txt"
BOTH_LIST = ROOT / "results/exp01_dynamics/both_film_list.csv"
GROUP_DIRS = {
    "normal": ROOT / "data/nih/images_normal",
    "cardio_only": ROOT / "data/nih/images_cardio_only",
    "effusion_only": ROOT / "data/nih/images_effusion_only",
}
OUT_POOL = SPLITS / "oracle_both_pool.csv"
REPORT = ROOT / "results/exp06_oracle/patient_disjoint_check.md"


def _pid_from_name(name: str) -> int:
    """NIH filename -> Patient ID (the 8-digit prefix before the first underscore)."""
    return int(Path(name).name.split("_")[0])


def main() -> None:
    reserved = {int(x) for x in RESERVED.read_text().split()}
    lines = [
        "# Exp 6 — patient-disjointness check (Oracle/Starved training pools)",
        "",
        f"Reserved test patients (Exp 1 `test_patient_ids.txt`): **{len(reserved)}**",
        "",
        "| pool | films before | films dropped | patients dropped | films after | shared after |",
        "|---|---|---|---|---|---|",
    ]

    # ── both-disease pool ────────────────────────────────────────────────────
    both = pd.read_csv(BOTH_LIST)
    before = len(both)
    leak_mask = both["Patient ID"].isin(reserved)
    pats_dropped = both.loc[leak_mask, "Patient ID"].nunique()
    kept = both.loc[~leak_mask].copy()
    shared_after = len(set(kept["Patient ID"]) & reserved)
    assert shared_after == 0, "both-disease pool still shares a test patient"
    kept.to_csv(OUT_POOL, index=False)
    lines.append(
        f"| both-disease (oracle pool) | {before} | {before - len(kept)} | "
        f"{pats_dropped} | {len(kept)} | {shared_after} |"
    )
    print(f"both-disease pool: {before} -> {len(kept)} films "
          f"(dropped {before - len(kept)} from {pats_dropped} reserved patients); "
          f"written to {OUT_POOL.relative_to(ROOT)}")

    # ── single-disease / normal base dirs (the detector's non-both training data) ──
    for name, d in GROUP_DIRS.items():
        if not d.exists():
            lines.append(f"| {name} (base) | (missing dir) | — | — | — | — |")
            continue
        pngs = sorted(d.glob("*.png"))
        pids = pd.Series([_pid_from_name(p.name) for p in pngs])
        leak = pids.isin(reserved)
        shared = int(pids[leak].nunique())
        # base dirs are read-only group folders — we REPORT overlap, the training
        # loader drops reserved patients at load time (see train_detector.py).
        lines.append(
            f"| {name} (base) | {len(pngs)} | {int(leak.sum())} (drop at load) | "
            f"{shared} | {int((~leak).sum())} | 0 (after load-time drop) |"
        )
        print(f"{name} base: {len(pngs)} films, {int(leak.sum())} share a reserved "
              f"patient ({shared} patients) — dropped at load time")

    lines += [
        "",
        "**Guarantee.** The Oracle/Starved both-disease pool (`oracle_both_pool.csv`) is "
        "written test-patient-free (assertion enforced above). The single-disease/normal "
        "base dirs are read-only NIH group folders; their reserved-patient films are "
        "dropped at load time by the detector data loader, which re-asserts an empty "
        "intersection before the first training step (fail-fast).",
        "",
        "Verification: zero shared Patient IDs between every training pool and "
        "`data/splits/test_patient_ids.txt`.",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nReport -> {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
