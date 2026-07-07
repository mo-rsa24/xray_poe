"""Exp02 task 2 — assemble the recon set: frozen separable-both + matched control.

The codec ceiling is read on the frozen separable-both test set (Exp 1) and on a
size-matched SINGLE-disease control, so we can show the VAE doesn't specifically
destroy the both-disease appearance (separable local LPIPS ≤ single-disease control).

Control films carry exactly ONE of the two diseases — drawn from the patient-
filtered deconfounded splits (Exp 4), PA only, balanced cardio/effusion, and thus
already patient-disjoint from the separable test set.

    python scripts/build_recon_set.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SPLITS = Path("data/splits")
SEPARABLE = SPLITS / "pixel_verifiable_separable_test.csv"
CARDIO = SPLITS / "cardio_not_effusion.csv"
EFFUSION = SPLITS / "effusion_not_cardio.csv"
OUT = Path("results/exp02_ceiling/recon_set_manifest.csv")
SEED = 42


def main() -> None:
    sep = pd.read_csv(SEPARABLE)
    n = len(sep)
    sep_rows = sep[["Image Index", "Patient ID", "View Position", "path"]].copy()
    sep_rows["group"] = "separable"

    # size-matched single-disease control, PA only, balanced across the two diseases
    half = n // 2
    ctrl = []
    for csv, grp, k in [(CARDIO, "control_cardio", n - half), (EFFUSION, "control_effusion", half)]:
        d = pd.read_csv(csv)
        d = d[d["View Position"] == "PA"]
        d = d.sample(min(k, len(d)), random_state=SEED)
        r = d[["Image Index", "Patient ID", "View Position", "path"]].copy()
        r["group"] = grp
        ctrl.append(r)
    manifest = pd.concat([sep_rows, *ctrl], ignore_index=True)

    # verify: control films patient-disjoint from separable + carry one disease only
    test_pat = set(sep["Patient ID"])
    ctrl_pat = set(pd.concat(ctrl)["Patient ID"])
    assert not (test_pat & ctrl_pat), f"control shares {len(test_pat & ctrl_pat)} patients with test"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT, index=False)

    vc = manifest["group"].value_counts()
    print(f"separable N={vc.get('separable', 0)}, "
          f"control N={vc.get('control_cardio', 0) + vc.get('control_effusion', 0)} "
          f"(cardio {vc.get('control_cardio', 0)} / effusion {vc.get('control_effusion', 0)})")
    print("control patient-disjoint from separable test ✓ · single-disease by construction ✓")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
