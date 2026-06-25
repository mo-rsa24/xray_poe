"""Freeze the pixel-verifiable separable test set + reserve its patients.

Reads results/exp01_dynamics/per_film_ctr_cp.csv and writes, under data/splits/:
  pixel_verifiable_separable_test.csv  PA separable films (CTR valid on PA only)
  entangled_scoped_out.csv             all entangled films (logged, NOT deleted)
  ap_separable_logged.csv              AP separable (separable signs, AP-magnified CTR)
  test_patient_ids.txt                 unique Patient IDs across the PA-separable films
  latent_refilter_manifest.csv         which data/latents/test_both/*.pt are in/out
                                       (manifest only — no latent is moved or deleted)

Also checks patient-level leakage against the existing data/latents/train/*.pt set.

    python scripts/freeze_separable_test.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

PER_FILM = Path("results/exp01_dynamics/per_film_ctr_cp.csv")
SPLITS = Path("data/splits")
LAT_TEST = Path("data/latents/test_both")
LAT_TRAIN = Path("data/latents/train")
DATA_ENTRY = "data/nih/Data_Entry_2017.csv"


def main() -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PER_FILM)
    sep = df["class"] == "separable"
    pa = df["View Position"] == "PA"

    pa_sep = df[sep & pa].copy()
    ap_sep = df[sep & ~pa].copy()
    entangled = df[~sep].copy()
    assert len(pa_sep) + len(ap_sep) + len(entangled) == len(df)

    pa_sep.to_csv(SPLITS / "pixel_verifiable_separable_test.csv", index=False)
    ap_sep.to_csv(SPLITS / "ap_separable_logged.csv", index=False)
    entangled[["Image Index", "Patient ID", "View Position", "reason", "path"]].to_csv(
        SPLITS / "entangled_scoped_out.csv", index=False)

    # ── reserve patients (every PA-separable test film) ──────────────────────
    patients = sorted(pa_sep["Patient ID"].unique())
    (SPLITS / "test_patient_ids.txt").write_text("\n".join(map(str, patients)) + "\n")

    print(f"FROZEN pixel-verifiable test set (PA separable): {len(pa_sep)} films")
    print(f"  entangled scoped-out (logged, not deleted)   : {len(entangled)}")
    print(f"  AP separable logged (CTR not PA-comparable)   : {len(ap_sep)}")
    print(f"test patients reserved: {len(patients)} unique Patient IDs across {len(pa_sep)} films")

    # ── latent re-filter manifest (no mutation) ──────────────────────────────
    if LAT_TEST.exists():
        on_disk = {Path(p).stem for p in glob.glob(str(LAT_TEST / "*.pt"))}
        keep = {Path(n).stem for n in pa_sep["Image Index"]}
        man = pd.DataFrame({"stem": sorted(on_disk)})
        man["keep_separable_PA"] = man["stem"].isin(keep)
        man.to_csv(SPLITS / "latent_refilter_manifest.csv", index=False)
        print(f"latent manifest: {int(man['keep_separable_PA'].sum())}/{len(man)} "
              f"test_both latents are PA-separable (rest scoped out — NOT deleted)")

    # ── patient-leakage check vs existing train latents ──────────────────────
    if LAT_TRAIN.exists():
        de = pd.read_csv(DATA_ENTRY); de.columns = [c.strip() for c in de.columns]
        idx2pat = dict(zip(de["Image Index"], de["Patient ID"]))
        train_idx = [Path(p).name.replace(".pt", ".png") for p in glob.glob(str(LAT_TRAIN / "*.pt"))]
        train_patients = {idx2pat.get(i) for i in train_idx} - {None}
        leak = set(patients) & train_patients
        print(f"\nLEAKAGE CHECK vs data/latents/train: "
              f"{len(leak)} reserved test patients also present in train")
        if leak:
            print("  ⚠ existing train set PREDATES this hold-out and leaks these patients —")
            print("    training builders must subtract data/splits/test_patient_ids.txt on the")
            print("    next (rebalanced) retrain. Existing latents NOT mutated here.")
            (SPLITS / "leaked_patient_ids.txt").write_text(
                "\n".join(map(str, sorted(leak))) + "\n")
        else:
            print("  ✓ no overlap — hold-out is clean against current train latents")

    print(f"\n→ {SPLITS}/")


if __name__ == "__main__":
    main()
