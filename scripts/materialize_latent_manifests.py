"""Exp04 task 3 — materialize the deconfounded splits into latent manifests.

The VAE latents already exist under data/latents/train/<ImageIndex stem>.pt — so we
SELECT, never re-encode. For each (patient-filtered) split CSV, write a manifest of
the latent paths whose film is in that split AND present in the latent store.

    data/splits/cardio_not_effusion_latents.txt
    data/splits/effusion_not_cardio_latents.txt

Also surfaces latent-store contamination: how many data/latents/train latents trace
to a reserved test patient (these must be purged before any retrain re-materialize —
the manifests below are already clean because the split CSVs were patient-filtered).

    python scripts/materialize_latent_manifests.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SPLITS = Path("data/splits")
LAT_TRAIN = Path("data/latents/train")
RESERVED = SPLITS / "test_patient_ids.txt"
TARGETS = {
    "cardio_not_effusion.csv": "cardio_not_effusion_latents.txt",
    "effusion_not_cardio.csv": "effusion_not_cardio_latents.txt",
}


def main() -> None:
    have = {p.stem for p in LAT_TRAIN.glob("*.pt")}          # available latent stems
    print(f"latent store: {len(have)} .pt under {LAT_TRAIN}/")

    for csv, manifest in TARGETS.items():
        df = pd.read_csv(SPLITS / csv)
        stems = [Path(n).stem for n in df["Image Index"]]
        present = [s for s in stems if s in have]
        missing = len(stems) - len(present)
        paths = [str(LAT_TRAIN / f"{s}.pt") for s in present]
        assert all(Path(p).exists() for p in paths), "a manifest path does not resolve"
        (SPLITS / manifest).write_text("\n".join(paths) + "\n")
        label = csv.replace(".csv", "")
        print(f"materialized: {label} = {len(present)} latents "
              f"({missing} of {len(stems)} split films have no latent in store) → {manifest}")

    # ── latent-store contamination (informational; not part of the manifests) ──
    reserved = {int(x) for x in RESERVED.read_text().split()}
    de = pd.read_csv("data/nih/Data_Entry_2017.csv")
    de.columns = [c.strip() for c in de.columns]
    idx2pat = dict(zip(de["Image Index"], de["Patient ID"]))
    store_pat = {idx2pat.get(f"{s}.pt".replace(".pt", ".png")) for s in have} - {None}
    contam_films = sum(1 for s in have if idx2pat.get(f"{s}.png") in reserved)
    contam_pat = len(store_pat & reserved)
    print(f"\nlatent-store contamination vs reserved test patients: "
          f"{contam_films} latents across {contam_pat} reserved patients still live in "
          f"{LAT_TRAIN}/ — purge before retrain re-materialize (manifests above are clean).")


if __name__ == "__main__":
    main()
