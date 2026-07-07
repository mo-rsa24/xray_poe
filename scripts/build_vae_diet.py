"""Build the combined NIH+VinDr border-visible manifest for VAE-v2 fine-tuning.

Inputs:
  results/exp05_decision/split_border_visible.csv   NIH border-visible cache
                                                    (stem, label, border_visible)
  data/latents/vindr/border_visible_manifest.csv    VinDr border-visible set
                                                    (image_id, class, ...)
  data/splits/test_patient_ids.txt                  317 reserved NIH patient IDs

Output:
  data/vae_v2_diet/train_manifest.csv               columns: image_path, source, label,
                                                    border_visible
  data/vae_v2_diet/leakage_check.txt                intersection=0 assertion log

No latent_path column — latent encoding is plan 05's concern, not ours.

Usage:
  python scripts/build_vae_diet.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NIH_BV          = Path("results/exp05_decision/split_border_visible.csv")
NIH_IMAGE_DIR   = Path("data/nih/images")
VINDR_BV        = Path("data/latents/vindr/border_visible_manifest.csv")
VINDR_IMAGE_DIR = Path("data/vindr512/vinbigdata/train")
TEST_PIDS       = Path("data/splits/test_patient_ids.txt")
OUT_MANIFEST    = Path("data/vae_v2_diet/train_manifest.csv")
OUT_LEAKAGE     = Path("data/vae_v2_diet/leakage_check.txt")


def _patient_id_from_stem(stem: str) -> str:
    # NIH stems: "00000008_000" → patient ID is the numeric part before "_"
    return str(int(stem.split("_")[0]))


def load_nih(reserved_pids: set[str]) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(NIH_BV)
    bv  = raw[raw["border_visible"] == True].copy()

    bv["patient_id"] = bv["stem"].map(_patient_id_from_stem)
    leaked = bv[bv["patient_id"].isin(reserved_pids)]
    clean  = bv[~bv["patient_id"].isin(reserved_pids)].copy()

    clean["image_path"]    = clean["stem"].map(lambda s: str(NIH_IMAGE_DIR / f"{s}.png"))
    clean["source"]        = "nih"
    clean["border_visible"]= True

    return clean[["image_path", "source", "label", "border_visible"]], len(leaked)


def load_vindr() -> pd.DataFrame:
    bv = pd.read_csv(VINDR_BV)

    # Structural disjointness assertion: VinDr IDs are 32-char hex; NIH IDs are integers
    non_hex = bv["image_id"].str.fullmatch(r"[0-9a-f]{32}").sum()
    assert non_hex == len(bv), "VinDr image_id format changed — re-check disjointness"
    print(f"  VinDr ID format: all {len(bv)} IDs are hex-32 strings "
          f"(structurally disjoint from NIH integer patient IDs) ✓")

    bv["image_path"]    = bv["image_id"].map(lambda i: str(VINDR_IMAGE_DIR / f"{i}.png"))
    bv["source"]        = "vindr"
    bv["label"]         = bv["class"]
    bv["border_visible"]= True

    return bv[["image_path", "source", "label", "border_visible"]]


def write_leakage_check(n_excluded: int) -> None:
    lines = [
        "NIH patient-disjoint leakage check",
        f"Reserved test patient IDs: {TEST_PIDS} (N=317)",
        f"NIH rows excluded (Patient ID in reserved set): {n_excluded}",
        f"intersection={n_excluded}",
        "",
        "VinDr disjointness: structural (hex-32 IDs vs integer NIH patient IDs) — "
        "no intersection possible by format.",
        "",
        f"PASS: intersection=0  {'✓' if n_excluded == 0 else '✗ — UNEXPECTED, investigate'}",
    ]
    OUT_LEAKAGE.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    reserved = set(open(TEST_PIDS).read().split())
    print(f"Reserved patient IDs loaded: {len(reserved)}")

    print("Loading NIH border-visible set …")
    nih_df, n_excluded = load_nih(reserved)
    print(f"  NIH border-visible: {len(nih_df) + n_excluded} total → "
          f"{n_excluded} excluded (test patients) → {len(nih_df)} clean")

    print("Loading VinDr border-visible set …")
    vindr_df = load_vindr()
    print(f"  VinDr border-visible: {len(vindr_df)}")

    combined = (
        pd.concat([nih_df, vindr_df], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    combined.to_csv(OUT_MANIFEST, index=False)
    write_leakage_check(n_excluded)

    print(f"\nN_nih={len(nih_df)} N_vindr={len(vindr_df)} "
          f"N_total={len(combined)} N_excluded_leaked={n_excluded}")
    print(f"\nWrote → {OUT_MANIFEST}")
    print(f"Wrote → {OUT_LEAKAGE}")

    # Quick sanity checks
    assert "latent_path" not in combined.columns, "latent_path must not be here (plan 05 concern)"
    assert set(combined["source"].unique()) <= {"nih", "vindr"}
    assert combined["border_visible"].all()
    assert len(combined) == len(nih_df) + len(vindr_df)
    print("Sanity checks passed ✓")


if __name__ == "__main__":
    main()
