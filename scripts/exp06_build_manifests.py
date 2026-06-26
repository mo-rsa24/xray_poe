"""Exp 6 — build train/test manifests for the detector + Oracle/Starved arms.

Centralises label derivation and the patient-leakage guard in one place, so
train_detector.py just reads CSV manifests (path, cardio, effusion, stratum).

Label policy (grounded in the Exp 1 CTR/CP instrument, not NIH text labels):
  - normal           -> [0, 0]
  - cardio_only       -> [1, 0]   (group identity is the label)
  - effusion_only     -> [0, 1]
  - both  (pool+test) -> CTR/CP-DERIVED per film:
        cardio   = 1 iff (ctr > 0.5 AND border_visible)        # CTR head
        effusion = 1 iff (cp_left_blunted OR cp_right_blunted)  # CP head
    This is what lets BOTH sigmoids fire on one image (the property softmax denies).

Leakage guard: every training film is patient-disjoint from the 317 reserved test
patients (data/splits/test_patient_ids.txt). Base groups (normal/cardio/effusion)
are patient-split into train + held-out test strata so AUC-by-stratum is well posed;
the both-separable test stratum is the frozen Exp 1 set (378 films, reserved patients).

All films resolve under data/nih/images/<name> (the full 112k corpus).

    python scripts/exp06_build_manifests.py
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data/splits"
IMAGES = ROOT / "data/nih/images"
OUT = SPLITS / "exp06_manifests"

CTR_CP = ROOT / "results/exp01_dynamics/per_film_ctr_cp.csv"
BOTH_POOL = SPLITS / "oracle_both_pool.csv"                       # 557, reserved-free
BOTH_TEST = SPLITS / "pixel_verifiable_separable_test.csv"        # 378 frozen
RESERVED = SPLITS / "test_patient_ids.txt"
ENTRY = ROOT / "data/nih/Data_Entry_2017.csv"

# base-group sizes (knobs; pos_weight handles residual imbalance during training)
N_NORMAL = 4000
N_EFFUSION = 4000        # cap the 11.7k effusion_only pool
TEST_PATIENT_FRAC = 0.12  # per base group, held out by PATIENT for the test strata
SEED = 42


def _pid(name: str) -> int:
    return int(Path(name).name.split("_")[0])


def _ctr_cp_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive [cardio, effusion] from instrument columns."""
    cardio = ((df["ctr"] > 0.5) & (df["border_visible"].astype(bool))).astype(int)
    eff = (df["cp_left_blunted"].astype(bool) | df["cp_right_blunted"].astype(bool)).astype(int)
    return pd.DataFrame({"cardio": cardio.values, "effusion": eff.values})


def _patient_split(names: list[str], frac: float, seed: int) -> tuple[set[int], set[int]]:
    """Split a list of films into (train_patients, test_patients) disjoint by patient."""
    pats = sorted({_pid(n) for n in names})
    rng = random.Random(seed)
    rng.shuffle(pats)
    n_test = max(1, int(len(pats) * frac))
    return set(pats[n_test:]), set(pats[:n_test])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reserved = {int(x) for x in RESERVED.read_text().split()}
    rng = random.Random(SEED)

    train_rows: list[dict] = []
    test_rows: list[dict] = []

    def add(rows, name, cardio, effusion, stratum):
        rows.append({"path": f"data/nih/images/{name}", "Image Index": name,
                     "patient_id": _pid(name), "cardio": int(cardio),
                     "effusion": int(effusion), "stratum": stratum})

    # ── both pool (train) — CTR/CP-derived labels ────────────────────────────
    pool = pd.read_csv(BOTH_POOL).merge(
        pd.read_csv(CTR_CP)[["Image Index", "ctr", "border_visible",
                             "cp_left_blunted", "cp_right_blunted"]],
        on="Image Index", how="left")
    pool = pool.dropna(subset=["ctr"])
    lab = _ctr_cp_labels(pool)
    pool_rows = []
    for (_, r), (_, l) in zip(pool.iterrows(), lab.iterrows()):
        assert r["Patient ID"] not in reserved, "both pool leaked a reserved patient"
        pool_rows.append({"path": r["path"], "Image Index": r["Image Index"],
                          "patient_id": int(r["Patient ID"]), "cardio": int(l["cardio"]),
                          "effusion": int(l["effusion"]), "stratum": "both"})
    pd.DataFrame(pool_rows).to_csv(OUT / "both_pool.csv", index=False)
    co = sum(1 for r in pool_rows if r["cardio"] and r["effusion"])
    print(f"both_pool: {len(pool_rows)} films "
          f"(CTR/CP co-present [1,1]: {co} = {co/len(pool_rows):.0%})")

    # ── both-separable TEST stratum (frozen 378) ─────────────────────────────
    bt = pd.read_csv(BOTH_TEST)
    lab = _ctr_cp_labels(bt)
    for (_, r), (_, l) in zip(bt.iterrows(), lab.iterrows()):
        test_rows.append({"path": r["path"], "Image Index": r["Image Index"],
                          "patient_id": int(r["Patient ID"]), "cardio": int(l["cardio"]),
                          "effusion": int(l["effusion"]), "stratum": "both_separable"})
    print(f"test both_separable: {len(bt)} films")

    # ── base groups: cardio_only, effusion_only, normal ──────────────────────
    # Gather all base films first, then split ONCE by patient GLOBALLY so a patient
    # never lands in train for one stratum and test for another (cross-stratum leak).
    cardio_names = sorted(p.name for p in (ROOT / "data/nih/images_cardio_only").glob("*.png"))
    eff_names = sorted(p.name for p in (ROOT / "data/nih/images_effusion_only").glob("*.png"))
    if len(eff_names) > N_EFFUSION:
        eff_names = rng.sample(eff_names, N_EFFUSION)
    entry = pd.read_csv(ENTRY)
    nf = entry[(entry["Finding Labels"] == "No Finding") &
               (entry["View Position"] == "PA")]
    nf = nf[~nf["Patient ID"].isin(reserved)]
    normal_names = sorted(nf["Image Index"].tolist())
    if len(normal_names) > N_NORMAL:
        normal_names = rng.sample(normal_names, N_NORMAL)

    base: list[tuple[str, int, int, str]] = (
        [(n, 1, 0, "cardio_only") for n in cardio_names] +
        [(n, 0, 1, "effusion_only") for n in eff_names] +
        [(n, 0, 0, "normal") for n in normal_names]
    )
    base = [(n, c, e, s) for (n, c, e, s) in base if _pid(n) not in reserved]

    # both-pool patients are always train-side; draw the base test split only from
    # patients NOT in the both pool (and not reserved) so both_pool ⟂ test holds.
    pool_pats = {r["patient_id"] for r in pool_rows}
    eligible = sorted({_pid(n) for (n, *_) in base} - pool_pats)
    rng_p = random.Random(SEED)
    rng_p.shuffle(eligible)
    n_test_pat = max(1, int(len(eligible) * TEST_PATIENT_FRAC))
    test_pats = set(eligible[:n_test_pat])

    counts: dict[str, list[int]] = {}
    for n, c, e, s in base:
        counts.setdefault(s, [0, 0])
        if _pid(n) in test_pats:
            add(test_rows, n, c, e, s); counts[s][1] += 1
        else:
            add(train_rows, n, c, e, s); counts[s][0] += 1
    for s, (tr, te) in counts.items():
        print(f"{s}: train {tr}, test {te} (global patient split, reserved-dropped)")

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    train_df.to_csv(OUT / "train_base.csv", index=False)
    test_df.to_csv(OUT / "test.csv", index=False)

    # leakage re-assertion across the whole thing (fail fast)
    shared = set(train_df["patient_id"]) & set(test_df["patient_id"])
    assert not shared, f"train/test share patients: {sorted(shared)[:5]}"
    assert not (set(train_df["patient_id"]) & reserved), "train pool hit a reserved patient"
    assert not (set(pd.DataFrame(pool_rows)["patient_id"]) & set(test_df["patient_id"])), \
        "both pool shares a patient with test"

    print(f"\ntrain_base: {len(train_df)} films  |  test: {len(test_df)} films")
    print("test strata:", test_df["stratum"].value_counts().to_dict())
    print(f"manifests -> {OUT.relative_to(ROOT)}/  (train_base.csv, both_pool.csv, test.csv)")
    print("leakage assertions PASSED (train ⟂ test ⟂ reserved)")


if __name__ == "__main__":
    main()
