"""Table 2 / Section 4 — does augmenting real with synthetic help comorbidity? (D7).

Compares the augmentation arms (+Individual, +PoE) against the Starved baseline on the
both-separable (comorbid) stratum: per-head AUC + sens/spec, and ΔAUC(arm − starved)
with film-level bootstrap 95% CIs. Reads the per-film prediction dumps written by
train_detector.py (results/exp06_oracle/preds/{starved50,aug_indiv,aug_poe}_s*.csv),
ensembling probabilities across seeds per arm.

Decision (pre-registered, EXP-07): +PoE helps iff ΔAUC(PoE − starved) ≥ 0.05 on
comorbid with a CI clear of 0; otherwise a (likely) clean null — reported, per §6.

    python scripts/section4_table2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "results/exp06_oracle/preds"
OUT = ROOT / "results/tables/table2_augmentation.md"
STRATUM = "both_separable"   # the comorbid test stratum
ARMS = {"starved50": "Starved", "aug_indiv": "+Individual", "aug_poe": "+PoE"}
HEADS = ["cardio", "effusion"]
B = 2000


def _load(prefix):
    fs = sorted(PREDS.glob(f"{prefix}_s*.csv"))
    if not fs:
        return None
    ds = [pd.read_csv(f) for f in fs]
    base = ds[0][["Image Index", "stratum", "cardio", "effusion"]].copy()
    base["cardio_prob"] = np.mean([d["cardio_prob"].values for d in ds], axis=0)
    base["effusion_prob"] = np.mean([d["effusion_prob"].values for d in ds], axis=0)
    base["n_seeds"] = len(ds)
    return base


def _auc(df, head, stratum=STRATUM):
    s = df if stratum is None else df[df.stratum == stratum]
    return roc_auc_score(s[head], s[f"{head}_prob"]) if s[head].nunique() == 2 else float("nan")


def _delta_overall(arm, base, head):
    """ΔAUC across ALL strata (general improvement) — contrast for point 2."""
    m = arm.merge(base, on="Image Index", suffixes=("_a", "_b"))
    y = m[f"{head}_a"].values
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, m[f"{head}_prob_a"].values) - roc_auc_score(y, m[f"{head}_prob_b"].values)


def _sens_spec(df, head):
    s = df[df.stratum == STRATUM]
    fpr, tpr, thr = roc_curve(s[head], s[f"{head}_prob"]); t = thr[np.argmax(tpr - fpr)]
    pred = (s[f"{head}_prob"] >= t).astype(int); lab = s[head].values
    pos, neg = lab == 1, lab == 0
    return (pred[pos].mean() if pos.any() else float("nan"),
            (1 - pred[neg]).mean() if neg.any() else float("nan"))


def _boot_delta(arm, base, head):
    a = arm[arm.stratum == STRATUM]; b = base[base.stratum == STRATUM]
    m = a.merge(b, on="Image Index", suffixes=("_a", "_b"))
    y = m[f"{head}_a"].values
    if len(np.unique(y)) < 2:
        return float("nan"), (float("nan"), float("nan"))
    pa, pb = m[f"{head}_prob_a"].values, m[f"{head}_prob_b"].values
    point = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    rng = np.random.default_rng(0); d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    lo, hi = np.percentile(d, [2.5, 97.5]) if d else (float("nan"), float("nan"))
    return point, (float(lo), float(hi))


def main():
    arms = {k: _load(k) for k in ARMS}
    have = {k: v for k, v in arms.items() if v is not None}
    if "starved50" not in have:
        raise SystemExit("need starved50 preds (the baseline) — run Exp 6 / train the starved arm")
    base = have["starved50"]
    print("arms present:", {ARMS[k]: f"{v.n_seeds.iloc[0]} seeds" for k, v in have.items()})

    L = ["# Table 2 — augmentation vs Starved on the comorbid stratum (Section 4, D7)", "",
         "Frozen real test set, both-separable (comorbid) stratum. Probabilities ensembled "
         "across seeds. ΔAUC = arm − Starved, film-level bootstrap 95% CI.", "",
         "Δ(overall) = general improvement across all strata; Δ(comorbid) = the claim. "
         "A positive Δ(overall) with a flat/negative Δ(comorbid) is still a NULL for our "
         "claim (supervisor point 2 — comorbidity-specific, not general, is the criterion).", "",
         "| arm | seeds | head | comorbid AUC | sens | spec | Δ(overall) | **Δ(comorbid)** | 95% CI | helps comorbid? |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    verdict_lines = []
    for k, df in have.items():
        for h in HEADS:
            auc = _auc(df, h); sens, spec = _sens_spec(df, h)
            if k == "starved50":
                L.append(f"| {ARMS[k]} | {df.n_seeds.iloc[0]} | {h} | {auc:.3f} | {sens:.3f} | {spec:.3f} | — | — | — | (baseline) |")
            else:
                d, (lo, hi) = _boot_delta(df, base, h)
                d_all = _delta_overall(df, base, h)
                helps = "YES" if (d >= 0.05 and lo > 0) else ("~" if d > 0 else "no")
                L.append(f"| {ARMS[k]} | {df.n_seeds.iloc[0]} | {h} | {auc:.3f} | {sens:.3f} | {spec:.3f} | "
                         f"{d_all:+.3f} | **{d:+.3f}** | [{lo:+.3f},{hi:+.3f}] | {helps} |")
                if h == "cardio":
                    verdict_lines.append(f"{ARMS[k]} cardio: Δ(overall)={d_all:+.3f}, Δ(comorbid)={d:+.3f} [{lo:+.3f},{hi:+.3f}] → {helps}")
    L += ["", "## Verdict", ""]
    L += [f"- {v}" for v in verdict_lines] or ["- (augmentation arms not yet trained)"]
    L += ["", "_Interpretation: a positive ΔAUC with a CI clear of 0 means the synthetic "
          "augmentation improved comorbid detection. Given synth cardio is under-rendered "
          "(Table 1) and the test set is saturated (Exp 6 null), a flat/negative ΔAUC is the "
          "expected, pre-registered null (§6) — augmentation in latent space did not translate "
          "to disease space._"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")
    for v in verdict_lines:
        print("  " + v)


if __name__ == "__main__":
    main()
