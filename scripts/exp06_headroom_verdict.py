"""Exp 6 — oracle - starved headroom verdict (Gate 6, DoD 6, D10).

Reads the per-film prediction dumps written by train_detector.py
(results/exp06_oracle/preds/{oracle,starved50}_s*.csv), ensembles probabilities
across seeds per arm, and reports the oracle - starved ΔAUC on the both-separable
stratum with film-level bootstrap CIs, plus co-presence sensitivity/specificity (D9).

Decision rule (pre-registered, EXP-06 falsify_condition):
    support (headroom exists)  if ΔAUC ≥ 0.05 on both-separable
    null    (test saturated)   if ΔAUC < 0.02   -> Exp 7 uninterpretable, revisit test set
    inconclusive               otherwise        -> more seeds

Writes results/exp06_oracle/verdict.md + figures/exp06_oracle_vs_starved_auc.png.

    python scripts/exp06_headroom_verdict.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "results/exp06_oracle/preds"
VERDICT = ROOT / "results/exp06_oracle/verdict.md"
FIG = ROOT / "figures/exp06_oracle_vs_starved_auc.png"
STRATA = ["normal", "cardio_only", "effusion_only", "both_separable"]
HEADLINE_STRATUM = "both_separable"
B = 2000  # bootstrap resamples


def _load_arm(prefix: str) -> pd.DataFrame | None:
    files = sorted(PREDS.glob(f"{prefix}_s*.csv"))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    base = dfs[0][["Image Index", "stratum", "cardio", "effusion"]].copy()
    base["cardio_prob"] = np.mean([d["cardio_prob"].values for d in dfs], axis=0)
    base["effusion_prob"] = np.mean([d["effusion_prob"].values for d in dfs], axis=0)
    base["n_seeds"] = len(dfs)
    return base


def _auc(df, head, stratum=None):
    sub = df if stratum is None else df[df.stratum == stratum]
    if sub[head].nunique() < 2:
        return float("nan")
    return roc_auc_score(sub[head], sub[f"{head}_prob"])


def _youden(df, head):
    fpr, tpr, thr = roc_curve(df[head], df[f"{head}_prob"])
    return float(thr[np.argmax(tpr - fpr)])


def _sens_spec(df, head, thr, stratum):
    sub = df[df.stratum == stratum]
    pred = (sub[f"{head}_prob"] >= thr).astype(int)
    tp = int(((pred == 1) & (sub[head] == 1)).sum()); fn = int(((pred == 0) & (sub[head] == 1)).sum())
    tn = int(((pred == 0) & (sub[head] == 0)).sum()); fp = int(((pred == 1) & (sub[head] == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return sens, spec


def _boot_delta(oracle, starved, head, stratum, seed=0):
    o = oracle[oracle.stratum == stratum].reset_index(drop=True)
    s = starved[starved.stratum == stratum].reset_index(drop=True)
    # both arms cover the same test films (same manifest) -> align on Image Index
    m = o.merge(s, on="Image Index", suffixes=("_o", "_s"))
    y = m[f"{head}_o"].values
    if len(np.unique(y)) < 2:
        return float("nan"), (float("nan"), float("nan"))
    po, ps = m[f"{head}_prob_o"].values, m[f"{head}_prob_s"].values
    point = roc_auc_score(y, po) - roc_auc_score(y, ps)
    rng = np.random.default_rng(seed); deltas = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], po[idx]) - roc_auc_score(y[idx], ps[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5]) if deltas else (float("nan"), float("nan"))
    return point, (float(lo), float(hi))


def main() -> None:
    oracle = _load_arm("oracle")
    starved = _load_arm("starved50")
    if oracle is None or starved is None:
        raise SystemExit(
            "Missing prediction dumps. Train the arms first via scripts/launch_exp06.sh "
            "(expects results/exp06_oracle/preds/{oracle,starved50}_s*.csv)."
        )
    print(f"oracle: {oracle.n_seeds.iloc[0]} seeds   starved50: {starved.n_seeds.iloc[0]} seeds")

    # headline: ΔAUC on both_separable, per head + mean, with bootstrap CIs
    heads = ["cardio", "effusion"]
    deltas = {}
    for h in heads:
        d, (lo, hi) = _boot_delta(oracle, starved, h, HEADLINE_STRATUM)
        deltas[h] = (d, lo, hi)
    valid = [deltas[h][0] for h in heads if deltas[h][0] == deltas[h][0]]
    mean_delta = float(np.mean(valid)) if valid else float("nan")

    rule_val = mean_delta
    if rule_val != rule_val:
        verdict = "inconclusive"
    elif rule_val >= 0.05:
        verdict = "support"
    elif rule_val < 0.02:
        verdict = "null"
    else:
        verdict = "inconclusive"

    # per-stratum AUC for both arms (figure + table)
    rows = []
    for s in STRATA:
        for h in heads:
            rows.append({"stratum": s, "head": h,
                         "oracle_auc": _auc(oracle, h, s), "starved_auc": _auc(starved, h, s)})
    auc_tbl = pd.DataFrame(rows)

    # sens/spec on both_separable at each arm's Youden threshold (D9)
    ss = {}
    for arm, df in [("oracle", oracle), ("starved", starved)]:
        for h in heads:
            thr = _youden(df, h)
            sens, spec = _sens_spec(df, h, thr, HEADLINE_STRATUM)
            ss[(arm, h)] = (sens, spec, thr)

    # ── figure ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, h in zip(axes, heads):
            sub = auc_tbl[auc_tbl["head"] == h]
            x = np.arange(len(STRATA)); w = 0.38
            ax.bar(x - w/2, [sub[sub.stratum==s].oracle_auc.values[0] for s in STRATA], w, label="oracle")
            ax.bar(x + w/2, [sub[sub.stratum==s].starved_auc.values[0] for s in STRATA], w, label="starved")
            ax.set_xticks(x); ax.set_xticklabels(STRATA, rotation=30, ha="right", fontsize=8)
            ax.axhline(0.5, ls=":", c="grey", lw=0.8); ax.set_title(f"{h} head AUC")
            ax.set_ylim(0, 1.0)
        axes[0].set_ylabel("AUC"); axes[0].legend(fontsize=8)
        fig.suptitle(f"Exp 6 — oracle vs starved AUC by stratum  "
                     f"(both-separable ΔAUC={mean_delta:.3f}, {verdict})", fontsize=10)
        fig.tight_layout()
        FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG, dpi=130); plt.close(fig)
        print(f"figure -> {FIG.relative_to(ROOT)}")
    except Exception as e:
        print(f"[warn] figure skipped: {e}")

    # ── verdict.md ─────────────────────────────────────────────────────────────
    L = [
        "# Exp 6 — Oracle headroom verdict (Gate 6, DoD 6)", "",
        f"**VERDICT: {verdict}** — both-separable oracle−starved ΔAUC (mean of heads) = "
        f"**{mean_delta:.3f}**.", "",
        "Pre-registered rule: support ≥ 0.05 · inconclusive 0.02–0.05 · null < 0.02.", "",
        "## ΔAUC on the both-separable stratum (the headline)", "",
        "| head | oracle AUC | starved AUC | ΔAUC | 95% bootstrap CI |",
        "|---|---|---|---|---|",
    ]
    for h in heads:
        d, lo, hi = deltas[h]
        o = _auc(oracle, h, HEADLINE_STRATUM); s = _auc(starved, h, HEADLINE_STRATUM)
        L.append(f"| {h} | {o:.3f} | {s:.3f} | {d:.3f} | [{lo:.3f}, {hi:.3f}] |")
    L += ["", "## Sensitivity / specificity on both-separable (D9, at each arm's Youden threshold)", "",
          "| arm | head | sensitivity | specificity | threshold |",
          "|---|---|---|---|---|"]
    for arm in ("oracle", "starved"):
        for h in heads:
            sens, spec, thr = ss[(arm, h)]
            L.append(f"| {arm} | {h} | {sens:.3f} | {spec:.3f} | {thr:.3f} |")
    L += [
        "", "## Role — the null-interpretability guard (D10)", "",
        "The oracle is **not** a utility competitor; it is the headroom-guard that makes an "
        "Exp 7 null *interpretable*. If real headroom exists here (support), a PoE-≤-overlay "
        "result in Exp 7 is a **clean, publishable null** rather than an artifact of a dead "
        "test set. If the test set is saturated (null verdict here), an Exp 7 null is "
        "uninterpretable and the test set / stratum must be revisited before Exp 7 is read.", "",
        "**Headroom bound (known-unknown 3).** The both-disease pool is capped by NIH "
        "(~1063 both-labelled films; 506 fall to the 317 reserved test patients, leaving an "
        f"abundant oracle pool of ~555). Within the 378-film both-separable test stratum the "
        "instrument labels only ~36 films as co-present [1,1] (cardio 286+/effusion 40+), so "
        "the effusion-head AUC rests on few positives — thin headroom weakens the null's "
        "interpretability and is reported honestly, not hidden.", "",
        f"Inputs: results/exp06_oracle/preds/ (oracle {oracle.n_seeds.iloc[0]} seeds, "
        f"starved {starved.n_seeds.iloc[0]} seeds). Figure: {FIG.name}.",
    ]
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    VERDICT.write_text("\n".join(L) + "\n")
    print(f"\noracle−starved ΔAUC(both-separable) = {mean_delta:.3f}, verdict={verdict}")
    print(f"verdict -> {VERDICT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
