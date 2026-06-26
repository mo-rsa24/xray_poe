"""Table 1 / Table 2 — pretrained-classifier accuracy/sens/spec, real vs synth (D4, D7, D9).

Eval-only (no training). Runs the FROZEN DenseNet presence classifier-under-test over
real and synthetic image sets, and reports — per disease head (cardio, effusion) — the
accuracy, sensitivity and specificity at the saved Youden thresholds, with bootstrap
95% CIs. Also reports the per-category positive rate (the §7.6 comorbidity breakdown:
does the cardio head fire as often on comorbid films as on cardio-only?).

Sources (each = a list of (path, cardio_label, effusion_label, category)):
  real        — the Exp 6 test manifest (data/splits/exp06_manifests/test.csv), real films
  synth/<tag> — results/synth_sets/<tag>/{cardio,effusion,comorbid}/ from gen_synth_sets.py
                (cardio→[1,0], effusion→[0,1], comorbid→[1,1])

Reuses metrics.presence_classifier (load_model, predict_probs) — the same frozen judge
used by Exp 9. This module is the D9 computation half (sens/spec everywhere).

    python -m metrics.classifier_report --synth-tags before after \\
        --out results/tables/table1_classifier_report.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metrics.presence_classifier import load_model, predict_probs  # noqa: E402

MANIFEST = ROOT / "data/splits/exp06_manifests/test.csv"
SYNTH = ROOT / "results/synth_sets"
STRATUM2CAT = {"normal": "normal", "cardio_only": "cardio",
               "effusion_only": "effusion", "both_separable": "comorbid"}
SYNTH_LABELS = {"cardio": (1, 0), "effusion": (0, 1), "comorbid": (1, 1)}
CATS = ["normal", "cardio", "effusion", "comorbid"]


def _boot_ci(mask_pos, pred, n_boot=1000, seed=0):
    """Bootstrap 95% CI for a rate over the items selected by mask_pos."""
    idx = np.where(mask_pos)[0]
    if len(idx) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = [pred[rng.choice(idx, len(idx), replace=True)].mean() for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _metrics(df, probs, head, thr):
    """sens/spec/acc + per-category positive rate for one head."""
    lab = df[head].values
    pred = (probs >= thr).astype(int)
    pos, neg = lab == 1, lab == 0
    sens = pred[pos].mean() if pos.any() else float("nan")
    spec = (1 - pred[neg]).mean() if neg.any() else float("nan")
    acc = (pred == lab).mean()
    sens_ci = _boot_ci(pos, pred); spec_ci = _boot_ci(neg, 1 - pred)
    percat = {c: float(pred[df["category"].values == c].mean())
              if (df["category"].values == c).any() else float("nan") for c in CATS}
    return {"sens": sens, "sens_ci": sens_ci, "spec": spec, "spec_ci": spec_ci,
            "acc": float(acc), "percat_pos_rate": percat}


def _load_real(cap: int = 300) -> pd.DataFrame:
    """LEAKAGE-CLEAN real eval set (the frozen classifier must never have trained on it):
      cardio / effusion = the classifier's HELD-OUT val split (seed 42, val_frac 0.2);
      comorbid          = the both-disease test (both-disease held out of classifier training);
      normal            = fresh No-Finding PA sample, reserved-patient-free (best-effort —
                          the classifier's original images_normal/ dir is gone on this clone).
    The old version drew single-disease films from the Exp-6 manifest, which overlapped the
    classifier's TRAINING set (effusion 403/403, cardio 91/127) — inflating those columns.
    """
    import random as _r
    from metrics.presence_classifier import _val_split_paths
    rows = []
    for grp, c, e, cat in [("images_cardio_only", 1, 0, "cardio"),
                           ("images_effusion_only", 0, 1, "effusion")]:
        for p in _val_split_paths(ROOT / "data/nih" / grp, val_frac=0.2, seed=42, n=cap):
            rows.append({"path": str(p), "cardio": c, "effusion": e, "category": cat})
    mb = pd.read_csv(MANIFEST)
    mb = mb[mb["stratum"] == "both_separable"]
    for _, r in mb.iterrows():
        rows.append({"path": str(ROOT / r["path"]), "cardio": int(r["cardio"]),
                     "effusion": int(r["effusion"]), "category": "comorbid"})
    reserved = {int(x) for x in (ROOT / "data/splits/test_patient_ids.txt").read_text().split()}
    entry = pd.read_csv(ROOT / "data/nih/Data_Entry_2017.csv")
    nf = entry[(entry["Finding Labels"] == "No Finding") & (entry["View Position"] == "PA")]
    nf = nf[~nf["Patient ID"].isin(reserved)]
    names = sorted(nf["Image Index"]); _r.Random(7).shuffle(names)
    for n in names[:cap]:
        rows.append({"path": str(ROOT / "data/nih/images" / n), "cardio": 0, "effusion": 0,
                     "category": "normal"})
    return pd.DataFrame(rows)


def _load_synth(tag: str) -> pd.DataFrame:
    rows = []
    for cond, (c, e) in SYNTH_LABELS.items():
        d = SYNTH / tag / cond
        for p in sorted(d.glob("*.png")):
            rows.append({"path": str(p), "cardio": c, "effusion": e, "category": cond})
    if not rows:
        raise RuntimeError(f"no synth images under {SYNTH / tag} — run gen_synth_sets.py --tag {tag}")
    return pd.DataFrame(rows)


def _run_source(name, df, model, ci, ei, thr_c, thr_e, device):
    probs = predict_probs(model, df["path"].tolist(), ci, ei, device)
    df = df.reset_index(drop=True)
    return {"source": name, "n": len(df),
            "cardio": _metrics(df, probs[:, 0], "cardio", thr_c),
            "effusion": _metrics(df, probs[:, 1], "effusion", thr_e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-tags", nargs="*", default=["before", "after"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/tables/table1_classifier_report.md")
    args = ap.parse_args()
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, ci, ei, ckpt = load_model(device)
    thr_c, thr_e = ckpt["youden_threshold_cardio"], ckpt["youden_threshold_effusion"]
    print(f"classifier thresholds: cardio>{thr_c:.3f} effusion>{thr_e:.3f}")

    results = [_run_source("real", _load_real(), model, ci, ei, thr_c, thr_e, device)]
    for tag in args.synth_tags:
        try:
            results.append(_run_source(f"synth-{tag}", _load_synth(tag), model, ci, ei, thr_c, thr_e, device))
        except RuntimeError as e:
            print(f"[skip] {e}")

    # ── table ────────────────────────────────────────────────────────────────
    L = ["# Table 1 — pretrained classifier: accuracy / sensitivity / specificity (D4, D9)",
         "",
         f"Frozen DenseNet judge, Youden thresholds cardio>{thr_c:.3f} effusion>{thr_e:.3f}. "
         "Both-disease held out of training (struggles on comorbidity by design). "
         "95% bootstrap CIs.", "",
         "**Leakage-clean (D9):** real single-disease = the classifier's held-out val split "
         "(never trained on); comorbid = held-out both-disease test; synth = generated. No real "
         "eval film was in the classifier's training set.", "",
         "## Per-head sens / spec / acc", "",
         "| source | n | head | sensitivity | specificity | accuracy |",
         "|---|---|---|---|---|---|"]
    for r in results:
        for h in ("cardio", "effusion"):
            m = r[h]
            L.append(f"| {r['source']} | {r['n']} | {h} | "
                     f"{m['sens']:.3f} [{m['sens_ci'][0]:.2f},{m['sens_ci'][1]:.2f}] | "
                     f"{m['spec']:.3f} [{m['spec_ci'][0]:.2f},{m['spec_ci'][1]:.2f}] | {m['acc']:.3f} |")
    L += ["", "## §7.6 comorbidity breakdown — head positive-rate by category", "",
          "| source | head | normal | cardio | effusion | comorbid |",
          "|---|---|---|---|---|---|"]
    for r in results:
        for h in ("cardio", "effusion"):
            pc = r[h]["percat_pos_rate"]
            L.append(f"| {r['source']} | {h} | " +
                     " | ".join(f"{pc[c]:.3f}" if pc[c] == pc[c] else "—" for c in CATS) + " |")
    L += ["", "_Read: if the cardio head's positive-rate drops from the `cardio` column to the "
          "`comorbid` column, the classifier under-detects cardiomegaly when effusion co-occurs "
          "(the comorbidity weakness §7.4/7.6 targets)._"]

    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"-> {out.relative_to(ROOT)}")
    # also dump machine-readable
    flat = []
    for r in results:
        for h in ("cardio", "effusion"):
            m = r[h]
            flat.append({"source": r["source"], "head": h, "n": r["n"],
                         "sens": m["sens"], "spec": m["spec"], "acc": m["acc"],
                         **{f"posrate_{c}": m["percat_pos_rate"][c] for c in CATS}})
    pd.DataFrame(flat).to_csv(out.with_suffix(".csv"), index=False)
    print(f"-> {out.with_suffix('.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
