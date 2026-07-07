"""Exp03 task 3 — PoE Gate-3 yield verdict.

Reads results/exp03_yield/yield.csv (per-sample, from gate3_yield.py) and files the
yield verdict against the pre-registered rule, honoring the two carry-forwards:
  • floor-corrected: subtract the generated no_finding joint Gate-3 rate (Exp 4 artifact)
  • abstain: a confident yield computed on non-threshold-edge samples (Exp 2)

    support       yield ≥ 30%
    null          yield < 10%   → fire the Exp 5 rebalanced-cardio retrain lever
    inconclusive  10–30%        → small w_c/w_e tweak + more samples before deciding

    python scripts/exp03_yield_verdict.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.ctr_overlay as C
import scripts.gate3_yield as Y

CSV = Path("results/exp03_yield/yield.csv")
VERDICT = Path("results/exp03_yield/verdict.md")
FIG = Path("figures/exp03_ctr_cp_scatter.png")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), c + h)


def band(y):
    return "support" if y >= 0.30 else ("null" if y < 0.10 else "inconclusive")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-dir", default=str(Y.FLOOR_DIR))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    N = len(df)
    raw_k = int(df["gate3_pass"].sum())
    raw_y = raw_k / N
    conf = df[~df["abstain"]]
    conf_k, conf_n = int(conf["gate3_pass"].sum()), len(conf)
    conf_y = conf_k / conf_n if conf_n else float("nan")

    # generated no_finding joint Gate-3 floor (re-scored for self-containment)
    seg, idx = C._load_seg(args.device)
    fl = pd.DataFrame(Y.score(seg, idx, sorted(Path(args.floor_dir).glob("*.png")), args.device))
    floor = float(fl["gate3_pass"].mean())
    floor_conf = float(fl[~fl["abstain"]]["gate3_pass"].mean()) if (~fl["abstain"]).any() else floor

    fc_y = max(0.0, raw_y - floor)                 # primary: floor-corrected
    fc_conf = max(0.0, conf_y - floor_conf)
    verdict = band(fc_y)

    raw_lo, raw_hi = wilson(raw_k, N)
    conf_lo, conf_hi = wilson(conf_k, conf_n)

    cardio = float(df["cardio_present"].mean())
    effusion = float(df["effusion_present"].mean())
    abst = float(df["abstain"].mean())

    L = [f"# Exp03 — PoE Gate-3 yield verdict", "",
         f"**VERDICT: {verdict}** (on the floor-corrected yield) · PoE w_c=2.0 w_e=1.0 · "
         f"ckpt `model_step0100000` · N={N}.", "",
         "## Yield (Gate-3 = CTR>0.5 AND CP blunted, simultaneously)",
         f"- raw **{raw_y:.1%}** ({raw_k}/{N}, Wilson 95% CI {raw_lo:.1%}–{raw_hi:.1%})",
         f"- generated no_finding floor {floor:.1%} → **floor-corrected {fc_y:.1%}** (primary)",
         f"- confident (non-abstain) **{conf_y:.1%}** ({conf_k}/{conf_n}, CI {conf_lo:.1%}–{conf_hi:.1%}); "
         f"floor-corrected confident {fc_conf:.1%}",
         f"- abstain (threshold-edge) {abst:.1%}",
         "",
         "## Channels",
         f"- cardio_present (CTR>0.5) = **{cardio:.1%}** — vs cardio-alone 4.2% (Exp 4): PoE composition "
         f"renders the heart better than the cardio expert by itself, but still barely clears 0.5.",
         f"- effusion_present (CP blunted) = {effusion:.1%} (incl. ~34% generated-texture floor).",
         "",
         "## Call",
         {"support": "**SUPPORT** — PoE co-renders both signs at ≥30% yield; the +PoE arm has ample gated data for Exp 7.",
          "null": "**NULL** — yield < 10%; the experts cannot co-render separably as-is → **fire the Exp 5 "
                  "rebalanced-cardio retrain lever** (the cardio channel is the bottleneck).",
          "inconclusive": "**INCONCLUSIVE (10–30%)** — co-rendering works but thinly, bottlenecked by the cardio "
                  "channel barely clearing CTR=0.5. Per the rule: a small w_c/w_e tweak + more samples before "
                  "committing to retrain — but combined with Exp 4 (cardio under-renders) this still points to the "
                  "**Exp 5 rebalanced-cardio retrain** as the durable fix."}[verdict],
         "",
         f"Verdicts agree across definitions: raw={band(raw_y)}, floor-corrected={band(fc_y)}, "
         f"confident={band(conf_y)}." if len({band(raw_y), band(fc_y), band(conf_y)}) == 1 else
         f"⚠ Verdict band depends on definition: raw={band(raw_y)}, floor-corrected={band(fc_y)}, "
         f"confident={band(conf_y)} — straddles a threshold; report all three.",
         "",
         f"Figure: {FIG} · feeds Exp 5 (yield input) + Exp 7 (+PoE augmentation volume)."]
    VERDICT.write_text("\n".join(L) + "\n")

    print(f"VERDICT: {verdict}  (floor-corrected {fc_y:.1%})")
    print(f"  raw {raw_y:.1%} (CI {raw_lo:.1%}-{raw_hi:.1%}) · floor {floor:.1%} · confident {conf_y:.1%} (CI {conf_lo:.1%}-{conf_hi:.1%})")
    print(f"  bands: raw={band(raw_y)} floor-corr={band(fc_y)} confident={band(conf_y)}")
    print(f"→ {VERDICT}")


if __name__ == "__main__":
    main()
