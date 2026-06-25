"""Auto-classify the both-disease films into separable vs entangled, and emit
failure-mode contact sheets for human eyeballing.

separable  = cardiac border traceable (CTR measurable) AND at least one
             costophrenic recess readable (blunted or sharp — we just need to
             SEE it). This is the pixel-verifiable test population.
entangled  = something hides a sign. Reason is recorded per film:
   whiteout_obscures_border  — peri-cardiac opacity high, silhouette sign,
                               "lots of whiteness around the heart" (Gate-0)
   heart_seg_low_conf        — PSPNet not confident about the heart
   seg_failed                — no heart or no lung mask at all
   no_cp_recess_readable     — neither costophrenic angle gives a usable vertex

Writes results/exp01_dynamics/per_film_ctr_cp.csv and contact-sheet PNGs under
results/exp01_dynamics/failure_sheets/ so the worst cases can be reviewed by eye
instead of hand-labelling from scratch.

    python scripts/classify_separable.py                 # all 1063
    python scripts/classify_separable.py --limit 80      # quick smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
import scripts.ctr_overlay as C

LIST = Path("results/exp01_dynamics/both_film_list.csv")
OUT_CSV = Path("results/exp01_dynamics/per_film_ctr_cp.csv")
SHEETS = Path("results/exp01_dynamics/failure_sheets")


def classify(m: dict) -> tuple[str, str]:
    cp_l = m["cp_left"]["blunted"] is not None
    cp_r = m["cp_right"]["blunted"] is not None
    if m["border_visible"] and (cp_l or cp_r):
        return "separable", ""
    if not m["border_visible"]:
        if not m["ok"]:
            return "entangled", "seg_failed"
        op = m.get("peri_cardiac_opacity")
        if op is not None and np.isfinite(op) and op >= C.TAU_OPACITY:
            return "entangled", "whiteout_obscures_border"
        return "entangled", "heart_seg_low_conf"
    return "entangled", "no_cp_recess_readable"


def _sheet(rows: pd.DataFrame, seg, idx, dev: str, title: str, n: int = 16) -> None:
    if not len(rows):
        print(f"  (no films for sheet '{title}')")
        return
    sub = rows.head(n)
    cells = []
    for _, r in sub.iterrows():
        m = C.ctr_for_image(seg, idx, Path(r["path"]), dev)
        cells.append(C._overlay(Path(r["path"]), m))
    SHEETS.mkdir(parents=True, exist_ok=True)
    out = SHEETS / f"{title}.png"
    C._grid(cells, cols=4, cell=512).save(out)
    print(f"  sheet → {out}  ({len(sub)} films)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="cap films (smoke test)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    films = pd.read_csv(LIST)
    if args.limit:
        films = films.head(args.limit)
    seg, idx = C._load_seg(args.device)
    print(f"PSPNet on {args.device}; classifying {len(films)} both-disease films")

    recs = []
    for i, r in films.iterrows():
        m = C.ctr_for_image(seg, idx, Path(r["path"]), args.device)
        klass, reason = classify(m)
        cl, cr = m["cp_left"], m["cp_right"]
        recs.append({
            "Image Index": r["Image Index"], "Patient ID": r["Patient ID"],
            "View Position": r["View Position"], "path": r["path"],
            "ctr": round(m["ctr"], 3) if np.isfinite(m["ctr"]) else np.nan,
            "heart_conf": m.get("heart_conf"),
            "peri_cardiac_opacity": m.get("peri_cardiac_opacity"),
            "border_visible": m["border_visible"],
            "cp_left_angle": cl["angle_deg"], "cp_left_blunted": cl["blunted"],
            "cp_right_angle": cr["angle_deg"], "cp_right_blunted": cr["blunted"],
            "class": klass, "reason": reason,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(films)}")
    df = pd.DataFrame(recs)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # ── headline counts ──────────────────────────────────────────────────────
    def report(d, tag):
        n = len(d); sep = int((d["class"] == "separable").sum())
        print(f"{tag:10s} N={n:4d}  separable={sep:4d} ({sep / n:.1%})  entangled={n - sep}")
    print("\n=== auto-classification (PROVISIONAL thresholds) ===")
    report(df, "ALL")
    report(df[df["View Position"] == "PA"], "PA")
    report(df[df["View Position"] == "AP"], "AP")
    print("\nentangled reasons:", df[df["class"] == "entangled"]["reason"].value_counts().to_dict())
    print(f"\n→ {OUT_CSV}")

    # ── failure-mode contact sheets (worst-first) ────────────────────────────
    print("\n=== failure-mode contact sheets ===")
    ent = df[df["class"] == "entangled"]
    whiteout = ent[ent["reason"] == "whiteout_obscures_border"].sort_values(
        "peri_cardiac_opacity", ascending=False)
    cp_fail = ent[ent["reason"] == "no_cp_recess_readable"]
    lowconf = ent[ent["reason"].isin(["heart_seg_low_conf", "seg_failed"])]
    borderline = df[(df["ctr"] >= 0.48) & (df["ctr"] <= 0.52)]
    separable = df[df["class"] == "separable"].sample(min(16, int((df["class"] == "separable").sum())),
                                                      random_state=42) if (df["class"] == "separable").any() else df.head(0)
    _sheet(whiteout, seg, idx, args.device, "entangled_whiteout_around_heart")
    _sheet(cp_fail, seg, idx, args.device, "entangled_cp_unreadable")
    _sheet(lowconf, seg, idx, args.device, "entangled_low_conf")
    _sheet(borderline, seg, idx, args.device, "borderline_ctr_0p48_0p52")
    _sheet(separable, seg, idx, args.device, "separable_sample")

    Path(SHEETS / "summary.json").write_text(json.dumps({
        "n_total": len(df),
        "by_view": {v: int((df["View Position"] == v).sum()) for v in ["PA", "AP"]},
        "separable_all": int((df["class"] == "separable").sum()),
        "separable_PA": int(((df["View Position"] == "PA") & (df["class"] == "separable")).sum()),
        "entangled_reasons": df[df["class"] == "entangled"]["reason"].value_counts().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
