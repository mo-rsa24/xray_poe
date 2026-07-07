"""Leaf 07 — bake-off panel: pick the cleanliness scorer (A vs B) on 5 films.

5 films = 2 hand-anchors (one definitely-clean, one definitely-hazy) + 3 max-disagreement
(keep_A != keep_B at a matched keep-rate). Renders each film's ctr_overlay sanity image with
A/B scores + both keep/drop verdicts, and auto-recommends the scorer whose verdicts match the
2 anchors (you confirm or override — it never auto-applies the filter).

    python scripts/basal_clean_bakeoff.py --anchor-clean <id> --anchor-hazy <id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/clean_cardio/scores.csv")
    ap.add_argument("--anchor-clean", default=None, help="image_id of a definitely-clean film")
    ap.add_argument("--anchor-hazy", default=None, help="image_id of a definitely-hazy film")
    ap.add_argument("--out-fig", default="figures/clean_cardio_bakeoff.png")
    ap.add_argument("--out-md", default="results/clean_cardio/SCORER.md")
    a = ap.parse_args()

    df = pd.read_csv(a.scores)
    df = df.dropna(subset=["B_ratio"]).reset_index(drop=True)

    # keep_B at a keep-rate matched to keep_A (compare WHICH films, not the rate):
    k = int(round(df["keep_A"].mean() * len(df)))
    thr_B = df["B_ratio"].sort_values(ascending=False).iloc[max(0, k - 1)] if k else df["B_ratio"].max() + 1
    df["keep_B"] = df["B_ratio"] >= thr_B

    # auto-suggest anchors if not given: highest/lowest by BOTH scores
    df["rankA"] = df["A_clean"].rank(pct=True)
    df["rankB"] = df["B_ratio"].rank(pct=True)
    df["both"] = (df["rankA"] + df["rankB"]) / 2
    clean_id = a.anchor_clean or df.loc[df["both"].idxmax(), "image_id"]
    hazy_id = a.anchor_hazy or df.loc[df["both"].idxmin(), "image_id"]
    auto = a.anchor_clean is None or a.anchor_hazy is None

    # 3 max-disagreement films (keep_A != keep_B), spanning both directions
    dis = df[df["keep_A"] != df["keep_B"]].copy()
    dis["gap"] = (df["rankA"] - df["rankB"]).abs()
    picks = list(dis.sort_values("gap", ascending=False)["image_id"].head(3))
    chosen = [clean_id, hazy_id] + picks
    sel = df[df["image_id"].isin(chosen)].drop_duplicates("image_id")

    # auto-recommend: which scorer's verdict matches the 2 anchors (clean→keep, hazy→drop)
    def match(col):
        m = 0
        rc = df[df["image_id"] == clean_id]
        rh = df[df["image_id"] == hazy_id]
        if len(rc) and bool(rc[col].iloc[0]):
            m += 1
        if len(rh) and not bool(rh[col].iloc[0]):
            m += 1
        return m
    mA, mB = match("keep_A"), match("keep_B")
    rec = "A (ctr_overlay basal-opacity+CP)" if mA > mB else ("B (basal lower-zone clarity)" if mB > mA else "A (tie — A reuses the validated instrument)")

    # ---- render panel from the saved ctr_overlay sanity overlays ----
    cells = []
    for iid in chosen:
        r = df[df["image_id"] == iid]
        if not len(r):
            continue
        r = r.iloc[0]
        ov = r["overlay"]
        im = Image.open(ov).convert("RGB").resize((420, 420)) if ov and Path(str(ov)).exists() \
            else Image.new("RGB", (420, 420), "gray")
        canvas = Image.new("RGB", (420, 500), "white")
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        role = "ANCHOR-clean" if iid == clean_id else ("ANCHOR-hazy" if iid == hazy_id else "disagree")
        d.text((6, 424), f"{r['source']} · {role}", fill="black", font=_font(15))
        kA = "KEEP" if r["keep_A"] else "drop"
        kB = "KEEP" if r["keep_B"] else "drop"
        cA = "#12a05f" if r["keep_A"] else "#d2452f"
        cB = "#12a05f" if r["keep_B"] else "#d2452f"
        d.text((6, 446), f"A: {kA}  (opac={r['opac']}, B-clean={r['A_clean']})", fill=cA, font=_font(14))
        d.text((6, 466), f"B: {kB}  (ratio={r['B_ratio']}, bright={r['B_bright']})", fill=cB, font=_font(14))
        cells.append(canvas)

    cols = len(cells)
    grid = Image.new("RGB", (cols * 426 + 6, 540), "white")
    for i, c in enumerate(cells):
        grid.paste(c, (6 + i * 426, 30))
    dd = ImageDraw.Draw(grid)
    dd.text((8, 6), f"Cleanliness bake-off — keep-rate matched {df['keep_A'].mean():.2f}  ·  "
                    f"▶ recommend: scorer {rec}  (anchors A {mA}/2, B {mB}/2)",
            fill="black", font=_font(17))
    Path(a.out_fig).parent.mkdir(parents=True, exist_ok=True)
    grid.save(a.out_fig)

    Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_md).write_text(
        f"# Cleanliness-scorer bake-off (leaf 07)\n\n"
        f"- Anchors: clean=`{clean_id}` hazy=`{hazy_id}`{' (AUTO-suggested — confirm/override)' if auto else ''}\n"
        f"- Anchor match: **scorer A {mA}/2**, **scorer B {mB}/2**\n"
        f"- keep-rate (matched): {df['keep_A'].mean():.2f}  ·  B threshold ratio ≥ {thr_B:.3f}\n"
        f"- 3 max-disagreement films shown: {picks}\n\n"
        f"## ▶ Recommended scorer: **{rec}**\n\n"
        f"Panel: `{a.out_fig}` — confirm or override, then leaf 08 filters the whole pool with the chosen scorer.\n"
    )
    print(f"[bake-off] panel → {a.out_fig}\n  ▶ recommend scorer {rec}  (A {mA}/2, B {mB}/2 anchor match)")
    if auto:
        print("  ⚠️ anchors AUTO-suggested — eyeball the panel + override with --anchor-clean/--anchor-hazy")


if __name__ == "__main__":
    main()
