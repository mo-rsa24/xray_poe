"""Side-by-side FID comparison for the N2 'composition beats overlay' claim.

Reads two or more FID result JSONs (each written by `python -m metrics.fid`, i.e.
each one already scored set B against the SAME real set A) and prints them in one
table sorted by realism, with the real-vs-real floor for context and a pairwise
verdict. Lower FID = more realistic; the composition wins the N2 claim when its FID
sits clearly below the naive overlay's (and both sit well above the floor).

It does NOT compute FID — it collates results metrics.fid already produced, so it is
torch-free and instant. Fairness guard: it warns if the sets were scored at
different sample sizes or in different embedding spaces, since FID is biased by N and
incomparable across embeddings.

CLI:
    python scripts/compare_fid.py \\
        --set "PoE compose=results/fid.json" \\
        --set "Naive overlay=results/overlay_fid.json" \\
        --floor results/floor.json --out results/n2_compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> dict:
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    if not p.is_file():
        raise SystemExit(f"no such FID JSON: {p}")
    return json.loads(p.read_text())


def _extract(label: str, path: str) -> dict:
    """Pull (fid, kid, kid_std, n, embed) from a metrics.fid result JSON."""
    d = _read(path)
    fid_blk = d.get("fid", {})
    kid_blk = d.get("kid", {})
    return {
        "label": label,
        "path": path,
        "fid": fid_blk.get("fid"),
        "kid": kid_blk.get("kid"),
        "kid_std": kid_blk.get("kid_std"),
        "n": fid_blk.get("n_b"),           # the scored (generated/overlay) set size
        "embed": d.get("embed"),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collate metrics.fid results into one N2 comparison table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--set", action="append", default=[], dest="sets", metavar="LABEL=PATH",
                   help="A labelled FID JSON, e.g. 'PoE compose=results/fid.json'. Repeatable.")
    p.add_argument("--floor", default=None, help="results/floor.json for the noise-floor row")
    p.add_argument("--out", default=None, help="Write the collated comparison JSON here")
    args = p.parse_args()

    if len(args.sets) < 1:
        raise SystemExit("pass at least one --set 'label=path.json' (usually two: compose + overlay)")

    rows = []
    for spec in args.sets:
        if "=" not in spec:
            raise SystemExit(f"--set must be LABEL=PATH (got {spec!r})")
        label, path = spec.split("=", 1)
        rows.append(_extract(label.strip(), path.strip()))

    floor_fid = None
    if args.floor:
        floor_fid = _read(args.floor).get("fid", {}).get("fid")

    # sort by FID ascending (most realistic first); None sinks to the bottom
    rows.sort(key=lambda r: (r["fid"] is None, r["fid"]))

    # ── table ────────────────────────────────────────────────────────────────────
    print(f"\n{'Set':<22}{'FID ↓':>10}   {'KID ± std':>20}   {'n':>5}   embed")
    print("─" * 72)
    for r in rows:
        fid = f"{r['fid']:.3f}" if isinstance(r["fid"], (int, float)) else "—"
        if isinstance(r["kid"], (int, float)):
            kid = f"{r['kid']:+.5f} ± {r['kid_std']:.5f}" if isinstance(r["kid_std"], (int, float)) \
                  else f"{r['kid']:+.5f}"
        else:
            kid = "—"
        n = r["n"] if r["n"] is not None else "—"
        print(f"{r['label']:<22}{fid:>10}   {kid:>20}   {str(n):>5}   {r['embed'] or '—'}")
    if isinstance(floor_fid, (int, float)):
        print(f"{'(real-vs-real floor)':<22}{floor_fid:>10.3f}   {'—':>20}   {'—':>5}   —")
    print("─" * 72)

    # ── fairness guards ──────────────────────────────────────────────────────────
    ns = {r["n"] for r in rows if r["n"] is not None}
    embeds = {r["embed"] for r in rows if r["embed"] is not None}
    if len(ns) > 1:
        print(f"⚠️  sets scored at different N {sorted(ns)} — FID is N-biased; re-score at one N.")
    if len(embeds) > 1:
        print(f"⚠️  mixed embeddings {sorted(embeds)} — FID is not comparable across feature spaces.")

    # ── verdict (best two) ───────────────────────────────────────────────────────
    verdict = None
    scored = [r for r in rows if isinstance(r["fid"], (int, float))]
    if len(scored) >= 2:
        best, second = scored[0], scored[1]
        gap = second["fid"] - best["fid"]
        factor = second["fid"] / best["fid"] if best["fid"] else float("inf")
        verdict = (f"{best['label']} beats {second['label']} by ΔFID = {gap:.3f} "
                   f"({factor:.1f}× lower) → more realistic.")
        print(f"\n{verdict}")
        if isinstance(floor_fid, (int, float)) and best["fid"] <= floor_fid:
            print("⚠️  best FID is at/under the floor — suspiciously low; check the sets.")
    elif len(scored) == 1:
        print(f"\nonly one scored set ({scored[0]['label']} FID={scored[0]['fid']:.3f}); "
              f"add the other --set to compare.")

    if args.out:
        out = {"sets": rows, "floor_fid": floor_fid, "verdict": verdict}
        out_path = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nComparison → {out_path}")


if __name__ == "__main__":
    main()
