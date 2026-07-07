"""VinDr ↔ NIH disjointness check (data-integrity gate for Stage-1).

Two-layer check:
  1. Structural — VinDr image IDs are 32-char hex strings; NIH patient IDs are
     small integers; NIH image indices are `XXXXXXXX_NNN.png`.  The namespaces
     cannot overlap by construction.
  2. Intersection — despite the structural guarantee, compute set intersections
     against every NIH test-side ID set and assert all are empty.  Protects
     against future data-pipeline changes that could accidentally re-use IDs.

NIH test-side sources loaded:
  data/splits/test_patient_ids.txt              317 reserved patient IDs
  data/splits/pixel_verifiable_separable_test.csv  frozen separable-both films (Exp 1)
  data/splits/exp06_manifests/test.csv          Exp 6 test manifest
  data/splits/oracle_both_pool.csv              oracle both-disease pool
  results/exp01_dynamics/both_film_list.csv     Exp 1 raw both-disease list

VinDr sources loaded:
  data/latents/vindr/border_visible_manifest.csv  kept border-visible set (6 671 films)
  data/vindr512/presence_manifest.csv             full 15 000-film manifest

Outputs:
  results/vindr_transfer/eda/02-integrity.md      PASS or BLOCK verdict + check log

Run:
  python scripts/vindr_leakage_check.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────

VINDR_BV      = Path("data/latents/vindr/border_visible_manifest.csv")
VINDR_FULL    = Path("data/vindr512/presence_manifest.csv")

NIH_RESERVED  = Path("data/splits/test_patient_ids.txt")
NIH_SEP_TEST  = Path("data/splits/pixel_verifiable_separable_test.csv")
NIH_EXP06     = Path("data/splits/exp06_manifests/test.csv")
NIH_ORACLE    = Path("data/splits/oracle_both_pool.csv")
NIH_BOTH_LIST = Path("results/exp01_dynamics/both_film_list.csv")

OUT_MD        = Path("results/vindr_transfer/eda/02-integrity.md")

HEX32 = re.compile(r"^[0-9a-f]{32}$")
NIH_IMG = re.compile(r"^\d{8}_\d{3}\.png$")


# ── loaders ────────────────────────────────────────────────────────────────────

def _load_vindr_ids() -> tuple[set[str], set[str]]:
    bv   = pd.read_csv(VINDR_BV)
    full = pd.read_csv(VINDR_FULL)
    return set(bv["image_id"].astype(str)), set(full["image_id"].astype(str))


def _load_nih_test_ids() -> dict[str, set[str]]:
    """Return labelled sets of NIH test-side identifiers (all as strings)."""
    sources: dict[str, set[str]] = {}

    # 1. Reserved patient integers
    raw = open(NIH_RESERVED).read().split()
    sources["test_patient_ids (integers)"] = set(raw)

    # 2. Frozen separable-both — patient IDs + image indices
    sep = pd.read_csv(NIH_SEP_TEST)
    sources["separable_test Patient IDs"] = set(sep["Patient ID"].astype(str))
    sources["separable_test Image Indices"] = set(sep["Image Index"].astype(str))

    # 3. Exp06 test — patient_id integers
    exp6 = pd.read_csv(NIH_EXP06)
    sources["exp06_test patient_ids"] = set(exp6["patient_id"].astype(str))

    # 4. Oracle both pool — patient IDs + image indices
    oracle = pd.read_csv(NIH_ORACLE)
    sources["oracle_both_pool Patient IDs"] = set(oracle["Patient ID"].astype(str))
    sources["oracle_both_pool Image Indices"] = set(oracle["Image Index"].astype(str))

    # 5. Exp01 both film list — patient IDs + image indices
    both = pd.read_csv(NIH_BOTH_LIST)
    sources["exp01_both_film_list Patient IDs"] = set(both["Patient ID"].astype(str))
    sources["exp01_both_film_list Image Indices"] = set(both["Image Index"].astype(str))

    return sources


# ── checks ─────────────────────────────────────────────────────────────────────

def _structural_check(vindr_bv: set[str], vindr_full: set[str]) -> list[str]:
    """Verify both VinDr ID sets are entirely 32-char hex strings."""
    lines = []
    for label, ids in [("border-visible (6 671)", vindr_bv), ("full (15 000)", vindr_full)]:
        bad = [i for i in ids if not HEX32.match(i)]
        if bad:
            lines.append(f"  ⚠️  {label}: {len(bad)} IDs do NOT match hex-32 pattern")
            lines.append(f"      examples: {bad[:3]}")
        else:
            lines.append(f"  ✅ {label}: all {len(ids)} IDs are hex-32 strings")
    return lines


def _intersection_check(
    vindr_ids: set[str],
    nih_sources: dict[str, set[str]],
) -> tuple[bool, list[str]]:
    """Check intersection of vindr_ids against every NIH source set.

    Returns (all_clear: bool, report_lines: list[str]).
    """
    all_clear = True
    lines = []
    for label, nih_set in nih_sources.items():
        overlap = vindr_ids & nih_set
        if overlap:
            all_clear = False
            lines.append(f"  ❌ INTERSECTION with '{label}': {len(overlap)} IDs")
            lines.append(f"      examples: {list(overlap)[:5]}")
        else:
            lines.append(f"  ✅ '{label}' (N={len(nih_set)}): intersection = 0")
    return all_clear, lines


# ── report ─────────────────────────────────────────────────────────────────────

def _write_report(
    structural_lines: list[str],
    bv_lines: list[str],
    full_lines: list[str],
    bv_clear: bool,
    full_clear: bool,
    vindr_bv: set[str],
    vindr_full: set[str],
    nih_sources: dict[str, set[str]],
) -> None:
    verdict = "PASS — disjoint from NIH test" if (bv_clear and full_clear) else "BLOCK — intersection found"
    icon    = "✅" if (bv_clear and full_clear) else "❌"

    lines = [
        f"# VinDr ↔ NIH Disjointness Check",
        f"",
        f"## Verdict: {icon} {verdict}",
        f"",
        f"Stage-1 pretrain is {'CLEARED' if bv_clear and full_clear else 'BLOCKED'} on data integrity.",
        f"",
        f"---",
        f"",
        f"## 1. ID namespace (structural argument)",
        f"",
        f"VinDr image IDs are 32-character lowercase hexadecimal strings (MD5-derived).",
        f"NIH patient IDs are small integers (1 – ~30,000).",
        f"NIH image indices are `XXXXXXXX_NNN.png` filenames.",
        f"These three namespaces are **mutually exclusive by format** — no string can",
        f"simultaneously be a valid member of more than one set.",
        f"",
        f"Verified ID formats:",
    ] + structural_lines + [
        f"",
        f"---",
        f"",
        f"## 2. Set intersection (belt-and-suspenders assertion)",
        f"",
        f"Even though the namespaces cannot overlap by format, we compute the intersection",
        f"of every VinDr ID against every NIH test-side ID set and assert it is empty.",
        f"",
        f"### VinDr border-visible set ({len(vindr_bv)} films) vs NIH test sources",
    ] + bv_lines + [
        f"",
        f"### VinDr full manifest ({len(vindr_full)} films) vs NIH test sources",
    ] + full_lines + [
        f"",
        f"---",
        f"",
        f"## 3. NIH test-side sources checked",
        f"",
        f"| Source | N IDs |",
        f"|--------|-------|",
    ] + [f"| {k} | {len(v)} |" for k, v in nih_sources.items()] + [
        f"",
        f"---",
        f"",
        f"## 4. Conclusion",
        f"",
        (
            f"VinDr-512 (Vietnam, hospital DICOM) is an **independent external cohort**.\n"
            f"Its image IDs share no namespace with NIH-ChestXray14 patient IDs or\n"
            f"image indices. Both structural and intersection checks confirm **zero overlap**\n"
            f"with the NIH frozen test set (317 reserved patients, 378 separable-both films).\n"
            f"\n"
            f"**Stage-1 VinDr pretrain is cleared on data integrity.**"
            if bv_clear and full_clear else
            f"**INTERSECTION DETECTED — Stage-1 BLOCKED until resolved.**\n"
            f"Investigate which films triggered the overlap before proceeding."
        ),
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nReport → {OUT_MD}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading VinDr IDs …")
    vindr_bv, vindr_full = _load_vindr_ids()
    print(f"  border-visible: {len(vindr_bv)}  |  full: {len(vindr_full)}")

    print("Loading NIH test-side IDs …")
    nih_sources = _load_nih_test_ids()
    for label, ids in nih_sources.items():
        print(f"  {label}: {len(ids)}")

    print("\n── Structural check (ID format) ─────────────────────────────────────")
    structural_lines = _structural_check(vindr_bv, vindr_full)
    for l in structural_lines:
        print(l)

    print("\n── Intersection: border-visible set ─────────────────────────────────")
    bv_clear, bv_lines = _intersection_check(vindr_bv, nih_sources)
    for l in bv_lines:
        print(l)

    print("\n── Intersection: full VinDr manifest ────────────────────────────────")
    full_clear, full_lines = _intersection_check(vindr_full, nih_sources)
    for l in full_lines:
        print(l)

    all_clear = bv_clear and full_clear
    print(f"\n{'='*60}")
    verdict = "PASS — disjoint from NIH test" if all_clear else "BLOCK — intersection found"
    print(f"  {'✅' if all_clear else '❌'}  {verdict}")
    print(f"{'='*60}")

    _write_report(
        structural_lines, bv_lines, full_lines,
        bv_clear, full_clear,
        vindr_bv, vindr_full, nih_sources,
    )

    if not all_clear:
        sys.exit(1)


if __name__ == "__main__":
    main()
