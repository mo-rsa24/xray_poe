#!/usr/bin/env python3
"""
Exp1 floor-power check — dataset decision: NIH ChestX-ray14 vs VinBigData (VinDr-CXR).

Settles the dataset choice on the number that actually governs it: the both-disease
(cardiomegaly AND effusion) count N_AB, which sets the *floor* — the smallest gap the
Exp6 two-sample test can resolve. A large noisy floor set (NIH) and a small clean one
(VinDr) are compared on the same axis.

What it computes, per dataset, side by side:
  1. Image-level counts: N, normal, A(=cardiomegaly), B(=effusion), A-only, B-only,
     A∧B (the floor/target population), and the "pure" variants (exactly {A}, {A,B}).
  2. The A×B association: phi coefficient, odds ratio + 95% CI, chi-square p — the
     EXPERIMENTS.md Exp1 metric, restricted to the treatment pair.
  3. Floor-power: given N_AB, the 95% upper bound of the real-vs-real null of C2ST-AUC
     (analytic + Monte-Carlo cross-check), the floor's own spread, the floor-vs-N curve,
     and the inverse: how many both-disease images are needed to clear the detection bar.
  4. A per-dataset verdict (POWERED / MARGINAL / UNDERPOWERED) against the Exp6 prereg
     thresholds (real-gap bar 0.65; expected treatment ~0.78, control ~0.57).

Statistical notes (read before trusting the floor numbers):
  - The C2ST detection metric is AUC. Under H0 (both halves same distribution) the
    *rank* AUC (Mann-Whitney) null is DISTRIBUTION-FREE, so it can be simulated from
    labels alone, before any image or extractor exists. That is what makes this a
    labels-only gate.
  - A *learned* C2ST classifier can overfit and push AUC above the rank null at small
    N. So the floor here is OPTIMISTIC (narrowest possible). Use --c2st-inflation to
    add a realism margin; the true floor is >= what this prints.
  - The half-split uses m = N_AB / 2 per side. If your real floor procedure holds out
    a test fraction, pass --eval-frac to shrink the effective m (widens the floor).

Usage:
  # See the full report shape now, on clearly-labelled SYNTHETIC counts (no downloads):
  python eda/floor_power_check.py --selftest

  # Real run, once the two label tables are on disk (see acquisition block below):
  python eda/floor_power_check.py \
      --nih   data/nih/Data_Entry_2017_v2020.csv \
      --vindr data/vindr/train.csv \
      --vindr-agree 1 \
      --out eda/out/floor_power_report.md

Acquisition (labels only — a few MB, no images needed for this gate):
  NIH ChestX-ray14:  Data_Entry_2017_v2020.csv
     kaggle datasets download -d nih-chest-xrays/data -f Data_Entry_2017_v2020.csv -p data/nih --unzip
  VinBigData/VinDr:  train.csv (long format, 1 row per box per radiologist)
     kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection -f train.csv -p data/vindr
     #  the xhlulu PNG mirror (...resized-png-1024x1024) also bundles a copy of train.csv.

Deps: numpy, pandas.   pip install numpy pandas
"""
from __future__ import annotations
import argparse
import math
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# Canonical pair for this study. A = cardiomegaly, B = effusion.
DATASETS = {
    "nih": dict(
        parser="wide", id="Image Index", labels="Finding Labels", sep="|",
        A="Cardiomegaly", B="Effusion", normal="No Finding",
    ),
    "vindr": dict(
        parser="long", id="image_id", cls="class_name", rad="rad_id",
        A="Cardiomegaly", B="Pleural effusion", normal="No finding",
    ),
}

# Exp6 prereg thresholds (from EXPERIMENTS.md), used only for the verdict.
REAL_GAP_BAR = 0.65        # two-sample score at/above this == "real gap"
EXPECTED_TREATMENT = 0.78  # predicted treatment/∅ score
EXPECTED_CONTROL = 0.57    # predicted control/∅ score (≈ floor)


# --------------------------------------------------------------------------- #
#  Parsing: both label formats -> per-image (A, B, finding-set)
# --------------------------------------------------------------------------- #
@dataclass
class ImageLabels:
    name: str
    A: np.ndarray            # bool, cardiomegaly present
    B: np.ndarray            # bool, effusion present
    n_abnormal: np.ndarray   # int, number of distinct abnormal findings
    exact_A: np.ndarray      # bool, finding-set == {A}
    exact_AB: np.ndarray     # bool, finding-set == {A, B}

    @property
    def N(self) -> int:
        return len(self.A)


def load_wide(path: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NIH-style: one row per image, pipe-separated 'Finding Labels'."""
    df = pd.read_csv(path)
    sets = df[cfg["labels"]].fillna("").map(
        lambda s: {x.strip() for x in s.split(cfg["sep"]) if x.strip()}
    )
    A = sets.map(lambda s: cfg["A"] in s).to_numpy()
    B = sets.map(lambda s: cfg["B"] in s).to_numpy()
    abn = sets.map(lambda s: s - {cfg["normal"]})
    n_abn = abn.map(len).to_numpy()
    exact_A = abn.map(lambda s: s == {cfg["A"]}).to_numpy()
    exact_AB = abn.map(lambda s: s == {cfg["A"], cfg["B"]}).to_numpy()
    return A, B, n_abn, exact_A, exact_AB


def load_long(path: str, cfg: dict, agree_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """VinDr-style: one row per box per radiologist. A finding is 'present' iff at
    least `agree_k` distinct radiologists marked it. Normal == no abnormal finding."""
    df = pd.read_csv(path)
    # distinct radiologists per (image, class)
    votes = (df.groupby([cfg["id"], cfg["cls"]])[cfg["rad"]]
               .nunique().reset_index(name="n_rad"))
    present = votes[votes["n_rad"] >= agree_k]
    present = present[present[cfg["cls"]] != cfg["normal"]]       # drop 'No finding'
    per_img = present.groupby(cfg["id"])[cfg["cls"]].apply(set)
    all_ids = df[cfg["id"]].unique()
    per_img = per_img.reindex(all_ids).map(lambda s: s if isinstance(s, set) else set())
    A = per_img.map(lambda s: cfg["A"] in s).to_numpy()
    B = per_img.map(lambda s: cfg["B"] in s).to_numpy()
    n_abn = per_img.map(len).to_numpy()
    exact_A = per_img.map(lambda s: s == {cfg["A"]}).to_numpy()
    exact_AB = per_img.map(lambda s: s == {cfg["A"], cfg["B"]}).to_numpy()
    return A, B, n_abn, exact_A, exact_AB


def load(path: str, kind: str, agree_k: int) -> ImageLabels:
    cfg = DATASETS[kind]
    if pd is None:
        sys.exit("pandas required for real CSVs: pip install pandas")
    if cfg["parser"] == "wide":
        out = load_wide(path, cfg)
    else:
        out = load_long(path, cfg, agree_k)
    return ImageLabels(kind, *out)


# --------------------------------------------------------------------------- #
#  Association stats on the 2x2 (A x B)
# --------------------------------------------------------------------------- #
@dataclass
class Assoc:
    n11: int; n10: int; n01: int; n00: int
    phi: float; odds_ratio: float; or_lo: float; or_hi: float; chi2: float; p: float


def association(A: np.ndarray, B: np.ndarray) -> Assoc:
    n11 = int(np.sum(A & B)); n10 = int(np.sum(A & ~B))
    n01 = int(np.sum(~A & B)); n00 = int(np.sum(~A & ~B))
    r1, r0 = n11 + n10, n01 + n00
    c1, c0 = n11 + n01, n10 + n00
    N = n11 + n10 + n01 + n00
    denom = math.sqrt(r1 * r0 * c1 * c0) if r1 and r0 and c1 and c0 else float("nan")
    phi = (n11 * n00 - n10 * n01) / denom if denom and not math.isnan(denom) else float("nan")
    # Haldane-Anscombe 0.5 correction for the OR if any zero cell
    a, b, c, d = n11, n10, n01, n00
    if 0 in (a, b, c, d):
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    odds = (a * d) / (b * c)
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    or_lo, or_hi = math.exp(math.log(odds) - 1.96 * se), math.exp(math.log(odds) + 1.96 * se)
    # chi-square (1 dof) with continuity, p via erfc (exact for 1 dof)
    chi2 = (N * (abs(n11 * n00 - n10 * n01) - N / 2) ** 2) / (r1 * r0 * c1 * c0) if denom else float("nan")
    chi2 = max(chi2, 0.0)
    p = math.erfc(math.sqrt(chi2 / 2)) if not math.isnan(chi2) else float("nan")
    return Assoc(n11, n10, n01, n00, phi, odds, or_lo, or_hi, chi2, p)


# --------------------------------------------------------------------------- #
#  Floor-power: real-vs-real null of C2ST-AUC at a given both-disease N
# --------------------------------------------------------------------------- #
def floor_upper_analytic(m: int, z: float = 1.96, inflation: float = 0.0) -> float:
    """95% upper bound of the rank-AUC null for two samples of size m (distribution-free)."""
    if m < 2:
        return 1.0
    var = (2 * m + 1) / (12 * m * m)
    return 0.5 + inflation + z * math.sqrt(var)


def floor_mc(m: int, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Monte-Carlo cross-check of the same null. Returns (p2.5, p50, p97.5) of AUC."""
    if m < 2:
        return (0.0, 0.5, 1.0)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        x = rng.standard_normal(m)   # 'real' half 1
        y = rng.standard_normal(m)   # 'real' half 2  (same dist -> H0)
        # AUC = P(x > y) via rank of x within pooled
        pooled = np.concatenate([x, y])
        ranks = pooled.argsort().argsort() + 1
        r_x = ranks[:m].sum()
        u = r_x - m * (m + 1) / 2
        aucs[i] = u / (m * m)
    return tuple(np.percentile(aucs, [2.5, 50, 97.5]))  # type: ignore


def required_NAB(target: float, z: float = 1.96, inflation: float = 0.0,
                 eval_frac: float = 1.0) -> int:
    """Smallest N_AB whose floor 95% upper bound <= target. m = eval_frac * N_AB/2."""
    for n_ab in range(4, 200_000, 2):
        m = int(eval_frac * n_ab / 2)
        if floor_upper_analytic(m, z, inflation) <= target:
            return n_ab
    return -1


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    name: str
    N: int; n_normal: int; n_A: int; n_B: int; n_A_only: int; n_B_only: int
    n_AB: int; n_A_pure: int; n_AB_pure: int
    assoc: Assoc
    m: int
    floor95: float
    floor_mc: tuple
    verdict: str
    margin_ok: bool


def analyse(lab: ImageLabels, inflation: float, eval_frac: float,
            n_boot: int, rng: np.random.Generator) -> Result:
    A, B = lab.A, lab.B
    n_AB = int(np.sum(A & B))
    n_normal = int(np.sum(lab.n_abnormal == 0))
    assoc = association(A, B)
    m = max(int(eval_frac * n_AB / 2), 0)
    floor95 = floor_upper_analytic(m, inflation=inflation)
    fmc = floor_mc(m, n_boot, rng)
    # verdict vs the Exp6 prereg bar
    if floor95 <= 0.60:
        verdict = "POWERED"
    elif floor95 <= REAL_GAP_BAR:
        verdict = "MARGINAL"
    else:
        verdict = "UNDERPOWERED"
    # can the floor separate treatment (~0.78) from control (~0.57)?
    margin_ok = floor95 < EXPECTED_CONTROL + (EXPECTED_TREATMENT - EXPECTED_CONTROL) / 2
    return Result(
        name=lab.name, N=lab.N, n_normal=n_normal,
        n_A=int(A.sum()), n_B=int(B.sum()),
        n_A_only=int((A & ~B).sum()), n_B_only=int((~A & B).sum()),
        n_AB=n_AB, n_A_pure=int(lab.exact_A.sum()), n_AB_pure=int(lab.exact_AB.sum()),
        assoc=assoc, m=m, floor95=floor95, floor_mc=fmc,
        verdict=verdict, margin_ok=margin_ok,
    )


def render(results: list[Result], inflation: float, eval_frac: float, synthetic: bool) -> str:
    L = []
    w = L.append
    if synthetic:
        w("> ⚠️ **SYNTHETIC PLACEHOLDER DATA — NOT REAL COUNTS.** "
          "Run on the real CSVs (`--nih ... --vindr ...`) before deciding anything.\n")
    w("# Exp1 floor-power check — NIH vs VinBigData (treatment pair: cardiomegaly ∧ effusion)\n")

    # counts table
    w("## Image-level counts\n")
    w("| metric | " + " | ".join(r.name for r in results) + " |")
    w("|---|" + "|".join("---" for _ in results) + "|")
    rows = [
        ("N (total images)", "N"), ("normal (no finding)", "n_normal"),
        ("A = cardiomegaly (present)", "n_A"), ("B = effusion (present)", "n_B"),
        ("A-only (LDM cardio cond.)", "n_A_only"), ("B-only (LDM effusion cond.)", "n_B_only"),
        ("**A∧B (floor/target N_AB)**", "n_AB"),
        ("A pure (exactly {A})", "n_A_pure"), ("A∧B pure (exactly {A,B})", "n_AB_pure"),
    ]
    for label, attr in rows:
        w(f"| {label} | " + " | ".join(f"{getattr(r, attr):,}" for r in results) + " |")

    # association
    w("\n## A×B association (Exp1 metric, treatment pair only)\n")
    w("| stat | " + " | ".join(r.name for r in results) + " |")
    w("|---|" + "|".join("---" for _ in results) + "|")
    w("| phi coefficient | " + " | ".join(f"{r.assoc.phi:+.3f}" for r in results) + " |")
    w("| odds ratio (95% CI) | " + " | ".join(
        f"{r.assoc.odds_ratio:.2f} [{r.assoc.or_lo:.2f}, {r.assoc.or_hi:.2f}]" for r in results) + " |")
    w("| chi-square p | " + " | ".join(
        (f"{r.assoc.p:.1e}" if r.assoc.p >= 1e-300 else "<1e-300") for r in results) + " |")

    # floor power
    w(f"\n## Floor power (C2ST-AUC real-vs-real null; m = N_AB/2 × eval_frac={eval_frac}, "
      f"inflation={inflation:+.2f})\n")
    w("| quantity | " + " | ".join(r.name for r in results) + " |")
    w("|---|" + "|".join("---" for _ in results) + "|")
    w("| per-side m | " + " | ".join(f"{r.m:,}" for r in results) + " |")
    w("| floor 95% upper bound (analytic) | " + " | ".join(f"{r.floor95:.3f}" for r in results) + " |")
    w("| floor null AUC MC [2.5, 50, 97.5] | " + " | ".join(
        f"[{r.floor_mc[0]:.3f}, {r.floor_mc[1]:.3f}, {r.floor_mc[2]:.3f}]" for r in results) + " |")
    w(f"| treatment must exceed (bar {REAL_GAP_BAR}) | " +
      " | ".join(f"{max(r.floor95, REAL_GAP_BAR):.3f}" for r in results) + " |")
    w("| **verdict** | " + " | ".join(f"**{r.verdict}**" for r in results) + " |")

    # required N
    w("\n## How many both-disease images each bar needs\n")
    for target, what in [(0.60, "POWERED"), (REAL_GAP_BAR, "detectable at all")]:
        need = required_NAB(target, inflation=inflation, eval_frac=eval_frac)
        w(f"- floor 95% ≤ **{target:.2f}** ({what}) requires **N_AB ≥ {need:,}** both-disease images.")
    w("")

    # floor-vs-N curve
    w("## Floor-vs-N curve (floor 95% upper bound)\n")
    w("| N_AB | " + " | ".join(str(n) for n in [50, 100, 200, 400, 800, 1600, 3200]) + " |")
    w("|---|" + "|".join("---" for _ in range(7)) + "|")
    curve = [floor_upper_analytic(int(eval_frac * n / 2), inflation=inflation)
             for n in [50, 100, 200, 400, 800, 1600, 3200]]
    w("| floor95 | " + " | ".join(f"{c:.3f}" for c in curve) + " |")
    for r in results:
        w(f"| ↳ {r.name} sits at N_AB={r.n_AB:,} → floor95={r.floor95:.3f} |"
          + " |" * 6 + " |")

    # verdict prose
    w("\n## Verdict\n")
    for r in results:
        sep = "can" if r.margin_ok else "**cannot** confidently"
        w(f"- **{r.name}**: N_AB={r.n_AB:,}, phi={r.assoc.phi:+.3f}, floor95={r.floor95:.3f} "
          f"→ {r.verdict}. Floor {sep} separate the predicted treatment (~{EXPECTED_TREATMENT}) "
          f"from control (~{EXPECTED_CONTROL}).")
    w("\n> Decision rule: pick the dataset that is **POWERED with a strongly positive phi** "
      "for the treatment pair. A clean small floor beats a large noisy one **iff** it clears "
      "the bar. If both clear it, prefer VinDr (radiologist boxes validate the Exp5/Exp6 "
      "extractor). If neither clears it, the controlled comparison is underpowered → widen "
      "the pair definition, merge sources cautiously, or relax the both-disease N requirement.")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  Synthetic self-test (no downloads) — clearly fake, to show report shape
# --------------------------------------------------------------------------- #
def synth_labels(name: str, N: int, pA: float, pB: float, rho: float,
                 rng: np.random.Generator) -> ImageLabels:
    """Correlated Bernoulli A,B via a shared latent factor; a few extra findings for 'pure'."""
    g = rng.standard_normal(N)                      # shared severity factor
    A = (rng.standard_normal(N) + rho * g) > -_z(pA)
    B = (rng.standard_normal(N) + rho * g) > -_z(pB)
    extra = rng.random(N) < 0.15                    # some other finding present
    n_abn = A.astype(int) + B.astype(int) + extra.astype(int)
    exact_A = A & ~B & ~extra
    exact_AB = A & B & ~extra
    return ImageLabels(name, A, B, n_abn, exact_A, exact_AB)


def _z(p: float) -> float:
    """Inverse normal CDF via Acklam's rational approx (avoids scipy)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= ph:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nih", help="NIH Data_Entry CSV (wide, pipe-separated)")
    ap.add_argument("--vindr", help="VinDr train.csv (long, box-per-radiologist)")
    ap.add_argument("--vindr-agree", type=int, default=1,
                    help="min radiologists to call a VinDr finding present (default 1; try 2 for majority)")
    ap.add_argument("--c2st-inflation", type=float, default=0.0,
                    help="additive realism margin on the floor (learned C2ST overfit); floor is optimistic at 0")
    ap.add_argument("--eval-frac", type=float, default=1.0,
                    help="fraction of each half used for evaluation (default 1.0 = cross-fitted)")
    ap.add_argument("--n-boot", type=int, default=2000, help="MC reps for the floor cross-check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the markdown report here too")
    ap.add_argument("--selftest", action="store_true",
                    help="run on SYNTHETIC NIH-like and VinDr-like counts (no downloads)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    synthetic = False
    if args.selftest:
        synthetic = True
        labs = [
            synth_labels("NIH~(synthetic)", 112_120, pA=0.025, pB=0.119, rho=0.6, rng=rng),
            synth_labels("VinDr~(synthetic)", 15_000, pA=0.13, pB=0.08, rho=0.7, rng=rng),
        ]
    else:
        if not (args.nih or args.vindr):
            ap.error("give --nih and/or --vindr, or use --selftest")
        labs = []
        if args.nih:
            labs.append(load(args.nih, "nih", args.vindr_agree))
        if args.vindr:
            labs.append(load(args.vindr, "vindr", args.vindr_agree))

    results = [analyse(l, args.c2st_inflation, args.eval_frac, args.n_boot, rng) for l in labs]
    report = render(results, args.c2st_inflation, args.eval_frac, synthetic)
    print(report)
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\n[written] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
