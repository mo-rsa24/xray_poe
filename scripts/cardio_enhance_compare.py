"""Compare every FREE inference-time cardio-enhance lever on BOTH instruments.

Plan 07-ldm-enhance asks: can we push fake-cardiomegaly detectability toward 0.70
without a retrain? Check A proved the guidance knob moves geometric CTR; the detector
confirmation showed w=4 -> detector 0.625. This script sweeps ALL the cheap levers
(hotter guidance, `normal` anchor, best-of-N rejection sampling, and their combos)
and scores each on the SAME two instruments used everywhere else:

  * CTR      — geometric heart-to-chest ratio (cardiomegaly <=> CTR > 0.5, real median 0.536)
  * detector — DenseNet presence classifier cardio sensitivity (real 0.80; raw fakes 0.365)

best-of-N here = draw K per slot, keep the biggest-HEART (highest-CTR) draw, then read the
detector on the KEPT set. Selection is by CTR (an instrument independent of the detector),
so the reported detector sensitivity is honest, not selected-on-the-judge.

Usage (env: micromamba activate jaxstack; GPU free):
    CKPT=ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors
    VAE=ckpts/vae_v2/vae_v2_step0075000.pt
    SF=data/latents_v2/joint_perch/scale_factor.pt
    python scripts/cardio_enhance_compare.py --ckpt $CKPT --vae $VAE --scale-factor $SF --n 160
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import scripts.generate as G
from scripts.validate_individual_synthetics import (
    load_classifier, classify, generate_disease, load_real)
import metrics.eval_checkpoint as EC
import scripts.ctr_overlay as C

CARDIO = 1
REAL_CTR = 0.536
OUT = Path("results/section4/cardio_enhance")

# (name, w, anchor, best_of) — every free inference-time lever + combos.
ARMS = [
    ("w1_null",       1.0, "null",   1),   # baseline (what the default set uses)
    ("w4_null",       4.0, "null",   1),   # current pick (geometric gap closed)
    ("w6_null",       6.0, "null",   1),   # hotter still — does the detector keep climbing or saturate?
    ("w4_normal",     4.0, "normal", 1),   # healthy->sick anchor direction
    ("w4_bo8",        4.0, "null",   8),   # rejection sampling: keep biggest heart of 8
    ("w4_normal_bo8", 4.0, "normal", 8),   # anchor + rejection sampling stacked
]


def _gray512(im) -> np.ndarray:
    a = im.convert("L")
    if a.size != (512, 512):
        a = a.resize((512, 512))
    return np.asarray(a, dtype=np.uint8)


def _ctr(seg, idx, imgs, dev) -> np.ndarray:
    return np.array([EC.score_decoded_array(seg, idx, _gray512(im), str(dev),
                                            with_masks=False)["ctr"] for im in imgs], dtype=float)


def _safe_argmax_ctr(row: np.ndarray) -> int:
    """argmax over a slot's K CTRs; if all NaN, keep draw 0."""
    if np.all(np.isnan(row)):
        return 0
    return int(np.nanargmax(row))


def _score(name, imgs, ctrs, prob_c, thr_c, rows):
    valid = np.isfinite(ctrs)
    cv = ctrs[valid]
    ctr_pos = ctrs > 0.5
    clf_pos = prob_c > thr_c
    sens = float(np.mean(clf_pos))
    med = float(np.median(cv)) if cv.size else float("nan")
    pct05 = float(np.mean(ctr_pos[valid])) if valid.any() else float("nan")
    caught = float(np.mean(clf_pos[valid & ctr_pos])) if (valid & ctr_pos).any() else float("nan")
    print(f"{name:<15} n={len(imgs):3d} valid={int(valid.sum()):3d}  "
          f"CTR med={med:.3f}  >0.5={pct05*100:4.1f}%  "
          f"detector sens={sens*100:4.1f}%  (of enlarged caught {caught*100:4.1f}%)")
    rows.append([name, len(imgs), int(valid.sum()), round(med, 4),
                 round(pct05, 4), round(sens, 4), round(caught, 4)])
    return sens, med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--scale-factor", required=True)
    ap.add_argument("--classifier", default="ckpts/presence_classifier_finetuned.pt")
    ap.add_argument("--real-csv", default="data/splits/cardio_not_effusion.csv")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    unet = G._load_unet(Path(args.ckpt), dev)
    vae = G._load_vae(Path(args.vae), dev)
    sf = torch.load(args.scale_factor, map_location="cpu", weights_only=False).float()
    clf, thr_c, _ = load_classifier(Path(args.classifier), dev)
    seg, idx = C._load_seg(str(dev))
    print(f"detector cardio threshold={thr_c:.3f}  CTR cutoff=0.5  real ref CTR={REAL_CTR}\n")

    rows = [["arm", "n", "ctr_valid", "ctr_median", "pct_gt_0.5", "detector_sens", "enlarged_caught"]]
    results = {}

    # real reference row (same instruments)
    real, _ = load_real(Path(args.real_csv), args.n, args.seed)
    r_ctr = _ctr(seg, idx, real, dev)
    r_prob = classify(clf, real, dev)[:, 0]
    _score("REAL_cardio", real, r_ctr, r_prob, thr_c, rows)

    for name, w, anchor, k in ARMS:
        if k == 1:
            imgs = generate_disease(unet, vae, sf, CARDIO, args.n, w, anchor, args.steps, args.seed, dev)
            ctrs = _ctr(seg, idx, imgs, dev)
            prob = classify(clf, imgs, dev)[:, 0]
        else:
            # draw n*k, keep the highest-CTR draw per slot, read detector on the kept set
            pool = generate_disease(unet, vae, sf, CARDIO, args.n * k, w, anchor,
                                    args.steps, args.seed + 1, dev)
            pool_ctr = _ctr(seg, idx, pool, dev).reshape(args.n, k)
            keep = [i * k + _safe_argmax_ctr(pool_ctr[i]) for i in range(args.n)]
            imgs = [pool[j] for j in keep]
            ctrs = np.array([pool_ctr[i, keep[i] - i * k] for i in range(args.n)], dtype=float)
            prob = classify(clf, imgs, dev)[:, 0]
        sens, med = _score(name, imgs, ctrs, prob, thr_c, rows)
        results[name] = (sens, med)

    with open(OUT / "compare_levers.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # verdict: best arm by detector sensitivity (the open gate), CTR gate = median >= 0.53
    best = max(results, key=lambda a: results[a][0])
    bs, bm = results[best]
    print(f"\nBEST detector arm: {best}  sens={bs*100:.1f}%  CTR median={bm:.3f}")
    print(f"detector >=0.70 target {'MET' if bs >= 0.70 else 'NOT met'};  "
          f"CTR gate (>=0.53) {'MET' if bm >= 0.53 else 'NOT met'}")
    print(f"wrote {OUT}/compare_levers.csv")


if __name__ == "__main__":
    main()
