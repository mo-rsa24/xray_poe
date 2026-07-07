"""Build a final synthetic cardio pile for plan 08's augment-probe (utility experiment).

Generates N fake cardiomegaly films at a chosen inference setting, scores each film's CTR
(geometric heart-to-chest ratio) with the same PSPNet instrument used everywhere, saves the
PNGs, and writes two manifests:
  * manifest_all.csv        — every kept film (unfiltered)           -> plan-08 arm A1
  * manifest_ctr_gt0.5.csv  — only the big-heart films (CTR > 0.5)   -> plan-08 arm A2

Settings (from plan 07 lever comparison, results/section4/cardio_enhance/compare_levers.csv):
  --w 6                     natural pile, detector 0.750  (1x cost)
  --w 4 --best-of 8         premium pile, detector 0.875  (8x cost, keep biggest-heart of 8)

best-of-K keeps the highest-CTR draw per slot (selection by the geometric instrument, which is
independent of the DenseNet detector, so any downstream detector number stays honest). Memory
is bounded: images stream in chunks, only the kept films are held.

Seeding: each chunk's base seed is offset by the cumulative draw count, so _noise_batch's
per-image seeds (seed+i) never collide across the whole pile — every film is distinct.

Usage (env: micromamba activate jaxstack; GPU free):
    CKPT=ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors
    VAE=ckpts/vae_v2/vae_v2_step0075000.pt
    SF=data/latents_v2/joint_perch/scale_factor.pt
    # natural w6 pile
    python scripts/build_augment_probe_set.py --ckpt $CKPT --vae $VAE --scale-factor $SF \
        --w 6 --n 10000 --out results/section4/augment_probe/synth/w6
    # premium best-of-8 pile
    python scripts/build_augment_probe_set.py --ckpt $CKPT --vae $VAE --scale-factor $SF \
        --w 4 --best-of 8 --n 10000 --out results/section4/augment_probe/synth/w4_bo8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import scripts.generate as G
from scripts.validate_individual_synthetics import generate_disease
import metrics.eval_checkpoint as EC
import scripts.ctr_overlay as C

CARDIO = 1


def _gray512(im) -> np.ndarray:
    a = im.convert("L")
    if a.size != (512, 512):
        a = a.resize((512, 512))
    return np.asarray(a, dtype=np.uint8)


def _ctr(seg, idx, imgs, dev) -> np.ndarray:
    return np.array([EC.score_decoded_array(seg, idx, _gray512(im), str(dev),
                                            with_masks=False)["ctr"] for im in imgs], dtype=float)


def _safe_argmax(row: np.ndarray) -> int:
    return 0 if np.all(np.isnan(row)) else int(np.nanargmax(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--scale-factor", required=True)
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--anchor", choices=["null", "normal"], default="null")
    ap.add_argument("--best-of", type=int, default=1)
    ap.add_argument("--n", type=int, default=10000, help="number of KEPT films")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=500, help="films per generation call (best-of=1)")
    ap.add_argument("--slot-block", type=int, default=64, help="slots per block (best-of>1)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    out = Path(args.out); img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    unet = G._load_unet(Path(args.ckpt), dev)
    vae = G._load_vae(Path(args.vae), dev)
    sf = torch.load(args.scale_factor, map_location="cpu", weights_only=False).float()
    seg, idx = C._load_seg(str(dev))
    print(f"building pile: w={args.w} anchor={args.anchor} best-of={args.best_of} n={args.n} -> {out}")

    rows = []               # (rel_path, ctr, label)
    kept = 0
    draws = 0               # cumulative draws (for seed offset uniqueness)
    t0 = time.time()

    def save(im, ctr):
        nonlocal kept
        fn = f"img_{kept:06d}.png"
        im.convert("L").save(img_dir / fn)
        rows.append((str((img_dir / fn)), round(float(ctr), 4), CARDIO))
        kept += 1

    if args.best_of <= 1:
        while kept < args.n:
            m = min(args.chunk, args.n - kept)
            imgs = generate_disease(unet, vae, sf, CARDIO, m, args.w, args.anchor,
                                    args.steps, args.seed + draws, dev)
            ctrs = _ctr(seg, idx, imgs, dev)
            for im, c in zip(imgs, ctrs):
                save(im, c)
            draws += m
            el = time.time() - t0
            print(f"  {kept}/{args.n} kept  ({kept/el:.1f} films/s, {el/60:.1f} min)", flush=True)
    else:
        K = args.best_of
        while kept < args.n:
            sb = min(args.slot_block, args.n - kept)
            pool = generate_disease(unet, vae, sf, CARDIO, sb * K, args.w, args.anchor,
                                    args.steps, args.seed + draws, dev)
            pool_ctr = _ctr(seg, idx, pool, dev).reshape(sb, K)
            for s in range(sb):
                j = _safe_argmax(pool_ctr[s])
                save(pool[s * K + j], pool_ctr[s, j])
            draws += sb * K
            el = time.time() - t0
            print(f"  {kept}/{args.n} kept  ({draws} draws, {draws/el:.1f} draws/s, {el/60:.1f} min)",
                  flush=True)

    ctr_all = np.array([r[1] for r in rows], dtype=float)
    valid = np.isfinite(ctr_all)
    big = valid & (ctr_all > 0.5)

    with open(out / "manifest_all.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "ctr", "label"]); w.writerows(rows)
    with open(out / "manifest_ctr_gt0.5.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "ctr", "label"])
        w.writerows([r for r in rows if np.isfinite(r[1]) and r[1] > 0.5])

    med = float(np.median(ctr_all[valid])) if valid.any() else float("nan")
    print(f"\nDONE {out.name}: kept={kept}  CTR median={med:.3f}  "
          f"CTR>0.5={int(big.sum())} ({big.mean()*100:.1f}%)  draws={draws}  "
          f"wall={(time.time()-t0)/60:.1f} min")
    print(f"  manifest_all.csv ({kept}) -> A1 unfiltered;  "
          f"manifest_ctr_gt0.5.csv ({int(big.sum())}) -> A2 big-heart")


if __name__ == "__main__":
    main()
