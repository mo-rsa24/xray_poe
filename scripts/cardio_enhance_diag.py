"""Cardio "bigger hearts" diagnostics A / B / C for plan 07-ldm-enhance.

All three tell you WHERE the too-small-heart problem lives before you spend any GPU
on a retrain. They reuse the exact generate + CTR building blocks from
ctr_vs_detector.py, so the numbers are directly comparable (synth cardio CTR median
0.482 vs real 0.536 is the reference gap).

  A  cfg-sweep     generate cardio at several guidance strengths (w), CTR median at each.
                   Rising CTR by w~3  -> free inference fix (just generate final set hotter).
  B  train-ctr     decode the TRAINING cardio latents, measure their CTR.
                   Training ~0.53 but generated ~0.48 -> model under-shoots its data.
                   Training ~0.48                      -> the DATA is borderline (curate a severe subset).
  C  reachability  (i) anchor=null vs anchor=normal cardio guidance; (ii) best-of-K per slot.
                   Either reaches ~0.53 -> big hearts ARE reachable (free inference fix).
                   Neither              -> the learned distribution is capped (data / codec).

Usage (env: micromamba activate jaxstack; GPU free):
    CKPT=ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors
    VAE=ckpts/vae_v2/vae_v2_step0075000.pt
    SF=data/latents_v2/joint_perch/scale_factor.pt

    python scripts/cardio_enhance_diag.py a --ckpt $CKPT --vae $VAE --scale-factor $SF \
        --w-list 1 2 3 4 --n 60
    python scripts/cardio_enhance_diag.py b --vae $VAE --scale-factor $SF \
        --latent-dir data/latents_v2/joint_perch --n 200
    python scripts/cardio_enhance_diag.py c --ckpt $CKPT --vae $VAE --scale-factor $SF \
        --n 80 --best-of 8
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import scripts.generate as G
from scripts.validate_individual_synthetics import generate_disease
import metrics.eval_checkpoint as EC
import scripts.ctr_overlay as C

CARDIO = 1              # class index for cardiomegaly
REAL_CARDIO_CTR = 0.536  # real cardiomegaly CTR median (reference gap)
OUT = Path("results/section4/cardio_enhance")


def _gray512(im) -> np.ndarray:
    a = im.convert("L")
    if a.size != (512, 512):
        a = a.resize((512, 512))
    return np.asarray(a, dtype=np.uint8)


def _ctr(seg, idx, imgs, dev) -> np.ndarray:
    """CTR per image (NaN where the segmenter finds no valid heart/lung)."""
    vals = [EC.score_decoded_array(seg, idx, _gray512(im), str(dev), with_masks=False)["ctr"]
            for im in imgs]
    return np.array(vals, dtype=float)


def _summ(name: str, ctrs: np.ndarray) -> str:
    v = ctrs[np.isfinite(ctrs)]
    line = (f"{name:<28} n={len(ctrs):3d} valid={len(v):3d}  "
            f"median={np.median(v):.3f} mean={np.mean(v):.3f}  "
            f">0.5={np.mean(v > 0.5)*100:4.1f}%  >0.53={np.mean(v > 0.53)*100:4.1f}%  "
            f">0.55={np.mean(v > 0.55)*100:4.1f}%   (real ref {REAL_CARDIO_CTR})")
    print(line)
    return line


def _load_gen(args, dev):
    unet = G._load_unet(Path(args.ckpt), dev)
    vae = G._load_vae(Path(args.vae), dev)
    sf = torch.load(args.scale_factor, map_location="cpu", weights_only=False).float()
    seg, idx = C._load_seg(str(dev))
    return unet, vae, sf, seg, idx


# ── Diagnostic A — CFG-CTR sweep ─────────────────────────────────────────────
def diag_a(args, dev):
    unet, vae, sf, seg, idx = _load_gen(args, dev)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = ["w,n,ctr_median,ctr_mean,pct_gt_0.5"]
    for w in args.w_list:
        imgs = generate_disease(unet, vae, sf, CARDIO, args.n, w, "null",
                                 args.steps, args.seed, dev)
        ctrs = _ctr(seg, idx, imgs, dev)
        _summ(f"w={w}", ctrs)
        v = ctrs[np.isfinite(ctrs)]
        rows.append(f"{w},{args.n},{np.median(v):.4f},{np.mean(v):.4f},{np.mean(v > 0.5):.4f}")
    (OUT / "cfg_ctr_sweep.csv").write_text("\n".join(rows) + "\n")
    print(f"\nDECIDE: CTR climbs toward {REAL_CARDIO_CTR} by w~3 -> free inference fix "
          f"(generate final set at that w). Flat -> not a guidance knob.")
    print(f"wrote {OUT}/cfg_ctr_sweep.csv")


# ── Diagnostic B — training-data cardio CTR ──────────────────────────────────
def diag_b(args, dev):
    vae = G._load_vae(Path(args.vae), dev)
    seg, idx = C._load_seg(str(dev))
    train_dir = Path(args.latent_dir) / "train"
    cache = torch.load(train_dir / "_labels_cache.pt", map_location="cpu", weights_only=False)
    cardio_names = [n for n, l in zip(cache["names"], cache["labels"]) if int(l) == CARDIO]
    if not cardio_names:
        raise SystemExit(f"no cardio (label={CARDIO}) latents found in {train_dir}")
    random.seed(args.seed)
    pick = random.sample(cardio_names, min(args.n, len(cardio_names)))
    print(f"decoding {len(pick)} of {len(cardio_names)} training cardio latents from {train_dir}")

    imgs = []
    DECODE_BS = 4                                  # fp32 VAE decoder ~7.4 GB/img at 512² -> 4 fits 32 GB
    for i in range(0, len(pick), DECODE_BS):
        zs = [torch.load(train_dir / n, map_location="cpu", weights_only=True)["z"]
              for n in pick[i:i + DECODE_BS]]
        z = torch.stack(zs).to(dev)               # (b,4,128,128), raw mu latent
        imgs += G._decode_latents(z, vae, 1.0)    # sf=1.0 -> decode raw latent as-is
    ctrs = _ctr(seg, idx, imgs, dev)
    OUT.mkdir(parents=True, exist_ok=True)
    line = _summ("TRAIN cardio", ctrs)
    (OUT / "train_cardio_ctr.txt").write_text(
        line + "\n\nDECIDE: training ~0.53 but generated ~0.482 -> model UNDER-SHOOTS its data "
        "(fix via A / longer / capacity). training ~0.48 -> BORDERLINE data "
        "(border-visible filter kept mild cases) -> curate a CTR>0.55 severe subset.\n")
    print(f"wrote {OUT}/train_cardio_ctr.txt")


# ── Diagnostic C — inference reachability (anchor + best-of-N) ────────────────
def diag_c(args, dev):
    unet, vae, sf, seg, idx = _load_gen(args, dev)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    # (i) anchor null vs normal
    for anchor in ("null", "normal"):
        imgs = generate_disease(unet, vae, sf, CARDIO, args.n, args.w, anchor,
                                args.steps, args.seed, dev)
        lines.append(_summ(f"anchor={anchor}", _ctr(seg, idx, imgs, dev)))
    # (ii) best-of-K: generate K draws per slot, keep the highest-CTR draw
    k = args.best_of
    imgs = generate_disease(unet, vae, sf, CARDIO, args.n * k, args.w, "null",
                            args.steps, args.seed + 1, dev)
    ctrs = _ctr(seg, idx, imgs, dev).reshape(args.n, k)
    best = np.nanmax(ctrs, axis=1)
    lines.append(_summ(f"best-of-{k}", best))
    (OUT / "reachability.txt").write_text("\n".join(lines) +
        f"\n\nDECIDE: anchor=normal or best-of-{k} reaching ~{REAL_CARDIO_CTR} -> big hearts are "
        "REACHABLE (free inference fix: anchor / rejection-sampling). Neither -> distribution "
        "is capped -> data-curation or codec.\n")
    print(f"wrote {OUT}/reachability.txt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p, need_ckpt=True):
        if need_ckpt:
            p.add_argument("--ckpt", required=True)
        p.add_argument("--vae", required=True)
        p.add_argument("--scale-factor", required=True)
        p.add_argument("--n", type=int, default=80)
        p.add_argument("--steps", type=int, default=50)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    pa = sub.add_parser("a", help="CFG-CTR sweep"); common(pa)
    pa.add_argument("--w-list", type=float, nargs="+", default=[1, 2, 3, 4])

    pb = sub.add_parser("b", help="training-data cardio CTR"); common(pb, need_ckpt=False)
    pb.add_argument("--latent-dir", default="data/latents_v2/joint_perch")

    pc = sub.add_parser("c", help="reachability (anchor + best-of-N)"); common(pc)
    pc.add_argument("--w", type=float, default=1.0)
    pc.add_argument("--best-of", type=int, default=8)

    args = ap.parse_args()
    dev = torch.device(args.device)
    {"a": diag_a, "b": diag_b, "c": diag_c}[args.mode](args, dev)


if __name__ == "__main__":
    main()
