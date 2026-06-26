"""Generate + save synthetic image sets from one LDM checkpoint, for the
before/after synthetic figures (D3 Fig1 GradCAM, D4 Table1 report, D5 Fig2 CTR).

Three conditions per checkpoint:
  cardio    — single-disease cardiomegaly  (cfg_single)
  effusion  — single-disease effusion       (cfg_single)
  comorbid  — PoE composition of both       (cfg_compose)

Saves PNGs to <out>/<tag>/{cardio,effusion,comorbid}/img_XXXX.png so the figure
scripts consume real-vs-synth from a stable on-disk layout. Reuses the proven
generation path (scripts.generate); modest gen-batch (default 4) keeps peak GPU
alloc ~9.5 GB so it COEXISTS with a concurrent training run (verified 2026-06-26).

    python scripts/gen_synth_sets.py --ldm ckpts/model_step0040000.safetensors --tag before --n 120
    python scripts/gen_synth_sets.py --ldm ckpts_rebalanced/model_step0050000.safetensors --tag after --n 120
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DDIMScheduler

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.generate import _load_unet, _load_vae, _decode_latents, _noise_batch, DISEASE_MAP  # noqa: E402
from src.inference.cfg import cfg_single, cfg_compose  # noqa: E402


def _gen(unet, vae, sf, kind, n, w, anchor, steps, seed, device, gen_bs, ddim, null_idx):
    imgs = []
    for start in range(0, n, gen_bs):
        bn = min(gen_bs, n - start)
        noise = _noise_batch(bn, seed + start, device)
        if kind == "comorbid":
            z0 = cfg_compose(unet, noise, w, ddim, steps=steps, anchor=anchor,
                             null_token_idx=null_idx)
        else:
            z0 = cfg_single(unet, noise, DISEASE_MAP[kind], w, ddim, steps=steps,
                            anchor=anchor, null_token_idx=null_idx)
        imgs.extend(_decode_latents(z0, vae, sf))
    return imgs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ldm", required=True)
    p.add_argument("--tag", required=True, help="before | after | arm1 ...")
    p.add_argument("--n", type=int, default=120, help="images per condition")
    p.add_argument("--conditions", default="cardio,effusion,comorbid")
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--anchor", choices=["null", "normal"], default="null")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen-bs", type=int, default=4)
    p.add_argument("--vae-ckpt", default="ckpts/vae_step0025000.pt")
    p.add_argument("--scale-factor", default="data/latents/scale_factor.pt")
    p.add_argument("--out", default="results/synth_sets")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = _load_unet(Path(args.ldm), device)
    vae = _load_vae(Path(args.vae_ckpt), device)
    sf = torch.load(args.scale_factor, map_location="cpu")
    sf = sf.item() if hasattr(sf, "item") else float(sf)
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    null_idx = unet.null_token_idx
    name_map = {"cardio": "cardiomegaly", "effusion": "effusion", "comorbid": "comorbid"}

    print(f"[{args.tag}] ldm={args.ldm}  n={args.n}/cond  conditions={args.conditions}")
    for cond in args.conditions.split(","):
        kind = "cardiomegaly" if cond == "cardio" else ("effusion" if cond == "effusion" else "comorbid")
        out_dir = Path(args.out) / args.tag / cond
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(list(out_dir.glob("*.png"))) >= args.n:
            print(f"  {cond:9s}: already has ≥{args.n} imgs -> skip (resume-friendly)")
            continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        imgs = _gen(unet, vae, sf, kind, args.n, args.w, args.anchor, args.steps,
                    args.seed, device, args.gen_bs, ddim, null_idx)
        for i, im in enumerate(imgs):
            im.save(out_dir / f"img_{i:04d}.png")
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"  {cond:9s}: {len(imgs)} imgs -> {out_dir}  (peak {peak:.2f} GB)")

    print(f"done: {args.out}/{args.tag}/")


if __name__ == "__main__":
    main()
