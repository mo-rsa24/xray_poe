"""CFG × anchor sweep grid — does stronger guidance pull a heart edge out of the haze?

Diagnostic for the cardio under-rendering plateau (Exp 5). The live recon panels are
sampled at w=1.0 (raw conditional, no guidance push) with the `null` anchor — which is
now ~75% diseased after the 25/40/35 rebalance, so the cardio direction (ε_cardio − ε_null)
is weak.  This script holds the *initial noise fixed per row* and sweeps the CFG weight
across columns, for a chosen anchor, so you can SEE — for the identical sample — whether a
stronger / cleaner cardio signal sharpens the cardiac border or whether the lower zone is
genuinely structureless mush that guidance cannot conjure.

    rows = seeds (same starting noise reused across every column)
    cols = CFG weight w  (w=1 raw conditional → w=4 strong guidance)
    anchor = 'normal' (healthy no-finding baseline) and/or 'null' (learned ∅ marginal)

It reuses the canonical loaders + sampler from scripts/generate.py (no divergence from the
training-time / yield-harness inference path).

Run it ONLY when the GPU is free (not alongside the live train — single-box OOM).

Example:
    python scripts/cfg_anchor_sweep.py --both-anchors \
        --seeds 42 137 256 --w-sweep 1.0 1.5 2.0 3.0 4.0
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference.cfg import cfg_single
# reuse the EXACT canonical loaders / decode used by generate.py + the yield harness
from scripts.generate import _load_unet, _load_vae, _decode_latents, DISEASE_MAP


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _latest_ckpt(ckpt_dir: str) -> str:
    cands = glob.glob(str(Path(ckpt_dir) / "model_step*.safetensors"))
    if not cands:
        raise SystemExit(f"no model_step*.safetensors in {ckpt_dir}")
    # sort by the integer step embedded in the filename
    cands.sort(key=lambda p: int("".join(c for c in Path(p).stem if c.isdigit()) or 0))
    return cands[-1]


def _step_of(ckpt: str) -> int:
    digits = "".join(c for c in Path(ckpt).stem if c.isdigit())
    return int(digits) if digits else 0


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _fixed_noise(seeds: list[int], device: torch.device) -> torch.Tensor:
    """One latent-noise tensor per seed, stacked into a batch (reused across all w)."""
    out = []
    for s in seeds:
        g = torch.Generator().manual_seed(s)
        out.append(torch.randn(1, 4, 128, 128, generator=g))
    return torch.cat(out, 0).to(device)


def _grid(cells: dict[tuple[int, int], Image.Image], seeds: list[int], ws: list[float],
          title: str, cell: int = 256) -> Image.Image:
    """cells[(row, col)] → PIL image.  row indexes seeds, col indexes ws."""
    pad, lm, tm, ttl = 8, 96, 34, 30
    nrow, ncol = len(seeds), len(ws)
    W = lm + ncol * cell + (ncol + 1) * pad
    H = ttl + tm + nrow * cell + (nrow + 1) * pad
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((pad, 6), title, fill="black", font=_font(19))
    fw, fs = _font(17), _font(15)
    # column headers (w)
    for c, w in enumerate(ws):
        x = lm + c * cell + (c + 1) * pad
        label = f"w={w:g}"
        tw = d.textlength(label, font=fw)
        d.text((x + cell / 2 - tw / 2, ttl + 8), label, fill="#1a3a8a", font=fw)
    # row headers (seed) + cells
    for r, s in enumerate(seeds):
        y = ttl + tm + r * cell + (r + 1) * pad
        d.text((6, y + cell / 2 - 8), f"seed\n{s}", fill="#444", font=fs)
        for c, _w in enumerate(ws):
            x = lm + c * cell + (c + 1) * pad
            im = cells[(r, c)].convert("RGB").resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (x, y))
    return canvas


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CFG × anchor sweep grid (fixed noise per row)")
    p.add_argument("--ckpt", default=None,
                   help="model.safetensors (default: latest in --ckpt-dir)")
    p.add_argument("--ckpt-dir", default="ckpts_rebalanced",
                   help="dir to auto-pick the latest model_step*.safetensors from")
    p.add_argument("--vae-ckpt", default="ckpts/vae_step0025000.pt")
    p.add_argument("--scale-factor", default="data/latents/scale_factor.pt")
    p.add_argument("--disease", default="cardiomegaly", choices=list(DISEASE_MAP.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 256],
                   help="one row per seed; the starting noise is fixed across the w columns")
    p.add_argument("--w-sweep", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0, 4.0])
    p.add_argument("--anchor", choices=["null", "normal"], default="normal")
    p.add_argument("--both-anchors", action="store_true",
                   help="render a grid for BOTH normal and null (direct A/B of the anchor)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("-o", "--output", default="outputs/exp05_cfg_sweep")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = args.ckpt or _latest_ckpt(args.ckpt_dir)
    step = _step_of(ckpt)
    label_idx = DISEASE_MAP[args.disease]
    anchors = ["normal", "null"] if args.both_anchors else [args.anchor]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg-sweep] ckpt={ckpt} (step {step})  device={device}")
    print(f"           disease={args.disease}(idx {label_idx})  seeds={args.seeds}"
          f"  w={args.w_sweep}  anchors={anchors}  steps={args.steps}")

    unet = _load_unet(Path(ckpt), device)
    vae = _load_vae(Path(args.vae_ckpt), device)
    sf_path = Path(args.scale_factor)
    scale_factor = torch.load(sf_path, map_location="cpu").item() if sf_path.exists() else 1.0
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    null_idx = unet.null_token_idx
    noise = _fixed_noise(args.seeds, device)  # (n_seed, 4,128,128) — fixed per row

    written = []
    for anchor in anchors:
        cells: dict[tuple[int, int], Image.Image] = {}
        t0 = time.perf_counter()
        for c, w in enumerate(args.w_sweep):
            z0 = cfg_single(unet, noise, label_idx, float(w), ddim,
                            steps=args.steps, anchor=anchor, null_token_idx=null_idx)
            imgs = _decode_latents(z0, vae, scale_factor)  # one per seed (row)
            for r, im in enumerate(imgs):
                cells[(r, c)] = im
            print(f"  anchor={anchor:6s} w={w:<4g} done ({len(imgs)} imgs)")
        title = f"{args.disease} · anchor={anchor} · step {step} · fixed-noise CFG sweep"
        grid = _grid(cells, args.seeds, args.w_sweep, title)
        out = out_dir / f"cfg_sweep_{args.disease}_{anchor}_step{step}.png"
        grid.save(out)
        written.append(str(out))
        print(f"  [{time.perf_counter()-t0:.1f}s] saved → {out}")

    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "ckpt": ckpt, "step": step, "vae_ckpt": args.vae_ckpt, "disease": args.disease,
        "seeds": args.seeds, "w_sweep": args.w_sweep, "anchors": anchors,
        "steps": args.steps, "grids": written,
    }, indent=2))
    print(f"[cfg-sweep] done · {len(written)} grid(s) · MANIFEST.json written")


if __name__ == "__main__":
    main()
