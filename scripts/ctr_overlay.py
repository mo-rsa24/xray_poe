"""Cardiothoracic ratio (CTR) — a classifier-INDEPENDENT heart-size instrument.

Segments Heart + Left/Right Lung with the torchxrayvision ChestX-Det PSPNet
(pretrained, anatomical segmentation — NOT the presence classifier), then:
    heart_width  = max horizontal extent of the Heart mask (over rows)
    thorax_width = max horizontal extent of (Left ∪ Right Lung) mask (over rows)
    CTR          = heart_width / thorax_width
CTR > ~0.5 is the classic cardiomegaly sign. This adjudicates the pilot question
the presence classifier + Grad-CAM could not: do the 100k generated cardiomegaly
images actually have enlarged hearts (→ classifier mis-calibration on synthetic)
or normal-sized hearts (→ the LDM under-renders the cardiac silhouette)?

Substitutes for the plan's HybridGNet/CheXmask CTR — same geometry, no torch_geometric
/ no GB-scale mask download. The geometry is reusable on any heart/lung masks.

Usage (repeatable --group label:dir:n; --pgroup label:partition_group:n for the
full-corpus groups like normal):
    python scripts/ctr_overlay.py \
        --group cardio_real:data/nih/images_cardio_only:60 \
        --group both_real:data/nih/images_both:60 \
        --pgroup normal_real:normal:60 \
        --group cardio_gen100k:outputs/ckpt_eval/0100000/single/cardiomegaly:100 \
        --group effusion_gen100k:outputs/ckpt_eval/0100000/single/effusion:60 \
        --out-json results/ctr_calibration.json \
        --out-fig figures/instrument_calibration/ctr_overlay_real.png --fig-n 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image, ImageDraw

HEART = "Heart"
LLUNG = "Left Lung"
RLUNG = "Right Lung"
_PARTITION = "data/nih/partition.parquet"
_FULL_IMAGES = "data/nih/images"


# ── segmentation ───────────────────────────────────────────────────────────────

def _load_seg(device: str):
    seg = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    idx = {t: i for i, t in enumerate(seg.targets)}
    return seg, idx


def _to_input(path: Path, device: str) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((512, 512))
    arr = np.array(img, dtype=np.float32)
    arr = xrv.datasets.normalize(arr, maxval=255)          # → [-1024, 1024]
    t = torch.from_numpy(arr)[None, None].to(device)       # (1,1,512,512)
    return t


def _mask(prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return prob > thr


def _max_width(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (width, row, x_lo, x_hi) of the widest row of a binary mask."""
    best = (0, -1, 0, 0)
    rows = np.where(mask.any(axis=1))[0]
    for y in rows:
        xs = np.where(mask[y])[0]
        w = int(xs[-1] - xs[0] + 1)
        if w > best[0]:
            best = (w, int(y), int(xs[0]), int(xs[-1]))
    return best


# ── CTR per image ────────────────────────────────────────────────────────────────

def ctr_for_image(seg, idx, path: Path, device: str) -> dict:
    with torch.no_grad():
        logits = seg(_to_input(path, device))
        probs = torch.sigmoid(logits)[0].cpu().numpy()     # (n_targets, 512, 512)
    heart = _mask(probs[idx[HEART]])
    lungs = _mask(probs[idx[LLUNG]]) | _mask(probs[idx[RLUNG]])
    if not heart.any() or not lungs.any():
        return {"ctr": float("nan"), "heart_w": 0, "thorax_w": 0, "ok": False}
    hw, hy, hx0, hx1 = _max_width(heart)
    tw, ty, tx0, tx1 = _max_width(lungs)
    return {
        "ctr": float(hw / tw) if tw else float("nan"),
        "heart_w": hw, "thorax_w": tw, "ok": True,
        "heart_line": [hx0, hy, hx1, hy],
        "thorax_line": [tx0, ty, tx1, ty],
    }


def _overlay(path: Path, m: dict) -> Image.Image:
    im = Image.open(path).convert("RGB").resize((512, 512))
    d = ImageDraw.Draw(im)
    if m["ok"]:
        d.line(m["heart_line"], fill=(255, 60, 60), width=4)     # heart width (red)
        d.line(m["thorax_line"], fill=(60, 160, 255), width=4)   # thorax width (blue)
    d.rectangle([0, 496, 512, 512], fill=(0, 0, 0))
    d.text((4, 498), f"{path.name[:22]}  CTR={m['ctr']:.3f}", fill=(255, 255, 0))
    return im


def _grid(cells: list[Image.Image], cols: int = 4, cell: int = 256) -> Image.Image:
    rows = (len(cells) + cols - 1) // cols
    g = 4
    canvas = Image.new("RGB", (cols * cell + (cols + 1) * g, rows * cell + (rows + 1) * g), (24, 24, 24))
    for i, im in enumerate(cells):
        r, c = divmod(i, cols)
        canvas.paste(im.resize((cell, cell)), (g + c * (cell + g), g + r * (cell + g)))
    return canvas


# ── path collection ──────────────────────────────────────────────────────────────

def _from_dir(d: str, n: int, seed: int = 42) -> list[Path]:
    paths = sorted(Path(d).glob("*.png")) + sorted(Path(d).glob("*.jpg"))
    if n < len(paths):
        paths = random.Random(seed).sample(paths, n)
    return paths


def _from_partition(group: str, n: int, seed: int = 42) -> list[Path]:
    import pandas as pd
    df = pd.read_parquet(_PARTITION)
    names = df[df["group"] == group]["Image Index"].tolist()
    names = random.Random(seed).sample(names, min(n, len(names)))
    return [Path(_FULL_IMAGES) / nm for nm in names]


def _summary(ctrs: list[float]) -> dict:
    a = np.array([c for c in ctrs if np.isfinite(c)], dtype=np.float64)
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)), "mean": float(a.mean()), "std": float(a.std()),
        "median": float(np.median(a)), "frac_gt_0.5": float((a > 0.5).mean()),
        "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="CTR via xrv ChestX-Det segmentation")
    p.add_argument("--group", action="append", default=[], help="label:dir:n")
    p.add_argument("--pgroup", action="append", default=[], help="label:partition_group:n")
    p.add_argument("--out-json", default="results/ctr_calibration.json")
    p.add_argument("--out-fig", default=None)
    p.add_argument("--fig-n", type=int, default=12, help="example overlays per group in the fig")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seg, idx = _load_seg(args.device)
    print(f"PSPNet loaded on {args.device}; heart idx={idx[HEART]}")

    specs = []
    for g in args.group:
        label, d, n = g.split(":")
        specs.append((label, _from_dir(d, int(n), args.seed), d))
    for g in args.pgroup:
        label, grp, n = g.split(":")
        specs.append((label, _from_partition(grp, int(n), args.seed), f"partition:{grp}"))

    out = {"instrument": "xrv ChestX-Det PSPNet CTR (heart_w / thorax_w)", "groups": {}}
    fig_cells: list[Image.Image] = []
    for label, paths, src in specs:
        per = [ctr_for_image(seg, idx, pth, args.device) for pth in paths]
        ctrs = [m["ctr"] for m in per]
        s = _summary(ctrs)
        out["groups"][label] = {"src": src, **s,
                                "n_failed_seg": int(sum(not m["ok"] for m in per))}
        print(f"{label:20s} n={s.get('n',0):3d}  mean CTR={s.get('mean',float('nan')):.3f}"
              f"  median={s.get('median',float('nan')):.3f}  frac>0.5={s.get('frac_gt_0.5',float('nan')):.2f}")
        if args.out_fig:
            for pth, m in list(zip(paths, per))[: args.fig_n]:
                fig_cells.append(_overlay(pth, m))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nResults → {args.out_json}")
    if args.out_fig and fig_cells:
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        _grid(fig_cells, cols=args.fig_n).save(args.out_fig)
        print(f"Overlay grid → {args.out_fig}")


if __name__ == "__main__":
    main()
