"""Grad-CAM over the fine-tuned 2-head DenseNet-121 presence classifier.

Produces a per-head class-activation overlay on the input chest X-ray:
  cardiomegaly head  → red channel
  effusion head      → blue channel
so a single figure shows where each head fires. Same weights that drive the
quantitative presence scores (metrics/presence_classifier.py) drive this
visualisation — no new model is trained.

The CAM target layer is `model.features` (DenseNet output, post-norm5 — the same
spatial map as the final denseblock4 conv). Gradients of each head logit w.r.t.
that feature map are global-average-pooled into channel weights, applied to the
activations, ReLU'd, upsampled to 512², and normalised to [0,1].

Input preprocessing matches the classifier exactly (xrv normalize → center-crop →
resize 224), so the CAM aligns with the field of view the model actually saw.

CLI
    # single image (sanity check)
    python scripts/grad_cam.py --image <path.png> --out <out.png>

    # one-directory grid
    python scripts/grad_cam.py --dir <dir> --n 16 --out <grid.png>

    # two-directory grid (e.g. cardio rows then effusion rows)
    python scripts/grad_cam.py --dir <dirA> --n 8 --dir2 <dirB> --n2 8 --out <grid.png>

Optional:
    --bbox        draw a red (cardio) / blue (effusion) box around cam >= 0.5*max
    --cols N      grid columns (default 4)
    --ckpt PATH   classifier checkpoint (default ckpts/presence_classifier_finetuned.pt)
    --device ...  cuda / cpu (default: auto)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# make the repo root importable when run as `python scripts/grad_cam.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Shared Grad-CAM hook (load_model, preprocess, GradCAM, bbox_from_cam) — the same
# machinery drives metrics/extractors.py, including the border-artifact fix.
from metrics.grad_cam_utils import _CKPT_PATH, GradCAM, bbox_from_cam, load_model, preprocess  # noqa: E402

_ALPHA = 0.5             # heatmap blend strength


# ── overlay ──────────────────────────────────────────────────────────────────────

def overlay(disp: np.ndarray, cam_c: np.ndarray, cam_e: np.ndarray,
            bbox: bool = False) -> Image.Image:
    """Blend cardio CAM → red, effusion CAM → blue onto the grayscale display."""
    gray = np.stack([disp, disp, disp], axis=-1)                  # (H, W, 3)
    heat = np.stack([cam_c, np.zeros_like(cam_c), cam_e], axis=-1)  # red + blue
    m = np.maximum(cam_c, cam_e)[..., None]                        # blend only where active
    out = gray * (1.0 - _ALPHA * m) + _ALPHA * heat
    out = np.clip(out, 0.0, 1.0)
    im = Image.fromarray((out * 255).astype(np.uint8), mode="RGB")

    if bbox:
        d = ImageDraw.Draw(im)
        bc = bbox_from_cam(cam_c)
        be = bbox_from_cam(cam_e)
        if bc:
            d.rectangle(bc, outline=(255, 60, 60), width=3)
        if be:
            d.rectangle(be, outline=(60, 60, 255), width=3)
    return im


def _label(im: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.rectangle([0, im.height - 16, im.width, im.height], fill=(0, 0, 0))
    d.text((3, im.height - 14), text[:42], fill=(220, 220, 220))
    return im


# ── grid ─────────────────────────────────────────────────────────────────────────

def make_grid(cells: list[Image.Image], cols: int, cell: int = 256) -> Image.Image:
    n = len(cells)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    gap = 4
    W = cols * cell + (cols + 1) * gap
    H = rows * cell + (rows + 1) * gap
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    for i, im in enumerate(cells):
        r, c = divmod(i, cols)
        x = gap + c * (cell + gap)
        y = gap + r * (cell + gap)
        canvas.paste(im.resize((cell, cell), Image.BILINEAR), (x, y))
    return canvas


# ── runners ──────────────────────────────────────────────────────────────────────

def _collect(dir_path: str, n: int) -> list[Path]:
    d = Path(dir_path)
    paths = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No PNG/JPG images in {dir_path}")
    return paths[:n]


def render_one(cam_engine: GradCAM, cardio_idx: int, effusion_idx: int,
               path: Path, bbox: bool, label: bool = True) -> Image.Image:
    model_in, disp = preprocess(path)
    cam_c = cam_engine.cam(model_in, cardio_idx)
    cam_e = cam_engine.cam(model_in, effusion_idx)
    im = overlay(disp, cam_c, cam_e, bbox=bbox)
    if label:
        im = _label(im, path.name)
    return im


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM over the 2-head presence classifier")
    ap.add_argument("--image", default=None, help="Single image → single 512² overlay")
    ap.add_argument("--dir", default=None, help="Directory of images for a grid")
    ap.add_argument("--n", type=int, default=16, help="Images from --dir")
    ap.add_argument("--dir2", default=None, help="Second directory (appended to grid)")
    ap.add_argument("--n2", type=int, default=8, help="Images from --dir2")
    ap.add_argument("--out", "-o", required=True, help="Output PNG path")
    ap.add_argument("--cols", type=int, default=4, help="Grid columns")
    ap.add_argument("--bbox", action="store_true", help="Draw cam>=0.5*max boxes")
    ap.add_argument("--ckpt", default=_CKPT_PATH)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.image and not args.dir:
        ap.error("provide --image or --dir")

    print(f"Loading classifier from {args.ckpt} on {args.device} ...")
    model, cardio_idx, effusion_idx = load_model(args.ckpt, args.device)
    cam_engine = GradCAM(model, args.device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.image:
        im = render_one(cam_engine, cardio_idx, effusion_idx, Path(args.image),
                        bbox=args.bbox, label=False)
        im.save(out_path)
        print(f"Saved single overlay → {out_path}  (red=cardio, blue=effusion)")
        return

    paths = _collect(args.dir, args.n)
    if args.dir2:
        paths += _collect(args.dir2, args.n2)
    print(f"Rendering {len(paths)} overlays ...")
    cells = [render_one(cam_engine, cardio_idx, effusion_idx, p, bbox=args.bbox)
             for p in paths]
    grid = make_grid(cells, cols=args.cols)
    grid.save(out_path)
    print(f"Saved grid ({len(cells)} cells) → {out_path}  (red=cardio, blue=effusion)")


if __name__ == "__main__":
    main()
