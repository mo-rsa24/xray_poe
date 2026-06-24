"""Qualitative co-morbidity panel — 3 sources × 5 samples, Grad-CAM cardio/effusion.

Rows:
  REAL both    5 real both-disease X-rays      (data/nih/images_both)
  PoE compose  5 product-of-experts composites (outputs/compose/compose/w1p0)
  Gen overlay  5 generator-controlled overlays (outputs/overlay_gen/overlay)

Each cell carries TWO instruments read off the SAME fine-tuned 2-head presence
classifier (no new model):
  * head logits  — sigmoid(logit) per head, printed as  c=…  e=…  (red / blue)
  * box          — cam>=0.5*max bounding box, red=cardio / blue=effusion
plus the Grad-CAM heat itself (red=cardio head, blue=effusion head).

The panel exists to answer, BEFORE any generator work: looking at REAL
both-disease, can we even tell both diseases are present, and which instrument
shows it (head logits vs box; CTR lives in the quantitative track, not here)?

    python scripts/comorbidity_panel.py --out figures/comorbidity_panel.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.grad_cam_utils import _CKPT_PATH, GradCAM, bbox_from_cam, load_model, preprocess  # noqa: E402
from scripts.grad_cam import overlay  # noqa: E402  (reuse the exact blend)

ROWS = [
    ("REAL both",    "data/nih/images_both"),
    ("PoE compose",  "outputs/compose/compose/w1p0"),
    ("Gen overlay",  "outputs/overlay_gen/overlay"),
]
N = 5
CELL = 320
GAP = 6
LABEL_W = 132          # left column for row headers
HEADER_H = 64          # top title + legend strip
FOOT_H = 18            # per-cell probability strip (baked into the cell)


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _collect(d: str, n: int) -> list[Path]:
    paths = sorted(Path(d).glob("*.png")) + sorted(Path(d).glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No images in {d}")
    return paths[:n]


def render_cell(cam_engine, model, c_idx, e_idx, path, device):
    """Overlay (with box) + head-logit probabilities for one image."""
    model_in, disp = preprocess(path)
    with torch.no_grad():
        probs = torch.sigmoid(model(model_in.to(device)))[0]
    pc, pe = float(probs[c_idx]), float(probs[e_idx])
    cam_c = cam_engine.cam(model_in, c_idx)
    cam_e = cam_engine.cam(model_in, e_idx)
    im = overlay(disp, cam_c, cam_e, bbox=True).resize((CELL, CELL), Image.BILINEAR)

    # bake a probability strip onto the bottom of the cell
    d = ImageDraw.Draw(im)
    d.rectangle([0, CELL - FOOT_H, CELL, CELL], fill=(0, 0, 0))
    f = _font(15)
    d.text((4, CELL - FOOT_H + 1), f"c={pc:.2f}", fill=(255, 110, 110), font=f)
    d.text((CELL // 2 + 4, CELL - FOOT_H + 1), f"e={pe:.2f}", fill=(120, 150, 255), font=f)
    return im, pc, pe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", "-o", default="figures/comorbidity_panel.png")
    ap.add_argument("--ckpt", default=_CKPT_PATH)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading classifier {args.ckpt} on {args.device} ...")
    model, c_idx, e_idx = load_model(args.ckpt, args.device)
    cam_engine = GradCAM(model, args.device)

    n_rows = len(ROWS)
    W = LABEL_W + N * CELL + (N + 1) * GAP
    H = HEADER_H + n_rows * CELL + (n_rows + 1) * GAP
    canvas = Image.new("RGB", (W, H), (20, 20, 22))
    draw = ImageDraw.Draw(canvas)

    # title + legend
    draw.text((10, 8), "Co-morbidity panel — Grad-CAM over the 2-head presence classifier",
              fill=(235, 235, 235), font=_font(20))
    lg = _font(15)
    draw.text((10, 36), "red = cardiomegaly head", fill=(255, 110, 110), font=lg)
    draw.text((240, 36), "blue = effusion head", fill=(120, 150, 255), font=lg)
    draw.text((430, 36), "box = cam>=0.5*max   c/e = sigmoid(head logit)",
              fill=(200, 200, 200), font=lg)

    means = {}
    for r, (name, d) in enumerate(ROWS):
        y0 = HEADER_H + GAP + r * (CELL + GAP)
        draw.text((8, y0 + CELL // 2 - 10), name, fill=(235, 235, 235), font=_font(17))
        pcs, pes = [], []
        for k, path in enumerate(_collect(d, N)):
            im, pc, pe = render_cell(cam_engine, model, c_idx, e_idx, path, args.device)
            x0 = LABEL_W + GAP + k * (CELL + GAP)
            canvas.paste(im, (x0, y0))
            pcs.append(pc); pes.append(pe)
            print(f"  {name:11s} {path.name:20s}  c={pc:.2f} e={pe:.2f}")
        means[name] = (float(np.mean(pcs)), float(np.mean(pes)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"\nSaved panel -> {out}")
    print("Row means (sigmoid head logit):")
    for name, (mc, me) in means.items():
        print(f"  {name:11s}  cardio={mc:.2f}  effusion={me:.2f}")


if __name__ == "__main__":
    main()
