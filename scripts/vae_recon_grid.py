"""Visualize VAE reconstruction quality as a labeled grid.

Columns: Input | Reconstruction | |Difference| (×5 amplified)
Rows: one per image (default 8, from local VinDr PNGs or a custom glob).

Usage:
    python scripts/vae_recon_grid.py                          # defaults
    python scripts/vae_recon_grid.py --ckpt ckpts/vae_step0025000.pt --n 12
    python scripts/vae_recon_grid.py --images data/vindr/images/*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vae.model import VAE
from vae.eval import recon_metrics


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str, res: int = 512) -> torch.Tensor:
    """Load a PNG as (1, 1, res, res) float32 in [-1, 1]."""
    img = Image.open(path).convert("L")
    if img.size != (res, res):
        img = img.resize((res, res), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [0,255] → [-1,1]
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,res,res)


# ---------------------------------------------------------------------------
# Grid building
# ---------------------------------------------------------------------------

CELL_SIZE = 256   # downsample to 256² to keep the file a sane size
COL_W = 80        # label sidebar width
ROW_H = 22        # per-row label height
HEADER_H = 28     # top header height


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, res, res) float32 in [-1,1] → uint8 (CELL_SIZE, CELL_SIZE)."""
    arr = t[0].float().clamp(-1, 1).cpu().numpy()
    arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "L").resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
    return np.array(img)


def _diff_uint8(x: torch.Tensor, r: torch.Tensor) -> np.ndarray:
    """Auto-normalized absolute difference → uint8 (CELL_SIZE, CELL_SIZE).

    Zero difference → black.  Worst pixel in this image (99th percentile) → white.
    Stretches the actual error range to [0, 255] so even small errors are visible
    and the user can immediately see WHERE the reconstruction fails.
    """
    arr = (r - x).abs()[0].float().cpu().numpy()  # (H, W), values in [0, 2]
    p99 = float(np.percentile(arr, 99))
    if p99 > 1e-8:
        arr = (arr / p99).clip(0, 1) * 255
    arr = arr.clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "L").resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
    return np.array(img)


def build_grid(
    inputs: list[torch.Tensor],
    recons: list[torch.Tensor],
    image_names: list[str],
    metrics: list[dict],
    step: int,
) -> Image.Image:
    n = len(inputs)
    COL_HEADERS = ["Input", "Reconstruction", "|Diff| (auto-scaled)"]
    n_cols = len(COL_HEADERS)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = font_sm = ImageFont.load_default()

    # Extra height per row to show the metric line below each cell strip
    METRIC_H = 18
    canvas_w = COL_W + CELL_SIZE * n_cols
    canvas_h = HEADER_H + (CELL_SIZE + ROW_H + METRIC_H) * n

    canvas = Image.new("L", (canvas_w, canvas_h), color=20)
    draw = ImageDraw.Draw(canvas)

    # Top header bar
    draw.rectangle([(0, 0), (canvas_w, HEADER_H - 1)], fill=10)
    title = f"VAE reconstruction  (step {step:,})"
    draw.text((COL_W + 4, 6), title, fill=200, font=font)

    # Column headers
    for ci, header in enumerate(COL_HEADERS):
        x = COL_W + ci * CELL_SIZE + CELL_SIZE // 2 - len(header) * 4
        draw.text((x, HEADER_H - 18), header, fill=190, font=font_sm)

    row_stride = CELL_SIZE + ROW_H + METRIC_H
    for ri, (x_t, r_t, name, m) in enumerate(zip(inputs, recons, image_names, metrics)):
        y_label  = HEADER_H + ri * row_stride
        y_cell   = y_label + ROW_H
        y_metric = y_cell + CELL_SIZE

        # Row label bar
        draw.rectangle([(0, y_label), (canvas_w, y_label + ROW_H - 1)], fill=35)
        short = Path(name).stem[:30]
        draw.text((4, y_label + 4), short, fill=170, font=font_sm)

        cells = [
            _to_uint8(x_t[0]),
            _to_uint8(r_t[0]),
            _diff_uint8(x_t[0], r_t[0]),
        ]
        for ci, cell_arr in enumerate(cells):
            cell_img = Image.fromarray(cell_arr, "L")
            canvas.paste(cell_img, (COL_W + ci * CELL_SIZE, y_cell))

        # Metric bar below the cells
        draw.rectangle([(COL_W, y_metric), (canvas_w, y_metric + METRIC_H - 1)], fill=28)
        metric_txt = (
            f"SSIM {m['ssim']:.4f}  |  LPIPS {m['lpips']:.4f}"
            f"  |  MAE {m['mae']:.4f}"
        )
        draw.text((COL_W + 6, y_metric + 3), metric_txt, fill=200, font=font_sm)

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VAE reconstruction grid")
    parser.add_argument("--ckpt", default=None,
                        help="Path to .pt checkpoint (default: latest in ckpts/)")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of images to show")
    parser.add_argument("--images", nargs="*", default=None,
                        help="Image paths (default: first N from data/vindr/images/)")
    parser.add_argument("--out", default="figures/vae_recon_grid.png",
                        help="Output path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # --- resolve checkpoint --------------------------------------------------
    if args.ckpt is None:
        ckpt_dir = ROOT / "ckpts"
        pts = sorted(ckpt_dir.glob("vae_step*.pt"))
        if not pts:
            sys.exit(f"No VAE checkpoints found in {ckpt_dir}")
        ckpt_path = pts[-1]
    else:
        ckpt_path = Path(args.ckpt)

    step_str = ckpt_path.stem.replace("vae_step", "")
    step = int(step_str) if step_str.isdigit() else 0
    print(f"Loading checkpoint: {ckpt_path}  (step {step:,})")

    device = torch.device(args.device)
    model = VAE()
    state = torch.load(ckpt_path, map_location=device)
    # support bare state dict or wrapped {"model": ...}
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()

    # --- resolve images -------------------------------------------------------
    if args.images:
        image_paths = args.images[: args.n]
    else:
        img_dir = ROOT / "data" / "vindr" / "images"
        image_paths = sorted(img_dir.glob("*.png"))[: args.n]
        if not image_paths:
            sys.exit(f"No PNGs found in {img_dir}")

    image_paths = [Path(p) for p in image_paths]
    print(f"Processing {len(image_paths)} images …")

    inputs, recons, metrics_list = [], [], []
    with torch.no_grad():
        for p in image_paths:
            x = load_image(str(p)).to(device)   # (1,1,512,512)
            r = model.reconstruct(x)             # posterior mean → (1,1,512,512)
            x_cpu, r_cpu = x.cpu(), r.cpu()
            m = recon_metrics(x_cpu, r_cpu)
            m["mae"] = (r_cpu - x_cpu).abs().mean().item()
            inputs.append(x_cpu)
            recons.append(r_cpu)
            metrics_list.append(m)
            print(f"  {p.stem[:40]:40s}  SSIM={m['ssim']:.4f}  LPIPS={m['lpips']:.4f}  MAE={m['mae']:.4f}")

    # --- build and save grid --------------------------------------------------
    grid = build_grid(inputs, recons, [str(p) for p in image_paths], metrics_list, step)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(out))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
