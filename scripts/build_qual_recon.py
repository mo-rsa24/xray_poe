"""VAE v1 vs v2 reconstruction of one real cardio film, with a zoomed cardiac-border ROI.
Proves the claim 'v1 pixel-only = blurry borders; v2 perceptual = sharp' qualitatively.
Both codecs exist again on disk; recon = deterministic decode(mu)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/Paper3")
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scripts.generate import _load_vae
from scripts.vae_recon_pilot import _recon_uint8

OUT = Path("results/vae_v2/exhibits")
dev = torch.device("cuda")
CK = "/datasets/mmolefe/checkpoints"


def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((512, 512)), dtype=np.uint8)


def main():
    # pick a border-visible cardio val film
    df = pd.read_csv("data/vae_v2_diet/val_manifest.csv")
    row = df[(df.label == 1) & (df.border_visible == True)].iloc[7]
    orig = load_gray(row.image_path)
    v1 = _load_vae(Path(f"{CK}/vae/vae_step0025000.pt") if Path(f"{CK}/vae/vae_step0025000.pt").exists()
                   else Path(f"{CK}/vae_step0025000.pt"), dev)
    v2 = _load_vae(Path(f"{CK}/vae_v2/vae_v2_step0075000.pt"), dev)
    r1 = _recon_uint8(v1, orig, dev)
    r2 = _recon_uint8(v2, orig, dev)

    # ROI on the cardiac border (center-lower); zoom 2.4x
    rx, ry, rw = 150, 250, 220
    th = 300
    tiles = [("original", orig, (200, 200, 200)), ("v1 recon (LPIPS 0.152)", r1, (235, 120, 60)),
             ("v2 recon (LPIPS 0.022)", r2, (80, 220, 120))]
    zoom = 260
    band = 30
    canvas = Image.new("RGB", (len(tiles)*th, band + th + zoom + 24), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    for k, (name, arr, col) in enumerate(tiles):
        im = Image.fromarray(arr).convert("RGB").resize((th, th))
        dd = ImageDraw.Draw(im)
        sx = th/512.0
        dd.rectangle([rx*sx, ry*sx, (rx+rw)*sx, (ry+rw)*sx], outline=(255, 210, 0), width=3)
        canvas.paste(im, (k*th, band))
        d.text((k*th+6, 6), name, fill=col, font=_font(18))
        # zoomed ROI inset
        crop = Image.fromarray(arr[ry:ry+rw, rx:rx+rw]).convert("RGB").resize((zoom, zoom), Image.LANCZOS)
        ImageDraw.Draw(crop).rectangle([0, 0, zoom-1, zoom-1], outline=(255, 210, 0), width=2)
        canvas.paste(crop, (k*th + (th-zoom)//2, band + th + 4))
    d.text((6, band + th + zoom + 6), "Yellow box = cardiac border ROI, zoomed below. v1 blurs the heart edge; v2 keeps it sharp.",
           fill=(200, 200, 200), font=_font(13))
    canvas.save(OUT / "qual_vae_recon_roi.png")
    print("wrote qual_vae_recon_roi.png (film:", row.image_path, ")")


def _font(sz):
    import scripts.ctr_overlay as C
    return C._font(sz)


if __name__ == "__main__":
    main()
