"""Exp02 task 3 — real-vs-recon eyeball panel for metric-vs-eye calibration.

For ~20 separable films, shows REAL | RECON with the CTR/CP overlay on each and
the localized verdict (|ΔCTR|, CP preserved?) captioned, so the printed metric can
be checked against whether the heart border / CP corner *looks* preserved.

    python scripts/recon_eyeball_panel.py --n 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.ctr_overlay as C
import scripts.vae_recon_pilot as V
from vae.model import VAE
from vae.config import DEFAULT_CONFIG as vae_cfg

MANIFEST = Path("results/exp02_ceiling/recon_set_manifest.csv")
OUT = Path("results/exp02_ceiling/eyeball_panel.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    model = VAE(vae_cfg).to(dev)
    ck = torch.load("ckpts/vae_step0025000.pt", map_location=dev)
    model.load_state_dict(ck.get("model", ck.get("model_state", ck)))
    model.eval()
    seg, idx = C._load_seg(args.device)

    sep = pd.read_csv(MANIFEST)
    sep = sep[sep["group"] == "separable"].sample(args.n, random_state=args.seed).reset_index(drop=True)
    tmp = Path("outputs/exp02_ceiling/_tmp_panel.png"); tmp.parent.mkdir(parents=True, exist_ok=True)

    cells = []
    for _, r in sep.iterrows():
        orig = np.array(Image.open(r["path"]).convert("L").resize((512, 512)), dtype=np.uint8)
        recon = V._recon_uint8(model, orig, dev)
        Image.fromarray(recon, mode="L").save(tmp)
        mo = C.ctr_for_image(seg, idx, Path(r["path"]), dev)
        mr = C.ctr_for_image(seg, idx, tmp, dev)
        dctr = abs(mo["ctr"] - mr["ctr"]) if np.isfinite(mo["ctr"]) and np.isfinite(mr["ctr"]) else float("nan")
        cpok = (mo["cp_left"]["blunted"] == mr["cp_left"]["blunted"]) and \
               (mo["cp_right"]["blunted"] == mr["cp_right"]["blunted"])
        cell = Image.new("RGB", (512, 290), (20, 20, 20))
        cell.paste(C._overlay(Path(r["path"]), mo).resize((256, 256)), (0, 28))
        cell.paste(C._overlay(tmp, mr).resize((256, 256)), (256, 28))
        d = ImageDraw.Draw(cell)
        d.text((4, 5), f"{r['Image Index'][:16]}  |dCTR|={dctr:.3f}  CP={'ok' if cpok else 'FLIP'}",
                fill=(255, 235, 0), font=C._font(18))
        d.text((96, 286 - 16), "REAL", fill=(170, 170, 170), font=C._font(13))
        d.text((348, 286 - 16), "RECON", fill=(170, 170, 170), font=C._font(13))
        cells.append(cell)

    cols, cw, chh, g = 3, 512, 290, 6
    rows = (len(cells) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cw + (cols + 1) * g, rows * chh + (rows + 1) * g), (10, 10, 10))
    for i, c in enumerate(cells):
        rr, cc = divmod(i, cols)
        canvas.paste(c, (g + cc * (cw + g), g + rr * (chh + g)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"→ {OUT}  ({len(cells)} real-vs-recon pairs)")


if __name__ == "__main__":
    main()
