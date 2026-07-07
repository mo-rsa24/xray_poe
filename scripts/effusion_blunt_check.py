"""v4-60k effusion blunt-rate sanity check (plan 06, last open task).

Generates effusion-only films from the winning LDM and measures the share whose costophrenic
corner reads as blunted (CP angle > 80°, the effusion instrument in scripts/ctr_overlay.py).
Pass mark: blunt-rate ≥ 80% of samples. Writes a summary + a most-blunted montage to
results/vae_v2/ldm_sanity/v4_60k_final/.

This is the geometric twin of the classifier evidence that effusion already renders well
(Exp-09 v4 effusion synth sensitivity 0.905→0.960 PASS) and of the CP blunting the Exp-03
co-presence run already measures per composed film — it just files the dedicated effusion-only
artifact plan 06 asks for.

Usage (env: micromamba activate jaxstack; needs a free GPU):
    python scripts/effusion_blunt_check.py --n 200 --w 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import scripts.generate as G
from scripts.validate_individual_synthetics import generate_disease
import metrics.eval_checkpoint as EC
import scripts.ctr_overlay as C

EFFUSION = 2
OUT = Path("results/vae_v2/ldm_sanity/v4_60k_final")


def _g512(im) -> np.ndarray:
    a = im.convert("L")
    if a.size != (512, 512):
        a = a.resize((512, 512))
    return np.asarray(a, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors")
    ap.add_argument("--vae", default="ckpts/vae_v2/vae_v2_step0075000.pt")
    ap.add_argument("--scale-factor", default="data/latents_v2/joint_perch/scale_factor.pt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    unet = G._load_unet(Path(args.ckpt), dev)
    vae = G._load_vae(Path(args.vae), dev)
    sf = torch.load(args.scale_factor, map_location="cpu", weights_only=False).float()
    seg, idx = C._load_seg(str(dev))

    imgs = generate_disease(unet, vae, sf, EFFUSION, args.n, args.w, "null", args.steps, args.seed, dev)
    res = [EC.score_decoded_array(seg, idx, _g512(im), str(dev)) for im in imgs]
    # failed segmentations return a minimal dict lacking these keys → count as NaN (excluded from rate)
    eff = np.array([float(r["effusion_present"]) if "effusion_present" in r else np.nan for r in res], dtype=float)
    cp = np.array([r.get("cp_max_angle", float("nan")) for r in res], dtype=float)
    valid = np.isfinite(cp)
    blunt_rate = float(np.nanmean(eff))

    OUT.mkdir(parents=True, exist_ok=True)
    line = (f"v4-60k effusion blunt-rate sanity (n={args.n}, w={args.w}, effusion-only):\n"
            f"  blunt-rate (effusion_present, CP>80°) = {blunt_rate*100:.1f}%   PASS-mark >= 80%  -> "
            f"{'PASS' if blunt_rate >= 0.80 else 'FAIL'}\n"
            f"  CP max-angle: median={np.nanmedian(cp):.1f}  valid={int(valid.sum())}/{args.n}\n")
    (OUT / "effusion_blunt_rate.txt").write_text(line)
    print(line)

    th, cols, rows = 200, 6, 2
    sheet = Image.new("L", (cols * th, rows * th), 0)
    order = np.argsort(-np.nan_to_num(cp, nan=-1))  # most-blunted first
    for k in range(min(cols * rows, args.n)):
        sheet.paste(imgs[int(order[k])].convert("L").resize((th, th)), ((k % cols) * th, (k // cols) * th))
    sheet.save(OUT / "effusion_panel.png")
    print(f"wrote {OUT}/effusion_blunt_rate.txt and effusion_panel.png")


if __name__ == "__main__":
    main()
