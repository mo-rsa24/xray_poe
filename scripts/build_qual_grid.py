"""Qualitative v1->v4 cardio grid, each version at its FAIR deployment checkpoint, scored by the
detector. Proves the version-ladder visually and pairs it with the sensitivity number.

Checkpoints (peak/deployment, NOT the decayed 100k end-state — showing 100k would slander v1):
  v1 40k, v2 20k (its final), v3 40k (peak), v4 60k (canonical). Global scale for v1-v3, per-channel for v4.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/Paper3")
import numpy as np
import torch
from PIL import Image, ImageDraw
import scripts.ctr_overlay as C
from metrics.grad_cam_utils import load_model, preprocess

OUT = Path("results/vae_v2/exhibits"); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
VERS = [
    ("v1", "/tmp/qual/v1_0040000/single/cardiomegaly", "renders cardio,\nbut effusion collapsed"),
    ("v2", "/tmp/qual/v2/single/cardiomegaly",         "cardio killed\n(share too low)"),
    ("v3", "/tmp/qual/v3_40k/single/cardiomegaly",     "both render,\nbut washed-out"),
    ("v4", "/tmp/qual/v4/single/cardiomegaly",         "crisp, all classes\nSELECTED"),
]


def main():
    seg, idx = C._load_seg(dev)
    model, ci, ei = load_model("ckpts/presence_classifier_finetuned.pt", dev)
    th = 200
    n = len(VERS)
    band = 78
    canvas = Image.new("RGB", (n*th, th+band), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    for k, (name, folder, note) in enumerate(VERS):
        files = sorted(Path(folder).glob("*.png"))
        # score each film: detector cardio prob + CTR; pick representative (median CTR)
        scored = []
        for f in files:
            x, _ = preprocess(f); x = x.to(dev)
            with torch.no_grad():
                pc = float(torch.sigmoid(model(x)).squeeze()[ci])
            m = C.ctr_for_image(seg, idx, f, dev)
            ctr = m["ctr"] if m["ok"] and np.isfinite(m["ctr"]) else np.nan
            scored.append((f, pc, ctr))
        probs = [p for _, p, _ in scored]
        mean_p = float(np.mean(probs))
        valid = [(f, p, c) for f, p, c in scored if np.isfinite(c)]
        pick = sorted(valid, key=lambda t: abs(t[2]-np.nanmedian([c for _, _, c in valid])))[0] if valid else (scored[0][0], scored[0][1], np.nan)
        img = Image.open(pick[0]).convert("L").resize((th, th))
        canvas.paste(img, (k*th, band))
        col = (80, 220, 120) if name == "v4" else (235, 235, 235)
        d.text((k*th+6, 4), name, fill=col, font=C._font(22))
        d.text((k*th+6, 30), note, fill=(200, 200, 200), font=C._font(12))
        # detector line under the version label
        pcol = (80, 220, 120) if mean_p >= 0.5 else (235, 150, 90)
        d.text((k*th+6, band-16), f"detector cardio {mean_p:.2f}", fill=pcol, font=C._font(13))
    canvas.save(OUT / "qual_ldm_v1_v4_grid.png")
    print("wrote qual_ldm_v1_v4_grid.png; per-version detector means:",
          {v[0]: None for v in VERS})


if __name__ == "__main__":
    main()
