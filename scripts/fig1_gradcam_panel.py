"""Fig 1 — GradCAM panel: real vs synth, correct vs incorrect (D3).

Supervisor layout: columns = Normal / Effusion / Cardio / Comorbid; rows = 2 correctly
classified + 2 incorrectly classified; real and synth side-by-side. Shows WHERE the
frozen DenseNet looks (corner-artifact-fixed grad_cam_utils). For correct cases the
saliency should sit on the right anatomy (heart for cardio, costophrenic for effusion);
incorrect cases expose the shortcut / comorbidity confusion.

Per source (real, synth-<tag>) a 4-col × 4-row panel is written. "Correct/incorrect" is
judged by the relevant head's prediction (Youden threshold) vs the category's expected
label (normal→both neg, cardio→cardio+, effusion→effusion+, comorbid→both+).

GPU (GradCAM backward + classifier) — run when the GPU is free of training.

    python scripts/fig1_gradcam_panel.py --sources real before after
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metrics.grad_cam_utils import GradCAM, load_model, preprocess  # noqa: E402

MANIFEST = ROOT / "data/splits/exp06_manifests/test.csv"
SYNTH = ROOT / "results/synth_sets"
STRATUM2CAT = {"normal": "normal", "cardio_only": "cardio",
               "effusion_only": "effusion", "both_separable": "comorbid"}
CATS = ["normal", "effusion", "cardio", "comorbid"]          # column order (supervisor)
EXPECT = {"normal": (0, 0), "cardio": (1, 0), "effusion": (0, 1), "comorbid": (1, 1)}
# which head drives the saliency we render for each column
HEAD_FOR = {"normal": 0, "cardio": 0, "effusion": 1, "comorbid": 0}


def _paths_for(source, cat, ckpt_lab=None):
    if source == "real":
        df = pd.read_csv(MANIFEST); df["category"] = df["stratum"].map(STRATUM2CAT)
        return [str(ROOT / p) for p in df[df["category"] == cat]["path"]]
    tag = source
    cond = cat if cat != "normal" else None
    if cat == "normal":
        return []   # synth sets are cardio/effusion/comorbid only; no synth 'normal'
    return [str(p) for p in sorted((SYNTH / tag / cat).glob("*.png"))]


def _classify_one(model, ci, ei, path, device):
    import torch
    x, _ = preprocess(path)
    with torch.no_grad():
        p = torch.sigmoid(model(x.to(device)))[0].cpu().numpy()
    return p[ci], p[ei]


def _overlay(disp, cam):
    import matplotlib.cm as cm
    rgb = np.stack([disp] * 3, -1)
    heat = cm.jet(cam)[..., :3]
    return np.clip(0.55 * rgb + 0.45 * heat, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["real", "before", "after"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--scan", type=int, default=40, help="images to scan per cell for correct/incorrect")
    ap.add_argument("--ckpt", default="ckpts/presence_classifier_finetuned.pt",
                    help="classifier to GradCAM (Fig 3: an augmented arm, e.g. ckpts/aug_poe_s01.pt)")
    ap.add_argument("--label", default="", help="suffix for output filenames (e.g. _aug_poe → fig1_gradcam_<src>_aug_poe.png)")
    args = ap.parse_args()
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ci, ei = load_model(args.ckpt, device)
    ck = torch.load(args.ckpt, map_location="cpu")
    thr = {0: ck["youden_threshold_cardio"], 1: ck["youden_threshold_effusion"]}
    cam_engine = GradCAM(model, device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for source in args.sources:
        fig, axes = plt.subplots(4, 4, figsize=(13, 13))
        for col, cat in enumerate(CATS):
            paths = _paths_for(source, cat)[: args.scan]
            head = HEAD_FOR[cat]
            exp = EXPECT[cat][head]
            correct, incorrect = [], []
            for p in paths:
                pc, pe = _classify_one(model, ci, ei, p, device)
                prob = pc if head == 0 else pe
                pred = int(prob >= thr[head])
                (correct if pred == exp else incorrect).append(p)
            chosen = (correct[:2] + ["__none__"] * 2)[:2] + (incorrect[:2] + ["__none__"] * 2)[:2]
            for row, p in enumerate(chosen):
                ax = axes[row][col]; ax.axis("off")
                if p == "__none__":
                    ax.text(0.5, 0.5, "—", ha="center", va="center"); continue
                x, disp = preprocess(p)
                cam = cam_engine.cam(x, head)
                ax.imshow(_overlay(disp, cam))
                tag = "correct" if row < 2 else "INCORRECT"
                ax.set_title(f"{cat} · {tag}", fontsize=8)
        for r, lab in enumerate(["correct", "correct", "incorrect", "incorrect"]):
            axes[r][0].set_ylabel(lab, fontsize=9)
        fig.suptitle(f"Fig 1 — GradCAM · source={source} "
                     f"(cols: Normal/Effusion/Cardio/Comorbid; rows: 2 correct + 2 incorrect)",
                     fontsize=11)
        fig.tight_layout()
        out = Path(ROOT / args.out_dir) / f"fig1_gradcam_{source}{args.label}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
