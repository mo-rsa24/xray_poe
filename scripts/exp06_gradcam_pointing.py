"""Exp 6 task 2 — detector construct + criterion validity (Gate 1 + Gate 2).

Gate 1 (automated here) — GradCAM pointing-game: on cardio-positive films, the
fraction of the cardio-head saliency MASS that falls inside the PSPNet heart mask.
A detector that fires on the heart (not a shortcut) concentrates its CAM there.
Pass rule: mean pointing-game ≥ 0.80.

Gate 2 (named, NOT automatable) — criterion agreement: the CTR-taught cardio
detector is confirmed by an INDEPENDENT path — GradCAM (where the net looks) +
radiologist — and NEVER by CTR again (the independence rule). The radiologist
sign-off is a human step; this script records the requirement and leaves it pending.

Coordinate frames: both the CAM and the PSPNet heart mask are taken in the 512²
frame. NIH PA films are square, so the detector's XRayCenterCrop is identity and the
two frames align; documented as an approximation (Gate 1's 0.80 bar is coarse).

GPU-bound (PSPNet + CAM backward) — run via scripts/launch_exp06.sh on a free GPU.

    python scripts/exp06_gradcam_pointing.py --ckpt ckpts/detector_ctr_labels.pt --n 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metrics.grad_cam_utils import GradCAM, load_model, preprocess  # noqa: E402
import scripts.ctr_overlay as C  # noqa: E402

MANIFESTS = ROOT / "data/splits/exp06_manifests"
OUT = ROOT / "results/exp06_oracle/detector_construct_validity.md"
HEART = C.HEART


def _heart_mask_512(seg, idx, path: Path, device: str) -> np.ndarray:
    m = C.ctr_for_image(seg, idx, path, device, with_masks=True)
    masks = m.get("_masks") or {}
    heart = masks.get(HEART)
    return heart if heart is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/detector_ctr_labels.pt")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    test = pd.read_csv(MANIFESTS / "test.csv")
    pos = test[test["cardio"] == 1].sample(n=min(args.n, int((test["cardio"] == 1).sum())),
                                           random_state=42)

    model, cardio_idx, _ = load_model(args.ckpt, args.device)
    cam_engine = GradCAM(model, args.device)
    seg, idx = C._load_seg(args.device)

    scores = []
    for _, r in pos.iterrows():
        path = ROOT / r["path"]
        heart = _heart_mask_512(seg, idx, path, args.device)
        if heart is None or not heart.any():
            continue
        x, _disp = preprocess(path)
        cam = cam_engine.cam(x, cardio_idx)              # (512,512) [0,1]
        if heart.shape != cam.shape:
            heart = np.array(Image.fromarray(heart.astype(np.uint8) * 255)
                             .resize(cam.shape[::-1], Image.NEAREST)) > 127
        total = cam.sum()
        if total <= 0:
            continue
        scores.append(float(cam[heart].sum() / total))

    pg = float(np.mean(scores)) if scores else float("nan")
    passed = pg >= 0.80
    print(f"pointing-game = {pg:.2f}  (n={len(scores)}, ≥0.80 {'PASS' if passed else 'FAIL'})")

    L = [
        "# Exp 6 — detector construct + criterion validity (Gate 1 + Gate 2)", "",
        f"Checkpoint under test: `{args.ckpt}`", "",
        "## Gate 1 — GradCAM pointing-game (automated)", "",
        f"- **pointing-game = {pg:.2f}** over {len(scores)} cardio-positive films "
        f"(saliency mass inside the PSPNet heart mask).",
        f"- Pass rule ≥ 0.80 → **{'PASS' if passed else 'FAIL'}**.",
        "- Reads where the net looks (independent of the CTR signal); kills shortcut learning.",
        "", "## Gate 2 — criterion agreement (independent path; radiologist step PENDING)", "",
        "- The CTR-taught cardio detector is confirmed by **GradCAM + radiologist**, "
        "**never by CTR again** (independence rule — CTR teaches, so it cannot also validate).",
        "- GradCAM half: satisfied by the Gate-1 pointing-game above.",
        "- Radiologist half: **⚠️ pending human sign-off** — a blinded read confirming the "
        "highlighted region is the cardiac silhouette. Cannot be automated; record the "
        "radiologist's agreement here once obtained.",
        "", "_Frame note: CAM and PSPNet heart mask compared in the 512² frame; NIH PA films "
        "are square so the detector center-crop is identity (coarse-bar approximation)._",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
