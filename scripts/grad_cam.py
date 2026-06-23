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
import torch.nn.functional as F
from PIL import Image, ImageDraw

try:
    import torchxrayvision as xrv
except ImportError:
    raise ImportError("TorchXRayVision not installed. Run: pip install torchxrayvision") from None

# PresenceClassifier lives in scripts/finetune_classifier.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finetune_classifier import PresenceClassifier  # noqa: E402

_CKPT_PATH = "ckpts/presence_classifier_finetuned.pt"
_DISPLAY = 512            # overlay resolution
_ALPHA = 0.5             # heatmap blend strength


# ── model ──────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PresenceClassifier(backbone_weights=ckpt["backbone"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, int(ckpt["cardio_idx"]), int(ckpt["effusion_idx"])


# ── preprocessing (must mirror metrics/presence_classifier._preprocess) ─────────

def preprocess(path: str | Path):
    """Return (model_input (1,1,224,224) float tensor, display (512,512) [0,1] array)."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)                          # (H, W)
    arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)    # (1, H, W) in [-1024, 1024]
    arr = xrv.datasets.XRayCenterCrop()(arr)                       # (1, S, S)
    arr = xrv.datasets.XRayResizer(224)(arr)                       # (1, 224, 224)
    arr = arr.copy()
    model_in = torch.from_numpy(arr).unsqueeze(0)                  # (1, 1, 224, 224)

    # display image: same FOV, min-max to [0,1], upsampled to 512
    disp = arr[0]
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
    disp = np.array(
        Image.fromarray((disp * 255).astype(np.uint8)).resize((_DISPLAY, _DISPLAY), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return model_in, disp


# ── Grad-CAM ────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._act = {}
        model.features.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inp, out):
        out.retain_grad()
        self._act["v"] = out

    def cam(self, x: torch.Tensor, head_idx: int) -> np.ndarray:
        """Return a (512,512) [0,1] CAM for one head."""
        x = x.to(self.device)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                       # fires the forward hook
        act = self._act["v"]                         # (1, C, h, w)
        logits[0, head_idx].backward()
        grad = act.grad                              # (1, C, h, w)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))      # (1, 1, h, w)
        cam = F.interpolate(cam, size=(_DISPLAY, _DISPLAY), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


# ── overlay ──────────────────────────────────────────────────────────────────────

def _bbox_from_cam(cam: np.ndarray, frac: float = 0.5):
    mask = cam >= frac * cam.max()
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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
        bc = _bbox_from_cam(cam_c)
        be = _bbox_from_cam(cam_e)
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
    ap.add_argument("--out", required=True, help="Output PNG path")
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
