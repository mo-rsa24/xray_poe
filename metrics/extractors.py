"""Heart-size & pleural-blunting extractors (Grad-CAM proxy, Option B — locked 2026-06-19).

Two scalar measurements per chest X-ray, read off the same fine-tuned 2-head
DenseNet-121 that produces the quantitative presence scores — no new model, no new
dependency. Using one instrument for both the presence gate (H1) and the coupling
measurement is the property Exp6 needs: the joint is measured by the model that gates
the marginals.

  heart_size   Grad-CAM on the cardio head → bounding box of the activated region
               (cam >= 0.5*max) → box_width / image_width. A cardiothoracic-ratio
               proxy; real images sit ~0.3–0.7, higher = larger heart.
  blunting     Grad-CAM on the effusion head → mean activation in the bottom-lateral
               quadrants (bottom 40% × outer 30% each side — the costophrenic angles).
               0.0–0.5, higher = more pleural blunting / fluid.

Both maps come from `metrics.grad_cam_utils.GradCAM`, which carries the top-border
artifact fix (see that module) — without it the effusion CAM's apical glow anchored
the normalisation and added noise to the basal blunting read.

The coupling Exp6 must capture
    In real both-disease images, heart_size and blunting rise *together* with disease
    severity (a dilated heart and basal fluid co-occur in decompensated cardiac
    failure). Named here as the **cardio–effusion severity coupling**: the positive
    covariation of (heart_size, blunting). PoE composition multiplies two independently
    trained single-disease conditionals, so it has no mechanism to reproduce this joint
    covariation — measuring whether it does is the point of Exp6. `coupling_correlation`
    reports the Pearson r that names it quantitatively.

CLI:
    # single image
    python -m metrics.extractors --image data/nih/images/00000001_000.png

    # batch → JSON arrays (used by Exp6)
    python -m metrics.extractors --dir data/nih/images/ --n 100 --out results/extractor_sample.json

    # validation: cardio vs normal (heart_size separates), effusion vs normal (blunting)
    python -m metrics.extractors --dir data/nih/images_cardio_only/ --n 50 --out /tmp/cardio_ext.json
    python -m metrics.extractors --dir data/nih/images_normal/      --n 50 --out /tmp/normal_ext.json

    # name the coupling: Pearson r of (heart_size, blunting) on the both set
    python -m metrics.extractors --dir data/nih/images/ --n 200 --coupling

Public API:
    extract(image_path) -> dict[str, float]              # {"heart_size", "blunting"}
    extract_batch(paths) -> dict[str, list[float]]       # {"heart_size":[...], "blunting":[...]}
    coupling_correlation(paths) -> dict                  # pearson r + n
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch / torchxrayvision live behind metrics.grad_cam_utils; it is imported lazily
# inside the engine/extract functions so the numeric core (heart_size_from_cam,
# blunting_from_cam, coupling math) stays import-light and CPU-testable.
_CKPT_PATH = "ckpts/presence_classifier_finetuned.pt"

# blunting ROI: bottom 40% (rows >= 0.6 H), outer 30% on each side (costophrenic angles)
_ROI_TOP_FRAC = 0.6
_ROI_SIDE_FRAC = 0.3

# heart-size: horizontal extent of cardio activation measured within the cardiac
# silhouette band (mid-thorax), mirroring how the cardiothoracic ratio is read off a
# film. Restricting to this band is the fix for the full-image bbox failing: the cardio
# head fires diffusely (apex-to-base) on normals, so a whole-image width is *wider* for
# normals than for true cardiomegaly; measuring width only across the cardiac level
# recovers the correct ordering (cardio > normal by ~0.12) and is the right CTR proxy.
_HEART_BAND_TOP = 0.35
_HEART_BAND_BOT = 0.75


# ── scalar measurements from a CAM ────────────────────────────────────────────────

def heart_size_from_cam(cam_c: np.ndarray, frac: float = 0.5) -> float:
    """Width of the cardio activation across the cardiac band / image width (CTR proxy).

    Horizontal extent of `cam_c >= frac*max` measured only within the cardiac silhouette
    band (rows `_HEART_BAND_TOP`–`_HEART_BAND_BOT`). Real images sit ~0.3–0.7; higher =
    larger heart.
    """
    if cam_c.max() <= 0:                        # no activation → no measurable heart
        return 0.0
    h, w = cam_c.shape
    band = cam_c[int(h * _HEART_BAND_TOP):int(h * _HEART_BAND_BOT)]
    mask = band >= frac * cam_c.max()
    if not mask.any():
        return 0.0
    cols = np.where(mask.any(axis=0))[0]
    return float((cols.max() - cols.min() + 1) / w)


def blunting_from_cam(cam_e: np.ndarray) -> float:
    """Mean effusion-head activation in the bottom-lateral (costophrenic) quadrants."""
    h, w = cam_e.shape
    top = int(h * _ROI_TOP_FRAC)
    side = int(w * _ROI_SIDE_FRAC)
    left = cam_e[top:, :side]
    right = cam_e[top:, w - side:]
    return float(np.concatenate([left.ravel(), right.ravel()]).mean())


# ── engine (load model once) ──────────────────────────────────────────────────────

def _build_engine(ckpt_path: str = _CKPT_PATH, device: str = "cpu"):
    from metrics.grad_cam_utils import GradCAM, load_model  # lazy: pulls in torch
    model, cardio_idx, effusion_idx = load_model(ckpt_path, device)
    return GradCAM(model, device), cardio_idx, effusion_idx


# module-level lazy singleton so `extract()` is usable as a one-liner
_ENGINE: tuple | None = None


def _get_engine(ckpt_path: str = _CKPT_PATH, device: str | None = None):
    global _ENGINE
    if _ENGINE is None:
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _ENGINE = _build_engine(ckpt_path, device)
    return _ENGINE


# ── public extraction API ─────────────────────────────────────────────────────────

def extract(image_path: str | Path, engine: tuple | None = None) -> dict[str, float]:
    """heart_size and blunting scalars for a single image."""
    from metrics.grad_cam_utils import preprocess  # lazy: pulls in torch
    eng, cardio_idx, effusion_idx = engine or _get_engine()
    model_in, _ = preprocess(image_path)
    cam_c = eng.cam(model_in, cardio_idx)
    cam_e = eng.cam(model_in, effusion_idx)
    return {"heart_size": heart_size_from_cam(cam_c), "blunting": blunting_from_cam(cam_e)}


def extract_batch(
    paths: Sequence[str | Path],
    engine: tuple | None = None,
    progress: bool = False,
) -> dict[str, list[float]]:
    """heart_size and blunting arrays over a list of images (model loaded once)."""
    eng = engine or _get_engine()
    heart, blunt = [], []
    for i, p in enumerate(paths):
        try:
            r = extract(p, engine=eng)
        except Exception as exc:                       # corrupt/unreadable → NaN, keep going
            print(f"  [warn] skipping {Path(p).name}: {exc}")
            r = {"heart_size": float("nan"), "blunting": float("nan")}
        heart.append(r["heart_size"])
        blunt.append(r["blunting"])
        if progress and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(paths)} ...")
    return {"heart_size": heart, "blunting": blunt}


def coupling_correlation(paths: Sequence[str | Path], engine: tuple | None = None) -> dict:
    """Pearson r between heart_size and blunting — the cardio–effusion severity coupling."""
    out = extract_batch(paths, engine=engine)
    h = np.asarray(out["heart_size"], dtype=np.float64)
    b = np.asarray(out["blunting"], dtype=np.float64)
    ok = np.isfinite(h) & np.isfinite(b)
    h, b = h[ok], b[ok]
    if len(h) < 2 or h.std() == 0 or b.std() == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(h, b)[0, 1])
    return {
        "pearson_r": r,
        "n": int(len(h)),
        "heart_size_mean": float(h.mean()) if len(h) else float("nan"),
        "blunting_mean": float(b.mean()) if len(b) else float("nan"),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────────

def _gather_paths(dir_path: str | Path, n: int | None) -> list[Path]:
    d = Path(dir_path)
    if not d.is_dir():
        raise SystemExit(f"directory does not exist: {d}")
    paths = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no PNG/JPG images in {d}")
    return paths[:n] if n else paths


def _summary(arr: list[float]) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(len(a))}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Heart-size & pleural-blunting Grad-CAM extractors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", default=None, help="Single image → print both scalars")
    p.add_argument("--dir", default=None, help="Directory of images → batch")
    p.add_argument("--n", type=int, default=None, help="Cap images from --dir")
    p.add_argument("--coupling", action="store_true",
                   help="Also report Pearson r of (heart_size, blunting)")
    p.add_argument("--out", default=None, help="Write results JSON here")
    p.add_argument("--ckpt", default=_CKPT_PATH)
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    args = p.parse_args()

    if not args.image and not args.dir:
        p.error("provide --image or --dir")

    engine = _get_engine(args.ckpt, args.device)

    if args.image:
        r = extract(args.image, engine=engine)
        print(f"{Path(args.image).name}: heart_size={r['heart_size']:.3f}  blunting={r['blunting']:.3f}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"image": str(args.image), **r}, indent=2))
            print(f"Results → {args.out}")
        return

    paths = _gather_paths(args.dir, args.n)
    print(f"Extracting from {len(paths)} images in {args.dir} ...")
    res = extract_batch(paths, engine=engine, progress=True)

    hs, bl = _summary(res["heart_size"]), _summary(res["blunting"])
    print(f"  heart_size: mean={hs['mean']:.3f} ± {hs['std']:.3f}  (n={hs['n']})")
    print(f"  blunting:   mean={bl['mean']:.3f} ± {bl['std']:.3f}  (n={bl['n']})")

    out: dict = {
        "dir": args.dir,
        "n": len(paths),
        "heart_size": res["heart_size"],
        "blunting": res["blunting"],
        "summary": {"heart_size": hs, "blunting": bl},
    }
    if args.coupling:
        c = coupling_correlation(paths, engine=engine)
        out["coupling"] = c
        print(f"  coupling (cardio–effusion severity): pearson_r={c['pearson_r']:+.3f}  (n={c['n']})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"Results → {args.out}")


if __name__ == "__main__":
    main()
