"""Cardiothoracic ratio (CTR) — a classifier-INDEPENDENT heart-size instrument.

Segments Heart + Left/Right Lung with the torchxrayvision ChestX-Det PSPNet
(pretrained, anatomical segmentation — NOT the presence classifier), then:
    heart_width  = max horizontal extent of the Heart mask (over rows)
    thorax_width = max horizontal extent of (Left ∪ Right Lung) mask (over rows)
    CTR          = heart_width / thorax_width
CTR > ~0.5 is the classic cardiomegaly sign. This adjudicates the pilot question
the presence classifier + Grad-CAM could not: do the 100k generated cardiomegaly
images actually have enlarged hearts (→ classifier mis-calibration on synthetic)
or normal-sized hearts (→ the LDM under-renders the cardiac silhouette)?

Substitutes for the plan's HybridGNet/CheXmask CTR — same geometry, no torch_geometric
/ no GB-scale mask download. The geometry is reusable on any heart/lung masks.

Usage (repeatable --group label:dir:n; --pgroup label:partition_group:n for the
full-corpus groups like normal):
    python scripts/ctr_overlay.py \
        --group cardio_real:data/nih/images_cardio_only:60 \
        --group both_real:data/nih/images_both:60 \
        --pgroup normal_real:normal:60 \
        --group cardio_gen100k:outputs/ckpt_eval/0100000/single/cardiomegaly:100 \
        --group effusion_gen100k:outputs/ckpt_eval/0100000/single/effusion:60 \
        --out-json results/ctr_calibration.json \
        --out-fig figures/instrument_calibration/ctr_overlay_real.png --fig-n 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image, ImageDraw, ImageFont

HEART = "Heart"
LLUNG = "Left Lung"
RLUNG = "Right Lung"
_PARTITION = "data/nih/partition.parquet"
_FULL_IMAGES = "data/nih/images"

# ── calibrated decision thresholds ───────────────────────────────────────────────
# Validated by researcher visual sign-off on the Exp01 failure sheets (2026-06-25);
# the support verdict is robust to ±0.05 / ±10° perturbation (see calibration.md).
TAU_OPACITY = 0.55        # peri-cardiac opacity ≥ this → cardiac border erased
TAU_HEART_CONF = 0.45     # PSPNet mean heart-mask confidence floor
CP_BLUNT_DEG = 80.0       # costophrenic wedge angle > this → blunted
CP_SEV_BINS = (60.0, 90.0, 120.0)   # sharp | mild | moderate | severe

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _font(size: int) -> ImageFont.ImageFont:
    """A legible bold TrueType font (DejaVuSans-Bold), cached by size; falls back
    to PIL's bitmap default only if no TTF is found."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    import glob
    try:
        import matplotlib
        mpl = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf")
    except Exception:
        mpl = ""
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", mpl,
              *glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True)]:
        if p and Path(p).exists():
            try:
                f = ImageFont.truetype(p, size)
                _FONT_CACHE[size] = f
                return f
            except Exception:
                pass
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


# ── segmentation ───────────────────────────────────────────────────────────────

def _load_seg(device: str):
    seg = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    idx = {t: i for i, t in enumerate(seg.targets)}
    return seg, idx


def _to_input(path: Path, device: str) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((512, 512))
    arr = np.array(img, dtype=np.float32)
    arr = xrv.datasets.normalize(arr, maxval=255)          # → [-1024, 1024]
    t = torch.from_numpy(arr)[None, None].to(device)       # (1,1,512,512)
    return t


def _mask(prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return prob > thr


def _max_width(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (width, row, x_lo, x_hi) of the widest row of a binary mask."""
    best = (0, -1, 0, 0)
    rows = np.where(mask.any(axis=1))[0]
    for y in rows:
        xs = np.where(mask[y])[0]
        w = int(xs[-1] - xs[0] + 1)
        if w > best[0]:
            best = (w, int(y), int(xs[0]), int(xs[-1]))
    return best


# ── costophrenic-angle read (effusion instrument) ────────────────────────────────

def _cp_read(lung: np.ndarray, r: int = 28, n_ang: int = 360) -> dict:
    """Costophrenic-angle read off a SINGLE lung mask (geometric, mask-only).

    Locates the inferolateral vertex (the costophrenic recess — down + away from
    the image midline), then measures the wedge of lung subtended on a circle of
    radius ``r`` about that vertex. A sharp CP angle subtends a narrow wedge;
    pleural fluid fills/rounds the recess → a wider wedge. ``angle_deg`` is that
    wedge in degrees (frac-of-circle-inside-lung × 360).

    Thresholds here are PROVISIONAL — the hand-calibration task tightens them.
    """
    ys, xs = np.where(lung)
    if len(ys) < 50:
        return {"angle_deg": float("nan"), "blunted": None, "severity": "n/a", "vertex": None}
    H, W = lung.shape
    cx = float(xs.mean())
    score = (ys - xs) if cx < W / 2 else (ys + xs)   # bottom-outer corner
    k = int(np.argmax(score))
    xv, yv = int(xs[k]), int(ys[k])
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    px = np.clip((xv + r * np.cos(ang)).astype(int), 0, W - 1)
    py = np.clip((yv + r * np.sin(ang)).astype(int), 0, H - 1)
    angle_deg = float(lung[py, px].mean() * 360.0)
    b1, b2, b3 = CP_SEV_BINS
    if   angle_deg < b1: sev = "sharp"
    elif angle_deg < b2: sev = "mild"
    elif angle_deg < b3: sev = "moderate"
    else:                sev = "severe"
    return {"angle_deg": round(angle_deg, 1), "blunted": bool(angle_deg > CP_BLUNT_DEG),
            "severity": sev, "vertex": [xv, yv], "radius": r}


def _peri_cardiac_opacity(gray: np.ndarray, heart: np.ndarray, lungs: np.ndarray) -> float:
    """Whiteness immediately lateral to the heart, normalized to THIS film's own
    references. Aerated lung is dark on a CXR; pleural fluid is bright.

        dark   = 20th-pctile intensity inside the lung masks (aerated baseline)
        bright = median intensity inside the heart mask (solid-tissue baseline)
        opacity = (mean band intensity − dark) / (bright − dark)   clipped to [0,1]

    ~0 → crisp cardiac border (dark lung abuts the heart, CTR measurable);
    ~1 → whiteout (the band is as bright as the heart → silhouette sign, the
    cardiac border is erased and CTR is not trustworthy). This is the operational
    "lots of whiteness around the heart" → entangled signal.
    """
    rows = np.where(heart.any(axis=1))[0]
    if len(rows) == 0 or not lungs.any():
        return float("nan")
    W = gray.shape[1]
    band = []
    for y in rows:
        xs = np.where(heart[y])[0]
        xl, xr = int(xs[0]), int(xs[-1])
        band.append(gray[y, max(0, xl - 25):max(0, xl - 3)])
        band.append(gray[y, min(W, xr + 3):min(W, xr + 25)])
    band = np.concatenate([b for b in band if b.size]) if any(b.size for b in band) else np.array([])
    if band.size < 20:
        return float("nan")
    dark = float(np.percentile(gray[lungs], 20))
    bright = float(np.median(gray[heart]))
    if bright - dark < 1:
        return float("nan")
    return float(np.clip((float(band.mean()) - dark) / (bright - dark), 0.0, 1.0))


def _border_visible(opacity: float, heart_conf: float,
                    tau_op: float = TAU_OPACITY, tau_conf: float = TAU_HEART_CONF) -> bool:
    """Gate-0: is the cardiac border traceable (CTR measurable)? True when the
    peri-cardiac band is NOT whited-out and PSPNet is confident about the heart.
    Thresholds PROVISIONAL — tightened in the hand-calibration task."""
    if not np.isfinite(opacity):
        return False
    return bool(opacity < tau_op and heart_conf >= tau_conf)


# ── CTR per image ────────────────────────────────────────────────────────────────

def ctr_for_image(seg, idx, path: Path, device: str) -> dict:
    with torch.no_grad():
        logits = seg(_to_input(path, device))
        probs = torch.sigmoid(logits)[0].cpu().numpy()     # (n_targets, 512, 512)
    heart = _mask(probs[idx[HEART]])
    llung, rlung = _mask(probs[idx[LLUNG]]), _mask(probs[idx[RLUNG]])
    lungs = llung | rlung
    gray = np.asarray(Image.open(path).convert("L").resize((512, 512)), dtype=np.float32)
    if not heart.any() or not lungs.any():
        return {"ctr": float("nan"), "heart_w": 0, "thorax_w": 0, "ok": False,
                "heart_conf": 0.0, "peri_cardiac_opacity": float("nan"),
                "border_visible": False,
                "cp_left": _cp_read(llung), "cp_right": _cp_read(rlung)}
    hw, hy, hx0, hx1 = _max_width(heart)
    tw, ty, tx0, tx1 = _max_width(lungs)
    heart_conf = float(probs[idx[HEART]][heart].mean())
    opacity = _peri_cardiac_opacity(gray, heart, lungs)
    return {
        "ctr": float(hw / tw) if tw else float("nan"),
        "heart_w": hw, "thorax_w": tw, "ok": True,
        "heart_conf": round(heart_conf, 3),
        "peri_cardiac_opacity": round(opacity, 3) if np.isfinite(opacity) else float("nan"),
        "border_visible": _border_visible(opacity, heart_conf),
        "cp_left": _cp_read(llung), "cp_right": _cp_read(rlung),
        "heart_line": [hx0, hy, hx1, hy],
        "thorax_line": [tx0, ty, tx1, ty],
    }


def _cp_marker(d: ImageDraw.ImageDraw, cp: dict) -> None:
    if not cp or cp.get("vertex") is None:
        return
    x, y = cp["vertex"]
    r = cp.get("radius", 28)
    col = (255, 140, 0) if cp.get("blunted") else (60, 230, 120)  # orange=blunted, green=sharp
    d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3)


def _overlay(path: Path, m: dict) -> Image.Image:
    im = Image.open(path).convert("RGB").resize((512, 512))
    d = ImageDraw.Draw(im)
    if m["ok"]:
        d.line(m["heart_line"], fill=(255, 60, 60), width=4)     # heart width (red)
        d.line(m["thorax_line"], fill=(60, 160, 255), width=4)   # thorax width (blue)
    _cp_marker(d, m.get("cp_left"))                              # CP recesses (green/orange)
    _cp_marker(d, m.get("cp_right"))
    bv = "Y" if m.get("border_visible") else "N"
    cl, cr = m.get("cp_left", {}), m.get("cp_right", {})
    _sev = {"sharp": "S", "mild": "M", "moderate": "Mo", "severe": "Sv", "n/a": "-"}
    def _cp(c): return f"{c.get('angle_deg')}({_sev.get(c.get('severity'), '?')})"
    line1 = f"{path.name[:16]}  CTR={m['ctr']:.3f}  border={bv}"
    line2 = f"opac={m.get('peri_cardiac_opacity')}  CP L={_cp(cl)} R={_cp(cr)}"
    fnt = _font(20)
    d.rectangle([0, 512 - 62, 512, 512], fill=(0, 0, 0))          # taller label band
    d.text((6, 512 - 58), line1, fill=(255, 235, 0), font=fnt)
    d.text((6, 512 - 30), line2, fill=(255, 235, 0), font=fnt)
    return im


def _grid(cells: list[Image.Image], cols: int = 4, cell: int = 512) -> Image.Image:
    rows = (len(cells) + cols - 1) // cols
    g = 6
    canvas = Image.new("RGB", (cols * cell + (cols + 1) * g, rows * cell + (rows + 1) * g), (24, 24, 24))
    for i, im in enumerate(cells):
        r, c = divmod(i, cols)
        tile = im if im.size == (cell, cell) else im.resize((cell, cell), Image.LANCZOS)
        canvas.paste(tile, (g + c * (cell + g), g + r * (cell + g)))
    return canvas


# ── path collection ──────────────────────────────────────────────────────────────

def _from_dir(d: str, n: int, seed: int = 42) -> list[Path]:
    paths = sorted(Path(d).glob("*.png")) + sorted(Path(d).glob("*.jpg"))
    if n < len(paths):
        paths = random.Random(seed).sample(paths, n)
    return paths


def _from_partition(group: str, n: int, seed: int = 42) -> list[Path]:
    import pandas as pd
    df = pd.read_parquet(_PARTITION)
    names = df[df["group"] == group]["Image Index"].tolist()
    names = random.Random(seed).sample(names, min(n, len(names)))
    return [Path(_FULL_IMAGES) / nm for nm in names]


def _summary(ctrs: list[float]) -> dict:
    a = np.array([c for c in ctrs if np.isfinite(c)], dtype=np.float64)
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)), "mean": float(a.mean()), "std": float(a.std()),
        "median": float(np.median(a)), "frac_gt_0.5": float((a > 0.5).mean()),
        "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="CTR via xrv ChestX-Det segmentation")
    p.add_argument("--group", action="append", default=[], help="label:dir:n")
    p.add_argument("--pgroup", action="append", default=[], help="label:partition_group:n")
    p.add_argument("--out-json", default="results/ctr_calibration.json")
    p.add_argument("--out-fig", default=None)
    p.add_argument("--fig-n", type=int, default=12, help="example overlays per group in the fig")
    p.add_argument("--film", default=None, help="single-film mode: path to one PA image")
    p.add_argument("--out-overlay", default=None, help="single-film mode: overlay PNG path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seg, idx = _load_seg(args.device)
    print(f"PSPNet loaded on {args.device}; heart idx={idx[HEART]}")

    if args.film:                                              # single-film sanity mode
        m = ctr_for_image(seg, idx, Path(args.film), args.device)
        cl, cr = m.get("cp_left", {}), m.get("cp_right", {})
        print(f"\nfilm = {args.film}")
        print(f"CTR            = {m['ctr']:.3f}" if np.isfinite(m['ctr']) else "CTR            = NaN (seg failed)")
        print(f"border_visible = {m.get('border_visible')}  (heart_conf={m.get('heart_conf')}, "
              f"peri_cardiac_opacity={m.get('peri_cardiac_opacity')})")
        print(f"CP_blunted_left  = {cl.get('blunted')}  (angle={cl.get('angle_deg')}°, {cl.get('severity')})")
        print(f"CP_blunted_right = {cr.get('blunted')}  (angle={cr.get('angle_deg')}°, {cr.get('severity')})")
        if args.out_overlay:
            Path(args.out_overlay).parent.mkdir(parents=True, exist_ok=True)
            _overlay(Path(args.film), m).save(args.out_overlay)
            print(f"overlay → {args.out_overlay}")
        return

    specs = []
    for g in args.group:
        label, d, n = g.split(":")
        specs.append((label, _from_dir(d, int(n), args.seed), d))
    for g in args.pgroup:
        label, grp, n = g.split(":")
        specs.append((label, _from_partition(grp, int(n), args.seed), f"partition:{grp}"))

    out = {"instrument": "xrv ChestX-Det PSPNet CTR (heart_w / thorax_w)", "groups": {}}
    fig_cells: list[Image.Image] = []
    for label, paths, src in specs:
        per = [ctr_for_image(seg, idx, pth, args.device) for pth in paths]
        ctrs = [m["ctr"] for m in per]
        s = _summary(ctrs)
        out["groups"][label] = {"src": src, **s,
                                "n_failed_seg": int(sum(not m["ok"] for m in per))}
        print(f"{label:20s} n={s.get('n',0):3d}  mean CTR={s.get('mean',float('nan')):.3f}"
              f"  median={s.get('median',float('nan')):.3f}  frac>0.5={s.get('frac_gt_0.5',float('nan')):.2f}")
        if args.out_fig:
            for pth, m in list(zip(paths, per))[: args.fig_n]:
                fig_cells.append(_overlay(pth, m))

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\nResults → {args.out_json}")
    if args.out_fig and fig_cells:
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        _grid(fig_cells, cols=args.fig_n).save(args.out_fig)
        print(f"Overlay grid → {args.out_fig}")


if __name__ == "__main__":
    main()
