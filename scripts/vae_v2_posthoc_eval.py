"""Post-hoc eval for VAE-v2 — closes plan 03-vae-retrain tasks 2 & 4.

Runs encode→decode through the fine-tuned checkpoint(s) and the baseline,
computes |ΔCTR| on the held-out val split, and saves side-by-side overlay grids
to results/vae_v2/sanity/<tag>/.

Usage:
    python scripts/vae_v2_posthoc_eval.py \
        --checkpoints ckpts/vae_v2/vae_v2_step0030000.pt \
                      ckpts/vae_v2/vae_v2_step0050000.pt \
                      ckpts/vae_v2/vae_v2_step0075000.pt \
        --baseline    ckpts/vae_step0025000.pt \
        --manifest    data/vae_v2_diet/val_manifest.csv \
        --out         results/vae_v2/sanity \
        --n-per-class 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from vae.config import VAEConfig
from vae.model import VAE

# ---------------------------------------------------------------------------
# Instrument loading
# ---------------------------------------------------------------------------

def _load_seg(device: str):
    import scripts.ctr_overlay as co
    return co._load_seg(device)


def _gray512(tensor_1chw: torch.Tensor) -> np.ndarray:
    """Convert 1×H×W tensor ([-1,1] or [0,1]) → uint8 HxW numpy."""
    t = tensor_1chw.squeeze().float().cpu()
    t = (t + 1.0) / 2.0  # [-1,1] → [0,1]
    t = t.clamp(0, 1)
    return (t.numpy() * 255).astype(np.uint8)


def _load_vae(ckpt_path: str, device: str) -> VAE:
    cfg = VAEConfig(input_resolution=512, use_checkpoint=False)
    vae = VAE(cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("ema", ckpt))
    vae.load_state_dict(state, strict=False)
    return vae


# ---------------------------------------------------------------------------
# Per-image CTR scoring
# ---------------------------------------------------------------------------

def _score(seg, idx, gray: np.ndarray, device: str) -> dict:
    from metrics.eval_checkpoint import score_decoded_array
    return score_decoded_array(seg, idx, gray, device)


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def _draw_label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 0))
    return img


def _make_grid(images: list[Image.Image], ncols: int = 4) -> Image.Image:
    w, h = images[0].size
    nrows = (len(images) + ncols - 1) // ncols
    grid = Image.new("RGB", (w * ncols, h * nrows), (30, 30, 30))
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        grid.paste(img.convert("RGB"), (c * w, r * h))
    return grid


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

LABEL_NAMES = {0: "normal", 1: "cardio", 2: "effusion"}
RESIZE = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def run_eval(
    checkpoints: list[str],
    baseline: str,
    manifest: str,
    out_dir: Path,
    n_per_class: int,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading PSPNet on {device}…")
    seg, idx = _load_seg(device)

    df = pd.read_csv(manifest)
    # sample n_per_class per label
    samples = []
    for label in [0, 1, 2]:
        rows = df[df["label"] == label].sample(
            n=min(n_per_class, (df["label"] == label).sum()), random_state=42)
        samples.append(rows)
    sample_df = pd.concat(samples).reset_index(drop=True)
    print(f"Val films sampled: {len(sample_df)} ({n_per_class}×3 classes)")

    # load images
    imgs_t = []
    for _, row in sample_df.iterrows():
        img = Image.open(row["image_path"]).convert("L")
        imgs_t.append(RESIZE(img))
    imgs_t = torch.stack(imgs_t)  # N×1×512×512

    all_ckpts = [baseline] + list(checkpoints)
    delta_ctrs: dict[str, list[float]] = {}

    for ckpt_path in all_ckpts:
        tag = Path(ckpt_path).stem
        print(f"\n── {tag} ──")
        vae = _load_vae(ckpt_path, device)

        recons = []
        ctrs_orig, ctrs_recon = [], []

        with torch.no_grad():
            for i, x in enumerate(imgs_t):
                x_d = x.unsqueeze(0).to(device)
                mu, _ = vae.encode_moments(x_d)
                recon = vae.decode(mu)

                gray_orig  = _gray512(x.unsqueeze(0))
                gray_recon = _gray512(recon)

                s_orig  = _score(seg, idx, gray_orig,  device)
                s_recon = _score(seg, idx, gray_recon, device)

                ctr_o = s_orig["ctr"]  if s_orig["ok"]  else float("nan")
                ctr_r = s_recon["ctr"] if s_recon["ok"] else float("nan")
                ctrs_orig.append(ctr_o)
                ctrs_recon.append(ctr_r)

                label_name = LABEL_NAMES[int(sample_df.iloc[i]["label"])]
                delta = abs(ctr_r - ctr_o) if (s_orig["ok"] and s_recon["ok"]) else float("nan")
                print(f"  [{i:02d}] {label_name:8s}  ctr_orig={ctr_o:.3f}  "
                      f"ctr_recon={ctr_r:.3f}  |Δ|={delta:.3f}")

                # build side-by-side panel: orig | recon
                orig_pil  = Image.fromarray(gray_orig).resize((256, 256))
                recon_pil = Image.fromarray(gray_recon).resize((256, 256))
                orig_pil  = _draw_label(orig_pil,  f"orig  CTR={ctr_o:.3f}")
                recon_pil = _draw_label(recon_pil, f"recon CTR={ctr_r:.3f}  |Δ|={delta:.3f}")
                panel = Image.new("RGB", (512, 256))
                panel.paste(orig_pil, (0, 0))
                panel.paste(recon_pil.convert("RGB"), (256, 0))
                recons.append(_draw_label(panel, f"{label_name}  {tag}"))

        delta_arr = np.array([abs(r - o) for r, o in zip(ctrs_recon, ctrs_orig)
                               if np.isfinite(r) and np.isfinite(o)])
        med_delta = float(np.median(delta_arr)) if len(delta_arr) else float("nan")
        delta_ctrs[tag] = delta_arr.tolist()
        print(f"  delta_ctr_median = {med_delta:.4f}  "
              f"({'PASS ✓' if med_delta < 0.015 else 'FAIL ✗ — regression!'})")

        # save grid
        tag_dir = out_dir / tag
        tag_dir.mkdir(exist_ok=True)
        grid = _make_grid(recons, ncols=4)
        grid_path = tag_dir / "grid.png"
        grid.save(grid_path)
        print(f"  grid → {grid_path}")

        # save per-image panels
        for i, panel in enumerate(recons):
            label_name = LABEL_NAMES[int(sample_df.iloc[i]["label"])]
            panel.save(tag_dir / f"{label_name}_{i:02d}.png")

    # summary table
    print("\n═══ SUMMARY ═══")
    print(f"{'checkpoint':<40}  delta_ctr_median  pass?")
    print("-" * 62)
    for tag, deltas in delta_ctrs.items():
        med = float(np.median(deltas)) if deltas else float("nan")
        print(f"{tag:<40}  {med:.4f}            {'✓' if med < 0.015 else '✗'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--baseline",    default="ckpts/vae_step0025000.pt")  # flat path, no subdir
    p.add_argument("--manifest",    default="data/vae_v2_diet/val_manifest.csv")
    p.add_argument("--out",         default="results/vae_v2/sanity")
    p.add_argument("--n-per-class", type=int, default=4)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    run_eval(
        checkpoints=args.checkpoints,
        baseline=args.baseline,
        manifest=args.manifest,
        out_dir=Path(args.out),
        n_per_class=args.n_per_class,
        device=args.device,
    )


if __name__ == "__main__":
    main()
