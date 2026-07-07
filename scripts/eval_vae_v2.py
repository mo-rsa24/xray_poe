"""Four-gate eval for VAE-v2 — plan 04-vae-eval-gate.md.

Gates:
  1. delta_ctr_median   — |CTR_orig − CTR_recon| median  target < 0.01
  2. lower_zone_ssim    — SSIM on rows 340:512              target ≥ 0.88
  3. lpips_full         — AlexNet LPIPS 512×512             target < 0.10
  4. lpips_cardiac_crop — AlexNet LPIPS 256×256 cardiac     target < 0.08

Usage:
    python scripts/eval_vae_v2.py \
        --checkpoint ckpts/vae_v2/vae_v2_step0075000.pt \
        --baseline   ckpts/vae_step0025000.pt \
        --manifest   data/vae_v2_diet/val_manifest.csv \
        --output     results/vae_v2/eval_gate/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

import lpips as lpips_lib

from vae.config import VAEConfig
from vae.model import VAE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GATE_TARGETS = {
    "delta_ctr_median":   ("< 0.01",  lambda v: v < 0.01),
    "lower_zone_ssim":    (">= 0.88", lambda v: v >= 0.88),
    "lpips_full":         ("< 0.10",  lambda v: v < 0.10),
    "lpips_cardiac_crop": ("< 0.08",  lambda v: v < 0.08),
}

LOWER_ZONE_START = 340   # rows 340–512 of a 512-px image

RESIZE = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_vae(path: str, device: str) -> VAE:
    cfg = VAEConfig(input_resolution=512, use_checkpoint=False)
    vae = VAE(cfg).to(device).eval()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    vae.load_state_dict(state, strict=False)
    return vae


def _load_seg(device: str):
    import scripts.ctr_overlay as co
    return co._load_seg(device)


def _load_lpips(device: str):
    net = lpips_lib.LPIPS(net="alex").to(device).eval()
    return net

# ---------------------------------------------------------------------------
# Per-film scoring helpers
# ---------------------------------------------------------------------------

def _tensor_to_gray512(t: torch.Tensor) -> np.ndarray:
    """1×H×W or H×W tensor in [-1,1] → uint8 (512,512)."""
    arr = t.squeeze().float().cpu().numpy()
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def _score_ctr(seg, idx, gray: np.ndarray, device: str) -> dict:
    from metrics.eval_checkpoint import score_decoded_array
    return score_decoded_array(seg, idx, gray, device, with_masks=True)


def _lower_zone_ssim(orig: np.ndarray, recon: np.ndarray) -> float:
    o = orig[LOWER_ZONE_START:].astype(np.float32) / 255.0
    r = recon[LOWER_ZONE_START:].astype(np.float32) / 255.0
    val, _ = ssim(o, r, data_range=1.0, full=True)
    return float(val)


def _lpips_full(net, orig_t: torch.Tensor, recon_t: torch.Tensor, device: str) -> float:
    o = orig_t.unsqueeze(0).repeat(1, 3, 1, 1).to(device)   # LPIPS expects 3ch
    r = recon_t.unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        return float(net(o, r).item())


def _cardiac_crop(gray: np.ndarray, masks: dict | None) -> tuple[int, int] | None:
    """Return (cy, cx) centroid of heart mask, or None if mask unavailable."""
    if masks is None:
        return None
    import scripts.ctr_overlay as co
    heart = masks.get(co.HEART)
    if heart is None or not heart.any():
        return None
    ys, xs = np.where(heart)
    return int(ys.mean()), int(xs.mean())


def _lpips_cardiac(net, orig_t: torch.Tensor, recon_t: torch.Tensor,
                   centroid: tuple[int, int] | None, device: str) -> float:
    """256×256 AlexNet LPIPS centred on the cardiac border; falls back to centre."""
    H, W = orig_t.shape[-2], orig_t.shape[-1]
    crop = 256
    half = crop // 2
    if centroid is not None:
        cy, cx = centroid
    else:
        cy, cx = H // 2, W // 2
    y0 = max(0, min(cy - half, H - crop))
    x0 = max(0, min(cx - half, W - crop))
    o = orig_t[..., y0:y0+crop, x0:x0+crop].unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    r = recon_t[..., y0:y0+crop, x0:x0+crop].unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        return float(net(o, r).item())

# ---------------------------------------------------------------------------
# Per-checkpoint eval loop
# ---------------------------------------------------------------------------

def eval_checkpoint(
    ckpt_path: str,
    manifest_df: pd.DataFrame,
    seg,
    idx,
    lpips_net,
    device: str,
) -> pd.DataFrame:
    tag = Path(ckpt_path).stem
    print(f"\n{'─'*60}")
    print(f"  {tag}")
    print(f"{'─'*60}")

    vae = _load_vae(ckpt_path, device)
    rows = []

    for i, row in manifest_df.iterrows():
        img_path = row["image_path"]
        source   = row.get("source", "unknown")
        patient_id = Path(img_path).stem

        try:
            img = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"  [{i:04d}] SKIP  {patient_id}  ({e})")
            continue

        orig_t = RESIZE(img)   # 1×512×512, [-1,1]

        with torch.no_grad():
            x = orig_t.unsqueeze(0).to(device)
            mu, _ = vae.encode_moments(x)
            recon_t = vae.decode(mu).squeeze(0).cpu()   # 1×512×512

        gray_orig  = _tensor_to_gray512(orig_t)
        gray_recon = _tensor_to_gray512(recon_t)

        s_orig  = _score_ctr(seg, idx, gray_orig,  device)
        s_recon = _score_ctr(seg, idx, gray_recon, device)

        ctr_o = s_orig["ctr"]  if s_orig["ok"]  else float("nan")
        ctr_r = s_recon["ctr"] if s_recon["ok"] else float("nan")
        delta_ctr = abs(ctr_r - ctr_o) if (s_orig["ok"] and s_recon["ok"]) else float("nan")

        lz_ssim = _lower_zone_ssim(gray_orig, gray_recon)

        lp_full = _lpips_full(lpips_net, orig_t, recon_t, device)

        centroid = _cardiac_crop(gray_recon, s_recon.get("_masks"))
        lp_cardiac = _lpips_cardiac(lpips_net, orig_t, recon_t, centroid, device)

        rows.append({
            "patient_id":         patient_id,
            "source":             source,
            "label":              row.get("label", -1),
            "checkpoint":         tag,
            "ctr_orig":           round(ctr_o, 4),
            "ctr_recon":          round(ctr_r, 4),
            "delta_ctr":          round(delta_ctr, 4),
            "lower_zone_ssim":    round(lz_ssim, 4),
            "lpips_full":         round(lp_full, 4),
            "lpips_cardiac_crop": round(lp_cardiac, 4),
        })

        if (len(rows) % 50) == 0:
            print(f"  {len(rows):>4d}/{len(manifest_df)}  "
                  f"delta_ctr={delta_ctr:.3f}  ssim={lz_ssim:.3f}  "
                  f"lpips={lp_full:.3f}  cardiac_lpips={lp_cardiac:.3f}")

    df = pd.DataFrame(rows)
    _print_summary(tag, df)
    return df


def _print_summary(tag: str, df: pd.DataFrame) -> None:
    n = len(df)
    if n == 0:
        print(f"  {tag}: no rows scored")
        return
    metrics = {
        "delta_ctr_median":   df["delta_ctr"].median(),
        "lower_zone_ssim":    df["lower_zone_ssim"].median(),
        "lpips_full":         df["lpips_full"].median(),
        "lpips_cardiac_crop": df["lpips_cardiac_crop"].median(),
    }
    print(f"\n  ── {tag} (n={n}) ──")
    for gate, val in metrics.items():
        target_str, passes = GATE_TARGETS[gate]
        ok = "✓ PASS" if passes(val) else "✗ FAIL"
        print(f"  {gate:<25}  {val:.4f}  target {target_str:<8}  {ok}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _aggregate(df: pd.DataFrame) -> dict:
    return {
        "delta_ctr_median":   df["delta_ctr"].median(),
        "lower_zone_ssim":    df["lower_zone_ssim"].median(),
        "lpips_full":         df["lpips_full"].median(),
        "lpips_cardiac_crop": df["lpips_cardiac_crop"].median(),
        "n":                  len(df),
    }


def print_comparison(baseline_df: pd.DataFrame, ckpt_df: pd.DataFrame,
                     baseline_tag: str, ckpt_tag: str) -> None:
    b = _aggregate(baseline_df)
    c = _aggregate(ckpt_df)
    print(f"\n{'═'*72}")
    print(f"  GATE COMPARISON  baseline={baseline_tag}  checkpoint={ckpt_tag}")
    print(f"{'═'*72}")
    header = f"  {'Gate':<25}  {'Target':<10}  {'Baseline':>10}  {'vae-v2':>10}  {'Δ':>8}  Pass?"
    print(header)
    print(f"  {'-'*65}")
    for gate in ["delta_ctr_median", "lower_zone_ssim", "lpips_full", "lpips_cardiac_crop"]:
        target_str, passes = GATE_TARGETS[gate]
        bv, cv = b[gate], c[gate]
        delta = cv - bv
        ok = "✓" if passes(cv) else "✗"
        print(f"  {gate:<25}  {target_str:<10}  {bv:>10.4f}  {cv:>10.4f}  {delta:>+8.4f}  {ok}")
    print(f"{'═'*72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="VAE-v2 four-gate eval")
    p.add_argument("--checkpoint", required=True,
                   help="Fine-tuned checkpoint to evaluate")
    p.add_argument("--baseline",   default="ckpts/vae_step0025000.pt",
                   help="Baseline checkpoint for comparison")
    p.add_argument("--manifest",   default="data/vae_v2_diet/val_manifest.csv",
                   help="Val manifest CSV (image_path, source, label)")
    p.add_argument("--output",     default="results/vae_v2/eval_gate/",
                   help="Output directory for metrics.csv")
    p.add_argument("--limit",      type=int, default=None,
                   help="Score only the first N films (smoke-test)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    print(f"Val films: {len(manifest)}  device: {args.device}")

    print("Loading PSPNet…")
    seg, idx = _load_seg(args.device)

    print("Loading LPIPS (AlexNet)…")
    lpips_net = _load_lpips(args.device)

    baseline_df = eval_checkpoint(args.baseline,    manifest, seg, idx, lpips_net, args.device)
    ckpt_df     = eval_checkpoint(args.checkpoint,  manifest, seg, idx, lpips_net, args.device)

    print_comparison(baseline_df, ckpt_df,
                     Path(args.baseline).stem, Path(args.checkpoint).stem)

    # save per-film CSV (both checkpoints stacked)
    combined = pd.concat([baseline_df, ckpt_df], ignore_index=True)
    csv_path = out_dir / "metrics.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}  ({len(combined)} rows, "
          f"columns: {list(combined.columns)})")

    # save aggregate CSV
    agg_rows = []
    for df, tag in [(baseline_df, Path(args.baseline).stem),
                    (ckpt_df,     Path(args.checkpoint).stem)]:
        a = _aggregate(df)
        a["checkpoint"] = tag
        agg_rows.append(a)
    agg_df = pd.DataFrame(agg_rows)
    agg_path = out_dir / "metrics_agg.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved → {agg_path}")


if __name__ == "__main__":
    main()
