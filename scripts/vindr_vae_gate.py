"""VAE round-trip gate on VinDr-512 images (leaf 02 — external-transfer scope).

For each image_id in results/vindr_transfer/vae_gate/sample_ids.txt:
  • load PNG from /home/molefe/data/vindr512/vinbigdata/train/
  • encode → decode through ckpts/vae_step0025000.pt  (fp32, scale_factor applied)
  • score original + recon with ctr_overlay.py (PSPNet tint + CTR lines + text)
  • save 1024×512 side-by-side pair to results/vindr_transfer/vae_gate/pairs/
  • measure |ΔCTR|, CP-flip, lower-zone SSIM → roundtrip_metrics.csv

Usage:
    python scripts/vindr_vae_gate.py [--ids PATH] [--vindr-dir PATH] [--vae-ckpt PATH]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as _ssim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae.model import VAE
from vae.config import DEFAULT_CONFIG as vae_cfg
import scripts.ctr_overlay as C

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_IDS     = "results/vindr_transfer/vae_gate/sample_ids.txt"
DEFAULT_VINDR   = "/home/molefe/data/vindr512/vinbigdata/train"
DEFAULT_VAE     = "ckpts/vae_step0025000.pt"
DEFAULT_SF      = "data/latents/scale_factor.pt"
DEFAULT_CSV     = "results/vindr_transfer/vae_gate/roundtrip_metrics.csv"
DEFAULT_PAIRS   = "results/vindr_transfer/vae_gate/pairs"

DIVIDER_W = 6          # px gap between orig and recon in each pair image
DIVIDER_COL = (50, 50, 50)


# ── VAE helpers ───────────────────────────────────────────────────────────────

def _load_vae(ckpt_path: str, device: torch.device) -> VAE:
    model = VAE(vae_cfg).to(device).float()   # fp32 — avoids black-pixel bf16 underflow
    raw = torch.load(ckpt_path, map_location=device)
    state = raw.get("model", raw.get("model_state", raw))
    model.load_state_dict(state)
    model.eval()
    return model


def _roundtrip(model: VAE, arr_uint8: np.ndarray,
               scale_factor: float, device: torch.device) -> np.ndarray:
    """Encode → decode; returns uint8 (512, 512)."""
    x = torch.from_numpy(arr_uint8.astype(np.float32) / 127.5 - 1.0
                         )[None, None].to(device)
    with torch.no_grad():
        z_mu, _ = model.net.encode(x)
        recon = model.net.decode(z_mu).clamp(-1, 1)
    return ((recon[0, 0].float().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)


# ── per-image metrics ─────────────────────────────────────────────────────────

def _lower_zone_ssim(orig: np.ndarray, recon: np.ndarray, frac: float = 0.25) -> float:
    cut = int(orig.shape[0] * (1.0 - frac))
    a, b = orig[cut:], recon[cut:]
    return float(_ssim(a, b, data_range=255)) if min(a.shape) >= 7 else float("nan")


def score_film(seg, idx, vae_model, scale_factor, image_id: str,
               vindr_dir: Path, device: torch.device, tmp_dir: Path) -> dict:
    src = vindr_dir / f"{image_id}.png"
    orig_arr = np.array(Image.open(src).convert("L").resize((512, 512)), dtype=np.uint8)

    recon_arr = _roundtrip(vae_model, orig_arr, scale_factor, device)
    tmp_recon = tmp_dir / f"{image_id}_recon.png"
    Image.fromarray(recon_arr, "L").save(tmp_recon)

    # with_masks=True so _overlay can tint the heart/lung segments
    mo = C.ctr_for_image(seg, idx, src, str(device), with_masks=True)
    mr = C.ctr_for_image(seg, idx, tmp_recon, str(device), with_masks=True)

    ctr_o, ctr_r = mo["ctr"], mr["ctr"]
    dctr = abs(ctr_o - ctr_r) if (np.isfinite(ctr_o) and np.isfinite(ctr_r)) else float("nan")

    cl_o, cr_o = mo.get("cp_left", {}), mo.get("cp_right", {})
    cl_r, cr_r = mr.get("cp_left", {}), mr.get("cp_right", {})

    return {
        "image_id": image_id,
        "ctr_orig":  round(ctr_o, 4) if np.isfinite(ctr_o) else float("nan"),
        "ctr_recon": round(ctr_r, 4) if np.isfinite(ctr_r) else float("nan"),
        "abs_dCTR":  round(dctr, 4)  if np.isfinite(dctr)  else float("nan"),
        "ctr_crosses_0p5": bool(np.isfinite(dctr) and (ctr_o > 0.5) != (ctr_r > 0.5)),
        "border_visible_orig":  mo.get("border_visible"),
        "border_visible_recon": mr.get("border_visible"),
        "border_flip": mo.get("border_visible") != mr.get("border_visible"),
        "opac_orig":  mo.get("peri_cardiac_opacity"),
        "opac_recon": mr.get("peri_cardiac_opacity"),
        "cp_flip_left":  cl_o.get("blunted") != cl_r.get("blunted"),
        "cp_flip_right": cr_o.get("blunted") != cr_r.get("blunted"),
        "cp_angle_l_orig":  cl_o.get("angle_deg"),
        "cp_angle_l_recon": cl_r.get("angle_deg"),
        "cp_angle_r_orig":  cr_o.get("angle_deg"),
        "cp_angle_r_recon": cr_r.get("angle_deg"),
        "lower_zone_ssim": round(_lower_zone_ssim(orig_arr, recon_arr), 4),
        "seg_ok_orig":  mo["ok"],
        "seg_ok_recon": mr["ok"],
        # private — used only for pair rendering, stripped from CSV
        "_src":    str(src),
        "_tmp":    str(tmp_recon),
        "_mo":     mo,
        "_mr":     mr,
    }


# ── pair image ────────────────────────────────────────────────────────────────

def _delta_bar(draw: ImageDraw.ImageDraw, r: dict, canvas_w: int, canvas_h: int) -> None:
    """Draw a one-line delta summary across the bottom of the combined canvas."""
    dctr = r["abs_dCTR"]
    dctr_s = f"|ΔCTR|={dctr:.4f}" if np.isfinite(dctr) else "|ΔCTR|=n/a"
    cross = "CROSS-0.5!" if r["ctr_crosses_0p5"] else ""
    lz = r["lower_zone_ssim"]
    lz_s = f"lzSSIM={lz:.3f}" if np.isfinite(lz) else ""
    bflip = "bv-FLIP!" if r["border_flip"] else ""
    cpflip = "CP-FLIP!" if (r["cp_flip_left"] or r["cp_flip_right"]) else ""
    flags = "  ".join(f for f in [cross, bflip, cpflip] if f)
    text = f"  {dctr_s}   {lz_s}   {flags if flags else 'clean'}"
    col = (255, 80, 80) if flags else (100, 255, 100)
    h = 26
    draw.rectangle([0, canvas_h - h, canvas_w, canvas_h], fill=(10, 10, 10))
    draw.text((4, canvas_h - h + 4), text, fill=col, font=C._font(15))


def save_pair(r: dict, out_path: Path) -> None:
    """1024×(512+26) image: [orig overlay] | [divider] | [recon overlay] | [delta bar]."""
    orig_ov  = C._overlay(Path(r["_src"]), r["_mo"],  tint=True)   # 512×512 RGB
    recon_ov = C._overlay(Path(r["_tmp"]), r["_mr"], tint=True)    # 512×512 RGB

    W = 512 + DIVIDER_W + 512
    H = 512 + 26   # 26px delta bar at the bottom
    canvas = Image.new("RGB", (W, H), (15, 15, 15))
    canvas.paste(orig_ov,  (0, 0))
    canvas.paste(Image.new("RGB", (DIVIDER_W, 512), DIVIDER_COL), (512, 0))
    canvas.paste(recon_ov, (512 + DIVIDER_W, 0))

    draw = ImageDraw.Draw(canvas)
    _delta_bar(draw, r, W, H)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ── summary print ─────────────────────────────────────────────────────────────

def _print_summary(rows: list[dict]) -> None:
    ok   = [r for r in rows if r["seg_ok_orig"] and r["seg_ok_recon"]]
    dctr = [r["abs_dCTR"] for r in ok if np.isfinite(r["abs_dCTR"])]
    lz   = [r["lower_zone_ssim"] for r in ok if np.isfinite(r["lower_zone_ssim"])]
    n_cross   = sum(r["ctr_crosses_0p5"] for r in ok)
    n_cp_flip = sum(r["cp_flip_left"] or r["cp_flip_right"] for r in ok)
    n_bv_flip = sum(r["border_flip"] for r in rows)
    print(f"\n{'='*56}")
    print(f"  VinDr VAE round-trip gate  (N={len(rows)}, seg ok={len(ok)})")
    print(f"{'='*56}")
    if dctr:
        print(f"  |ΔCTR|      median={np.median(dctr):.4f}  mean={np.mean(dctr):.4f}"
              f"  p95={np.percentile(dctr, 95):.4f}")
    print(f"  CTR crosses 0.5:     {n_cross}/{len(ok)}")
    print(f"  CP-angle flip:       {n_cp_flip}/{len(ok)}")
    print(f"  border_visible flip: {n_bv_flip}/{len(rows)}")
    if lz:
        print(f"  lower-zone SSIM     median={np.median(lz):.4f}  mean={np.mean(lz):.4f}")
    med_dctr = np.median(dctr) if dctr else float("nan")
    if np.isfinite(med_dctr) and med_dctr < 0.02 and n_cross <= 1:
        verdict = "PASS  ✓ VAE is a faithful codec on VinDr-512"
    elif np.isfinite(med_dctr) and med_dctr < 0.03:
        verdict = "MARGINAL  — within abstain margin (±0.03 CTR)"
    else:
        verdict = "FAIL  ✗ codec degrades VinDr images — investigate"
    print(f"\n  Leaf-02 verdict: {verdict}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids",        default=DEFAULT_IDS)
    ap.add_argument("--vindr-dir",  default=DEFAULT_VINDR)
    ap.add_argument("--vae-ckpt",   default=DEFAULT_VAE)
    ap.add_argument("--scale-factor", default=DEFAULT_SF)
    ap.add_argument("--out-csv",    default=DEFAULT_CSV)
    ap.add_argument("--out-pairs",  default=DEFAULT_PAIRS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ids = [l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()]
    vindr_dir = Path(args.vindr_dir)
    print(f"Loaded {len(ids)} image IDs  |  device={device}")

    print("Loading VAE …")
    vae_model = _load_vae(args.vae_ckpt, device)
    sf_raw = torch.load(args.scale_factor, map_location="cpu")
    scale_factor = float(sf_raw) if sf_raw.ndim == 0 else float(sf_raw[0])
    print(f"  scale_factor = {scale_factor:.6f}")

    print("Loading PSPNet segmentor …")
    seg, idx = C._load_seg(str(device))

    pairs_dir = Path(args.out_pairs)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, image_id in enumerate(ids):
            print(f"  [{i+1:2d}/{len(ids)}] {image_id[:16]}…", end="  ", flush=True)
            r = score_film(seg, idx, vae_model, scale_factor, image_id,
                           vindr_dir, device, Path(tmp_dir))
            rows.append(r)

            # save individual side-by-side pair
            out_png = pairs_dir / f"{i+1:02d}_{image_id[:12]}.png"
            save_pair(r, out_png)

            dctr_s = f"|ΔCTR|={r['abs_dCTR']:.4f}" if np.isfinite(r["abs_dCTR"]) else "|ΔCTR|=n/a"
            lz_s   = f"lzSSIM={r['lower_zone_ssim']:.3f}" if np.isfinite(r["lower_zone_ssim"]) else ""
            print(f"CTR {r['ctr_orig']:.3f}→{r['ctr_recon']:.3f}  {dctr_s}  {lz_s}")

    _print_summary(rows)

    import pandas as pd
    csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"Metrics → {out_csv}")
    print(f"Pairs   → {pairs_dir}/  ({len(rows)} files)")


if __name__ == "__main__":
    main()
