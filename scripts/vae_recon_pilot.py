"""Rung 1 — VAE reconstruction baseline for the pilot ladder.

Reconstructs the SAME 100 both-group images that Rung 0.5 (presence_classifier
--mode joint) sampled, so the real→recon presence delta is paired. Uses the
canonical deterministic recon path VAE.reconstruct(x) = decode(mu); the scale
factor cancels in an encode→decode round-trip, so it is not applied here.

    python scripts/vae_recon_pilot.py \
        --images data/nih/images_both --vae-ckpt ckpts/vae_step0025000.pt \
        --n 100 --out outputs/vae_recon
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as _ssim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae.model import VAE
from vae.config import DEFAULT_CONFIG as vae_cfg
import scripts.ctr_overlay as C

HEART_HALF = 48   # witness box half-size at each lateral cardiac border (px @512)
CP_HALF = 40      # witness box half-size at each costophrenic recess


def _same_sample_as_joint(image_dir: Path, n: int, seed: int = 42) -> list[Path]:
    """Reproduce metrics.presence_classifier._gather_paths(dir, n, seed) exactly."""
    paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if n is not None and n < len(paths):
        paths = random.Random(seed).sample(paths, n)
    return paths


# ── localized witness metrics (Exp 2 codec ceiling) ─────────────────────────────

def _init_lpips(device):
    try:
        import lpips
        return lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as e:
        print(f"[lpips unavailable ({type(e).__name__}); using SSIM only]")
        return None


def _recon_uint8(model, arr_uint8: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(arr_uint8.astype(np.float32) / 127.5 - 1.0)[None, None].to(device)
    with torch.no_grad():
        r = model.reconstruct(x).clamp(-1, 1)
    return ((r[0, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)


def _crop(arr: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    H, W = arr.shape
    return arr[max(0, cy - half):min(H, cy + half), max(0, cx - half):min(W, cx + half)]


def _box_metrics(orig, recon, cx, cy, half, lpips_fn, device) -> tuple[float, float]:
    """(SSIM, LPIPS) on a witness crop — SSIM always; LPIPS if the net loaded."""
    a, b = _crop(orig, cx, cy, half), _crop(recon, cx, cy, half)
    if a.shape != b.shape or min(a.shape) < 7:
        return float("nan"), float("nan")
    s = float(_ssim(a, b, data_range=255))
    lp = float("nan")
    if lpips_fn is not None:
        ta = torch.from_numpy(a.astype(np.float32) / 127.5 - 1)[None, None].repeat(1, 3, 1, 1).to(device)
        tb = torch.from_numpy(b.astype(np.float32) / 127.5 - 1)[None, None].repeat(1, 3, 1, 1).to(device)
        with torch.no_grad():
            lp = float(lpips_fn(ta, tb).item())
    return s, lp


def localized_recon_metrics(model, seg, idx, lpips_fn, path: Path, device, tmp_png: Path) -> dict:
    """Encode→decode one film; read CTR/CP on original vs reconstruction and the
    local fidelity at the heart-border + CP-corner witness boxes."""
    orig = np.array(Image.open(path).convert("L").resize((512, 512)), dtype=np.uint8)
    recon = _recon_uint8(model, orig, device)
    Image.fromarray(recon, mode="L").save(tmp_png)
    mo = C.ctr_for_image(seg, idx, Path(path), device)
    mr = C.ctr_for_image(seg, idx, tmp_png, device)

    ctr_o, ctr_r = mo["ctr"], mr["ctr"]
    dctr = abs(ctr_o - ctr_r) if (np.isfinite(ctr_o) and np.isfinite(ctr_r)) else float("nan")
    clo, cro = mo["cp_left"]["blunted"], mo["cp_right"]["blunted"]
    clr, crr = mr["cp_left"]["blunted"], mr["cp_right"]["blunted"]
    res = {"ctr_orig": round(ctr_o, 3) if np.isfinite(ctr_o) else np.nan,
           "ctr_recon": round(ctr_r, 3) if np.isfinite(ctr_r) else np.nan,
           "abs_dCTR": round(dctr, 3) if np.isfinite(dctr) else np.nan,
           "ctr_crosses_0p5": bool(np.isfinite(dctr) and (ctr_o > 0.5) != (ctr_r > 0.5)),
           "cp_left_preserved": clo == clr, "cp_right_preserved": cro == crr,
           "cp_preserved": (clo == clr) and (cro == crr),
           "cp_left_orig_angle": mo["cp_left"].get("angle_deg"),
           "cp_left_recon_angle": mr["cp_left"].get("angle_deg"),
           "cp_right_orig_angle": mo["cp_right"].get("angle_deg"),
           "cp_right_recon_angle": mr["cp_right"].get("angle_deg"),
           "heart_ssim": np.nan, "heart_lpips": np.nan,
           "cp_ssim": np.nan, "cp_lpips": np.nan, "ok": mo["ok"]}
    if mo["ok"]:
        hx0, hy, hx1, _ = mo["heart_line"]
        hs, hl = zip(*[_box_metrics(orig, recon, cx, hy, HEART_HALF, lpips_fn, device)
                       for cx in (hx0, hx1)])
        res["heart_ssim"], res["heart_lpips"] = float(np.nanmean(hs)), float(np.nanmean(hl))
        cs, cl = [], []
        for cp in (mo["cp_left"], mo["cp_right"]):
            if cp.get("vertex"):
                s, l = _box_metrics(orig, recon, cp["vertex"][0], cp["vertex"][1], CP_HALF, lpips_fn, device)
                cs.append(s); cl.append(l)
        res["cp_ssim"] = float(np.nanmean(cs)) if cs else np.nan
        res["cp_lpips"] = float(np.nanmean(cl)) if cl else np.nan
    return res


def _run_localized(args) -> None:
    device = torch.device(args.device)
    model = VAE(vae_cfg).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt.get("model_state", ckpt)))
    model.eval()
    seg, idx = C._load_seg(args.device)
    lpips_fn = _init_lpips(device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp_recon.png"

    if args.film:
        m = localized_recon_metrics(model, seg, idx, lpips_fn, Path(args.film), device, tmp)
        print(f"\nfilm = {args.film}")
        print(f"CTR  {m['ctr_orig']} → {m['ctr_recon']}  (|ΔCTR|={m['abs_dCTR']}, "
              f"crosses 0.5={m['ctr_crosses_0p5']})")
        print(f"CP preserved: L={m['cp_left_preserved']} R={m['cp_right_preserved']} "
              f"(both={m['cp_preserved']})")
        print(f"heart-border  SSIM={m['heart_ssim']:.3f}  LPIPS={m['heart_lpips']:.3f}")
        print(f"CP-corner     SSIM={m['cp_ssim']:.3f}  LPIPS={m['cp_lpips']:.3f}")
        return

    import pandas as pd
    df = pd.read_csv(args.manifest)
    rows = []
    for i, r in df.iterrows():
        m = localized_recon_metrics(model, seg, idx, lpips_fn, Path(r["path"]), device, tmp)
        rows.append({"path": r["path"], "group": r.get("group", ""), **m})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)}")
    res = pd.DataFrame(rows)
    res.to_csv(args.out_csv, index=False)
    ok = res[res["ok"]]
    print(f"\nlocalized recon metrics: {len(res)} films ({len(ok)} segmented)")
    print(f"  median |ΔCTR| = {ok['abs_dCTR'].median():.4f}   CTR-cross-0.5 = {ok['ctr_crosses_0p5'].mean():.1%}")
    print(f"  CP preserved  = {ok['cp_preserved'].mean():.1%}")
    print(f"  heart-border  SSIM={ok['heart_ssim'].median():.3f}  LPIPS={ok['heart_lpips'].median():.3f}")
    print(f"  CP-corner     SSIM={ok['cp_ssim'].median():.3f}  LPIPS={ok['cp_lpips'].median():.3f}")
    print(f"→ {args.out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="VAE reconstruction baseline (Rung 1)")
    p.add_argument("--images", default="data/nih/images_both")
    p.add_argument("--vae-ckpt", "--vae", dest="vae_ckpt", default="ckpts/vae_step0025000.pt")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/vae_recon")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    # localized witness-metric mode (Exp 2 codec ceiling)
    p.add_argument("--localized", action="store_true",
                   help="measure local fidelity at the heart border + CP corner")
    p.add_argument("--film", default=None, help="localized single-film mode")
    p.add_argument("--manifest", default=None, help="localized batch: CSV with a 'path' column")
    p.add_argument("--out-csv", default="results/exp02_ceiling/recon_witness_metrics.csv")
    args = p.parse_args()

    if args.localized:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        _run_localized(args)
        return

    device = torch.device(args.device)
    model = VAE(vae_cfg).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt.get("model_state", ckpt)))
    model.eval()

    paths = _same_sample_as_joint(Path(args.images), args.n, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reconstructing {len(paths)} images from {args.images} → {out}")

    saved = 0
    for start in range(0, len(paths), args.batch_size):
        chunk = paths[start : start + args.batch_size]
        arrs = []
        for p_ in chunk:
            arr = np.array(Image.open(p_).convert("L").resize((512, 512)), dtype=np.float32)
            arrs.append(arr / 127.5 - 1.0)
        x = torch.from_numpy(np.stack(arrs)).unsqueeze(1).to(device)  # (B,1,512,512)
        with torch.no_grad():
            r = model.reconstruct(x).clamp(-1, 1)
        for i in range(r.shape[0]):
            img = ((r[i, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(out / f"recon_{saved:04d}.png")
            saved += 1
    print(f"Saved {saved} reconstructions to {out}")


if __name__ == "__main__":
    main()
