"""Visual leakage harness — assemble the 5-stage pipeline sample set (plan 06-07).

Five stages per matched sample, exposing where composition fails and where the VAE
loses faithfulness:

    real          a real both-disease image (cardiomegaly + effusion)
    vae_recon     that SAME image round-tripped through the VAE (encode → decode)
    ldm_cardio    a single-disease LDM cardiomegaly sample (fresh from noise)
    ldm_effusion  a single-disease LDM effusion sample (fresh from noise)
    poe           a PoE-composed both-disease sample (fresh from noise)

Asymmetry (recorded in the manifest): real + vae_recon are the *same* image — they test
reconstruction faithfulness, so they are pixel-aligned. ldm_* and poe are independent
draws from noise — there is no per-image seed shared with the real, so they are selected
deterministically by index (seed) for reproducibility, not matched content.

Tasks 7.1 (build-set) and 7.2 (overlay --view dev) are implemented; 7.5 (blind-pack) is a
separate task that consumes this set.

Instruments for the dev view (7.2): the cardiomegaly instrument is selected by
02-pre-evaluation/05.6 from {Grad-CAM proxy, HybridGNet CTR, MedSAM-cut}. That decision is
NOT yet made and CTR/MedSAM are instrument-blocked (SPRINT deferred), so this uses the
locked-default Grad-CAM proxy (cardio bbox + heart_size) and the effusion Grad-CAM box +
blunting (MedSAM mask demoted to box-only as it is unavailable/unvalidated). The overlay is
instrument-pluggable via --cardio_instrument so CTR/MedSAM can swap in once 05 lands.

All stages are brought to 512×512 grayscale and saved under --out as:

    outputs/visual_leakage/
      real/sample00.png  vae_recon/sample00.png  ldm_cardio/sample00.png
      ldm_effusion/sample00.png  poe/sample00.png
      manifest.json          # per-sample source paths, seed, vae ckpt, stage list

CLI:
    python scripts/visual_leakage.py build-set \\
      --real data/nih/images/ --vae_ckpt ckpts/vae_step0025000.pt \\
      --ldm_cardio outputs/single/cardiomegaly/ --ldm_effusion outputs/single/effusion/ \\
      --poe outputs/compose/w1p0/ --n 8 --out outputs/visual_leakage/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_RES = 512
_STAGES = ("real", "vae_recon", "ldm_cardio", "ldm_effusion", "poe")
_DEFAULT_VAE = "ckpts/vae_step0025000.pt"   # canonical VAE (see hippo-shipped-artifacts)


# ── image io ──────────────────────────────────────────────────────────────────────

def _load_neg1_1(path: str | Path, res: int = _RES) -> torch.Tensor:
    """PNG → (1, 1, res, res) float32 in [-1, 1] (the VAE input space)."""
    img = Image.open(path).convert("L")
    if img.size != (res, res):
        img = img.resize((res, res), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _save_neg1_1(t: torch.Tensor, path: Path) -> None:
    """(1, 1, res, res) float32 [-1,1] → grayscale PNG at full resolution."""
    arr = t[0, 0].float().clamp(-1, 1).cpu().numpy()
    arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "L").save(path)


def _save_resized(src: str | Path, path: Path, res: int = _RES) -> None:
    """Copy a generated image to the set as 512² grayscale (no VAE round-trip)."""
    img = Image.open(src).convert("L")
    if img.size != (res, res):
        img = img.resize((res, res), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# ── selection ─────────────────────────────────────────────────────────────────────

def _pick(dir_path: str | Path, n: int, seed: int) -> list[Path]:
    """Deterministically pick n images from a dir (sorted → seeded sample)."""
    d = Path(dir_path)
    if not d.is_dir():
        raise SystemExit(f"directory does not exist: {d}")
    paths = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no PNG/JPG images in {d}")
    if n >= len(paths):
        return paths
    return sorted(random.Random(seed).sample(paths, n))


# ── VAE ─────────────────────────────────────────────────────────────────────────

def _load_vae(ckpt_path: str | Path, device: str):
    from vae.model import VAE
    model = VAE()
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


# ── build-set (task 7.1) ───────────────────────────────────────────────────────────

def build_set(args: argparse.Namespace) -> None:
    out = Path(args.out)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    real_paths = _pick(args.real, args.n, args.seed)
    n = len(real_paths)
    cardio_paths = _pick(args.ldm_cardio, n, args.seed)
    eff_paths = _pick(args.ldm_effusion, n, args.seed)
    poe_paths = _pick(args.poe, n, args.seed)
    # generated dirs may have fewer than n images → align all stages to the common count
    n = min(n, len(cardio_paths), len(eff_paths), len(poe_paths))
    real_paths, cardio_paths = real_paths[:n], cardio_paths[:n]
    eff_paths, poe_paths = eff_paths[:n], poe_paths[:n]

    print(f"Loading VAE {args.vae_ckpt} on {device} ...")
    vae = _load_vae(args.vae_ckpt, device)

    print(f"Building {n} matched 5-stage samples → {out}")
    manifest: dict = {
        "n": n, "seed": args.seed, "res": _RES, "stages": list(_STAGES),
        "vae_ckpt": str(args.vae_ckpt),
        "asymmetry": "real+vae_recon are the same image round-tripped (faithfulness); "
                     "ldm_*/poe are independent draws from noise (not content-matched)",
        "samples": [],
    }

    with torch.no_grad():
        for i in range(n):
            tag = f"sample{i:02d}.png"
            # real (resized to 512) + its VAE reconstruction (pixel-aligned)
            x = _load_neg1_1(real_paths[i], _RES).to(device)
            r = vae.reconstruct(x)
            _save_neg1_1(x, out / "real" / tag)
            _save_neg1_1(r, out / "vae_recon" / tag)
            # generated stages: independent draws, just normalized to 512²
            _save_resized(cardio_paths[i], out / "ldm_cardio" / tag)
            _save_resized(eff_paths[i], out / "ldm_effusion" / tag)
            _save_resized(poe_paths[i], out / "poe" / tag)

            manifest["samples"].append({
                "sample": tag,
                "real": str(real_paths[i]),
                "vae_recon": f"{real_paths[i]} → VAE({Path(args.vae_ckpt).name})",
                "ldm_cardio": str(cardio_paths[i]),
                "ldm_effusion": str(eff_paths[i]),
                "poe": str(poe_paths[i]),
            })
            print(f"  {tag}: real={real_paths[i].name}  cardio={cardio_paths[i].name}"
                  f"  eff={eff_paths[i].name}  poe={poe_paths[i].name}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved 5×{n} images + manifest.json → {out}")
    print(f"  stages: {', '.join(_STAGES)}")
    print("  NOTE: ldm_*/poe provenance is whatever --ldm_*/--poe point at; if those are a "
          "decoy/non-final model, regenerate before the radiologist read (7.4–7.6).")


# ── overlay --view dev (task 7.2) ──────────────────────────────────────────────────

def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def overlay_dev(args: argparse.Namespace) -> None:
    """Apply the cardio + effusion instruments to all 5 stages → a rows=stages dev grid."""
    import torch
    from PIL import Image, ImageDraw

    from scripts.grad_cam import overlay as blend_overlay
    from metrics.grad_cam_utils import GradCAM, load_model, preprocess
    from metrics.extractors import blunting_from_cam, heart_size_from_cam

    if args.cardio_instrument != "gradcam":
        raise SystemExit(
            f"--cardio_instrument {args.cardio_instrument!r} not available: CTR/MedSAM are "
            "instrument-blocked and 02-pre-evaluation/05.6 has not chosen yet. Use 'gradcam'.")

    set_dir = Path(args.set)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    samples = sorted((set_dir / "real").glob("sample*.png"))[: args.n_cols]
    if not samples:
        raise SystemExit(f"no sample*.png under {set_dir}/real (run build-set first)")
    n_cols = len(samples)

    print(f"Loading classifier on {device} ...")
    model, ci, ei = load_model(device=device)
    cam_engine = GradCAM(model, device)

    cell = 256
    cap_h, label_w, header_h = 20, 96, 30
    canvas_w = label_w + n_cols * cell
    canvas_h = header_h + len(_STAGES) * (cell + cap_h)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    f_lab, f_cap, f_hdr = _font(13), _font(11), _font(13)

    draw.text((6, 8),
              "Visual leakage dev view — cardio=Grad-CAM proxy (bbox=red, hs), "
              "effusion=Grad-CAM box (bbox=blue, bl); C/E = classifier presence prob "
              "[provisional: 05.6 instrument decision pending; gen stages may be non-final model]",
              fill=(210, 210, 210), font=f_hdr)

    for ri, stage in enumerate(_STAGES):
        y0 = header_h + ri * (cell + cap_h)
        draw.text((4, y0 + cell // 2 - 6), stage, fill=(230, 230, 120), font=f_lab)
        for cj, s in enumerate(samples):
            img_path = set_dir / stage / s.name
            model_in, disp = preprocess(img_path)
            cam_c = cam_engine.cam(model_in, ci)
            cam_e = cam_engine.cam(model_in, ei)
            with torch.no_grad():
                probs = torch.sigmoid(model(model_in.to(device)))[0].cpu().numpy()
            pc, pe = float(probs[ci]), float(probs[ei])
            hs, bl = heart_size_from_cam(cam_c), blunting_from_cam(cam_e)

            cell_img = blend_overlay(disp, cam_c, cam_e, bbox=True).resize((cell, cell))
            x0 = label_w + cj * cell
            canvas.paste(cell_img, (x0, y0))
            cap = f"C={pc:.2f} hs={hs:.2f}  E={pe:.2f} bl={bl:.2f}"
            draw.rectangle([(x0, y0 + cell), (x0 + cell, y0 + cell + cap_h - 1)], fill=(30, 30, 30))
            draw.text((x0 + 4, y0 + cell + 4), cap, fill=(210, 210, 210), font=f_cap)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Saved dev view ({len(_STAGES)}×{n_cols}) → {out}")
    print("  rows = stages; red=cardio Grad-CAM bbox, blue=effusion Grad-CAM bbox; "
          "C/E=presence prob, hs=heart_size, bl=blunting")
    print("  7.3 reading: ldm_cardio row should fire cardio only; ldm_effusion row effusion only "
          "(+CTR normal); poe row both. Record verdicts in tracking.md.")


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Visual leakage harness (plan 06-07)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-set", help="7.1 — assemble the 5-stage matched sample set")
    b.add_argument("--real", required=True, help="Dir of real both-disease images")
    b.add_argument("--vae_ckpt", default=_DEFAULT_VAE, help="VAE checkpoint for the E→D recon")
    b.add_argument("--ldm_cardio", required=True, help="Dir of LDM cardiomegaly samples")
    b.add_argument("--ldm_effusion", required=True, help="Dir of LDM effusion samples")
    b.add_argument("--poe", required=True, help="Dir of PoE composed samples")
    b.add_argument("--n", type=int, default=8, help="Matched samples to build")
    b.add_argument("--seed", type=int, default=42, help="Deterministic selection seed")
    b.add_argument("--out", default="outputs/visual_leakage/", help="Output set dir")
    b.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    b.set_defaults(func=build_set)

    o = sub.add_parser("overlay", help="7.2 — apply instruments to all stages → dev grid")
    o.add_argument("--view", choices=["dev"], default="dev", help="Overlay view (dev only for now)")
    o.add_argument("--set", default="outputs/visual_leakage/", help="5-stage set dir (from build-set)")
    o.add_argument("--cardio_instrument", default="gradcam",
                   choices=["gradcam", "ctr", "medsam"],
                   help="Cardio instrument (05.6 decision; ctr/medsam instrument-blocked)")
    o.add_argument("--n_cols", type=int, default=4, help="Samples (columns) to show")
    o.add_argument("--out", default="figures/visual_leakage/dev_view.png", help="Output PNG")
    o.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    o.set_defaults(func=overlay_dev)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
