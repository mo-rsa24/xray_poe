"""rFID ceiling-check gate (Sprint S1, Track A — VAE certification).

Reconstruct real both-disease (cardiomegaly+effusion) images through a frozen VAE
checkpoint and score how much the codec distorts them. The headline number is the
**reconstruction-FID (rFID)**: the domain-FID between the real set and its own VAE
reconstruction. Because the LDM samples in *this* VAE's latent space and decodes
through *this* codec, rFID is a hard lower bound on any generative FID the pipeline
can ever reach — the codec ceiling. The gate certifies that ceiling sits *far below*
the LDM's target gen-FID, i.e. the codec is not the bottleneck on the composition
claim. If it does not, the conditional fine-tune task (plan 10, task 2) fires.

No new metric math — this is glue over already-built, already-tested pieces:

    VAE codec            vae.model.VAE.reconstruct  (decode(μ), the canonical recon)
    SSIM / LPIPS         vae.eval.recon_metrics     (per-image codec fidelity)
    domain-FID + KID     metrics.fid (embed="xrv")  (the S4 harness, blocked-by dep)
    real-vs-real floor   results/floor.json         (context: the smallest gap that
                                                      is resolvable above noise)

Three FID quantities are reported, all in the xrv (NIH DenseNet-121, 1024-d) domain
feature space the rest of the eval uses:

    rFID            FID(real originals, recon)  — the headline ceiling. Real set is
                    the *original* images (what gen-FID in S6 also scores against),
                    so this is the honest lower bound including any input-resize loss
                    the codec pipeline imposes.
    rFID_codec      FID(model input @512, recon @512) — isolates the encode→decode
                    distortion alone (resize confound removed). Diagnostic: if
                    rFID ≫ rFID_codec the loss is mostly the 1024→512 resize, not the
                    codec, which changes the keep-or-fine-tune call.
    floor.fid       FID(real half, real half) from results/floor.json — the noise
                    floor. rFID must sit *above* it (recon ≠ identity) yet *far below*
                    the target gen-FID.

Pass: rFID <= target_gen_fid * margin  (default margin 1/3 → "far below").

CLI:
    # the gate (target gen-FID supplied → pass/fail verdict)
    python scripts/rfid_gate.py --ckpt ckpts/vae_step0025000.pt \\
        --real data/nih/images --n 512 --target-fid 12.0 \\
        --out results/rfid_gate.json

    # measure only (no target yet → verdict "indeterminate", numbers still recorded)
    python scripts/rfid_gate.py --n 128

    # quick smoke on a handful (cpu-friendly)
    python scripts/rfid_gate.py --n 16 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vae.model import VAE                       # noqa: E402
from vae.eval import recon_metrics              # noqa: E402
from metrics.fid import extract_features, fid_from_features, kid_poly  # noqa: E402
from metrics.c2st import _gather_paths          # noqa: E402  (shared deterministic sampler)

# The real both-disease set (cardiomegaly+effusion) — same reference the floor uses.
DEFAULT_REAL_DIR = "data/nih/images"
DEFAULT_CKPT = "ckpts/vae_step0025000.pt"
FLOOR_JSON = "results/floor.json"


# ── image <-> tensor ──────────────────────────────────────────────────────────────

def load_image(path: str | Path, res: int = 512) -> torch.Tensor:
    """PNG → (1, 1, res, res) float32 in [-1, 1] (LANCZOS, grayscale)."""
    img = Image.open(path).convert("L")
    if img.size != (res, res):
        img = img.resize((res, res), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def save_image(t: torch.Tensor, path: Path) -> None:
    """(1, 1, H, W) float32 in [-1, 1] → uint8 grayscale PNG at native resolution."""
    arr = t[0, 0].float().clamp(-1, 1).cpu().numpy()
    arr = ((arr + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, "L").save(path)


# ── VAE loading (mirrors scripts/vae_recon_grid.py) ────────────────────────────────

def load_vae(ckpt_path: Path, device: torch.device) -> VAE:
    model = VAE()
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


# ── reconstruct a real set → write input@512 + recon@512 PNGs, accumulate SSIM/LPIPS ─

@torch.no_grad()
def reconstruct_set(
    model: VAE,
    paths: list[Path],
    in_dir: Path,
    recon_dir: Path,
    device: torch.device,
    batch_size: int,
    res: int = 512,
) -> dict:
    """Reconstruct each real image; save the model-input (resized) and the recon as
    PNGs, and accumulate SSIM/LPIPS over the set. Returns mean recon metrics + the two
    PNG dirs to feed the FID harness."""
    in_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    n, ssim_sum, lpips_sum = 0, 0.0, 0.0
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        xs = torch.cat([load_image(p, res) for p in chunk], dim=0).to(device)  # (B,1,res,res)
        recons = model.reconstruct(xs)                                          # decode(μ)

        m = recon_metrics(xs.cpu(), recons.cpu())
        b = xs.shape[0]
        ssim_sum += m["ssim"] * b
        lpips_sum += m["lpips"] * b
        n += b

        for j, p in enumerate(chunk):
            stem = Path(p).stem
            save_image(xs[j : j + 1].cpu(), in_dir / f"{stem}.png")
            save_image(recons[j : j + 1].cpu(), recon_dir / f"{stem}.png")
        print(f"  reconstructed {n}/{len(paths)}  "
              f"(running SSIM={ssim_sum / n:.4f}  LPIPS={lpips_sum / n:.4f})")

    return {"n": n, "ssim": ssim_sum / max(n, 1), "lpips": lpips_sum / max(n, 1)}


# ── floor reference (context only) ──────────────────────────────────────────────────

def load_floor_fid() -> dict | None:
    p = ROOT / FLOOR_JSON
    if not p.is_file():
        return None
    try:
        f = json.loads(p.read_text())["fid"]
        return {"fid": f["fid"], "upper95": f.get("upper95")}
    except Exception:
        return None


# ── gate ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="rFID ceiling-check gate: certify the VAE codec is not the "
                    "bottleneck on the composition claim.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="VAE checkpoint (.pt)")
    p.add_argument("--real", default=DEFAULT_REAL_DIR,
                   help="Real both-disease image dir (cardiomegaly+effusion)")
    p.add_argument("--n", type=int, default=512, help="Cap real images (None=all)")
    p.add_argument("--res", type=int, default=512, help="VAE input resolution")
    p.add_argument("--target-fid", type=float, default=None, dest="target_fid",
                   help="LDM target gen-FID (xrv domain space). Required for a "
                        "pass/fail verdict; omit to measure only.")
    p.add_argument("--margin", type=float, default=1.0 / 3.0,
                   help="rFID must be <= target_fid * margin to pass ('≪'; default 1/3)")
    p.add_argument("--kid", action="store_true", help="Also report domain-KID")
    p.add_argument("--seed", type=int, default=42, help="Real-image sampling seed")
    p.add_argument("--batch_size", type=int, default=16, help="VAE recon batch size")
    p.add_argument("--fid_batch_size", type=int, default=32, help="Feature-extract batch")
    p.add_argument("--workdir", default="outputs/rfid_gate",
                   help="Where input@res / recon@res PNGs are written")
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    p.add_argument("--out", default="results/rfid_gate.json", help="Results JSON")
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    ckpt_path = ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    real_dir = ROOT / args.real if not Path(args.real).is_absolute() else Path(args.real)
    workdir = ROOT / args.workdir if not Path(args.workdir).is_absolute() else Path(args.workdir)

    print(f"Loading VAE: {ckpt_path}")
    model = load_vae(ckpt_path, device)
    step = ckpt_path.stem.replace("vae_step", "")

    real_paths = _gather_paths(real_dir, args.n, seed=args.seed)
    print(f"Reconstructing {len(real_paths)} real both-disease images "
          f"from {real_dir} (device={args.device}) ...")

    in_dir, recon_dir = workdir / "input", workdir / "recon"
    recon = reconstruct_set(model, real_paths, in_dir, recon_dir, device,
                            args.batch_size, res=args.res)

    # Free VAE before the feature extractor pulls in the xrv DenseNet.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nExtracting xrv (domain) features ...")
    feats_real = extract_features([str(q) for q in real_paths], embed="xrv",
                                  device=args.device, batch_size=args.fid_batch_size)
    feats_in = extract_features(sorted(str(q) for q in in_dir.glob("*.png")), embed="xrv",
                                device=args.device, batch_size=args.fid_batch_size)
    feats_recon = extract_features(sorted(str(q) for q in recon_dir.glob("*.png")), embed="xrv",
                                   device=args.device, batch_size=args.fid_batch_size)

    rfid = fid_from_features(feats_real, feats_recon)["fid"]          # headline ceiling
    rfid_codec = fid_from_features(feats_in, feats_recon)["fid"]      # codec-only (resize removed)

    out: dict = {
        "ckpt": str(ckpt_path),
        "vae_step": int(step) if step.isdigit() else step,
        "real_dir": str(real_dir),
        "n": recon["n"],
        "embed": "xrv",
        "rfid": rfid,
        "rfid_codec": rfid_codec,
        "ssim": recon["ssim"],
        "lpips": recon["lpips"],
        "target_fid": args.target_fid,
        "margin": args.margin,
    }

    if args.kid:
        k = kid_poly(feats_real, feats_recon, subset_size=min(200, recon["n"]), seed=args.seed)
        out["rkid"] = {"kid": k["kid"], "kid_std": k["kid_std"]}

    floor = load_floor_fid()
    if floor is not None:
        out["floor_fid"] = floor

    # ── verdict ──────────────────────────────────────────────────────────────────
    if args.target_fid is not None:
        threshold = args.target_fid * args.margin
        passed = rfid <= threshold
        out["threshold"] = threshold
        out["ratio_to_target"] = rfid / args.target_fid
        out["pass"] = bool(passed)
        out["verdict"] = (
            f"VAE certified — ships as-is (rFID {rfid:.3f} ≪ target {args.target_fid:.3f})"
            if passed else
            f"GATE FAILED (rFID {rfid:.3f} > {args.margin:.2f}×target {args.target_fid:.3f} "
            f"= {threshold:.3f}) — run the conditional fine-tune"
        )
    else:
        out["pass"] = None
        out["verdict"] = "indeterminate — pass --target-fid for a pass/fail decision"

    # ── report ───────────────────────────────────────────────────────────────────
    print(f"\n── rFID CEILING CHECK (vae_step {step}, n={recon['n']}, xrv domain space)")
    print(f"   rFID        = {rfid:.4f}   (real originals → recon; the ceiling)")
    print(f"   rFID_codec  = {rfid_codec:.4f}   (input@{args.res} → recon; resize removed)")
    print(f"   SSIM        = {recon['ssim']:.4f}   LPIPS = {recon['lpips']:.4f}")
    if floor is not None:
        print(f"   floor.fid   = {floor['fid']:.4f}   (real-vs-real noise floor — rFID sits above this)")
    if "rkid" in out:
        print(f"   rKID        = {out['rkid']['kid']:+.5f} ± {out['rkid']['kid_std']:.5f}")
    print(f"\n   {out['verdict']}")

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults → {out_path}")

    if args.target_fid is not None and not out["pass"]:
        sys.exit(1)   # non-zero exit so a watcher/CI can branch to the fine-tune task


if __name__ == "__main__":
    main()
