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

The FID is computed in the xrv (NIH DenseNet-121, 1024-d) domain feature space the
rest of the eval uses, and against the *original* real images — the same set S6's
gen-FID scores against — so rFID is the honest, apples-to-apples lower bound on
gen-FID. floor.fid (from results/floor.json) is the noise floor: rFID must sit
*above* it (recon ≠ identity) yet *far below* the target gen-FID.

Pass: rFID <= target_gen_fid * margin  (default margin 1/3 → "far below").
The target may be passed inline (--target-fid) or read from the S6 gen-FID JSON
(--target-from results/fid.json) so the checkpoint-watcher can close the loop with
no manual step.

CLI:
    # the gate, target read straight from the S6 gen-FID result
    python scripts/rfid_gate.py --ckpt ckpts/vae_step0025000.pt \\
        --real data/nih/images --n 1063 --kid \\
        --target-from results/fid.json --out results/rfid_gate.json

    # target supplied inline instead
    python scripts/rfid_gate.py --n 1063 --target-fid 12.0

    # measure only (no target yet → verdict "indeterminate", numbers still recorded)
    python scripts/rfid_gate.py --n 1063
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


# ── reconstruct a real set → write recon PNGs, accumulate SSIM/LPIPS ─────────────────

@torch.no_grad()
def reconstruct_set(
    model: VAE,
    paths: list[Path],
    recon_dir: Path,
    device: torch.device,
    batch_size: int,
    res: int = 512,
) -> dict:
    """Reconstruct each real image; save the recon as a PNG and accumulate SSIM/LPIPS
    over the set. Returns mean recon metrics; the recon dir is fed to the FID harness."""
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
            save_image(recons[j : j + 1].cpu(), recon_dir / f"{Path(p).stem}.png")
        print(f"  reconstructed {n}/{len(paths)}  "
              f"(running SSIM={ssim_sum / n:.4f}  LPIPS={lpips_sum / n:.4f})")

    return {"n": n, "ssim": ssim_sum / max(n, 1), "lpips": lpips_sum / max(n, 1)}


# ── reference loaders ───────────────────────────────────────────────────────────────

def load_floor_fid() -> dict | None:
    """The real-vs-real domain-FID floor from results/floor.json (context only)."""
    p = ROOT / FLOOR_JSON
    if not p.is_file():
        return None
    try:
        f = json.loads(p.read_text())["fid"]
        return {"fid": f["fid"], "upper95": f.get("upper95")}
    except Exception:
        return None


def resolve_target(path: str, key: str) -> float:
    """Read the LDM gen-FID from the S6 result JSON at a dotted `key`.

    Tolerates both the metrics.fid layout ({"fid": {"fid": X, ...}}) and a flat
    {"fid": X}: if the dotted key lands on the fid sub-dict, descend to its scalar.
    """
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.is_file():
        raise SystemExit(f"--target-from: no such file: {p}")
    cur = json.loads(p.read_text())
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            cur = None
            break
    if isinstance(cur, dict) and isinstance(cur.get("fid"), (int, float)):
        cur = cur["fid"]
    if not isinstance(cur, (int, float)):
        raise SystemExit(f"--target-from: could not resolve a numeric FID from {p} "
                         f"at key {key!r} (got {cur!r})")
    return float(cur)


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
                   help="LDM target gen-FID (xrv domain space), supplied inline.")
    p.add_argument("--target-from", default=None, dest="target_from",
                   help="Read the target gen-FID from this JSON instead (e.g. "
                        "results/fid.json from S6). Ignored if --target-fid is given.")
    p.add_argument("--target-key", default="fid", dest="target_key",
                   help="Dotted key into --target-from for the gen-FID (default 'fid')")
    p.add_argument("--margin", type=float, default=1.0 / 3.0,
                   help="rFID must be <= target_fid * margin to pass ('≪'; default 1/3)")
    p.add_argument("--kid", action="store_true", help="Also report domain-KID")
    p.add_argument("--seed", type=int, default=42, help="Real-image sampling seed")
    p.add_argument("--batch_size", type=int, default=16, help="VAE recon batch size")
    p.add_argument("--fid_batch_size", type=int, default=32, help="Feature-extract batch")
    p.add_argument("--workdir", default="outputs/rfid_gate",
                   help="Where recon PNGs are written")
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    p.add_argument("--out", default="results/rfid_gate.json", help="Results JSON")
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    ckpt_path = ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    real_dir = ROOT / args.real if not Path(args.real).is_absolute() else Path(args.real)
    workdir = ROOT / args.workdir if not Path(args.workdir).is_absolute() else Path(args.workdir)

    # Resolve the target up front so a bad --target-from fails before the GPU work.
    target_fid = args.target_fid
    target_src = "inline"
    if target_fid is None and args.target_from is not None:
        target_fid = resolve_target(args.target_from, args.target_key)
        target_src = f"{args.target_from}:{args.target_key}"

    print(f"Loading VAE: {ckpt_path}")
    model = load_vae(ckpt_path, device)
    step = ckpt_path.stem.replace("vae_step", "")

    real_paths = _gather_paths(real_dir, args.n, seed=args.seed)
    print(f"Reconstructing {len(real_paths)} real both-disease images "
          f"from {real_dir} (device={args.device}) ...")

    recon_dir = workdir / "recon"
    recon = reconstruct_set(model, real_paths, recon_dir, device, args.batch_size, res=args.res)

    # Free VAE before the feature extractor pulls in the xrv DenseNet.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nExtracting xrv (domain) features ...")
    feats_real = extract_features([str(q) for q in real_paths], embed="xrv",
                                  device=args.device, batch_size=args.fid_batch_size)
    feats_recon = extract_features(sorted(str(q) for q in recon_dir.glob("*.png")), embed="xrv",
                                   device=args.device, batch_size=args.fid_batch_size)

    rfid = fid_from_features(feats_real, feats_recon)["fid"]   # the codec ceiling

    out: dict = {
        "ckpt": str(ckpt_path),
        "vae_step": int(step) if step.isdigit() else step,
        "real_dir": str(real_dir),
        "n": recon["n"],
        "embed": "xrv",
        "rfid": rfid,
        "ssim": recon["ssim"],
        "lpips": recon["lpips"],
        "target_fid": target_fid,
        "target_src": target_src if target_fid is not None else None,
        "margin": args.margin,
    }

    if args.kid:
        k = kid_poly(feats_real, feats_recon, subset_size=min(200, recon["n"]), seed=args.seed)
        out["rkid"] = {"kid": k["kid"], "kid_std": k["kid_std"]}

    floor = load_floor_fid()
    if floor is not None:
        out["floor_fid"] = floor

    # ── verdict ──────────────────────────────────────────────────────────────────
    if target_fid is not None:
        threshold = target_fid * args.margin
        passed = rfid <= threshold
        out["threshold"] = threshold
        out["ratio_to_target"] = rfid / target_fid
        out["pass"] = bool(passed)
        out["verdict"] = (
            f"VAE certified — ships as-is (rFID {rfid:.3f} ≪ target {target_fid:.3f})"
            if passed else
            f"GATE FAILED (rFID {rfid:.3f} > {args.margin:.2f}×target {target_fid:.3f} "
            f"= {threshold:.3f}) — run the conditional fine-tune"
        )
    else:
        out["pass"] = None
        out["verdict"] = "indeterminate — pass --target-fid or --target-from for a verdict"

    # ── report ───────────────────────────────────────────────────────────────────
    print(f"\n── rFID CEILING CHECK (vae_step {step}, n={recon['n']}, xrv domain space)")
    print(f"   rFID   = {rfid:.4f}   (real originals → recon; the codec ceiling)")
    print(f"   SSIM   = {recon['ssim']:.4f}   LPIPS = {recon['lpips']:.4f}")
    if floor is not None:
        print(f"   floor  = {floor['fid']:.4f}   (real-vs-real noise floor — rFID sits above this)")
    if "rkid" in out:
        print(f"   rKID   = {out['rkid']['kid']:+.5f} ± {out['rkid']['kid_std']:.5f}")
    print(f"\n   {out['verdict']}")

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults → {out_path}")

    if target_fid is not None and not out["pass"]:
        sys.exit(1)   # non-zero exit so a watcher/CI can branch to the fine-tune task


if __name__ == "__main__":
    main()
