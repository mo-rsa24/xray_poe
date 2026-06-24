"""Generator-controlled naive-overlay baseline for N2 (one command).

The fair overlay baseline for "composition beats overlay": instead of blending two
REAL single-disease X-rays (which inherit real-pixel realism and sit at the codec
floor, making the comparison unwinnable by construction), generate the two single-
disease sets from the SAME LDM and overlay THOSE. Now both the compose set and the
overlay set are synthetic and pay the same VAE/codec tax, so the only difference is
how the two diseases are combined — PoE in noise-space vs blend in pixel-space —
which is the actual claim.

Pipeline (each step a subprocess to an already-tested script; no torch here):
    1. generate N cardiomegaly-only images from the LDM   (scripts/generate.py)
    2. generate N effusion-only images from the LDM        (scripts/generate.py, +seed)
    3. overlay the two sets                                 (scripts/naive_overlay.py)
    4. domain-FID(real both, overlay)                       (metrics.fid xrv)
    5. (optional) compare against the compose FID           (scripts/compare_fid.py)

CLI:
    python scripts/overlay_baseline.py --ckpt ckpts/model_step0040000.safetensors \\
        --n 500 --out results/overlay_gen_fid.json \\
        --compare-with results/ckpt_eval/0040000/fid.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_baseline(args: argparse.Namespace) -> int:
    ckpt = Path(args.ckpt) if Path(args.ckpt).is_absolute() else ROOT / args.ckpt
    if not ckpt.is_file() and not args.dry_run:
        raise SystemExit(f"--ckpt: no such file: {ckpt}")

    gen_out = (ROOT / args.gen_out) if not Path(args.gen_out).is_absolute() else Path(args.gen_out)
    cardio_dir = gen_out / "single" / "cardiomegaly"   # generate.py's single-disease layout
    effusion_dir = gen_out / "single" / "effusion"
    overlay_dir = gen_out / "overlay"
    out_json = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    py = sys.executable

    common = ["--ckpt", str(ckpt), "--vae-ckpt", args.vae_ckpt,
              "--scale-factor", args.scale_factor, "--steps", str(args.steps),
              "--gen-batch-size", str(args.gen_batch_size), "-o", str(gen_out)]

    steps: list[tuple[str, list[str]]] = [
        ("gen cardiomegaly", [
            py, "scripts/generate.py", "--disease", "cardiomegaly", "--w", str(args.w),
            "--n", str(args.n), "--seed", str(args.seed), *common]),
        ("gen effusion", [
            py, "scripts/generate.py", "--disease", "effusion", "--w", str(args.w),
            "--n", str(args.n), "--seed", str(args.seed + 100), *common]),   # +100 → independent anatomy
        ("overlay", [
            py, "scripts/naive_overlay.py", "--a", str(cardio_dir), "--b", str(effusion_dir),
            "--n", str(args.n), "--blend", args.blend, "--seed", str(args.seed),
            "--out", str(overlay_dir)]),
        ("overlay-FID", [
            py, "-m", "metrics.fid", "--embed", "xrv", "--kid", "--n", str(args.n),
            "--a", args.real, "--b", str(overlay_dir), "--out", str(out_json)]),
    ]
    if args.compare_with:
        steps.append((
            "compare", [
                py, "scripts/compare_fid.py",
                "--set", f"PoE compose={args.compare_with}",
                "--set", f"Overlay (gen)={out_json}",
                "--floor", args.floor]))

    if args.dry_run:
        for name, cmd in steps:
            print(f"[dry-run] {name}: {' '.join(cmd)}")
        return 0

    for name, cmd in steps:
        print(f"\n── {name} ──")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"  [error] {name} failed (exit {r.returncode}) — aborting")
            return r.returncode
    print(f"\nOverlay baseline FID → {out_json}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generator-controlled overlay baseline for the N2 claim (one command).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="LDM checkpoint (.safetensors)")
    p.add_argument("--vae-ckpt", default="ckpts/vae_step0025000.pt")
    p.add_argument("--real", default="data/nih/images", help="Real both-disease set")
    p.add_argument("--scale-factor", default="data/latents/scale_factor.pt")
    p.add_argument("--n", type=int, default=500, help="Images per single-disease set / overlays")
    p.add_argument("--w", type=float, default=1.0, help="Single-disease CFG weight")
    p.add_argument("--blend", choices=["mean", "max", "screen"], default="mean")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen-batch-size", type=int, default=4)
    p.add_argument("--gen-out", default="outputs/overlay_gen", help="Root for gen + overlay PNGs")
    p.add_argument("--out", default="results/overlay_gen_fid.json")
    p.add_argument("--compare-with", default=None,
                   help="Compose FID JSON to compare against (runs compare_fid.py)")
    p.add_argument("--floor", default="results/floor.json", help="Floor JSON for the compare")
    p.add_argument("--dry-run", action="store_true", help="Print commands; run nothing")
    args = p.parse_args()
    sys.exit(run_baseline(args))


if __name__ == "__main__":
    main()
