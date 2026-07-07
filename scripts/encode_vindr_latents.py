"""Encode VinDr border-visible films under a frozen VAE checkpoint.

Produces data/latents_v2/vindr/ so VinDr and NIH latents share one codec
before leaf 06 decides the LDM training regime.

IMPORTANT: data/latents/vindr/ (VAE-v1 space) is the VinDr-transfer Stage-1
input and must never be overwritten. This script writes to data/latents_v2/vindr/
(or whatever --out-dir specifies).

Usage:
    python scripts/encode_vindr_latents.py \
        --vae-ckpt     ckpts/vae_v2/vae_v2_step0075000.pt \
        --manifest     data/latents/vindr/border_visible_manifest.csv \
        --image-dir    data/vindr512/vinbigdata/train/ \
        --out-dir      data/latents_v2/vindr/ \
        --scale-factor data/latents_v2/scale_factor.pt \
        --batch-size   16 \
        --device       cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vae.config import VAEConfig
from vae.model import VAE

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

RESIZE = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


class VinDrEncodeDataset(Dataset):
    def __init__(self, jobs: list[tuple[str, int, str]]):
        # jobs: (image_path, label, out_path)
        self.jobs = jobs

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, i: int):
        img_path, label, out_path = self.jobs[i]
        try:
            from PIL import Image
            img = Image.open(img_path).convert("L")
            tensor = RESIZE(img)
        except Exception as e:
            print(f"  SKIP {img_path}: {e}", file=sys.stderr)
            tensor = None
        return tensor, label, out_path


def _collate(batch):
    valid = [(t, l, p) for t, l, p in batch if t is not None]
    if not valid:
        return None
    tensors, labels, paths = zip(*valid)
    return torch.stack(tensors), list(labels), list(paths)


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------

def _load_vae(ckpt_path: str, device: torch.device) -> VAE:
    cfg = VAEConfig(input_resolution=512, use_checkpoint=False)
    vae = VAE(cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # "ema" key = vae_v2 EMA weights; legacy keys for older checkpoints
    state = ckpt.get("ema", ckpt.get("model", ckpt.get("model_state", ckpt)))
    vae.load_state_dict(state, strict=False)
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"Loaded VAE from {ckpt_path}  (eval, frozen)", flush=True)
    return vae


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _save(z: torch.Tensor, label: int, out_path: str) -> None:
    tmp = out_path + ".tmp"
    torch.save({"z": z, "label": label}, tmp)
    os.replace(tmp, out_path)


def encode(
    jobs: list[tuple[str, int, str]],
    vae: VAE,
    device: torch.device,
    args: argparse.Namespace,
) -> int:
    loader = DataLoader(
        VinDrEncodeDataset(jobs),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=_collate,
        pin_memory=(str(device).startswith("cuda")),
    )
    writer = ThreadPoolExecutor(max_workers=args.write_workers)
    pending = []
    done = 0
    t0 = time.time()

    for batch in tqdm(loader, desc="Encoding VinDr", unit="batch"):
        if batch is None:
            continue
        x, labels, outs = batch
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            # Must MATCH the NIH path (precompute_latents_fast.py --encode-mode).
            # VAE-v2 is posterior-collapsed (σ≈1.0, signal ~3.4%): a sampled z is
            # ~96% prior noise → decodes to noise → LDM collapses. Use μ (the mode,
            # 100% signal, the VAE's clean reconstruction) for both datasets so the
            # joint latents are homogeneous and high-signal.
            if getattr(args, "encode_mode", "mode") == "mode":
                z, _ = vae.encode_moments(x)     # deterministic posterior mean μ
                z = z.cpu().float()
            else:
                z = vae.encode(x).cpu().float()  # sampled z ~ N(μ,σ) (legacy)
        for i in range(z.shape[0]):
            pending.append(writer.submit(_save, z[i].clone(), labels[i], outs[i]))
        done += z.shape[0]
        if len(pending) > 4096:
            for f in pending:
                f.result()
            pending = []

    for f in pending:
        f.result()
    writer.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"Encoded {done} images in {elapsed/60:.1f} min ({done/max(elapsed,1):.1f} img/s)", flush=True)
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Encode VinDr border-visible films under frozen VAE")
    p.add_argument("--vae-ckpt",     required=True,  help="VAE checkpoint (.pt)")
    p.add_argument("--manifest",     required=True,  help="border_visible_manifest.csv (image_id, class, …)")
    p.add_argument("--image-dir",    required=True,  help="Directory containing {image_id}.png files")
    p.add_argument("--out-dir",      required=True,  help="Output directory for latent .pt files")
    p.add_argument("--scale-factor", default=None,   help="Path to scale_factor.pt (optional — not applied during encoding, used for sanity print)")
    p.add_argument("--encode-mode",  choices=["sample", "mode"], default="mode",
                   help="'mode' = deterministic μ via encode_moments (REQUIRED for "
                        "posterior-collapsed VAE-v2; matches NIH --encode-mode mode); "
                        "'sample' = z~N(μ,σ) legacy.")
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--num-workers",  type=int, default=8)
    p.add_argument("--write-workers",type=int, default=8)
    p.add_argument("--overwrite",    action="store_true", help="Re-encode even if .pt already exists")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load manifest ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.manifest)
    print(f"Manifest: {len(df)} rows  cols={list(df.columns)}", flush=True)

    # resolve image paths — accept image_id (UUID) or full path in manifest
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        # try common alternative
        alt = image_dir.parent.parent / "vinbigdata" / "train"
        if alt.exists():
            print(f"  --image-dir {image_dir} not found; using {alt}", flush=True)
            image_dir = alt
        else:
            print(f"ERROR: image dir not found: {image_dir}", file=sys.stderr)
            sys.exit(1)

    jobs, skipped = [], 0
    for _, row in df.iterrows():
        img_id = str(row.get("image_id", row.get("image_path", ""))).strip()
        # strip extension if present
        stem = Path(img_id).stem if "." in img_id else img_id
        img_path = image_dir / f"{stem}.png"
        out_path = out_dir / f"{stem}.pt"
        label = int(row.get("class", row.get("label", 0)))
        if not args.overwrite and out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        jobs.append((str(img_path), label, str(out_path)))

    print(f"Jobs: {len(jobs)} to encode, {skipped} already cached", flush=True)
    if not jobs:
        print("Nothing to do.", flush=True)
        return

    # ── optional: print scale factor for awareness ─────────────────────────
    if args.scale_factor and Path(args.scale_factor).exists():
        sf = torch.load(args.scale_factor, map_location="cpu", weights_only=False)
        print(f"Scale factor (informational, NOT applied during encoding): {sf}", flush=True)

    # ── encode ────────────────────────────────────────────────────────────────
    vae = _load_vae(args.vae_ckpt, device)
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    done = encode(jobs, vae, device, args)

    # ── final count ──────────────────────────────────────────────────────────
    total = len(list(out_dir.glob("*.pt")))
    print(f"\nTotal .pt in {out_dir}: {total}  (expected ≈ {len(df)})", flush=True)

    # ── append to cache_registry.txt ─────────────────────────────────────────
    registry = Path("results/vae_v2/reencode/cache_registry.txt")
    if registry.exists():
        text = registry.read_text()
        old_line = "Film count VinDr: 0 (NOT YET ENCODED — leaf 05 Task 5 pending)"
        new_line = f"           Film count VinDr: {total} (encoded 2026-06-30)"
        if old_line in text:
            registry.write_text(text.replace(old_line, new_line))
            print(f"Updated {registry}", flush=True)

    # ── spot-check ───────────────────────────────────────────────────────────
    import random, glob as _glob
    files = _glob.glob(str(out_dir / "*.pt"))
    sample = random.sample(files, min(50, len(files)))
    errors = 0
    for f in sample:
        d = torch.load(f, map_location="cpu", weights_only=False)
        z = d["z"]
        if z.shape != (4, 128, 128) or z.dtype != torch.float32 or not torch.isfinite(z).all():
            print(f"  FAIL spot-check: {f}  shape={z.shape}  dtype={z.dtype}", file=sys.stderr)
            errors += 1
    if errors == 0:
        print(f"Spot-check OK: {len(sample)} files — shape (4,128,128), float32, finite", flush=True)
    else:
        print(f"Spot-check FAILED: {errors}/{len(sample)} files bad", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
