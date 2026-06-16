"""Fast precompute of VAE latents for NIH ChestX-ray14.

Functionally identical output to scripts/precompute_latents.py (per-image
{"z": (4,128,128) float32, "label": int} .pt files under {out_dir}/{train,val})
but pipelined for throughput on the RunPod MooseFS network volume:

  * Decode/resize PNGs in parallel via a DataLoader (num_workers) — the
    single-threaded loop in the original ran at ~3.4 img/s; 16 workers reach
    ~400 img/s.
  * Write the per-image .pt files from a ThreadPoolExecutor — serial torch.save
    to MooseFS runs at ~11 files/s; 16 threads reach ~490 files/s.
  * Encode stays in **fp32** (no autocast): bf16 is numerically fragile on this
    VAE and the latents become LDM training targets — not worth the risk.

With decode and writes overlapped, the floor is GPU encode (~19 img/s at B=16),
so the full ~74k-image cache builds in ~1 hour.

The run is **resumable**: images whose output .pt already exists are skipped, so
a disconnect mid-run costs only the in-flight batch.  Pass --overwrite to ignore
existing files.

Usage (background-safe):
    cd /workspace/Paper3
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    nohup python scripts/precompute_latents_fast.py \\
        --csv        data/nih/Data_Entry_2017.csv \\
        --image-dir  data/nih/images \\
        --vae-ckpt   ckpts/vae_step0025000.pt \\
        --out-dir    /workspace/Paper3/data/latents \\
        --batch-size 16 --num-workers 16 --write-workers 16 \\
        > runs/precompute_latents.log 2>&1 &

    # then:
    python scripts/compute_scale_factor.py \\
        --latent-dir data/latents/train --out data/latents/scale_factor.pt
"""

from __future__ import annotations

import os
# Must be set before torch initialises CUDA — avoids fragmentation OOM at B=16.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.precompute_latents import _collect_samples, _load_image  # reuse


class _DecodeDataset(Dataset):
    """Yields (image_tensor | None, label, out_path_str) for each job."""

    def __init__(self, jobs: list[tuple[str, int, str]], res: int):
        self.jobs = jobs
        self.res = res

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, i: int):
        path, label, out_path = self.jobs[i]
        try:
            img = _load_image(path, self.res)
        except Exception as exc:  # noqa: BLE001 — log and drop, don't kill the run
            print(f"  SKIP {path}: {exc}", file=sys.stderr)
            img = None
        return img, label, out_path


def _collate(batch):
    """Drop failed decodes; stack the rest."""
    imgs = [b[0] for b in batch if b[0] is not None]
    if not imgs:
        return None
    labels = [b[1] for b in batch if b[0] is not None]
    outs = [b[2] for b in batch if b[0] is not None]
    return torch.stack(imgs), labels, outs


def _build_jobs(
    samples: list[tuple[str, int]], out_dir: Path, overwrite: bool
) -> list[tuple[str, int, str]]:
    """Map (path, label) -> (path, label, out_path), skipping existing unless overwrite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, int, str]] = []
    skipped = 0
    for path, label in samples:
        out_path = out_dir / f"{Path(path).stem}.pt"
        if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        jobs.append((path, label, str(out_path)))
    return jobs, skipped


def encode_split(
    samples: list[tuple[str, int]],
    vae,
    out_dir: Path,
    device: torch.device,
    split_name: str,
    args: argparse.Namespace,
) -> None:
    jobs, skipped = _build_jobs(samples, out_dir, args.overwrite)
    print(
        f"[{split_name}] {len(samples)} total | {skipped} already cached | "
        f"{len(jobs)} to encode",
        flush=True,
    )
    if not jobs:
        return

    loader = DataLoader(
        _DecodeDataset(jobs, args.res),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=_collate,
        pin_memory=True,
    )

    writer = ThreadPoolExecutor(max_workers=args.write_workers)
    pending: list = []

    def _save(z_i: torch.Tensor, label: int, out_path: str) -> None:
        torch.save({"z": z_i, "label": int(label)}, out_path)

    t0 = time.time()
    done = 0
    for batch in tqdm(loader, desc=f"Encoding {split_name}", unit="batch"):
        if batch is None:
            continue
        x, labels, outs = batch
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            z = vae.encode(x).cpu().float()
        for i in range(z.shape[0]):
            pending.append(writer.submit(_save, z[i].clone(), labels[i], outs[i]))
        done += z.shape[0]
        # Keep the futures list (and any surfaced exceptions) bounded.
        if len(pending) > 4096:
            for f in pending:
                f.result()
            pending = []

    for f in pending:  # surface any write errors, wait for drain
        f.result()
    writer.shutdown(wait=True)
    dt = time.time() - t0
    print(
        f"[{split_name}] encoded {done} images in {dt/60:.1f} min "
        f"({done/max(dt,1e-9):.1f} img/s)",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast precompute of VAE latents (NIH CXR)")
    p.add_argument("--csv", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=16, help="DataLoader decode workers")
    p.add_argument("--write-workers", type=int, default=16, help="Threaded .pt writers")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-encode even if an output .pt already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    from vae.model import VAE
    from vae.config import DEFAULT_CONFIG as vae_cfg

    vae = VAE(vae_cfg).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt.get("model", ckpt.get("model_state", ckpt)))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"Loaded VAE from {args.vae_ckpt}", flush=True)

    t = time.time()
    train_samples, val_samples = _collect_samples(
        args.csv, args.image_dir, args.val_fraction, args.seed
    )
    print(
        f"Scan: {len(train_samples)} train | {len(val_samples)} val "
        f"({time.time()-t:.0f}s)",
        flush=True,
    )

    out_root = Path(args.out_dir)
    if args.split in ("train", "both"):
        encode_split(train_samples, vae, out_root / "train", device, "train", args)
    if args.split in ("val", "both"):
        encode_split(val_samples, vae, out_root / "val", device, "val", args)

    print(
        f"\nDone.  Now run:\n"
        f"  python scripts/compute_scale_factor.py \\\n"
        f"      --latent-dir {out_root}/train \\\n"
        f"      --out        {out_root}/scale_factor.pt",
        flush=True,
    )


if __name__ == "__main__":
    main()
