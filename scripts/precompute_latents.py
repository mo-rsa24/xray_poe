"""Precompute VAE latents for the NIH ChestX-ray14 dataset.

Encodes every frontal (PA/AP) PNG from the three clean classes (Normal,
Cardiomegaly, Effusion) through a frozen VAE and saves the latent tensors
to disk.  The resulting cache is consumed by LatentDataset in train_ldm.py
via --latent-cache, which removes the VAE from the GPU memory budget during
LDM training.

Output layout:
    {out_dir}/
        train/
            <Image_Index_stem>.pt   # {"z": (4,128,128) float32, "label": int}
            ...
        val/
            <Image_Index_stem>.pt
            ...
    (scale_factor.pt is NOT written here — run compute_scale_factor.py after.)

Usage:
    python scripts/precompute_latents.py \\
        --csv        data/nih/Data_Entry_2017.csv \\
        --image-dir  data/nih/images \\
        --vae-ckpt   ckpts/vae/best.pt \\
        --out-dir    data/latents \\
        --batch-size 16 \\
        --device     cuda

    # Then compute the scale factor:
    python scripts/compute_scale_factor.py \\
        --latent-dir data/latents/train \\
        --out        data/latents/scale_factor.pt
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GROUP_LABEL = {"normal": 0, "cardio": 1, "effusion": 2}


def _collect_samples(
    csv_path: str,
    image_dir: str,
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Return (train_samples, val_samples) as [(abs_path, label), ...].

    Excludes 'both' (cardio+effusion co-morbid) and 'other' groups to keep
    the label space clean.  Only frontal (PA/AP) views are included.
    """
    img_dir = Path(image_dir)
    groups: dict[str, list[str]] = {"normal": [], "cardio": [], "effusion": []}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("View Position", "").strip() not in ("PA", "AP"):
                continue
            fname = row["Image Index"].strip()
            p = img_dir / fname
            if not p.exists() or p.stat().st_size == 0:
                continue
            labels_set = set(row["Finding Labels"].split("|"))
            has_cardio = "Cardiomegaly" in labels_set
            has_effusion = "Effusion" in labels_set
            if has_cardio and has_effusion:
                continue  # exclude both-class
            elif has_cardio:
                groups["cardio"].append(str(p))
            elif has_effusion:
                groups["effusion"].append(str(p))
            elif labels_set == {"No Finding"}:
                groups["normal"].append(str(p))

    rng = random.Random(seed)
    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []
    for name, paths in groups.items():
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_fraction)) if paths else 0
        label = _GROUP_LABEL[name]
        val_samples.extend((p, label) for p in paths[:n_val])
        train_samples.extend((p, label) for p in paths[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def _load_image(path: str, res: int) -> torch.Tensor:
    """Load a PNG as a normalised float32 tensor of shape (1, res, res)."""
    img = Image.open(path).convert("L")
    if img.width != res or img.height != res:
        img = img.resize((res, res), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute VAE latents for NIH CXR")
    p.add_argument("--csv", required=True, help="Path to Data_Entry_2017.csv")
    p.add_argument("--image-dir", required=True, help="Directory with NIH PNGs")
    p.add_argument("--vae-ckpt", required=True, help="VAE checkpoint (.pt)")
    p.add_argument("--out-dir", required=True, help="Root output directory for latents")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split", choices=["train", "val", "both"], default="both",
        help="Which split(s) to encode (default: both)",
    )
    return p.parse_args()


def encode_split(
    samples: list[tuple[str, int]],
    vae,
    out_dir: Path,
    batch_size: int,
    res: int,
    device: torch.device,
    split_name: str,
) -> dict[str, int]:
    """Encode all samples and write per-file .pt dicts.  Returns class counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"normal": 0, "cardio": 0, "effusion": 0}
    label_to_name = {0: "normal", 1: "cardio", 2: "effusion"}

    # Process in batches.
    for start in tqdm(range(0, len(samples), batch_size), desc=f"Encoding {split_name}"):
        batch = samples[start : start + batch_size]
        imgs: list[torch.Tensor] = []
        valid_batch: list[tuple[str, int]] = []
        for path, label in batch:
            try:
                imgs.append(_load_image(path, res))
                valid_batch.append((path, label))
            except Exception as exc:
                print(f"  SKIP {path}: {exc}", file=sys.stderr)

        if not imgs:
            continue

        x = torch.stack(imgs).to(device)   # (B, 1, res, res), float32, [-1,1]
        with torch.no_grad():
            z = vae.encode(x).cpu().float()  # (B, 4, res//4, res//4)

        for i, (path, label) in enumerate(valid_batch):
            stem = Path(path).stem
            out_path = out_dir / f"{stem}.pt"
            torch.save({"z": z[i].clone(), "label": label}, out_path)
            counts[label_to_name[label]] += 1

    return counts


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # --- Load VAE -----------------------------------------------------------
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from vae.model import VAE
    from vae.config import DEFAULT_CONFIG as vae_cfg

    vae_model = VAE(vae_cfg).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae_model.load_state_dict(ckpt.get("model", ckpt.get("model_state", ckpt)))
    vae_model.eval()
    for p in vae_model.parameters():
        p.requires_grad_(False)
    print(f"Loaded VAE from {args.vae_ckpt}")

    # --- Collect samples ----------------------------------------------------
    train_samples, val_samples = _collect_samples(
        args.csv, args.image_dir, args.val_fraction, args.seed
    )
    print(f"Train: {len(train_samples)} samples | Val: {len(val_samples)} samples")

    out_root = Path(args.out_dir)

    # --- Encode -------------------------------------------------------------
    if args.split in ("train", "both"):
        counts = encode_split(
            train_samples, vae_model, out_root / "train",
            args.batch_size, args.res, device, "train",
        )
        print(f"Train encoded — normal={counts['normal']}, "
              f"cardio={counts['cardio']}, effusion={counts['effusion']}")

    if args.split in ("val", "both"):
        counts = encode_split(
            val_samples, vae_model, out_root / "val",
            args.batch_size, args.res, device, "val",
        )
        print(f"Val encoded   — normal={counts['normal']}, "
              f"cardio={counts['cardio']}, effusion={counts['effusion']}")

    print(
        f"\nDone.  Now run:\n"
        f"  python scripts/compute_scale_factor.py \\\n"
        f"      --latent-dir {out_root}/train \\\n"
        f"      --out        {out_root}/scale_factor.pt"
    )


if __name__ == "__main__":
    main()
