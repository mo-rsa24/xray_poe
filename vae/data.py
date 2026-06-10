"""Data sources for the VAE.

Synthetic (local profiling / gates):
  * ``NoiseDataset``       — random tensors, content-independent profiling.
  * ``fixed_overfit_batch``— deterministic low-freq sinusoids, overfit-sanity gate.

Real (RunPod training):
  * ``RealCXRDataset``     — NIH ChestX-ray14 PNGs → grayscale 512² → [-1,1].
  * ``real_cxr_loader()``  — DataLoader factory over train or val split.
  * ``make_splits()``      — stratified train/val split from Data_Entry_2017.csv.

All loaders share the same (B,1,res,res) float32 contract in [-1,1] so the
training loop is identical across data sources.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class NoiseDataset(Dataset):
    """Deterministic-per-index Gaussian noise in [-1,1], shaped (1, res, res)."""

    def __init__(self, res: int = 512, length: int = 4096, channels: int = 1, seed: int = 0):
        self.res = res
        self.length = length
        self.channels = channels
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed + idx)
        x = torch.randn(self.channels, self.res, self.res, generator=g)
        return x.clamp(-3, 3) / 3.0  # ~[-1,1]


def noise_loader(
    res: int = 512,
    batch: int = 8,
    length: int = 4096,
    channels: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    ds = NoiseDataset(res=res, length=length, channels=channels)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=num_workers, drop_last=True)


# ---------------------------------------------------------------------------
# Real data — NIH ChestX-ray14
# ---------------------------------------------------------------------------

def make_splits(
    csv_path: str,
    image_dir: str,
    val_fraction: float = 0.05,
    seed: int = 42,
    frontal_only: bool = True,
) -> tuple[list[str], list[str]]:
    """Return (train_paths, val_paths) from Data_Entry_2017.csv.

    Filters to frontal views (PA/AP) by default — lateral views are excluded
    because they look structurally different and the VAE trains on the majority.
    The split is stratified by the four-group label so each split has a
    representative mix.
    """
    img_dir = Path(image_dir)
    groups: dict[str, list[str]] = {"normal": [], "cardio": [], "effusion": [], "both": [], "other": []}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if frontal_only and row.get("View Position", "").strip() not in ("PA", "AP"):
                continue
            fname = row["Image Index"].strip()
            fpath = str(img_dir / fname)
            if not Path(fpath).exists():
                continue
            labels = set(row["Finding Labels"].split("|"))
            has_cardio = "Cardiomegaly" in labels
            has_effusion = "Effusion" in labels
            if has_cardio and has_effusion:
                groups["both"].append(fpath)
            elif has_cardio:
                groups["cardio"].append(fpath)
            elif has_effusion:
                groups["effusion"].append(fpath)
            elif labels == {"No Finding"}:
                groups["normal"].append(fpath)
            else:
                groups["other"].append(fpath)

    rng = random.Random(seed)
    train_paths, val_paths = [], []
    for paths in groups.values():
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_fraction)) if paths else 0
        val_paths.extend(paths[:n_val])
        train_paths.extend(paths[n_val:])

    rng.shuffle(train_paths)
    rng.shuffle(val_paths)
    return train_paths, val_paths


class RealCXRDataset(Dataset):
    """NIH ChestX-ray14 PNGs → (1, res, res) float32 in [-1, 1].

    Expects images already present on disk at the given paths (produced by
    ``scripts/download_nih.sh``). No augmentation — the VAE is label-blind and
    trains to reconstruct faithfully; augmentation would corrupt the recon ceiling.
    """

    def __init__(self, paths: list[str], res: int = 512):
        self.paths = paths
        self.res = res

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("L")  # grayscale
        if img.size != (self.res, self.res):
            img = img.resize((self.res, self.res), Image.LANCZOS)
        t = torch.from_numpy(__import__("numpy").array(img, dtype="float32"))
        t = t / 127.5 - 1.0          # [0,255] → [-1,1]
        return t.unsqueeze(0)         # (1, res, res)


def real_cxr_loader(
    paths: list[str],
    res: int = 512,
    batch: int = 8,
    num_workers: int = 4,
    split: Literal["train", "val"] = "train",
) -> DataLoader:
    ds = RealCXRDataset(paths, res=res)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )


def fixed_overfit_batch(batch: int = 8, res: int = 512, channels: int = 1, seed: int = 1234) -> torch.Tensor:
    """A single fixed batch (shuffle/aug off) for the overfit-sanity run.

    Deliberately **structured / low-frequency**, not white noise: the overfit gate
    asks whether the loop can drive recon→~0 on a *compressible* signal. At f=4 the
    codec discards ~75% of incompressible content, so pure Gaussian noise cannot be
    reconstructed to ~0 by construction and would falsely fail the gate. Each image
    is a sum of a few low-frequency sinusoids (deterministic per index) — smooth,
    compressible, and CXR-like in spatial scale.
    """
    ys = torch.linspace(-1, 1, res).view(res, 1)
    xs = torch.linspace(-1, 1, res).view(1, res)
    imgs = []
    for i in range(batch):
        g = torch.Generator().manual_seed(seed + i)
        img = torch.zeros(res, res)
        for _ in range(4):  # a few random low-frequency components
            fx, fy = torch.rand(2, generator=g) * 3 + 0.5
            phase = torch.rand(1, generator=g) * 6.283
            amp = torch.rand(1, generator=g) * 0.5 + 0.25
            img = img + amp * torch.sin(fx * 3.14159 * xs + fy * 3.14159 * ys + phase)
        img = img / img.abs().amax().clamp_min(1e-6)   # → [-1,1]
        imgs.append(img)
    x = torch.stack(imgs).unsqueeze(1)                 # (B,1,res,res)
    if channels != 1:
        x = x.repeat(1, channels, 1, 1)
    return x
