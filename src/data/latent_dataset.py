"""Dataset that loads pre-encoded latent tensors from disk.

Cache layout (produced by scripts/precompute_latents.py):

    {cache_dir}/
        scale_factor.pt          # scalar; loaded by train_ldm.py separately
        train/
            <stem>.pt            # {"z": (4,128,128) float32, "label": int}
            ...
        val/
            <stem>.pt
            ...

Each .pt file is a dict with keys "z" (the sampled VAE latent) and "label"
(int in {0,1,2}: 0=No Finding, 1=Cardiomegaly, 2=Effusion).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler


_LABEL_EFFUSION = 2


class LatentDataset(Dataset):
    """Map-style dataset over a directory of pre-encoded .pt latent files.

    Args:
        cache_dir: Root of the latent cache (contains train/ and val/).
        split: One of "train" or "val".
    """

    def __init__(self, cache_dir: str | Path, split: str = "train") -> None:
        split_dir = Path(cache_dir) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Latent cache split directory not found: {split_dir}\n"
                "Run scripts/precompute_latents.py first."
            )
        self._files = sorted(split_dir.glob("*.pt"))
        if not self._files:
            raise RuntimeError(f"No .pt files found in {split_dir}")

        # Cache labels in memory so make_sampler doesn't re-load every file.
        self._labels: list[int] = []
        for f in self._files:
            d = torch.load(f, map_location="cpu", weights_only=True)
            self._labels.append(int(d["label"]))

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        d = torch.load(self._files[idx], map_location="cpu", weights_only=True)
        z: Tensor = d["z"].float()          # (4, 128, 128)
        label = torch.tensor(d["label"], dtype=torch.long)
        return z, label

    def make_sampler(self, effusion_weight: float = 2.0) -> WeightedRandomSampler:
        """Weighted sampler that up-samples effusion to compensate for class imbalance."""
        weights = [
            effusion_weight if lbl == _LABEL_EFFUSION else 1.0
            for lbl in self._labels
        ]
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
