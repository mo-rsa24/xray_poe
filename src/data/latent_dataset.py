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
# Sidecar written into each split dir caching per-file labels (excluded from the
# .pt glob). Makes startup O(1) file reads instead of O(N) after the first run.
_LABELS_CACHE_NAME = "_labels_cache.pt"


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
        self._files = sorted(
            f for f in split_dir.glob("*.pt") if f.name != _LABELS_CACHE_NAME
        )
        if not self._files:
            raise RuntimeError(f"No .pt files found in {split_dir}")

        # Labels are needed up-front for make_sampler. Loading every .pt to read one
        # int is the dominant startup cost (~81k loads for the joint cache → minutes).
        # Cache them to a sidecar so subsequent launches read one small file instead.
        self._labels: list[int] = self._load_labels(split_dir)

    def _load_labels(self, split_dir: Path) -> list[int]:
        """Return per-file labels, using a sidecar cache when valid; else build it
        (with a progress bar) and write the sidecar for next time."""
        cache_path = split_dir / _LABELS_CACHE_NAME
        names = [f.name for f in self._files]

        if cache_path.exists():
            try:
                c = torch.load(cache_path, map_location="cpu", weights_only=False)
                if c.get("names") == names and len(c.get("labels", [])) == len(names):
                    return [int(x) for x in c["labels"]]
            except Exception:
                pass  # corrupt / stale → rebuild below

        # Build with a progress bar (falls back to a one-liner if tqdm is absent).
        try:
            from tqdm import tqdm
            iterator = tqdm(self._files, desc=f"Indexing labels [{split_dir.name}]",
                            unit="file", dynamic_ncols=True)
        except Exception:
            print(f"Indexing labels for {len(self._files)} files in {split_dir} ...",
                  flush=True)
            iterator = self._files

        labels: list[int] = []
        for f in iterator:
            d = torch.load(f, map_location="cpu", weights_only=True)
            labels.append(int(d["label"]))

        try:
            torch.save({"names": names, "labels": labels}, cache_path)
        except Exception as exc:  # read-only fs etc. — non-fatal, just slower next time
            print(f"(label-cache not written to {cache_path}: {exc})", flush=True)
        return labels

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        d = torch.load(self._files[idx], map_location="cpu", weights_only=True)
        z: Tensor = d["z"].float()          # (4, 128, 128)
        label = torch.tensor(d["label"], dtype=torch.long)
        return z, label

    def make_sampler(
        self,
        effusion_weight: float = 2.0,
        balance_classes: bool = False,
        class_mult: dict[int, float] | None = None,
    ) -> WeightedRandomSampler:
        """Weighted sampler for class balance.

        Two modes:
          • legacy (balance_classes=False): per-sample weight = effusion_weight for
            effusion, 1.0 otherwise. Combined with the raw 7:1 effusion:cardio count
            this starves cardiomegaly (~6% of batches) — see ldm_full.yaml history.
          • balanced (balance_classes=True): inverse-frequency weighting so the
            SAMPLED proportion of each class equals its class_mult share (normalised).
            class_mult={0:0.25, 1:0.40, 2:0.35} → 25% normal / 40% cardio / 35% effusion
            regardless of raw counts. Default (class_mult=None) → equal thirds.
        """
        if balance_classes:
            from collections import Counter

            counts = Counter(self._labels)
            mult = class_mult or {}
            weights = [mult.get(lbl, 1.0) / counts[lbl] for lbl in self._labels]
        else:
            weights = [
                effusion_weight if lbl == _LABEL_EFFUSION else 1.0
                for lbl in self._labels
            ]
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
