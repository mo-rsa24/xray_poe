"""PyTorch Dataset over raw NIH ChestX-ray14 PNGs with class labels.

Wraps vae.data.make_splits to reuse the same split logic (frontal-only,
stratified by pathology group, optional no_finding_cap) and exposes the
(image, label) contract that train_ldm.py expects.

Label mapping:
    0  No Finding  (normal)
    1  Cardiomegaly only
    2  Effusion only
    (both/other groups are excluded — label space stays clean for 3-class CFG)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler

from vae.data import make_splits


_LABEL_NORMAL = 0
_LABEL_CARDIO = 1
_LABEL_EFFUSION = 2

# Groups produced by make_splits that map directly to integer labels.
_GROUP_TO_LABEL: dict[str, int] = {
    "normal": _LABEL_NORMAL,
    "cardio": _LABEL_CARDIO,
    "effusion": _LABEL_EFFUSION,
}


def _build_samples(
    csv_path: str,
    image_dir: str,
    split: str,
    no_finding_cap: int | None,
    val_fraction: float,
    seed: int,
) -> list[tuple[str, int]]:
    """Return [(abs_path, label), ...] for the requested split, excluding both/other."""
    import csv as _csv

    img_dir = Path(image_dir)
    groups: dict[str, list[str]] = {
        "normal": [], "cardio": [], "effusion": [], "both": [], "other": []
    }

    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get("View Position", "").strip() not in ("PA", "AP"):
                continue
            fname = row["Image Index"].strip()
            p = img_dir / fname
            if not p.exists() or p.stat().st_size == 0:
                continue
            labels = set(row["Finding Labels"].split("|"))
            has_cardio = "Cardiomegaly" in labels
            has_effusion = "Effusion" in labels
            if has_cardio and has_effusion:
                groups["both"].append(str(p))
            elif has_cardio:
                groups["cardio"].append(str(p))
            elif has_effusion:
                groups["effusion"].append(str(p))
            elif labels == {"No Finding"}:
                groups["normal"].append(str(p))
            else:
                groups["other"].append(str(p))

    rng = random.Random(seed)
    train_groups: dict[str, list[str]] = {}
    val_groups: dict[str, list[str]] = {}
    for name, paths in groups.items():
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_fraction)) if paths else 0
        val_groups[name] = paths[:n_val]
        train_groups[name] = paths[n_val:]

    if no_finding_cap is not None and split == "train":
        n = min(no_finding_cap, len(train_groups["normal"]))
        train_groups["normal"] = train_groups["normal"][:n]

    chosen = train_groups if split == "train" else val_groups
    samples: list[tuple[str, int]] = []
    for group, label in _GROUP_TO_LABEL.items():
        for path in chosen[group]:
            samples.append((path, label))
    rng.shuffle(samples)
    return samples


class RealCXRDataset(Dataset):
    """NIH ChestX-ray14 frontal PNGs with 3-class labels for LDM training.

    Args:
        csv_path:      Path to Data_Entry_2017.csv.
        image_dir:     Directory containing the PNG files listed in the CSV.
        split:         "train" or "val".
        no_finding_cap: Cap on the number of normal (No Finding) training images.
        val_fraction:  Fraction of each group held out for validation.
        seed:          RNG seed for reproducible splits.
        res:           Target spatial resolution (images are resized to res×res).
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        split: str = "train",
        no_finding_cap: int | None = None,
        val_fraction: float = 0.05,
        seed: int = 42,
        res: int = 512,
    ) -> None:
        self._res = res
        self._samples = _build_samples(
            csv_path, image_dir, split, no_finding_cap, val_fraction, seed
        )
        if not self._samples:
            raise RuntimeError(
                f"RealCXRDataset: no samples found for split='{split}' "
                f"in {csv_path} / {image_dir}"
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        path, label = self._samples[idx]
        # Graceful fallback on corrupt files — cycle forward until a valid one loads.
        for attempt in range(len(self._samples)):
            try:
                img = Image.open(self._samples[(idx + attempt) % len(self._samples)][0])
                img = img.convert("L")
                if self._res != img.width or self._res != img.height:
                    img = img.resize((self._res, self._res), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
                x = torch.from_numpy(arr).unsqueeze(0)   # (1, res, res)
                lbl = self._samples[(idx + attempt) % len(self._samples)][1]
                return x, torch.tensor(lbl, dtype=torch.long)
            except Exception:
                continue
        raise RuntimeError(f"RealCXRDataset: could not load any image near idx={idx}")

    def make_sampler(self, effusion_weight: float = 2.0) -> WeightedRandomSampler:
        """Weighted sampler that up-samples effusion to compensate for class imbalance."""
        weights = [
            effusion_weight if lbl == _LABEL_EFFUSION else 1.0
            for _, lbl in self._samples
        ]
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
