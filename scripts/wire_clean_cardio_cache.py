"""Leaf 09 — assemble the clean-cardio-augmented latent cache for the Arm-1 retrain.

Reads results/clean_cardio/kept.csv (clean+enlarged cardio = NIH-kept + VinDr-kept, scorer A) and
builds data/latents/rebalanced_cleancardio/{train,val}/:

  - effusion + normal  → symlinked unchanged from data/latents/rebalanced/train/ (label != 1)
  - cardio (label 1)   → REPLACED by the clean+enlarged set:
        NIH-kept   → symlink the existing data/latents/train/<id>.pt
        VinDr-kept → encode the 512² PNG through our VAE, write {"z","label":1}
  - val/               → symlinked unchanged from rebalanced/val/ (loss-tracking only)
  - scale_factor.pt    → symlinked from rebalanced/

Every .pt is the SAME format as the rest of the cache — {"z": (4,128,128) float32 RAW vae.encode,
"label": int} — encoded identically to scripts/precompute_latents.py (no scale_factor here; that's
applied at train time). GPU for the VinDr encode (~1 min). Run after kept.csv exists, GPU free.

    python scripts/wire_clean_cardio_cache.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.generate import _load_vae


def _load_image(path: str, res: int = 512) -> torch.Tensor:
    """Identical to precompute_latents._load_image: PNG → (1,res,res) float32 in [-1,1]."""
    img = Image.open(path).convert("L")
    if img.width != res or img.height != res:
        img = img.resize((res, res), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).unsqueeze(0)


def _link(target: Path, link: Path):
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(target.resolve(), link)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", default="results/clean_cardio/kept.csv")
    ap.add_argument("--src-cache", default="data/latents/rebalanced")
    ap.add_argument("--nih-latents", default="data/latents/train")
    ap.add_argument("--out-cache", default="data/latents/rebalanced_cleancardio")
    ap.add_argument("--vae-ckpt", default="ckpts/vae_step0025000.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    src, out = Path(a.src_cache), Path(a.out_cache)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    # scale_factor (same VAE, same scale)
    if (src / "scale_factor.pt").exists():
        _link(src / "scale_factor.pt", out / "scale_factor.pt")

    # 1) carry over effusion+normal (train, label != 1) and the whole val split unchanged
    def carry(split: str, drop_cardio: bool) -> int:
        n = 0
        for p in sorted((src / split).glob("*.pt")):
            if drop_cardio:
                if int(torch.load(p, map_location="cpu")["label"]) == 1:
                    continue   # old cardio is replaced by the clean set
            _link(p, out / split / p.name)
            n += 1
        return n

    print("[wire-clean] carrying effusion+normal (train) + val …")
    n_eff_norm = carry("train", drop_cardio=True)
    n_val = carry("val", drop_cardio=False)

    # 2) clean cardio → train/
    kept = pd.read_csv(a.kept)
    nih = kept[kept["source"] == "nih"]
    vindr = kept[kept["source"] == "vindr"]

    n_nih = 0
    for iid in nih["image_id"].astype(str):
        latp = Path(a.nih_latents) / f"{iid}.pt"
        if latp.exists():
            _link(latp, out / "train" / f"{iid}.pt")
            n_nih += 1
        else:
            print(f"  WARN no NIH latent for {iid}", file=sys.stderr)

    vae = _load_vae(Path(a.vae_ckpt), torch.device(a.device))
    rows = list(vindr.itertuples())
    n_vindr = 0
    print(f"[wire-clean] encoding {len(rows)} VinDr-kept cardio on {a.device} …")
    for s in range(0, len(rows), a.batch_size):
        chunk = rows[s:s + a.batch_size]
        imgs, ids = [], []
        for r in chunk:
            try:
                imgs.append(_load_image(r.img_path))
                ids.append(str(r.image_id))
            except Exception as e:
                print(f"  SKIP {r.image_id}: {e}", file=sys.stderr)
        if not imgs:
            continue
        x = torch.stack(imgs).to(a.device)
        with torch.no_grad():
            z = vae.encode(x).cpu().float()        # (B,4,128,128) RAW — matches precompute
        for i, iid in enumerate(ids):
            torch.save({"z": z[i].clone(), "label": 1}, out / "train" / f"vindr_{iid}.pt")
            n_vindr += 1

    n_cardio = n_nih + n_vindr
    n_train = len(list((out / "train").glob("*.pt")))
    print(f"\n[wire-clean] effusion+normal: {n_eff_norm}  | val carried: {n_val}")
    print(f"[wire-clean] clean cardio: NIH-symlink {n_nih} + VinDr-encode {n_vindr} = {n_cardio}  (floor 854)")
    print(f"[wire-clean] cache → {out}/train ({n_train} pt), {out}/val ({n_val} pt), scale_factor.pt")
    print(f"  next: point relaunch_retrain.sh / the config at {out} and launch Arm 1 (GPU).")


if __name__ == "__main__":
    main()
