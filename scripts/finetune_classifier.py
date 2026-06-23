"""
Fine-tune a 2-class presence classifier on top of DenseNet-121.

Improvements over v1:
  - Unfreeze denseblock4 + norm5 with differential LR (backbone << head)
  - MLP head (Linear→ReLU→Dropout→Linear) instead of single linear layer
  - Data augmentation: random horizontal flip + affine
  - Label smoothing (0.1) to prevent overconfidence
  - Early stopping (patience 12 epochs)
  - Youden-optimal threshold computed on val set and saved with checkpoint

Labelled groups (images with known [cardio, effusion] binary labels):
  data/nih/images_normal/        → [0, 0]
  data/nih/images_cardio_only/   → [1, 0]
  data/nih/images_effusion_only/ → [0, 1]

Both-group images (data/nih/images/) are NOT included — kept as held-out
Rung 0 / Exp8 reference and must never touch the classifier training set.

Usage:
  python scripts/finetune_classifier.py
  python scripts/finetune_classifier.py --epochs 80 --backbone densenet121-res224-nih
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

try:
    import torchxrayvision as xrv
except ImportError:
    raise ImportError("pip install torchxrayvision")


# ── dataset ───────────────────────────────────────────────────────────────────

class NIHPresenceDataset(Dataset):
    GROUPS = {
        "images_normal":        [0.0, 0.0],
        "images_cardio_only":   [1.0, 0.0],
        "images_effusion_only": [0.0, 1.0],
    }

    def __init__(self, root="data/nih", split="train",
                 val_frac=0.2, seed=42, augment=False, smooth=0.0):
        self.augment = augment
        self.smooth  = smooth
        self.items: list[tuple[Path, list[float]]] = []

        for folder, label in self.GROUPS.items():
            paths = sorted(Path(root, folder).glob("*.png"))
            if not paths:
                raise RuntimeError(f"No PNGs in {root}/{folder}")
            rng = random.Random(seed)
            rng.shuffle(paths)
            n_val  = max(1, int(len(paths) * val_frac))
            chosen = paths[n_val:] if split == "train" else paths[:n_val]
            for p in chosen:
                self.items.append((p, label))

        print(f"  {split}: {len(self.items)} images "
              f"({sum(1 for _,l in self.items if l[0]==1)} cardio+, "
              f"{sum(1 for _,l in self.items if l[1]==1)} effusion+)")

    def _xrv_preprocess(self, path):
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
        arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)  # (1,H,W) [-1024,1024]
        arr = xrv.datasets.XRayCenterCrop()(arr)
        arr = xrv.datasets.XRayResizer(224)(arr)
        return torch.from_numpy(arr.copy())   # (1, 224, 224)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        x = self._xrv_preprocess(path)

        if self.augment:
            # Augment in [-1024,1024] space before converting to tensor operations
            # horizontal flip (standard for CXR)
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            # small rotation + scale via affine grid
            angle  = random.uniform(-10, 10)
            scale  = random.uniform(0.90, 1.10)
            x = T.functional.affine(x, angle=angle, translate=[0,0],
                                    scale=scale, shear=0,
                                    interpolation=T.InterpolationMode.BILINEAR)

        y = torch.tensor(label, dtype=torch.float32)
        if self.smooth > 0.0:
            y = y.clamp(self.smooth, 1.0 - self.smooth)
        return x, y


# ── model ─────────────────────────────────────────────────────────────────────

class PresenceClassifier(nn.Module):
    """DenseNet backbone (partly unfrozen) + MLP head → 2 logits [cardio, effusion]."""

    def __init__(self, backbone_weights="densenet121-res224-nih", dropout=0.4):
        super().__init__()
        base = xrv.models.DenseNet(weights=backbone_weights)

        # freeze everything first
        for p in base.parameters():
            p.requires_grad_(False)

        # unfreeze only last dense block + final normalisation
        for name, param in base.named_parameters():
            if "denseblock4" in name or "norm5" in name:
                param.requires_grad_(True)

        self.features = base.features

        # probe feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 224)
            f = base.features(dummy)
            f = nn.functional.relu(f, inplace=False)
            f = nn.functional.adaptive_avg_pool2d(f, (1, 1))
            feat_dim = f.view(1, -1).shape[1]

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.features(x)
        f = nn.functional.relu(f, inplace=False)
        f = nn.functional.adaptive_avg_pool2d(f, (1, 1))
        f = torch.flatten(f, 1)
        return self.head(f)   # raw logits


# ── helpers ───────────────────────────────────────────────────────────────────

def youden_threshold(labels_col, probs_col):
    """Threshold that maximises sensitivity + specificity on val set."""
    fpr, tpr, thresholds = roc_curve(labels_col, probs_col)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


# ── training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading datasets ...")
    train_ds = NIHPresenceDataset(split="train", val_frac=args.val_frac,
                                  seed=args.seed, augment=True, smooth=args.label_smooth)
    val_ds   = NIHPresenceDataset(split="val",   val_frac=args.val_frac,
                                  seed=args.seed, augment=False, smooth=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = PresenceClassifier(backbone_weights=args.backbone,
                                dropout=args.dropout).to(device)

    n_backbone = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
    n_head     = sum(p.numel() for p in model.head.parameters())
    print(f"Backbone trainable (denseblock4+norm5): {n_backbone:,}  Head: {n_head:,}")

    # pos_weight: negative count / positive count per head
    n_cardio_pos  = sum(1 for _,l in train_ds.items if l[0] == 1.0)
    n_effusion_pos = sum(1 for _,l in train_ds.items if l[1] == 1.0)
    n_total       = len(train_ds)
    pos_weight = torch.tensor([
        (n_total - n_cardio_pos)  / max(n_cardio_pos,  1),
        (n_total - n_effusion_pos) / max(n_effusion_pos, 1),
    ], device=device)
    print(f"pos_weight: cardio={pos_weight[0]:.2f}  effusion={pos_weight[1]:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # differential learning rates: backbone << head
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.features.parameters() if p.requires_grad],
         "lr": args.lr_backbone},
        {"params": model.head.parameters(),
         "lr": args.lr_head},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    Path("ckpts").mkdir(exist_ok=True)
    out_path = Path(args.out)
    best_auc, patience_count = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)
        scheduler.step()

        # ── val ──
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_logits.append(model(x.to(device)).cpu())
                all_labels.append(y)
        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs  = 1 / (1 + np.exp(-logits))

        auc_c = roc_auc_score(labels[:, 0], probs[:, 0])
        auc_e = roc_auc_score(labels[:, 1], probs[:, 1])
        mean_auc = (auc_c + auc_e) / 2

        normal_m   = (labels[:, 0] == 0) & (labels[:, 1] == 0)
        cardio_m   = (labels[:, 0] == 1) & (labels[:, 1] == 0)
        effusion_m = (labels[:, 0] == 0) & (labels[:, 1] == 1)

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}"
              f"  AUC cardio={auc_c:.3f} effusion={auc_e:.3f}"
              f"  | normal c={probs[normal_m,0].mean():.3f} e={probs[normal_m,1].mean():.3f}"
              f"  | cardio c={probs[cardio_m,0].mean():.3f}"
              f"  | effusion e={probs[effusion_m,1].mean():.3f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            patience_count = 0

            thr_c = youden_threshold(labels[:, 0], probs[:, 0])
            thr_e = youden_threshold(labels[:, 1], probs[:, 1])

            torch.save({
                "model_state":      model.state_dict(),
                "backbone":         args.backbone,
                "epoch":            epoch,
                "val_auc_cardio":   auc_c,
                "val_auc_effusion": auc_e,
                "cardio_idx":       0,
                "effusion_idx":     1,
                "youden_threshold_cardio":   thr_c,
                "youden_threshold_effusion": thr_e,
            }, out_path)
            print(f"  ✓ saved (AUC={best_auc:.4f}  thr_c={thr_c:.3f}  thr_e={thr_e:.3f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stop — no improvement for {args.patience} epochs")
                break

    print(f"\nDone. Best val AUC={best_auc:.4f}  →  {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",      default="densenet121-res224-nih")
    p.add_argument("--epochs",        type=int,   default=80)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr-backbone",   type=float, default=5e-5)
    p.add_argument("--lr-head",       type=float, default=5e-4)
    p.add_argument("--dropout",       type=float, default=0.4)
    p.add_argument("--val-frac",      type=float, default=0.2)
    p.add_argument("--label-smooth",  type=float, default=0.1)
    p.add_argument("--patience",      type=int,   default=12)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--out",           default="ckpts/presence_classifier_finetuned.pt")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
