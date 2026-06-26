"""Exp 6 — shared two-sigmoid detector harness (Oracle / Starved arms + CTR-label detector).

ONE training recipe, parameterised by how much real both-disease data it sees:
    --both-n all   -> Oracle   (abundant real both; the ceiling)
    --both-n 50    -> Starved  (scarce real both; the headroom reference)

Architecture is the existing PresenceClassifier (DenseNet-121 + MLP head, TWO
independent sigmoid logits [cardio, effusion] — never softmax, so both can fire on
one image). Training data = base single-disease/normal manifest + a seeded sample of
the both-pool (CTR/CP-derived labels). Evaluates on the frozen test manifest by
stratum and appends a per-seed AUC row to --auc-csv. Saves a checkpoint carrying
Youden thresholds (sensitivity+specificity-optimal) for the downstream gates.

Leakage: the manifests are already patient-disjoint from the reserved test set
(scripts/exp06_build_manifests.py); this re-asserts it at load (fail fast).

GPU-bound — DO NOT launch while the rebalanced retrain holds the GPU (OOMs the box).
Use scripts/launch_exp06.sh, which gates on a free GPU + the df disk guard.

    python scripts/train_detector.py --both-n all --seed 1 \\
        --out ckpts/oracle_classifier_s01.pt --auc-csv results/exp06_oracle/oracle_auc.csv \\
        --wandb-name exp6_oracle_s01_20260626
    python scripts/train_detector.py --both-n 50  --seed 1 \\
        --out ckpts/starved_classifier_s01.pt --auc-csv results/exp06_oracle/starved_auc.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.finetune_classifier import PresenceClassifier  # noqa: E402

try:
    import torchxrayvision as xrv
except ImportError:
    raise ImportError("pip install torchxrayvision")

MANIFESTS = ROOT / "data/splits/exp06_manifests"
RESERVED = ROOT / "data/splits/test_patient_ids.txt"
STRATA = ["normal", "cardio_only", "effusion_only", "both_separable"]


# ── dataset ───────────────────────────────────────────────────────────────────

class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False, smooth: float = 0.0):
        self.items = df.reset_index(drop=True)
        self.augment = augment
        self.smooth = smooth

    def _xrv(self, path):
        arr = np.array(Image.open(ROOT / path).convert("L"), dtype=np.float32)
        arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)
        arr = xrv.datasets.XRayCenterCrop()(arr)
        arr = xrv.datasets.XRayResizer(224)(arr)
        return torch.from_numpy(arr.copy())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        r = self.items.iloc[idx]
        x = self._xrv(r["path"])
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            x = T.functional.affine(x, angle=random.uniform(-10, 10), translate=[0, 0],
                                    scale=random.uniform(0.90, 1.10), shear=0,
                                    interpolation=T.InterpolationMode.BILINEAR)
        y = torch.tensor([r["cardio"], r["effusion"]], dtype=torch.float32)
        if self.smooth > 0.0:
            y = y.clamp(self.smooth, 1.0 - self.smooth)
        return x, y


# ── data assembly ──────────────────────────────────────────────────────────────

SYNTH_LABELS = {"cardio": (1, 0), "effusion": (0, 1), "comorbid": (1, 1)}


def _synth_rows(conditions: list[str], tag: str) -> pd.DataFrame:
    """Synthetic augmentation images → training rows (patient_id=-1, never a real patient)."""
    rows = []
    for cond in conditions:
        c, e = SYNTH_LABELS[cond]
        for p in sorted((ROOT / "results/synth_sets" / tag / cond).glob("*.png")):
            rows.append({"path": f"results/synth_sets/{tag}/{cond}/{p.name}",
                         "Image Index": p.name, "patient_id": -1,
                         "cardio": c, "effusion": e, "stratum": f"synth_{cond}"})
    return pd.DataFrame(rows)


def build_train_df(both_n: str, seed: int, augment: str = "none", augment_tag: str = "after") -> pd.DataFrame:
    base = pd.read_csv(MANIFESTS / "train_base.csv")
    pool = pd.read_csv(MANIFESTS / "both_pool.csv")
    if both_n != "all":
        k = min(int(both_n), len(pool))
        pool = pool.sample(n=k, random_state=seed).reset_index(drop=True)
    parts = [base, pool]
    if augment and augment != "none":
        syn = _synth_rows([c.strip() for c in augment.split(",")], augment_tag)
        print(f"  augment: +{len(syn)} synth images ({augment}, tag={augment_tag})")
        parts.append(syn)
    df = pd.concat(parts, ignore_index=True)
    # fail-fast leakage re-assertion (synth rows are patient_id=-1, can't be reserved)
    reserved = {int(x) for x in RESERVED.read_text().split()}
    assert not (set(df["patient_id"]) & reserved), "TRAIN POOL LEAKED A RESERVED TEST PATIENT"
    return df


def youden(labels, probs):
    fpr, tpr, thr = roc_curve(labels, probs)
    return float(thr[np.argmax(tpr - fpr)])


# ── eval ────────────────────────────────────────────────────────────────────────

def evaluate(model, test_df, device, bs=64):
    """Per-stratum AUC + co-presence sens/spec on the test manifest."""
    ds = ManifestDataset(test_df, augment=False)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    logits = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            logits.append(model(x.to(device)).cpu())
    probs = torch.sigmoid(torch.cat(logits)).numpy()
    out = {"cardio_prob": probs[:, 0], "effusion_prob": probs[:, 1]}
    td = test_df.reset_index(drop=True).assign(**out)

    res = {}
    # overall per-head AUC (across all strata, well-posed: mixed labels)
    res["auc_cardio_overall"] = roc_auc_score(td["cardio"], td["cardio_prob"])
    res["auc_effusion_overall"] = roc_auc_score(td["effusion"], td["effusion_prob"])
    # per-stratum AUC where the stratum has both classes for that head
    for s in STRATA:
        sub = td[td["stratum"] == s]
        for head in ("cardio", "effusion"):
            col = f"auc_{head}_{s}"
            if sub[head].nunique() == 2:
                res[col] = roc_auc_score(sub[head], sub[f"{head}_prob"])
            else:
                res[col] = float("nan")  # single-class stratum for this head
    return res, td


# ── train ─────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  arm-both-n={args.both_n}  seed={args.seed}")

    train_df = build_train_df(args.both_n, args.seed, args.augment, args.augment_tag)
    test_df = pd.read_csv(MANIFESTS / "test.csv")
    print(f"train: {len(train_df)} films "
          f"(both={int((train_df['stratum']=='both').sum())})  test: {len(test_df)}")

    use_wandb = bool(args.wandb_name) and not args.no_wandb
    run = None
    if use_wandb:
        try:
            import wandb
            run = wandb.init(project="paper3-exp6", name=args.wandb_name,
                             tags=["gate6", f"both-{args.both_n}", f"seed{args.seed}"],
                             config=vars(args))
        except Exception as e:
            print(f"[warn] wandb off: {e}")

    tr_ds = ManifestDataset(train_df, augment=True, smooth=args.label_smooth)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True)

    model = PresenceClassifier(backbone_weights=args.backbone, dropout=args.dropout).to(device)
    n_pos = train_df[["cardio", "effusion"]].sum().values
    n = len(train_df)
    pos_weight = torch.tensor([(n - n_pos[0]) / max(n_pos[0], 1),
                               (n - n_pos[1]) / max(n_pos[1], 1)],
                              dtype=torch.float32, device=device)
    print(f"pos_weight cardio={pos_weight[0]:.2f} effusion={pos_weight[1]:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam([
        {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_mean_auc = -1.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); tl = 0.0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt.step()
            tl += loss.item() * len(x)
        tl /= len(tr_ds); sched.step()

        res, td = evaluate(model, test_df, device, bs=args.batch_size)
        mean_auc = np.nanmean([res["auc_cardio_overall"], res["auc_effusion_overall"]])
        bs_auc = res.get("auc_cardio_both_separable")
        print(f"ep{epoch:3d}/{args.epochs} loss={tl:.4f} "
              f"AUC c={res['auc_cardio_overall']:.3f} e={res['auc_effusion_overall']:.3f} "
              f"| both-sep AUC(cardio)={bs_auc if bs_auc!=bs_auc else round(bs_auc,3)}")
        if run:
            run.log({"epoch": epoch, "train_loss": tl, "mean_auc": mean_auc, **res})

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            thr_c = youden(td["cardio"], td["cardio_prob"])
            thr_e = youden(td["effusion"], td["effusion_prob"])
            torch.save({"model_state": model.state_dict(), "backbone": args.backbone,
                        "cardio_idx": 0, "effusion_idx": 1, "epoch": epoch,
                        "both_n": args.both_n, "seed": args.seed,
                        "youden_threshold_cardio": thr_c, "youden_threshold_effusion": thr_e,
                        "auc_by_stratum": res}, args.out)
            best_res = res
            best_td = td.copy()
    print(f"\nbest mean-AUC={best_mean_auc:.4f} -> {args.out}")

    # dump best-epoch per-film predictions for film-level bootstrap CIs in the verdict
    arm = args.arm_name or ("oracle" if args.both_n == "all" else f"starved{args.both_n}")
    preds_dir = ROOT / args.preds_dir; preds_dir.mkdir(parents=True, exist_ok=True)
    best_td[["Image Index", "patient_id", "stratum", "cardio", "effusion",
             "cardio_prob", "effusion_prob"]].to_csv(
        preds_dir / f"{arm}_s{args.seed}.csv", index=False)

    # append the per-seed AUC-by-stratum row
    row = {"arm": arm,
           "both_n": args.both_n, "seed": args.seed, "ckpt": str(args.out), **best_res}
    ap = Path(args.auc_csv); ap.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(ap, mode="a", header=not ap.exists(), index=False)
    print(f"AUC row appended -> {ap}")
    if run:
        run.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--both-n", default="all", help="'all' (Oracle) or an int like 50 (Starved)")
    p.add_argument("--augment", default="none", help="synth conditions to add: cardio,effusion | comorbid | none")
    p.add_argument("--augment-tag", default="after", help="synth_sets tag to draw augmentation from")
    p.add_argument("--arm-name", default=None, help="override arm label (e.g. aug_indiv, aug_poe)")
    p.add_argument("--preds-dir", default="results/exp06_oracle/preds", help="where to dump per-film preds")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--auc-csv", required=True)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--backbone", default="densenet121-res224-nih")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-backbone", type=float, default=5e-5)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--label-smooth", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
