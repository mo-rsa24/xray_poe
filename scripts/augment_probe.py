"""Plan-08 augment-probe (EXPLORATORY, low-stakes): do the model's own best fake hearts
help a cardio detector?

Frozen backbone (densenet121-res224-nih — the fixed measuring instrument), retrain only the
top head. Three training arms, each x 3 seeds, all tested on the SAME leakage-held-out real
test set (data/splits/exp06_manifests/test.csv):

  A0 = real only            (train_base + both_pool)
  A1 = real + ALL fake      (+ synth manifest_all.csv)
  A2 = real + big-heart fake (+ synth manifest_ctr_gt0.5.csv)

Because the backbone is frozen it is a pure feature extractor, so we cache each image's 1024-d
pooled feature ONCE (to --cache) and train the tiny head on cached features — the whole 9-run
sweep is minutes, not hours. Reports cardio AUC (threshold-free, primary) and sensitivity at
fixed specificity 0.90, mean +/- 95% CI across the 3 seeds, every single run's value saved.

Read: A2 > A0 (fakes helped) AND A2 > A1 (the big-heart filter did it). Even a win is small and
only shows up after filtering — it backs the "model draws hearts too small" finding, it does not
overturn the paper's story.

Usage (env: micromamba activate jaxstack):
    python scripts/augment_probe.py \
        --synth-dir results/section4/augment_probe/synth/w6 \
        --out results/section4/augment_probe --tag w6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.finetune_classifier import PresenceClassifier

try:
    import torchxrayvision as xrv
except ImportError:
    raise ImportError("pip install torchxrayvision")

from PIL import Image

MAN = ROOT / "data/splits/exp06_manifests"
RESERVED = ROOT / "data/splits/test_patient_ids.txt"


def _xrv(path) -> torch.Tensor:
    arr = np.array(Image.open(ROOT / path).convert("L"), dtype=np.float32)
    arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)
    arr = xrv.datasets.XRayCenterCrop()(arr)
    arr = xrv.datasets.XRayResizer(224)(arr)
    return torch.from_numpy(arr.copy())


@torch.no_grad()
def extract_features(paths, backbone, device, cache_path, bs=64):
    """1024-d frozen-backbone feature per path, cached to disk keyed by path string."""
    cache = {}
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False)
    todo = [p for p in paths if p not in cache]
    if todo:
        print(f"  extracting features for {len(todo)} new images "
              f"({len(cache)} cached) ...", flush=True)
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            x = torch.stack([_xrv(p) for p in batch]).to(device)
            f = backbone(x)
            f = nn.functional.relu(f, inplace=False)
            f = nn.functional.adaptive_avg_pool2d(f, (1, 1))
            f = torch.flatten(f, 1).cpu()
            for p, v in zip(batch, f):
                cache[p] = v.clone()
            if (i // bs) % 20 == 0:
                print(f"    {i+len(batch)}/{len(todo)}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
    return cache


class Head(nn.Module):
    def __init__(self, feat_dim=1024, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(256, 2))
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def sens_at_spec(labels, probs, spec=0.90) -> float:
    """Sensitivity (TPR) at the threshold giving >= `spec` specificity on the negatives."""
    fpr, tpr, thr = roc_curve(labels, probs)
    ok = (1 - fpr) >= spec
    return float(tpr[ok].max()) if ok.any() else float("nan")


def train_head(Xtr, Ytr, seed, device, epochs=80, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    n = len(Ytr); npos = Ytr.sum(0).clamp(min=1)
    pos_weight = ((n - npos) / npos).to(device)
    head = Head().to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    head.train()
    idx = torch.arange(n, device=device)
    for _ in range(epochs):
        perm = idx[torch.randperm(n, device=device)]
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            opt.zero_grad(); loss = crit(head(Xtr[b]), Ytr[b]); loss.backward(); opt.step()
    return head


@torch.no_grad()
def eval_head(head, Xte, te_df, device):
    head.eval()
    prob = torch.sigmoid(head(Xte.to(device))).cpu().numpy()
    y = te_df["cardio"].values
    return {"auc": roc_auc_score(y, prob[:, 0]),
            "sens@spec0.9": sens_at_spec(y, prob[:, 0], 0.90)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", required=True, help="pile dir with manifest_all.csv + manifest_ctr_gt0.5.csv")
    ap.add_argument("--out", default="results/section4/augment_probe")
    ap.add_argument("--tag", default="w6", help="label for this synth source (w6 / w4 / w4_bo8)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--backbone", default="densenet121-res224-nih")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(MAN / "train_base.csv")
    pool = pd.read_csv(MAN / "both_pool.csv")
    test = pd.read_csv(MAN / "test.csv")
    real = pd.concat([base, pool], ignore_index=True)

    syn_all = pd.read_csv(Path(args.synth_dir) / "manifest_all.csv")
    syn_big = pd.read_csv(Path(args.synth_dir) / "manifest_ctr_gt0.5.csv")
    # synth rows -> repo-relative paths, cardio=1 effusion=0
    def _syn(df):
        d = df.copy()
        d["path"] = d["path"].apply(lambda p: str(Path(p).relative_to(ROOT)) if Path(p).is_absolute() else p)
        d["cardio"] = 1; d["effusion"] = 0
        return d[["path", "cardio", "effusion"]]
    syn_all, syn_big = _syn(syn_all), _syn(syn_big)
    print(f"[{args.tag}] real={len(real)}  synth_all={len(syn_all)}  synth_big={len(syn_big)}  test={len(test)}")

    # leakage re-assert: real train shares no reserved test patient
    reserved = {int(x) for x in RESERVED.read_text().split()}
    assert not (set(real["patient_id"]) & reserved), "TRAIN LEAKED A RESERVED TEST PATIENT"

    # frozen feature extractor + cache
    clf = PresenceClassifier(backbone_weights=args.backbone).to(dev).eval()
    backbone = clf.features
    for p in backbone.parameters():
        p.requires_grad = False
    all_paths = list(dict.fromkeys(
        real["path"].tolist() + test["path"].tolist() + syn_all["path"].tolist()))
    feats = extract_features(all_paths, backbone, dev, out / f"feat_cache_{args.tag}.pt")

    def mat(df):
        X = torch.stack([feats[p] for p in df["path"]])
        Y = torch.tensor(df[["cardio", "effusion"]].values, dtype=torch.float32)
        return X, Y

    Xr, Yr = mat(real)
    Xa, Ya = mat(syn_all)
    Xb, Yb = mat(syn_big)
    Xte = torch.stack([feats[p] for p in test["path"]])

    arms = {
        "A0_real":      (Xr, Yr),
        "A1_real+all":  (torch.cat([Xr, Xa]), torch.cat([Yr, Ya])),
        "A2_real+big":  (torch.cat([Xr, Xb]), torch.cat([Yr, Yb])),
    }

    rows = []
    for arm, (X, Y) in arms.items():
        for s in args.seeds:
            head = train_head(X, Y, s, dev, epochs=args.epochs)
            m = eval_head(head, Xte, test, dev)
            rows.append({"tag": args.tag, "arm": arm, "seed": s,
                         "auc": round(m["auc"], 4), "sens@spec0.9": round(m["sens@spec0.9"], 4),
                         "n_train": len(Y)})
            print(f"  {arm:<14} seed{s}  AUC={m['auc']:.3f}  sens@spec0.9={m['sens@spec0.9']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / f"probe_runs_{args.tag}.csv", index=False)

    # aggregate mean +/- 95% CI (t-ish, n=3 -> use 1.96*sd/sqrt(n) as a rough band)
    print(f"\n=== {args.tag} summary (mean +/- 95% band across {len(args.seeds)} seeds) ===")
    summ = []
    for arm in arms:
        g = df[df["arm"] == arm]
        for met in ("auc", "sens@spec0.9"):
            v = g[met].values
            mean, band = float(v.mean()), 1.96 * float(v.std(ddof=1)) / np.sqrt(len(v))
            summ.append({"tag": args.tag, "arm": arm, "metric": met,
                         "mean": round(mean, 4), "ci95": round(band, 4)})
            print(f"  {arm:<14} {met:<13} {mean:.3f} +/- {band:.3f}")
    pd.DataFrame(summ).to_csv(out / f"probe_summary_{args.tag}.csv", index=False)
    print(f"\nwrote {out}/probe_runs_{args.tag}.csv and probe_summary_{args.tag}.csv")


if __name__ == "__main__":
    main()
