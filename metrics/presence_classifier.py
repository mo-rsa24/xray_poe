"""Fine-tuned 2-class DenseNet-121 head for presence-rate measurement.

Loads a calibrated checkpoint (ckpts/presence_classifier_finetuned.pt) that was
trained on NIH single-disease images. Decision thresholds are the Youden-optimal
values stored in the checkpoint, not hardcoded constants.

Validates head calibration on real held-out images, then runs on composed outputs
to produce the Exp5 (H1 marginals gate) and Exp6 (joint co-presence) tables.

CLI:
    python -m metrics.presence_classifier --mode validate \\
        --real_cardio   data/nih/images_cardio_only/ \\
        --real_effusion data/nih/images_effusion_only/ \\
        --real_normal   data/nih/images_normal/

    python -m metrics.presence_classifier --mode eval \\
        --composed_cardio outputs/single/cardiomegaly/ \\
        --composed_effusion outputs/single/effusion/ \\
        --n 2000 --out results/exp5_presence.json

    python -m metrics.presence_classifier --mode joint \\
        --composed_both outputs/compose/w1.0/ \\
        --n 1000 --out results/exp5_joint.json

Public API (imported by scripts/eval_ldm.py):
    load_model(device)                      -> (model, cardio_idx, effusion_idx, ckpt)
    predict_probs(model, paths, ...)        -> np.ndarray (N, 2) — [cardio, effusion]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

try:
    import torchxrayvision as xrv
except ImportError:
    raise ImportError(
        "TorchXRayVision not installed. Run: pip install torchxrayvision"
    ) from None

# PresenceClassifier lives in scripts/finetune_classifier.py; import from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.finetune_classifier import PresenceClassifier  # noqa: E402

_CKPT_PATH = "ckpts/presence_classifier_finetuned.pt"
_H1_GAP_MAX = 0.05    # composed presence within 5pp of real → H1 supported


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(device: str = "cpu") -> tuple:
    """Load fine-tuned PresenceClassifier checkpoint.

    Returns (model, cardio_idx, effusion_idx, ckpt) where ckpt contains
    youden_threshold_cardio and youden_threshold_effusion for downstream gates.
    """
    ckpt = torch.load(_CKPT_PATH, map_location=device)
    model = PresenceClassifier(backbone_weights=ckpt["backbone"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model, ckpt["cardio_idx"], ckpt["effusion_idx"], ckpt


# ── image preprocessing ───────────────────────────────────────────────────────

def _preprocess(path: str | Path) -> torch.Tensor:
    """Load a grayscale image into a (1, 224, 224) xrv-normalised tensor."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)                         # (H, W)
    arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)   # (1, H, W), [-1024, 1024]
    arr = xrv.datasets.XRayCenterCrop()(arr)                      # (1, S, S)
    arr = xrv.datasets.XRayResizer(224)(arr)                      # (1, 224, 224)
    return torch.from_numpy(arr.copy())                            # copy: ensure writeable


# ── inference ─────────────────────────────────────────────────────────────────

def predict_probs(
    model,
    paths: Sequence[str | Path],
    cardio_idx: int,
    effusion_idx: int,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Run frozen classifier over paths.

    Returns (N, 2) float32 array — columns: [cardio_prob, effusion_prob].
    Corrupt or unreadable images are replaced with a zero tensor and a warning.
    """
    results: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        chunk = list(paths[i : i + batch_size])
        tensors: list[torch.Tensor] = []
        for p in chunk:
            try:
                tensors.append(_preprocess(p))
            except Exception as exc:
                print(f"  [warn] skipping {Path(p).name}: {exc}")
                tensors.append(torch.zeros(1, 224, 224))
        batch = torch.stack(tensors).to(device)          # (B, 1, 224, 224)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).cpu().numpy()  # (B, num_pathologies)
        results.append(
            np.column_stack([probs[:, cardio_idx], probs[:, effusion_idx]])
        )
    if not results:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(results, axis=0)


# ── statistics helpers ────────────────────────────────────────────────────────

def _gather_paths(dir_path: str | Path, n: int | None, seed: int = 42) -> list[Path]:
    dir_path = Path(dir_path)
    paths = sorted(dir_path.glob("*.png")) + sorted(dir_path.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No PNG/JPG images in {dir_path}")
    if n is not None and n < len(paths):
        rng = random.Random(seed)
        paths = rng.sample(paths, n)
    return paths


def _val_split_paths(dir_path: str | Path, val_frac: float = 0.2,
                     seed: int = 42, n: int | None = None) -> list[Path]:
    """Reproduce the held-out VAL subset used during fine-tuning.

    Mirrors scripts/finetune_classifier.NIHPresenceDataset exactly: sorted *.png →
    random.Random(seed).shuffle → first `int(len*val_frac)`. Validating on these (not
    the whole dir) keeps the check honest — the other 80% were training images.
    """
    dir_path = Path(dir_path)
    paths = sorted(dir_path.glob("*.png"))           # dataset globs *.png only
    if not paths:
        raise RuntimeError(f"No PNG images in {dir_path}")
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_frac))
    val = paths[:n_val]
    if n is not None and n < len(val):
        val = random.Random(seed).sample(val, n)
    return val


def _ci95(values: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    boot = np.array([
        rng.choice(values, len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _summarise(probs_col: np.ndarray) -> dict:
    lo, hi = _ci95(probs_col)
    return {"mean": float(probs_col.mean()), "ci95": [lo, hi], "n": len(probs_col)}


# ── CLI modes ─────────────────────────────────────────────────────────────────

def _run_validate(args: argparse.Namespace, model, cardio_idx, effusion_idx, ckpt: dict) -> bool:
    """Validate head calibration on real images. Prints PASS/FAIL per check.

    Gates use Youden-optimal thresholds from the checkpoint:
      - Disease images: mean probability ≥ Youden threshold (majority correctly classified +)
      - Normal images:  mean probability < Youden threshold for each head
    """
    thr_c = ckpt["youden_threshold_cardio"]
    thr_e = ckpt["youden_threshold_effusion"]

    checks = [
        ("cardio real",   args.real_cardio,   "cardio"),
        ("effusion real", args.real_effusion,  "effusion"),
        ("normal real",   args.real_normal,    "normal"),
    ]
    all_pass = True
    summary: dict = {}

    for name, dir_arg, split in checks:
        if dir_arg is None:
            continue
        if args.held_out:
            paths = _val_split_paths(dir_arg, args.val_frac, args.val_seed, args.n_validate)
            print(f"\n── {name}: {dir_arg}  (held-out val split: {len(paths)} imgs)")
        else:
            paths = _gather_paths(dir_arg, args.n_validate)
            # a materialized split dir (…/splits/val/…) is already held-out; only the raw
            # NIH group dirs mix train+val, so only warn there.
            pre_split = "split" in str(dir_arg).lower() or Path(dir_arg).parent.name == "val"
            note = "" if pre_split else "  — ⚠️ may include training data; use --held_out or data/splits/val/"
            print(f"\n── {name}: {dir_arg}  ({len(paths)} imgs){note}")
        probs = predict_probs(model, paths, cardio_idx, effusion_idx, args.device)

        if split == "cardio":
            s = _summarise(probs[:, 0])
            ok = s["mean"] >= thr_c
            print(f"  cardio  mean={s['mean']:.3f} 95%CI=[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]"
                  f"  (need ≥{thr_c:.3f})  {'PASS' if ok else 'FAIL'}")
            summary["cardio"] = {**s, "pass": ok}
            all_pass &= ok

        elif split == "effusion":
            s = _summarise(probs[:, 1])
            ok = s["mean"] >= thr_e
            print(f"  effusion mean={s['mean']:.3f} 95%CI=[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]"
                  f"  (need ≥{thr_e:.3f})  {'PASS' if ok else 'FAIL'}")
            summary["effusion"] = {**s, "pass": ok}
            all_pass &= ok

        elif split == "normal":
            sc = _summarise(probs[:, 0])
            se = _summarise(probs[:, 1])
            ok_c = sc["mean"] < thr_c
            ok_e = se["mean"] < thr_e
            print(f"  cardio  mean={sc['mean']:.3f}  (need <{thr_c:.3f})  {'PASS' if ok_c else 'FAIL'}")
            print(f"  effusion mean={se['mean']:.3f}  (need <{thr_e:.3f})  {'PASS' if ok_e else 'FAIL'}")
            summary["normal"] = {"cardio": sc, "effusion": se, "pass": bool(ok_c and ok_e)}
            all_pass &= ok_c and ok_e

    verdict = "VALIDATION PASS ✓" if all_pass else "VALIDATION FAIL"
    print(f"\n{verdict}")

    out_path = Path(args.out or "results/pilot_validate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "verdict": verdict,
        "held_out": bool(args.held_out),
        "val_frac": args.val_frac if args.held_out else None,
        "youden_threshold_cardio": thr_c,
        "youden_threshold_effusion": thr_e,
        "val_auc_cardio": ckpt.get("val_auc_cardio"),
        "val_auc_effusion": ckpt.get("val_auc_effusion"),
        "results": summary,
    }, indent=2))
    print(f"Results → {out_path}")

    return all_pass


def _run_eval(args: argparse.Namespace, model, cardio_idx, effusion_idx) -> None:
    """Presence rates on composed single-disease images (Exp5 H1 gate)."""
    out: dict = {}

    pairs = [
        ("cardio",   args.composed_cardio,   0, args.real_cardio_mean),
        ("effusion", args.composed_effusion,  1, args.real_effusion_mean),
    ]
    for label, dir_arg, col, real_mean in pairs:
        if dir_arg is None:
            continue
        print(f"\n── composed {label}: {dir_arg}")
        paths = _gather_paths(dir_arg, args.n)
        probs = predict_probs(model, paths, cardio_idx, effusion_idx, args.device)
        s = _summarise(probs[:, col])
        print(f"  n={s['n']}")
        print(f"  {label} presence  mean={s['mean']:.3f}  95%CI=[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]")

        gate = None
        if real_mean is not None:
            gap = abs(s["mean"] - real_mean)
            supported = gap < _H1_GAP_MAX
            gate = f"gap={gap:.3f} vs real={real_mean:.3f}  → H1 {'SUPPORTED' if supported else 'NOT SUPPORTED'} (threshold <{_H1_GAP_MAX})"
            print(f"  H1 gate: {gate}")

        out[f"composed_{label}"] = {**s, "h1_gate": gate}

    args.out = args.out or "results/exp5_presence.json"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults → {out_path}")


def _run_joint(args: argparse.Namespace, model, cardio_idx, effusion_idx, ckpt: dict) -> None:
    """Joint co-presence rate on composed both-disease images (sanity check for Exp6)."""
    if args.composed_both is None:
        raise ValueError("--composed_both is required for joint mode")
    print(f"\n── composed both: {args.composed_both}")
    paths = _gather_paths(args.composed_both, args.n)
    probs = predict_probs(model, paths, cardio_idx, effusion_idx, args.device)

    sc = _summarise(probs[:, 0])
    se = _summarise(probs[:, 1])
    thr_c = ckpt["youden_threshold_cardio"]
    thr_e = ckpt["youden_threshold_effusion"]
    co_rate = float(((probs[:, 0] >= thr_c) & (probs[:, 1] >= thr_e)).mean())

    print(f"  n={sc['n']}")
    print(f"  cardio   mean={sc['mean']:.3f}  95%CI={sc['ci95']}")
    print(f"  effusion mean={se['mean']:.3f}  95%CI={se['ci95']}")
    print(f"  co-presence (cardio≥{thr_c:.3f} & effusion≥{thr_e:.3f})  rate={co_rate:.3f}")

    out = {
        "n": sc["n"],
        "cardio": sc,
        "effusion": se,
        "co_presence_rate": co_rate,
        "co_presence_threshold_cardio": thr_c,
        "co_presence_threshold_effusion": thr_e,
    }
    out_path = Path(args.out or "results/exp5_joint.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Frozen NIH DenseNet-121 presence-rate classifier"
    )
    p.add_argument("--mode", choices=["validate", "eval", "joint"], required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    g = p.add_argument_group("validate")
    g.add_argument("--real_cardio",   default=None, help="Dir of real cardiomegaly PNGs")
    g.add_argument("--real_effusion", default=None, help="Dir of real effusion PNGs")
    g.add_argument("--real_normal",   default=None, help="Dir of real normal PNGs")
    g.add_argument("--n_validate",    type=int, default=None, help="Cap images per class")
    g.add_argument("--held_out", action="store_true",
                   help="Validate on the reproduced fine-tune VAL split only (not the whole dir)")
    g.add_argument("--val_frac", type=float, default=0.2, help="Val fraction (match fine-tune: 0.2)")
    g.add_argument("--val_seed", type=int, default=42, help="Val split seed (match fine-tune: 42)")

    g = p.add_argument_group("eval")
    g.add_argument("--composed_cardio",   default=None, help="Dir of composed cardio PNGs")
    g.add_argument("--composed_effusion", default=None, help="Dir of composed effusion PNGs")
    g.add_argument("--real_cardio_mean",  type=float, default=None,
                   help="Real cardio presence mean for H1 gap check (from --mode validate)")
    g.add_argument("--real_effusion_mean", type=float, default=None)
    g.add_argument("--n",   type=int,  default=2000, help="Max images per condition")
    g.add_argument("--out", default=None,
                   help="Results JSON (per-mode default: pilot_validate / exp5_presence / exp5_joint)")

    g = p.add_argument_group("joint")
    g.add_argument("--composed_both", default=None, help="Dir of composed both-disease PNGs")

    args = p.parse_args()

    print(f"Loading fine-tuned PresenceClassifier from {_CKPT_PATH} on {args.device} ...")
    model, cardio_idx, effusion_idx, ckpt = load_model(args.device)
    print(f"  epoch={ckpt['epoch']}  AUC cardio={ckpt['val_auc_cardio']:.4f}"
          f"  effusion={ckpt['val_auc_effusion']:.4f}")
    print(f"  youden thr_c={ckpt['youden_threshold_cardio']:.3f}"
          f"  thr_e={ckpt['youden_threshold_effusion']:.3f}\n")

    if args.mode == "validate":
        _run_validate(args, model, cardio_idx, effusion_idx, ckpt)
    elif args.mode == "eval":
        _run_eval(args, model, cardio_idx, effusion_idx)
    elif args.mode == "joint":
        _run_joint(args, model, cardio_idx, effusion_idx, ckpt)


if __name__ == "__main__":
    main()
