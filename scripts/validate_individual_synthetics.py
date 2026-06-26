"""Individual-disease validation — the body-or-no-body gate (Exp 09 / Supervisor D2).

Proves that synth-only cardio and synth-only effusion EACH classify *before* composition.
Generate single-disease synthetics from a given LDM checkpoint, run them through the
DenseNet presence classifier-under-test, and compare per-disease accuracy / sensitivity /
specificity against the REAL single-disease baseline (Exp-4 deconfounded splits). Emit an
explicit PASS/FAIL per disease and the gate consequence.

Gate (verbatim, Supervisor D2): "Synth-only cardio & synth-only effusion each run through
classifier; accuracy/sens/spec vs real single-disease baseline; explicit pass/fail per disease.
Gate: if individual synthetics fail, paper reports 'diffusion-not-composition.'"

Same harness, run twice — once per checkpoint:
  BEFORE (contaminated, the exposure read):
    python scripts/validate_individual_synthetics.py --ldm ckpts/model_step0040000.safetensors --tag before
  AFTER (rebalanced, when it lands):
    python scripts/validate_individual_synthetics.py --ldm ckpts_rebalanced/model_step0050000.safetensors --tag after

Wraps scripts/generate.py (generation) + scripts/finetune_classifier.py::PresenceClassifier
(the classifier). It does NOT reimplement either. Run when the GPU is free (generation +
classifier together — do not run alongside a live train: OOM).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference.cfg import cfg_single
# reuse the canonical generation loaders/decoder + the real classifier — no reimplementation
from scripts.generate import _load_unet, _load_vae, _decode_latents, _noise_batch, DISEASE_MAP
from scripts.finetune_classifier import PresenceClassifier

import torchxrayvision as xrv


# ---------------------------------------------------------------------------
# classifier side
# ---------------------------------------------------------------------------

def _xrv_preprocess(img) -> torch.Tensor:
    """Identical pipeline to NIHPresenceDataset._xrv_preprocess (224, [-1024,1024])."""
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("L"), dtype=np.float32)
    else:
        arr = np.asarray(img, dtype=np.float32)
    arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)   # (1,H,W)
    arr = xrv.datasets.XRayCenterCrop()(arr)
    arr = xrv.datasets.XRayResizer(224)(arr)
    return torch.from_numpy(arr.copy())                            # (1,224,224)


def load_classifier(path: Path, device):
    ck = torch.load(path, map_location=device)
    model = PresenceClassifier(backbone_weights=ck.get("backbone", "densenet121-res224-nih"))
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    thr_c = float(ck.get("youden_threshold_cardio", 0.5))
    thr_e = float(ck.get("youden_threshold_effusion", 0.5))
    return model, thr_c, thr_e


@torch.no_grad()
def classify(model, imgs, device, bs: int = 32) -> np.ndarray:
    """Return (N,2) sigmoid probabilities [cardio, effusion]."""
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([_xrv_preprocess(im) for im in imgs[i:i + bs]]).to(device)
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 2))


# ---------------------------------------------------------------------------
# generation side
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_disease(unet, vae, scale_factor, label_idx, n, w, anchor, steps, seed, device, gen_bs=8):
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    null_idx = unet.null_token_idx
    imgs = []
    for start in range(0, n, gen_bs):
        bn = min(gen_bs, n - start)
        noise = _noise_batch(bn, seed + start, device)
        z0 = cfg_single(unet, noise, label_idx, w, ddim, steps=steps,
                        anchor=anchor, null_token_idx=null_idx)
        imgs.extend(_decode_latents(z0, vae, scale_factor))
    return imgs


def load_real(csv_path: Path, n: int, seed: int):
    df = pd.read_csv(csv_path)
    col = "path" if "path" in df.columns else df.columns[-1]
    df = df.sample(min(n, len(df)), random_state=seed)
    imgs, ids = [], []
    for _, row in df.iterrows():
        p = Path(str(row[col]))
        if p.exists():
            imgs.append(Image.open(p).convert("L"))
            ids.append(p.name)
    return imgs, ids


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def per_disease(pos_pred: np.ndarray, neg_pred: np.ndarray):
    """sens on own-disease (positives), spec on the OTHER disease (negatives), balanced-ish acc."""
    sens = float(np.mean(pos_pred)) if len(pos_pred) else float("nan")
    spec = float(np.mean(1 - neg_pred)) if len(neg_pred) else float("nan")
    npos, nneg = len(pos_pred), len(neg_pred)
    acc = (sens * npos + spec * nneg) / (npos + nneg) if (npos + nneg) else float("nan")
    return acc, sens, spec


def _fmt(x):
    return f"{x:.3f}" if x == x else "  -  "


# ---------------------------------------------------------------------------
# verdict.md (per-tag block, idempotent)
# ---------------------------------------------------------------------------

_HEADER = ("# Exp 09 · Individual-Disease Validation — Body-or-No-Body Gate (D2)\n\n"
           "Supervisor gate: *\"if individual synthetics fail, paper reports "
           "'diffusion-not-composition.'\"* Per-disease PASS = synth sensitivity ≥ "
           "real-baseline sensitivity − margin. Baseline = Exp-4 deconfounded real single-disease "
           "(may include classifier-train images → an optimistic reference; the synth-vs-real "
           "delta is the signal).\n")


def update_verdict(path: Path, tag: str, block: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    s, e = f"<!-- TAG:{tag} -->", f"<!-- /TAG:{tag} -->"
    existing = path.read_text() if path.exists() else ""
    existing = re.sub(re.escape(s) + r".*?" + re.escape(e), "", existing, flags=re.S)
    if not existing.strip().startswith("#"):
        existing = _HEADER
    path.write_text(existing.rstrip() + "\n\n" + s + "\n" + block + "\n" + e + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Individual-disease validation gate (Exp 09 / D2)")
    p.add_argument("--ldm", default="ckpts/model_step0040000.safetensors",
                   help="LDM checkpoint to validate (default: the BEFORE contaminated 40k)")
    p.add_argument("--tag", default="before", help="checkpoint tag: before | after")
    p.add_argument("--vae-ckpt", default="ckpts/vae_step0025000.pt")
    p.add_argument("--scale-factor", default="data/latents/scale_factor.pt")
    p.add_argument("--classifier", default="ckpts/presence_classifier_finetuned.pt")
    p.add_argument("--cardio-csv", default="data/splits/cardio_not_effusion.csv")
    p.add_argument("--effusion-csv", default="data/splits/effusion_not_cardio.csv")
    p.add_argument("--n", type=int, default=200, help="synthetics per disease")
    p.add_argument("--n-real", type=int, default=None, help="real baseline per disease (default = --n)")
    p.add_argument("--w", type=float, default=1.0, help="CFG weight for single-disease generation")
    p.add_argument("--anchor", choices=["null", "normal"], default="null")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin", type=float, default=0.15,
                   help="PASS if synth sens ≥ real sens − margin")
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--out-dir", default="results/exp09_individual")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_real = args.n_real or args.n
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp09] tag={args.tag}  ldm={args.ldm}  device={device}")
    print(f"        n_synth={args.n}/disease  n_real={n_real}/disease  w={args.w}  anchor={args.anchor}")

    # --- load everything ---
    clf, thr_c, thr_e = load_classifier(Path(args.classifier), device)
    print(f"        classifier thresholds: cardio>{thr_c:.3f}  effusion>{thr_e:.3f}")
    unet = _load_unet(Path(args.ldm), device)
    vae = _load_vae(Path(args.vae_ckpt), device)
    sf = torch.load(args.scale_factor, map_location="cpu").item() if Path(args.scale_factor).exists() else 1.0

    # --- generate + load reals ---
    print("        generating synth cardio …")
    synth_c = generate_disease(unet, vae, sf, DISEASE_MAP["cardiomegaly"], args.n,
                               args.w, args.anchor, args.steps, args.seed, device, args.gen_batch_size)
    print("        generating synth effusion …")
    synth_e = generate_disease(unet, vae, sf, DISEASE_MAP["effusion"], args.n,
                               args.w, args.anchor, args.steps, args.seed + 10_000, device, args.gen_batch_size)
    real_c, real_c_ids = load_real(Path(args.cardio_csv), n_real, args.seed)
    real_e, real_e_ids = load_real(Path(args.effusion_csv), n_real, args.seed)
    print(f"        sets: synth c/e = {len(synth_c)}/{len(synth_e)}  real c/e = {len(real_c)}/{len(real_e)}")

    if args.save_images:
        for nm, imgs in (("synth_cardio", synth_c), ("synth_effusion", synth_e)):
            d = out_dir / "images" / args.tag / nm
            d.mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(imgs):
                im.save(d / f"{i:04d}.png")

    # --- classify ---
    p_sc = classify(clf, synth_c, device)   # synth cardio
    p_se = classify(clf, synth_e, device)   # synth effusion
    p_rc = classify(clf, real_c, device)    # real cardio
    p_re = classify(clf, real_e, device)    # real effusion

    def preds(probs):
        return (probs[:, 0] > thr_c).astype(int), (probs[:, 1] > thr_e).astype(int)

    sc_cpred, sc_epred = preds(p_sc)
    se_cpred, se_epred = preds(p_se)
    rc_cpred, rc_epred = preds(p_rc)
    re_cpred, re_epred = preds(p_re)

    # cardio head: positives = cardio images, negatives = effusion images
    syn_cardio = per_disease(sc_cpred, se_cpred)
    real_cardio = per_disease(rc_cpred, re_cpred)
    # effusion head: positives = effusion images, negatives = cardio images
    syn_eff = per_disease(se_epred, sc_epred)
    real_eff = per_disease(re_epred, rc_epred)

    pass_cardio = syn_cardio[1] >= (real_cardio[1] - args.margin)
    pass_eff = syn_eff[1] >= (real_eff[1] - args.margin)

    # --- per-image CSV ---
    rows = []
    def add(source, dset, ids, probs):
        cp, ep = preds(probs)
        for i in range(len(probs)):
            rows.append(dict(tag=args.tag, source=source, disease_set=dset,
                             image_id=ids[i] if ids else f"{dset}_{i:04d}",
                             cardio_prob=round(float(probs[i, 0]), 4),
                             effusion_prob=round(float(probs[i, 1]), 4),
                             cardio_pred=int(cp[i]), effusion_pred=int(ep[i])))
    add("synth", "cardio", None, p_sc)
    add("synth", "effusion", None, p_se)
    add("real", "cardio", real_c_ids, p_rc)
    add("real", "effusion", real_e_ids, p_re)
    csv_path = out_dir / f"validate_individual_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # --- inline output (matches the anticipated format) ---
    def line(name, m, ok):
        return (f"{name} synth: acc/sens/spec = {_fmt(m[0])} / {_fmt(m[1])} / {_fmt(m[2])} "
                f"({'PASS' if ok else 'FAIL'})")
    print(f"\n=== Exp 09 verdict · tag={args.tag} ===")
    print("  " + line("cardio", syn_cardio, pass_cardio) +
          f"   [real sens {_fmt(real_cardio[1])}, Δ {syn_cardio[1]-real_cardio[1]:+.3f}]")
    print("  " + line("effusion", syn_eff, pass_eff) +
          f" [real sens {_fmt(real_eff[1])}, Δ {syn_eff[1]-real_eff[1]:+.3f}]")
    both = pass_cardio and pass_eff
    gate = ("✅ both classify → synthetic BODY proceeds (composition is the question, plans 10/11/12)"
            if both else
            "❌ gate fires → paper reports 'diffusion-not-composition' for "
            + ", ".join([d for d, ok in (("cardio", pass_cardio), ("effusion", pass_eff)) if not ok])
            + " (§4 confirmatory of the null on this channel)")
    print(f"  GATE: {gate}")

    # --- verdict.md block ---
    block = (
        f"## tag = `{args.tag}` — `{args.ldm}`\n\n"
        f"Generation: w={args.w}, anchor={args.anchor}, n={args.n}/disease, steps={args.steps}. "
        f"Classifier thresholds cardio>{thr_c:.3f}, effusion>{thr_e:.3f}. margin={args.margin}.\n\n"
        f"| disease | source | acc | sens | spec | PASS/FAIL |\n"
        f"|---|---|---|---|---|---|\n"
        f"| cardio | synth | {_fmt(syn_cardio[0])} | {_fmt(syn_cardio[1])} | {_fmt(syn_cardio[2])} | "
        f"**{'PASS' if pass_cardio else 'FAIL'}** |\n"
        f"| cardio | real | {_fmt(real_cardio[0])} | {_fmt(real_cardio[1])} | {_fmt(real_cardio[2])} | — |\n"
        f"| effusion | synth | {_fmt(syn_eff[0])} | {_fmt(syn_eff[1])} | {_fmt(syn_eff[2])} | "
        f"**{'PASS' if pass_eff else 'FAIL'}** |\n"
        f"| effusion | real | {_fmt(real_eff[0])} | {_fmt(real_eff[1])} | {_fmt(real_eff[2])} | — |\n\n"
        f"**Gate consequence ({args.tag}):** {gate}\n\n"
        f"Artifact: `{csv_path}`.\n"
    )
    verdict = out_dir / "verdict.md"
    update_verdict(verdict, args.tag, block)
    print(f"\n  wrote {csv_path}\n  wrote {verdict} (tag block: {args.tag})")


if __name__ == "__main__":
    main()
