"""CTR (geometric) vs DenseNet detector on cardiomegaly — synth vs real.

Motivation: the presence classifier is trained on REAL images and can key on features
that don't transfer to synthetics (Grad-CAM has shown it attending to the wrong regions),
so it may under-count synthetic cardiomegaly. CTR (cardiothoracic ratio) is a direct
geometric measure — cardiomegaly ⇔ CTR > 0.5 (real median ≈ 0.535). This experiment runs
BOTH instruments on the same 200 synthetic and 200 real cardio films and reports:
  • CTR distribution (synth vs real) — are the generated hearts actually enlarged?
  • detector sensitivity (synth vs real)
  • agreement matrix — of the films CTR calls enlarged, how many does the detector catch?

Usage:
    python scripts/ctr_vs_detector.py \
      --ckpt ckpts/vae_v2_ldm_v4/model_ema_step0060000.safetensors \
      --vae ckpts/vae_v2/vae_v2_step0075000.pt \
      --scale-factor data/latents_v2/joint_perch/scale_factor.pt \
      --classifier ckpts/presence_classifier_finetuned.pt \
      --real-csv data/splits/cardio_not_effusion.csv --n 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import scripts.generate as G
from scripts.validate_individual_synthetics import load_classifier, classify, generate_disease, load_real
import metrics.eval_checkpoint as EC
import scripts.ctr_overlay as C

CARDIO_REAL_CTR = 0.535   # real cardiomegaly reference (metrics.eval_checkpoint.REF)


def _gray512(im) -> np.ndarray:
    a = im.convert("L")
    if a.size != (512, 512):
        a = a.resize((512, 512))
    return np.asarray(a, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--scale-factor", required=True)
    ap.add_argument("--classifier", default="ckpts/presence_classifier_finetuned.pt")
    ap.add_argument("--real-csv", default="data/splits/cardio_not_effusion.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/section4/ctr_vs_detector_v4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    unet = G._load_unet(Path(args.ckpt), dev)
    vae = G._load_vae(Path(args.vae), dev)
    sf = torch.load(args.scale_factor, map_location="cpu", weights_only=False).float()
    clf, thr_c, thr_e = load_classifier(Path(args.classifier), dev)
    seg, idx = C._load_seg(str(dev))
    print(f"detector cardio threshold = {thr_c:.3f} | CTR cardiomegaly cutoff = 0.5 "
          f"(real ref median {CARDIO_REAL_CTR})")

    print(f"generating {args.n} synth cardio (w={args.w}, steps={args.steps}) ...")
    synth = generate_disease(unet, vae, sf, 1, args.n, args.w, "null", args.steps, args.seed, dev)
    print(f"loading {args.n} real cardio from {args.real_csv} ...")
    real, _ = load_real(Path(args.real_csv), args.n, args.seed)

    def measure(name, imgs):
        ctrs = np.array([EC.score_decoded_array(seg, idx, _gray512(im), str(dev),
                                                with_masks=False)["ctr"] for im in imgs], dtype=float)
        prob_c = classify(clf, imgs, dev)[:, 0]
        valid = np.isfinite(ctrs)
        cv = ctrs[valid]
        ctr_pos = ctrs > 0.5
        clf_pos = prob_c > thr_c
        print(f"\n=== {name}  (n={len(imgs)}, CTR-valid={int(valid.sum())}) ===")
        print(f"  CTR      : median={np.median(cv):.3f}  mean={np.mean(cv):.3f}  std={np.std(cv):.3f}")
        print(f"  CTR>0.5  : {np.mean(ctr_pos[valid])*100:.1f}%   "
              f"CTR>0.53: {np.mean(cv>0.53)*100:.1f}%   CTR>0.55: {np.mean(cv>0.55)*100:.1f}%")
        print(f"  detector : sens (cardio prob>{thr_c:.2f}) = {np.mean(clf_pos)*100:.1f}%")
        v = valid
        both = float(np.mean((ctr_pos & clf_pos)[v]))
        ctronly = float(np.mean((ctr_pos & ~clf_pos)[v]))
        clfonly = float(np.mean((~ctr_pos & clf_pos)[v]))
        neither = float(np.mean((~ctr_pos & ~clf_pos)[v]))
        print(f"  agree    : both+={both*100:.0f}%  CTR-only={ctronly*100:.0f}%  "
              f"detector-only={clfonly*100:.0f}%  neither={neither*100:.0f}%")
        enl = v & ctr_pos
        if enl.sum():
            print(f"  KEY      : of geometrically-enlarged (CTR>0.5), detector catches "
                  f"{np.mean(clf_pos[enl])*100:.1f}%")
        return ctrs, prob_c

    sc, sp = measure("SYNTH v4 cardio", synth)
    rc, rp = measure("REAL cardio", real)

    import csv
    with open(out / "per_sample.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "i", "ctr", "cardio_prob"])
        for i, (a, b) in enumerate(zip(sc, sp)): w.writerow(["synth", i, a, b])
        for i, (a, b) in enumerate(zip(rc, rp)): w.writerow(["real", i, a, b])
    print(f"\nwrote {out}/per_sample.csv")


if __name__ == "__main__":
    main()
