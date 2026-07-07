"""Rebalanced-retrain monitor — the cardio-recovery instrument check per checkpoint.

For a given rebalanced checkpoint, generates a quick batch and runs the CTR/CP
instrument to track whether the cardio conditional is recovering:
  • single cardio  → cardio CTR median + frac>0.5   (target: rise 0.395 → real 0.547)
  • PoE compose    → Gate-3 yield + cardio_present   (target: rise 9% → ≥30%)
Appends a row to results/exp05_decision/retrain_tracking.md so the trajectory is visible.

    # one checkpoint:
    python scripts/monitor_retrain.py --ckpt ckpts_rebalanced/model_step0030000.safetensors
    # newest checkpoint:
    python scripts/monitor_retrain.py --latest
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.ctr_overlay as C
import scripts.generate as G
import scripts.gate3_yield as Y
import metrics.eval_checkpoint as EC

# OPEN GRAFT (deferred): KID / density-coverage from CheXGenBench (arXiv 2505.10496).
# This is an OFFLINE-only fidelity add-on — add it here (a subprocess to metrics.fid
# --kid plus a density/coverage pass over the same generated batch, appended as extra
# table columns) ONLY if a reviewer attacks generation fidelity. Not built.

TRACK = Path("results/exp05_decision/retrain_tracking.md")
CROSS_CSV = Path("results/exp04_leakage/cross_presence.csv")   # real reference distributions
# real-data targets the retrain should approach (measured in exp01/exp04)
REF = {
    "cardio_real_ctr_med": 0.535,    # generated cardio CTR median should rise toward this
    "cardio_real_frac_gt0p5": 0.758,
    "effusion_floor": 0.341,         # cardio→effusion leak should stay at/below this
    "yield_target": 0.30,            # PoE Gate-3 yield support bar
    "cardio_gen_baseline_ctr": 0.430,  # current @100k (the gap to close)
}
CARDIO_BASELINE, CARDIO_REAL = 0.430, 0.535   # gen-cardio CTR @100k vs real cardio (exp04)


def _gen_single(unet, vae, scale, ddim, label, n, steps, batch, seed, device, out_dir):
    """Single-disease generation (label 1=cardiomegaly, 2=effusion, 0=no_finding)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    k = 0
    for start in range(0, n, batch):
        bn = min(batch, n - start)
        noise = G._noise_batch(bn, seed + start, device)
        with torch.no_grad():
            z0 = G.cfg_single(unet, noise, label, 1.0, ddim, steps=steps, anchor="null",
                              null_token_idx=unet.null_token_idx)
        for img in G._decode_latents(z0, vae, scale):
            img.save(out_dir / f"s_{k:05d}.png"); k += 1
    return sorted(out_dir.glob("*.png"))


def _decoded_arrays(paths) -> list:
    """Read generated PNGs back as (512,512) uint8 arrays for the shared evaluator.

    Going through the saved PNGs (rather than holding latents in memory) keeps the
    images on disk for inspection AND guarantees the evaluator scores byte-identical
    input to the offline ctr_overlay path."""
    from PIL import Image
    out = []
    for p in paths:
        im = Image.open(p).convert("L")
        if im.size != (512, 512):
            im = im.resize((512, 512))
        out.append(np.asarray(im, dtype=np.uint8))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--latest", action="store_true", help="use newest ckpts_rebalanced/model_step*.safetensors")
    ap.add_argument("--ckpt-glob", default="ckpts_rebalanced/model_step*.safetensors")
    ap.add_argument("--vae", default="ckpts/vae_step0025000.pt")
    ap.add_argument("--scale-factor", default="data/latents/scale_factor.pt",
                    help="scale_factor.pt for the codec of --vae (V1: data/latents/; "
                         "VAE-v2 μ: data/latents_v2/scale_factor.pt)")
    ap.add_argument("--n-poe", type=int, default=300)
    ap.add_argument("--n-cardio", type=int, default=200)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--wandb", action="store_true", help="log histograms + recovery curves to W&B")
    ap.add_argument("--wandb-run", default="exp5_retrain_monitor", help="persistent monitor run name/id")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    ckpt = args.ckpt
    if args.latest or not ckpt:
        cks = sorted(glob.glob(args.ckpt_glob))
        if not cks:
            print(f"no checkpoints match {args.ckpt_glob}"); return
        ckpt = cks[-1]
    step = int("".join(filter(str.isdigit, Path(ckpt).stem)) or 0)
    print(f"monitor · {ckpt} (step {step})")

    unet = G._load_unet(Path(ckpt), device)
    vae = G._load_vae(Path(args.vae), device)
    scale = torch.load(args.scale_factor, map_location="cpu").item()
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    seg, idx = C._load_seg(args.device)
    tmp = Path("outputs/exp05_monitor")

    # Generate the three finding batches (PNGs kept on disk for inspection), then score
    # them all through the SHARED evaluator (metrics.eval_checkpoint) so the offline
    # re-eval returns numbers identical to the in-training W&B hook.
    cpaths = _gen_single(unet, vae, scale, ddim, 1, args.n_cardio, args.steps,
                         args.batch, args.seed, device, tmp / f"cardio_{step}")
    epaths = _gen_single(unet, vae, scale, ddim, 2, args.n_cardio, args.steps,
                         args.batch, args.seed + 9000, device, tmp / f"effusion_{step}")
    ppaths = Y._compose(unet, vae, scale, ddim, args.n_poe, 2.0, 1.0, args.steps,
                        args.batch, args.seed + 5000, device, tmp / f"poe_{step}")

    imgs = (_decoded_arrays(cpaths) + _decoded_arrays(epaths) + _decoded_arrays(ppaths))
    labels = ([EC.LABEL_CARDIO] * len(cpaths) + [EC.LABEL_EFFUSION] * len(epaths)
              + [EC.LABEL_POE] * len(ppaths))
    metrics, overlays = EC.evaluate_checkpoint(imgs, labels, (seg, idx), classifier=None)

    # Per-finding overlays (for the few aggregates the namespaced dict doesn't carry:
    # seg-fail, effusion blunt-rate / CTR, and the W&B histograms).
    ov_cardio = [o for o in overlays if o.stem.startswith("cardio_")]
    ov_eff = [o for o in overlays if o.stem.startswith("effusion_")]
    ov_poe = [o for o in overlays if o.stem.startswith("poe_")]

    def _cp_max(o):
        angs = [a for a in (o.cp_left.get("angle_deg"), o.cp_right.get("angle_deg"))
                if a is not None and np.isfinite(a)]
        return max(angs) if angs else np.nan

    def _eff_present(o):
        return bool(o.cp_left.get("blunted") or o.cp_right.get("blunted"))

    ctr_med = metrics["ctr_med/cardio"]
    ctr_gt = metrics["frac_gt0.5/cardio"]
    cardio_leak = metrics["leak/cardio_to_effusion"]        # effusion leaking into cardio
    seg_fail = float(np.mean([not np.isfinite(o.ctr) for o in ov_cardio])) if ov_cardio else float("nan")

    eff_blunt = float(np.mean([_eff_present(o) for o in ov_eff])) if ov_eff else float("nan")
    eff_ctrs = np.array([o.ctr for o in ov_eff if np.isfinite(o.ctr)], dtype=np.float64)
    eff_ctr_med = float(np.median(eff_ctrs)) if len(eff_ctrs) else float("nan")

    yld = metrics["yield/poe_gate3"]
    cardio_pres = float(np.mean([np.isfinite(o.ctr) and o.ctr > 0.5 for o in ov_poe])) if ov_poe else float("nan")
    ctr = pd.Series([o.ctr for o in ov_cardio], dtype="float64")
    poe_cp = pd.Series([_cp_max(o) for o in ov_poe], dtype="float64").dropna()

    print(f"  CARDIO  : CTR median {ctr_med:.3f} (baseline {CARDIO_BASELINE}, real {CARDIO_REAL}) · "
          f"frac>0.5 {ctr_gt:.1%} (real {REF['cardio_real_frac_gt0p5']:.0%}) · effusion-leak {cardio_leak:.1%} "
          f"(floor {REF['effusion_floor']:.0%}) · seg-fail {seg_fail:.1%}")
    print(f"  EFFUSION: blunt-rate {eff_blunt:.1%} (real ~36% — should HOLD) · "
          f"CTR median {eff_ctr_med:.3f} (should stay LOW; rise = cardio bleeding in)")
    print(f"  PoE     : Gate-3 yield {yld:.1%} (target ≥30%) · cardio_present {cardio_pres:.1%}")

    TRACK.parent.mkdir(parents=True, exist_ok=True)
    if not TRACK.exists():
        TRACK.write_text(
            "# Exp05 — rebalanced-retrain recovery tracking\n\n"
            f"It's ONE conditional model — cardio must RECOVER while effusion must HOLD (no regression).\n"
            f"Targets: cardio CTR med → {CARDIO_REAL} (base {CARDIO_BASELINE}); cardio frac>0.5 → "
            f"{REF['cardio_real_frac_gt0p5']:.0%}; PoE yield → ≥30%; cardio→eff leak ≤ {REF['effusion_floor']:.0%}; "
            f"effusion blunt-rate HOLD ~36%; effusion CTR stay LOW.\n\n"
            "| step | cardio CTR med | cardio frac>0.5 | eff→cardio leak | seg-fail | eff blunt-rate | eff CTR med | PoE yield | PoE cardio_pres |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    with TRACK.open("a") as f:
        f.write(f"| {step} | {ctr_med:.3f} | {ctr_gt:.1%} | {cardio_leak:.1%} | {seg_fail:.1%} | "
                f"{eff_blunt:.1%} | {eff_ctr_med:.3f} | {yld:.1%} | {cardio_pres:.1%} |\n")
    print(f"→ appended to {TRACK}")

    # ── W&B: distribution histograms + recovery curves vs real targets ───────
    if args.wandb:
        import wandb
        run = wandb.init(project="paper3-ldm", name=args.wandb_run, id=args.wandb_run,
                         resume="allow", job_type="retrain-monitor", reinit=True)
        log = {
            "gen/cardio_ctr_median": ctr_med, "gen/cardio_frac_gt0p5": ctr_gt,
            "gen/cardio_effusion_leak": cardio_leak, "gen/cardio_seg_fail": seg_fail,
            "gen/effusion_blunt_rate": eff_blunt, "gen/effusion_ctr_median": eff_ctr_med,
            "gen/poe_gate3_yield": yld, "gen/poe_cardio_present": cardio_pres,
            "gen/cardio_ctr_hist": wandb.Histogram(ctr.dropna().tolist()),
            "gen/poe_cp_angle_hist": wandb.Histogram(poe_cp.tolist()),
            # real-data targets as flat reference lines on the same panels
            "ref/cardio_real_ctr_median": REF["cardio_real_ctr_med"],
            "ref/cardio_real_frac_gt0p5": REF["cardio_real_frac_gt0p5"],
            "ref/effusion_floor": REF["effusion_floor"],
            "ref/effusion_blunt_hold": 0.36,
            "ref/yield_target": REF["yield_target"],
        }
        # overlay the real cardio CTR distribution once, for histogram comparison
        if CROSS_CSV.exists():
            cp = pd.read_csv(CROSS_CSV)
            rc = pd.to_numeric(cp[cp["set"] == "cardio_real"]["ctr"], errors="coerce").dropna()
            if len(rc):
                log["ref/cardio_real_ctr_hist"] = wandb.Histogram(rc.tolist())
        run.log(log, step=step)
        print(f"  → W&B run '{args.wandb_run}' updated (histograms + recovery curves @ step {step})")


if __name__ == "__main__":
    main()
