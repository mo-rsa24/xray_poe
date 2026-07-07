"""Exp03 — PoE Gate-3 co-presence yield harness.

PoE-composes the two single-disease experts (ε_uncond + w_c·(ε_cardio−ε_uncond) +
w_e·(ε_effusion−ε_uncond)), decodes, and tallies the Gate-3 pass-rate: a sample
passes iff it shows BOTH signs at once — CTR > 0.5 AND a blunted costophrenic corner.

Applies the two upstream carry-forwards:
  • generated-image FLOOR (Exp 4): the instrument false-positives on generated texture,
    so the joint Gate-3 rate on generated no_finding images is the floor; true yield is
    the excess over it.
  • ABSTAIN margin (Exp 2): binary sign-reads are fragile at the threshold, so samples
    with CTR within ±0.03 of 0.5 or a CP angle within ±10° of 80° are flagged abstain.

    python scripts/gate3_yield.py --n 10                       # task-1 smoke
    python scripts/gate3_yield.py --n 2000 --out results/exp03_yield   # full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.ctr_overlay as C
import scripts.generate as G

CTR_MARGIN = 0.03      # abstain band around CTR=0.5   (Exp 2)
CP_MARGIN = 10.0       # abstain band around CP=80°     (Exp 2)
FLOOR_DIR = Path("outputs/exp04_leakage/single/no_finding")   # Exp-4 generated floor


def _compose(unet, vae, scale, ddim, n, w_c, w_e, steps, batch, seed, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    k = 0
    for start in range(0, n, batch):
        bn = min(batch, n - start)
        noise = G._noise_batch(bn, seed + start, device)
        with torch.no_grad():
            z0 = G.cfg_compose(unet, noise, w_c, ddim, steps=steps, anchor="null",
                               null_token_idx=unet.null_token_idx, w_effusion=w_e)
        for img in G._decode_latents(z0, vae, scale):
            img.save(out_dir / f"img_{k:05d}.png"); k += 1
    return sorted(out_dir.glob("*.png"))


def score(seg, idx, paths, device) -> list[dict]:
    rows = []
    for p in paths:
        m = C.ctr_for_image(seg, idx, p, device)
        ctr, cl, cr = m["ctr"], m["cp_left"], m["cp_right"]
        cardio = bool(np.isfinite(ctr) and ctr > 0.5)
        effusion = bool(cl["blunted"] or cr["blunted"])
        angs = [a for a in (cl["angle_deg"], cr["angle_deg"]) if a is not None and np.isfinite(a)]
        cp_max = max(angs) if angs else np.nan
        abstain = bool((np.isfinite(ctr) and abs(ctr - 0.5) <= CTR_MARGIN) or
                       (angs and min(abs(a - 80) for a in angs) <= CP_MARGIN))
        rows.append({"path": str(p), "ctr": round(ctr, 3) if np.isfinite(ctr) else np.nan,
                     "cp_max_angle": round(cp_max, 1) if np.isfinite(cp_max) else np.nan,
                     "cardio_present": cardio, "effusion_present": effusion,
                     "gate3_pass": cardio and effusion, "abstain": abstain})
    return rows


def _scatter(df: pd.DataFrame, fig_out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.dropna(subset=["ctr", "cp_max_angle"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axvspan(0.5, max(0.6, d["ctr"].max() + 0.05), color="#2ca25f", alpha=0.05)
    ax.axhspan(80, max(130, d["cp_max_angle"].max() + 5), color="#2ca25f", alpha=0.05)
    p = d[d["gate3_pass"]]
    a = d[d["abstain"] & ~d["gate3_pass"]]
    f = d[~d["abstain"] & ~d["gate3_pass"]]
    ax.scatter(f["ctr"], f["cp_max_angle"], s=18, c="#b0b0b0", label="fail", alpha=0.7)
    ax.scatter(a["ctr"], a["cp_max_angle"], s=18, c="#e8a13a", label="abstain (edge)", alpha=0.85)
    ax.scatter(p["ctr"], p["cp_max_angle"], s=34, c="#2ca25f", edgecolor="k", lw=0.4, label="Gate-3 pass")
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.axhline(80, color="k", ls="--", lw=1)
    ax.set_xlabel("CTR  (cardiomegaly → right of 0.5)")
    ax.set_ylabel("max CP wedge angle  (effusion → above 80°)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/model_step0100000.safetensors")
    ap.add_argument("--vae", default="ckpts/vae_step0025000.pt")
    ap.add_argument("--scale-factor", default="data/latents/scale_factor.pt")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--w-cardio", type=float, default=2.0)
    ap.add_argument("--w-effusion", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/exp03_yield")
    ap.add_argument("--csv-out", default=None, help="per-sample CSV (default {out}/yield_per_sample.csv)")
    ap.add_argument("--fig-out", default=None, help="CTR×CP scatter PNG (skip if unset)")
    ap.add_argument("--floor-dir", default=str(FLOOR_DIR), help="generated no_finding floor ('' to skip)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    unet = G._load_unet(Path(args.ckpt), device)
    vae = G._load_vae(Path(args.vae), device)
    scale = (torch.load(args.scale_factor, map_location="cpu", weights_only=False).float()
             if Path(args.scale_factor).exists() else 1.0)  # scalar or per-channel (4,1,1); _decode_latents moves to device
    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    seg, idx = C._load_seg(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"PoE compose w_c={args.w_cardio} w_e={args.w_effusion} · ckpt {Path(args.ckpt).name} · n={args.n}")
    imgs = _compose(unet, vae, scale, ddim, args.n, args.w_cardio, args.w_effusion,
                    args.steps, args.batch, args.seed, device, out / "compose")
    df = pd.DataFrame(score(seg, idx, imgs, device))
    csv_out = Path(args.csv_out) if args.csv_out else out / "yield_per_sample.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)

    n = len(df)
    npass = int(df["gate3_pass"].sum())
    nabs = int(df["abstain"].sum())
    conf = df[~df["abstain"]]
    floor = float("nan")
    if args.floor_dir and Path(args.floor_dir).exists():
        fl = pd.DataFrame(score(seg, idx, sorted(Path(args.floor_dir).glob("*.png")), device))
        floor = float(fl["gate3_pass"].mean())

    if args.fig_out:
        _scatter(df, Path(args.fig_out),
                 f"Exp03 spot-check · PoE Gate-3 · n={n} · yield {npass/n:.0%} (abstain {nabs/n:.0%})")

    print(f"\nGate-3 yield (CTR>0.5 AND CP blunted):")
    print(f"  raw          = {npass}/{n} = {npass/n:.1%}")
    print(f"  cardio_present={df['cardio_present'].mean():.1%}  effusion_present={df['effusion_present'].mean():.1%}")
    print(f"  abstain (threshold-edge) = {nabs}/{n} = {nabs/n:.1%}  → confident yield "
          f"{int(conf['gate3_pass'].sum())}/{len(conf)} = {conf['gate3_pass'].mean():.1%}" if len(conf) else "")
    if np.isfinite(floor):
        print(f"  no_finding Gate-3 floor = {floor:.1%}  → floor-corrected yield = {max(0, npass/n - floor):.1%}")
    if args.fig_out:
        print(f"→ {args.fig_out}")
    print(f"→ {csv_out}  · images → {out}/compose/")


if __name__ == "__main__":
    main()
