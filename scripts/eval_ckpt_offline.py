"""Offline eval of a raw LDM checkpoint — reuses train_ldm.py's EXACT generation +
decode + instrument path, so the numbers are identical to the in-training dashboard
(no scale-factor / VAE-version traps). Purpose: read the REAL disease metrics off the
raw (non-EMA) weights, bypassing the EMA-warmup confound.

Usage:
    python scripts/eval_ckpt_offline.py \
        --ckpt      ckpts/vae_v2_ldm_v2/model_step0020000.safetensors \
        --vae-ckpt  ckpts/vae_v2/vae_v2_step0075000.pt \
        --n 48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.ldm_unet import build_unet
import scripts.train_ldm as T
import metrics.eval_checkpoint as EC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="raw model_step*.safetensors")
    ap.add_argument("--vae-ckpt", default="ckpts/vae_v2/vae_v2_step0075000.pt")
    ap.add_argument("--scale-factor", default="data/latents_v2/scale_factor.pt",
                    help="scale_factor.pt — generated latents are divided by this before decode")
    ap.add_argument("--model-channels", type=int, default=128)
    ap.add_argument("--n", type=int, default=48, help="samples per finding")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    # scale_factor is a scalar OR a per-channel (4,1,1) tensor — keep as a tensor on device
    # so the (z0 / scale_factor) decode broadcasts either way.
    scale_factor = torch.load(args.scale_factor, map_location=device, weights_only=False).float()
    _sf = scale_factor.flatten().tolist()
    print(f"scale_factor = {[round(v,3) for v in _sf]}  shape={tuple(scale_factor.shape)}  "
          f"({'per-channel whitening' if scale_factor.numel() > 1 else 'global scalar'}; "
          f"generated latents divided by this before decode)")

    # --- UNet from the raw checkpoint (same split as train_ldm resume) ---------
    unet = build_unet(num_classes=3, model_channels=args.model_channels).to(device)
    raw = load_file(args.ckpt, device=str(device))
    unet.unet.load_state_dict(
        {k: v for k, v in raw.items() if not k.startswith("class_embed.")})
    unet.class_embed.load_state_dict(
        {k[len("class_embed."):]: v for k, v in raw.items() if k.startswith("class_embed.")})
    unet.eval()
    null_idx = unet.null_token_idx
    print(f"loaded UNet from {args.ckpt}")

    # --- VAE fp32, loaded exactly as train_ldm does (bf16 → black images) ------
    from vae.model import VAE
    from vae.config import DEFAULT_CONFIG as vae_cfg_obj
    vae = VAE(vae_cfg_obj).float()
    key = T._load_vae_state(vae, args.vae_ckpt, map_location="cpu")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.to(device)
    print(f"loaded VAE (fp32) from {args.vae_ckpt} (key={key})")

    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")

    seg, classifier = T._eval_load_instruments()
    if seg is None:
        print("PSPNet instrument failed to load — cannot score."); return
    seg[0].to(device)
    if classifier is not None:
        classifier[0].to(device)

    # --- generate + decode via train_ldm's EXACT path -------------------------
    imgs, labels = [], []
    for mode, lab, base in (
        ("normal",   EC.LABEL_NORMAL,   T._EVAL_BASE_SEED_NORMAL),
        ("cardio",   EC.LABEL_CARDIO,   T._EVAL_BASE_SEED_CARDIO),
        ("effusion", EC.LABEL_EFFUSION, T._EVAL_BASE_SEED_EFFUSION),
        ("poe",      EC.LABEL_POE,      T._EVAL_BASE_SEED_POE),
    ):
        batch = T._eval_gen_decode(unet, vae, ddim, null_idx, device, mode, args.n, base,
                                   scale_factor=scale_factor)
        imgs.extend(batch)
        labels.extend([lab] * len(batch))
        print(f"  generated {len(batch)} {mode}")

    metrics, _ = EC.evaluate_checkpoint(imgs, labels, (seg[0], seg[1]), classifier=classifier)

    print("\n===== RAW-weight metrics (EMA-warmup bypassed) =====")
    for k in ("ctr_med/cardio", "frac_gt0.5/cardio", "border_frac/cardio", "presence/cardio",
              "ctr_med/effusion", "presence/effusion", "leak/cardio_to_effusion",
              "yield/poe_gate3", "presence/both"):
        v = metrics.get(k)
        print(f"  {k:28s} {v}")
    print("\nREF: real cardio CTR 0.535 | v1(6wj35ips)@20k ctr_c=0.508 presE=0.159 | "
          "noise-floor presE≈0.659")


if __name__ == "__main__":
    main()
