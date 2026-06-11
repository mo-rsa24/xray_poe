"""LDM inference CLI — single-disease and compositional generation.

Single-disease:
    python scripts/generate.py --ckpt ckpts/ldm/model.safetensors \\
        --vae-ckpt ckpts/vae/model.pt \\
        --disease cardiomegaly --n 200 --w 1.0 --seed 42

Compositional sweep (EXP-LDM-02):
    python scripts/generate.py --ckpt ckpts/ldm/model.safetensors \\
        --vae-ckpt ckpts/vae/model.pt \\
        --compose --w-sweep 0.5 1.0 1.5 2.0 3.0 --n 200 --seed 42

Anchor comparison (both modes, same noise):
    python scripts/generate.py --ckpt ... --compose --anchor null   --n 4 --seed 42 -o /tmp/null
    python scripts/generate.py --ckpt ... --compose --anchor normal --n 4 --seed 42 -o /tmp/norm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.ldm_unet import build_unet, LDMUNet
from src.inference.cfg import cfg_single, cfg_compose

DISEASE_MAP = {"no_finding": 0, "cardiomegaly": 1, "effusion": 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_unet(ckpt_path: Path, device: torch.device) -> LDMUNet:
    """Load LDMUNet from a safetensors checkpoint."""
    state = load_file(str(ckpt_path), device=str(device))

    # Separate class_embed keys
    unet_state = {k: v for k, v in state.items() if not k.startswith("class_embed.")}
    embed_state = {
        k[len("class_embed."):]: v
        for k, v in state.items()
        if k.startswith("class_embed.")
    }

    unet = build_unet(num_classes=3)
    unet.unet.load_state_dict(unet_state, strict=False)
    if embed_state:
        unet.class_embed.load_state_dict(embed_state)
    return unet.to(device).eval()


def _load_vae(vae_ckpt: Path, device: torch.device):
    from vae.model import VAE
    from vae.config import DEFAULT_CONFIG as vae_cfg
    model = VAE(vae_cfg).to(device)
    ckpt = torch.load(vae_ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _decode_latents(z0: torch.Tensor, vae, scale_factor: float) -> list[Image.Image]:
    """Decode latents → list of uint8 PIL images."""
    z0_scaled = z0 / scale_factor
    with torch.no_grad():
        decoded = vae.decode(z0_scaled)
    if hasattr(decoded, "sample"):
        decoded = decoded.sample
    decoded = decoded.float().clamp(-1, 1)
    decoded = (decoded + 1) / 2   # [0, 1]
    imgs = []
    for i in range(decoded.shape[0]):
        ch = decoded[i, 0].cpu().numpy()   # (H, W) — grayscale
        imgs.append(Image.fromarray((ch * 255).clip(0, 255).astype(np.uint8), mode="L"))
    return imgs


def _save_images(imgs: list[Image.Image], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(out_dir / f"img_{i:05d}.png")
    print(f"  Saved {len(imgs)} images → {out_dir}")


def _noise_batch(n: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate n reproducible noise tensors from consecutive seeds."""
    tensors = []
    for i in range(n):
        g = torch.Generator().manual_seed(seed + i)
        tensors.append(torch.randn(1, 4, 128, 128, generator=g))
    return torch.cat(tensors, dim=0).to(device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LDM inference — single-disease or compositional")
    p.add_argument("--ckpt", required=True, help="Path to model.safetensors")
    p.add_argument("--vae-ckpt", required=True, help="Path to VAE checkpoint .pt")
    p.add_argument("--scale-factor", default="data/latents/scale_factor.pt",
                   help="Path to scale_factor.pt (default: data/latents/scale_factor.pt)")
    p.add_argument("--n", type=int, default=8, help="Number of images to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("-o", "--output", default="outputs", help="Output root directory")
    p.add_argument("--device", default=None)

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--disease", choices=list(DISEASE_MAP.keys()),
                      help="Single-disease generation")
    mode.add_argument("--compose", action="store_true",
                      help="Compositional (PoE) generation")

    p.add_argument("--w", type=float, default=1.0, help="CFG weight for --disease mode")
    p.add_argument("--w-sweep", nargs="+", type=float, default=[1.0],
                   help="CFG weight(s) for --compose mode")
    p.add_argument("--anchor", choices=["null", "normal"], default="null",
                   help="Unconditional anchor: null token or no-finding class")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # --- load model + VAE + scale_factor -------------------------------------
    unet = _load_unet(Path(args.ckpt), device)
    vae = _load_vae(Path(args.vae_ckpt), device)

    scale_factor_path = Path(args.scale_factor)
    if scale_factor_path.exists():
        scale_factor = torch.load(scale_factor_path, map_location="cpu").item()
    else:
        scale_factor = 1.0
        print(f"[warn] scale_factor.pt not found at {scale_factor_path}; using 1.0")

    ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    null_token_idx: int = unet.null_token_idx

    out_root = Path(args.output)

    # --- single-disease ------------------------------------------------------
    if args.disease:
        label_idx = DISEASE_MAP[args.disease]
        print(f"Generating {args.n} '{args.disease}' images  w={args.w}  anchor={args.anchor}")
        noise = _noise_batch(args.n, args.seed, device)
        z0 = cfg_single(
            unet, noise, label_idx, args.w, ddim,
            steps=args.steps, anchor=args.anchor, null_token_idx=null_token_idx,
        )
        imgs = _decode_latents(z0, vae, scale_factor)
        _save_images(imgs, out_root / "single" / args.disease)

    # --- compositional sweep -------------------------------------------------
    elif args.compose:
        for w in args.w_sweep:
            print(f"Composing {args.n} images  w={w}  anchor={args.anchor}")
            noise = _noise_batch(args.n, args.seed, device)
            z0 = cfg_compose(
                unet, noise, w, ddim,
                steps=args.steps, anchor=args.anchor, null_token_idx=null_token_idx,
            )
            imgs = _decode_latents(z0, vae, scale_factor)
            w_str = f"{w:.1f}".replace(".", "p")
            _save_images(imgs, out_root / "compose" / f"w{w_str}")

    print("Done.")


if __name__ == "__main__":
    main()
