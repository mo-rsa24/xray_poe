"""LDM training loop — single-disease, CFG dropout, DDPM ε-prediction.

Usage:
    python scripts/train_ldm.py --config configs/ldm_full.yaml
    python scripts/train_ldm.py --config configs/ldm_debug.yaml  # smoke run, CPU-safe
    python scripts/train_ldm.py --config configs/ldm_full.yaml --max-steps 5  # dry-run

W&B metrics (§4 of plans/single-disease-ldm/EXPERIMENTS.md):
    train/loss              — every step
    train/loss_cls{0,1,2}   — every 100 steps (pre-dropout labels)
    train/lr                — every step
    train/grad_norm         — every 100 steps
    val/loss                — every 1 000 steps
    recon_grid              — wandb.Image every 1 000 steps; Artifact ldm-recon-grid:step{N}
    compose_grid            — wandb.Image every 5 000 steps (both anchors); Artifact ldm-compose-grid:step{N}

Checkpoints:
    model.safetensors + config.yaml in W&B Artifact ldm-ckpt:step{N}
    Tagged 'latest' on the final step.

Kill criteria:
    train/loss > 2× loss_at_step_10k → W&B alert + sys.exit(1)
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from safetensors.torch import save_file
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models.ldm_unet import build_unet
from src.monitor import MonitorBatch

import wandb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "no_finding", 1: "cardiomegaly", 2: "effusion"}
_COMPOSE_WEIGHTS = [1.0, 2.0]
_COMPOSE_ANCHORS = ["null", "normal"]
_COMPOSE_SEEDS = [42, 137, 256, 512]


def _make_run_name(cfg: dict) -> str:
    tag = cfg.get("tag", "full")
    cfg_p = str(cfg.get("cfg_dropout_p", 0.15)).replace(".", "")
    seed = cfg.get("seed", 42)
    from datetime import date
    d = date.today().strftime("%Y%m%d")
    template = cfg.get("wandb", {}).get("run_name_template", "ldm_{tag}_{cfg_p}_{seed}_{date}")
    return template.format(tag=tag, cfg_p=cfg_p, seed=seed, date=d)


def _save_checkpoint(
    unet: torch.nn.Module,
    class_embed: torch.nn.Module,
    cfg: dict,
    ckpt_dir: Path,
    step: int,
    config_path: Path,
    is_final: bool = False,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = ckpt_dir / f"model_step{step:07d}.safetensors"
    config_out = ckpt_dir / f"config_step{step:07d}.yaml"

    state = {
        **unet.state_dict(),
        **{f"class_embed.{k}": v for k, v in class_embed.state_dict().items()},
    }
    save_file(state, safetensors_path)

    with open(config_out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    if wandb.run is not None:
        aliases = [f"step{step}"]
        if is_final:
            aliases.append("latest")
        artifact = wandb.Artifact("ldm-ckpt", type="model")
        artifact.add_file(str(safetensors_path))
        artifact.add_file(str(config_out))
        wandb.log_artifact(artifact, aliases=aliases)


def _log_per_class_loss(
    unet: torch.nn.Module,
    scheduler: DDPMScheduler,
    vae_encode_fn,
    batch: tuple[torch.Tensor, torch.Tensor],
    scale_factor: float,
    device: torch.device,
    step: int,
) -> None:
    """Compute per-class MSE on the pre-dropout label batch."""
    x, labels = batch
    x = x.to(device)
    labels = labels.to(device)

    unet.eval()
    with torch.no_grad():
        z = vae_encode_fn(x) * scale_factor
        t = torch.randint(0, scheduler.config.num_train_timesteps, (z.shape[0],), device=device)
        noise = torch.randn_like(z)
        z_noisy = scheduler.add_noise(z, noise, t)

        per_class: dict[str, float] = {}
        for cls_idx in range(3):
            mask = labels == cls_idx
            if mask.sum() == 0:
                continue
            lbl_full = torch.full((z.shape[0],), cls_idx, device=device)
            eps_pred = unet(z_noisy, t, lbl_full)
            per_class[f"train/loss_cls{cls_idx}"] = F.mse_loss(eps_pred[mask], noise[mask]).item()

    unet.train()
    wandb.log(per_class, step=step)


def _log_recon_grid(
    unet: torch.nn.Module,
    monitor: MonitorBatch,
    ddim_scheduler: DDIMScheduler,
    vae,
    step: int,
    cfg_w: float,
    null_token_idx: int,
) -> None:
    """Decode the 4×3 monitor grid and log as wandb.Image + Artifact."""
    grid_pil = monitor.decode_grid(
        unet=unet,
        ddim_scheduler=ddim_scheduler,
        vae=vae,
        cfg_weight=cfg_w,
        null_token_idx=null_token_idx,
    )
    caption = f"step={step} | w={cfg_w}"
    wandb.log({"recon_grid": wandb.Image(grid_pil, caption=caption)}, step=step)

    # also save as artifact
    with tempfile.TemporaryDirectory() as td:
        grid_path = Path(td) / f"recon_grid_step{step:07d}.png"
        grid_pil.save(grid_path)
        artifact = wandb.Artifact("ldm-recon-grid", type="recon-grid")
        artifact.add_file(str(grid_path))
        wandb.log_artifact(artifact, aliases=[f"step{step}"])


def _log_compose_grid(
    unet: torch.nn.Module,
    ddim_scheduler: DDIMScheduler,
    vae,
    step: int,
    device: torch.device,
    null_token_idx: int,
) -> None:
    """Run cfg_compose for both anchors × both weights; log 8 images + Artifact.

    Skips gracefully if cfg_compose is not yet implemented (plan 09 dependency).
    """
    from src.inference.cfg import cfg_compose
    import numpy as np
    from PIL import Image

    images: list[wandb.Image] = []
    grid_cells: list[np.ndarray] = []

    for anchor in _COMPOSE_ANCHORS:
        for w in _COMPOSE_WEIGHTS:
            for seed in _COMPOSE_SEEDS:
                g = torch.Generator().manual_seed(seed)
                noise = torch.randn(1, 4, 128, 128, generator=g, device=device)
                try:
                    z0 = cfg_compose(
                        unet, noise, w, ddim_scheduler,
                        anchor=anchor, null_token_idx=null_token_idx,
                    )
                except NotImplementedError:
                    return   # plan 09 not done — skip silently

                decoded = vae.decode(z0)
                if hasattr(decoded, "sample"):
                    decoded = decoded.sample
                img_np = decoded[0, 0].float().clamp(-1, 1).cpu().numpy()
                img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                caption = f"step={step} | w={w} | anchor={anchor}"
                images.append(wandb.Image(Image.fromarray(img_np, "L"), caption=caption))
                grid_cells.append(img_np)

    if not images:
        return

    wandb.log({"compose_grid": images}, step=step)

    # artifact: stitch all cells into one PNG (2 anchors × 2 weights × 4 seeds = 8 wide × 1 tall)
    with tempfile.TemporaryDirectory() as td:
        W_px = grid_cells[0].shape[1]
        H_px = grid_cells[0].shape[0]
        canvas = np.zeros((H_px, W_px * len(grid_cells)), dtype=np.uint8)
        for i, cell in enumerate(grid_cells):
            canvas[:, i * W_px:(i + 1) * W_px] = cell
        grid_pil = Image.fromarray(canvas, "L")
        grid_path = Path(td) / f"compose_grid_step{step:07d}.png"
        grid_pil.save(grid_path)
        artifact = wandb.Artifact("ldm-compose-grid", type="recon-grid")
        artifact.add_file(str(grid_path))
        wandb.log_artifact(artifact, aliases=[f"step{step}"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--latent-cache", default=None, help="Path to pre-encoded latent .pt files")
    p.add_argument("--vae-ckpt", default=None, help="VAE checkpoint for online encoding")
    p.add_argument("--resume", default=None, help="W&B run ID to resume")
    p.add_argument("--cfg-w", type=float, default=1.0, help="CFG weight for monitor grid")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.get("bf16", False) and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # --- W&B init -------------------------------------------------------------
    wandb.init(
        project=cfg["wandb"]["project"],
        name=_make_run_name(cfg),
        config={k: v for k, v in cfg.items() if k != "wandb"},
        resume="allow" if args.resume else None,
        id=args.resume,
    )

    config_artifact = wandb.Artifact("ldm-config", type="config")
    config_artifact.add_file(str(config_path))
    wandb.log_artifact(config_artifact)

    # --- Model ----------------------------------------------------------------
    model_channels = cfg.get("model_channels", 128)
    cfg_dropout_p = cfg.get("cfg_dropout_p", 0.15)

    unet = build_unet(num_classes=3, model_channels=model_channels)
    null_token_idx: int = unet.null_token_idx
    unet = unet.to(device)
    unet.train()

    # --- Schedulers -----------------------------------------------------------
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.get("num_train_timesteps", 1000),
        beta_schedule="linear",
    )
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=cfg.get("num_train_timesteps", 1000),
        beta_schedule="linear",
    )

    # --- Monitor --------------------------------------------------------------
    monitor = MonitorBatch(device=device)

    # --- Optimiser ------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        list(unet.parameters()),
        lr=cfg.get("lr", 1e-4),
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    grad_accum = cfg.get("grad_accum", 4)
    max_steps = cfg.get("max_steps", 100_000)
    ckpt_every = cfg.get("ckpt_every", 10_000)
    ckpt_dir = Path(cfg.get("ckpt_dir", "ckpts/ldm"))

    # --- Data -----------------------------------------------------------------
    latent_cache = args.latent_cache or cfg.get("latent_cache")
    vae_ckpt_path = args.vae_ckpt or cfg.get("vae_ckpt")
    vae_model = None

    if latent_cache:
        from src.data.latent_dataset import LatentDataset
        scale_factor = torch.load(
            Path(latent_cache) / "scale_factor.pt", map_location=device
        ).float()
        train_ds = LatentDataset(latent_cache, split="train")
        val_ds = LatentDataset(latent_cache, split="val")
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg.get("batch_size", 4),
            sampler=train_ds.make_sampler(effusion_weight=cfg.get("effusion_weight", 2.0)),
            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg.get("batch_size", 4),
            shuffle=False, num_workers=2, drop_last=False,
        )

        def vae_encode_fn(x: torch.Tensor) -> torch.Tensor:
            return x   # already latents

    elif vae_ckpt_path:
        from vae.model import VAE
        from vae.config import DEFAULT_CONFIG as vae_cfg_obj
        scale_factor = torch.load(
            cfg.get("scale_factor_path", "data/latents/scale_factor.pt"), map_location=device
        ).float()
        vae_model = VAE(vae_cfg_obj).to(device)
        ckpt = torch.load(vae_ckpt_path, map_location=device)
        vae_model.load_state_dict(ckpt.get("model_state", ckpt))
        vae_model.eval()
        for p in vae_model.parameters():
            p.requires_grad_(False)

        from src.data.real_cxr_dataset import RealCXRDataset
        data_root = args.data_root or cfg.get("data_root", "data/vindr")
        train_ds = RealCXRDataset(
            csv_path=f"{data_root}/train.csv",
            image_dir=f"{data_root}/images",
            split="train",
            no_finding_cap=cfg.get("no_finding_cap", 4000),
        )
        val_ds = RealCXRDataset(
            csv_path=f"{data_root}/train.csv",
            image_dir=f"{data_root}/images",
            split="val",
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg.get("batch_size", 4),
            sampler=train_ds.make_sampler(effusion_weight=cfg.get("effusion_weight", 2.0)),
            num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg.get("batch_size", 4),
            shuffle=False, num_workers=2, drop_last=False,
        )

        def vae_encode_fn(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return vae_model.encode(x).latent_dist.sample()

    else:
        raise ValueError(
            "Provide either --latent-cache or --vae-ckpt. "
            "See plans/single-disease-ldm/plans/07-training-loop.md."
        )

    # --- Training loop --------------------------------------------------------
    step = 0
    loss_at_step_10k: float | None = None
    optimizer.zero_grad(set_to_none=True)

    def _infinite_loader():
        while True:
            yield from train_loader

    loader_iter = _infinite_loader()

    while step < max_steps:
        accum_loss = 0.0
        for _ in range(grad_accum):
            x, labels = next(loader_iter)
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pre_dropout_labels = labels.clone()

            with torch.no_grad():
                z = vae_encode_fn(x).float() * scale_factor

            t = torch.randint(0, scheduler.config.num_train_timesteps, (z.shape[0],), device=device)
            noise = torch.randn_like(z)
            z_noisy = scheduler.add_noise(z, noise, t)

            drop_mask = torch.bernoulli(
                torch.full((z.shape[0],), cfg_dropout_p, device=device)
            ).bool()
            labels[drop_mask] = null_token_idx

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_bf16):
                eps_pred = unet(z_noisy, t, labels)
                loss = F.mse_loss(eps_pred.float(), noise.float()) / grad_accum

            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        # --- per-step logging -------------------------------------------------
        log_dict: dict[str, float] = {
            "train/loss": accum_loss,
            "train/lr": optimizer.param_groups[0]["lr"],
        }
        if step % 100 == 0:
            log_dict["train/grad_norm"] = (
                grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            )
        wandb.log(log_dict, step=step)

        # per-class loss every 100 steps
        if step % 100 == 0:
            _log_per_class_loss(
                unet, scheduler, vae_encode_fn,
                (x, pre_dropout_labels), scale_factor.item(), device, step,
            )

        # val loss every 1 000 steps
        if step % 1000 == 0:
            unet.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for x_v, lbl_v in val_loader:
                    x_v, lbl_v = x_v.to(device), lbl_v.to(device)
                    z_v = vae_encode_fn(x_v).float() * scale_factor
                    t_v = torch.randint(0, scheduler.config.num_train_timesteps, (z_v.shape[0],), device=device)
                    n_v = torch.randn_like(z_v)
                    eps_v = unet(scheduler.add_noise(z_v, n_v, t_v), t_v, lbl_v)
                    val_losses.append(F.mse_loss(eps_v.float(), n_v.float()).item())
            unet.train()
            wandb.log({"val/loss": sum(val_losses) / len(val_losses)}, step=step)

            # recon grid every 1 000 steps
            if vae_model is not None:
                _log_recon_grid(unet, monitor, ddim_scheduler, vae_model, step, args.cfg_w, null_token_idx)

        # compose grid every 5 000 steps (both anchors, both weights)
        if step % 5000 == 0 and vae_model is not None:
            _log_compose_grid(unet, ddim_scheduler, vae_model, step, device, null_token_idx)

        # checkpoint every ckpt_every steps
        if step % ckpt_every == 0:
            _save_checkpoint(
                unet.unet, unet.class_embed, cfg, ckpt_dir, step, config_path,
                is_final=(step == max_steps),
            )

        # kill criteria
        if step == 10_000:
            loss_at_step_10k = accum_loss
        if loss_at_step_10k is not None and step > 10_000 and accum_loss > 2.0 * loss_at_step_10k:
            msg = (
                f"Loss divergence at step {step}: {accum_loss:.4f} > "
                f"2× step-10k value {loss_at_step_10k:.4f}"
            )
            wandb.alert(title="LDM training diverged", text=msg, level=wandb.AlertLevel.ERROR)
            print(f"\nKILL CRITERIA MET: {msg}", file=sys.stderr)
            wandb.finish(exit_code=1)
            sys.exit(1)

    # --- final checkpoint + finish --------------------------------------------
    _save_checkpoint(
        unet.unet, unet.class_embed, cfg, ckpt_dir, step, config_path, is_final=True
    )
    wandb.finish()
    print(f"Training complete — {step} steps.")


if __name__ == "__main__":
    main()
