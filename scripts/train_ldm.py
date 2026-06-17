"""LDM training loop — single-disease, CFG dropout, DDPM ε-prediction.

Usage:
    python scripts/train_ldm.py --config configs/ldm_full.yaml
    python scripts/train_ldm.py --config configs/ldm_debug.yaml  # smoke run, CPU-safe
    python scripts/train_ldm.py --config configs/ldm_full.yaml --max-steps 5  # dry-run

    # Resume from last checkpoint, continuing the same W&B run:
    python scripts/train_ldm.py --config configs/ldm_full.yaml \\
        --latent-cache data/latents --vae-decode-ckpt ckpts/vae_step0025000.pt \\
        --resume <wandb-run-id>

W&B metrics (§4 of plans/single-disease-ldm/EXPERIMENTS.md):
    train/loss              — every step
    train/loss_cls{0,1,2}   — every 100 steps (pre-dropout labels)
    train/lr                — every step
    train/grad_norm         — every 100 steps
    val/loss                — every 1 000 steps
    recon_grid              — wandb.Image every 1 000 steps; Artifact ldm-recon-grid:step{N}
    compose/null_anchor/w{w}    — wandb.Image every 5 000 steps; 4 panels (anchor×weight), 4 seeds each; Artifact ldm-compose-grid:step{N}
    compose/healthy_anchor/w{w} —

Checkpoints (every ckpt_every steps):
    model_step{N}.safetensors   — model weights (UNet + class_embed)
    training_state_step{N}.pt   — optimizer state, LR scheduler state, step counter,
                                   kill-criteria reference loss; everything needed for
                                   exact resume
    config_step{N}.yaml         — config snapshot
    All three files are uploaded together as W&B Artifact ldm-ckpt:step{N}.

Resume behaviour:
    --resume <wandb-run-id> continues the same W&B run AND loads the latest local
    checkpoint.  The training loop restarts exactly at the saved step with the
    optimizer's Adam momentum/variance and cosine LR position intact.

Kill criteria:
    train/loss > 2× loss_at_step_10k → W&B alert + sys.exit(1)
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from safetensors.torch import load_file as safetensors_load, save_file
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.models.ldm_unet import build_unet
from src.monitor import MonitorBatch

import wandb

_console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "no_finding", 1: "cardiomegaly", 2: "effusion"}
_COMPOSE_WEIGHTS = [1.0, 2.0]
_COMPOSE_ANCHORS = ["null", "normal"]
_COMPOSE_SEEDS = [42, 137, 256, 512]
# Human-readable labels for compose grid panels and artifact rows
_ANCHOR_DISPLAY = {
    "null":   "∅-anchor (null token / unconditional)",
    "normal": "healthy-anchor (no-finding class)",
}


def _load_vae_state(vae_model: torch.nn.Module, ckpt_path: str, map_location) -> str:
    """Load VAE weights from a training checkpoint with flexible key detection.

    VAE checkpoints saved during training wrap weights in a dict alongside
    optimizer state, step counter, and EMA weights.  Tries keys in preference
    order: ema (smoother for inference) → model → model_state → raw dict.
    Returns the key that was used, for logging.
    """
    raw = torch.load(ckpt_path, map_location=map_location)
    for key in ("ema", "model", "model_state"):
        if key in raw and isinstance(raw[key], dict):
            vae_model.load_state_dict(raw[key])
            return key
    vae_model.load_state_dict(raw)
    return "raw"


def _find_latest_checkpoint(ckpt_dir: Path) -> tuple[Path | None, int]:
    """Return (safetensors_path, step) for the highest-step checkpoint, or (None, 0)."""
    candidates = sorted(ckpt_dir.glob("model_step*.safetensors"))
    if not candidates:
        return None, 0
    latest = candidates[-1]
    step = int(latest.stem.replace("model_step", ""))
    return latest, step


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]Training[/bold blue]"),
        MofNCompleteColumn(),
        BarColumn(bar_width=35),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("  [dim]ep[/dim] [white]{task.fields[epoch]}[/white][dim]/{task.fields[total_epochs]}[/dim]"),
        TextColumn("  [yellow]loss[/yellow]=[green]{task.fields[loss]:.4f}[/green]"),
        TextColumn("  [yellow]lr[/yellow]=[cyan]{task.fields[lr]:.2e}[/cyan]"),
        TimeElapsedColumn(),
        TextColumn("[dim]eta[/dim]"),
        TimeRemainingColumn(),
        console=_console,
        refresh_per_second=4,
        transient=False,
    )


def _print_banner(cfg: dict, run_name: str, device: torch.device, max_steps: int,
                  resume_step: int = 0) -> None:
    eff_batch = cfg.get("batch_size", 4) * cfg.get("grad_accum", 4)
    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False, show_edge=False)
    table.add_column(style="bold cyan", no_wrap=True, min_width=16)
    table.add_column(style="white")
    table.add_row("Run", f"[bold]{run_name}[/bold]")
    table.add_row("Device", str(device))
    if resume_step:
        table.add_row("Steps", f"{resume_step:,} → {max_steps:,} [dim](resuming)[/dim]")
    else:
        table.add_row("Steps", f"{max_steps:,}")
    table.add_row("Batch", f"{cfg.get('batch_size', 4)} × {cfg.get('grad_accum', 4)} accum → {eff_batch} effective")
    table.add_row("LR", str(cfg.get("lr", 1e-4)))
    table.add_row("LR sched", f"cosine (warmup={cfg.get('lr_warmup_steps', 500)} steps)")
    table.add_row("BF16", "[green]✓[/green]" if cfg.get("bf16", False) else "[dim]✗[/dim]")
    table.add_row("CFG drop p", str(cfg.get("cfg_dropout_p", 0.15)))
    table.add_row("Model ch.", str(cfg.get("model_channels", 128)))
    table.add_row("Ckpt every", f"{cfg.get('ckpt_every', 10_000):,}")
    _console.print(Panel(table, title="[bold blue] LDM Training [/bold blue]", border_style="blue"))


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


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
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    loss_at_step_10k: float | None = None,
    is_final: bool = False,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = ckpt_dir / f"model_step{step:07d}.safetensors"
    config_out = ckpt_dir / f"config_step{step:07d}.yaml"
    training_state_path = ckpt_dir / f"training_state_step{step:07d}.pt"

    # Model weights (safetensors — fast, safe, tensor-only)
    state = {
        **unet.state_dict(),
        **{f"class_embed.{k}": v for k, v in class_embed.state_dict().items()},
    }
    save_file(state, safetensors_path)

    with open(config_out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Training state (separate .pt — safetensors can't store non-tensor objects
    # like Adam's step counter, exp_avg_sq scalars, or Python ints)
    if optimizer is not None:
        ts: dict = {"step": step, "optimizer": optimizer.state_dict()}
        if lr_scheduler is not None:
            ts["lr_scheduler"] = lr_scheduler.state_dict()
        if loss_at_step_10k is not None:
            ts["loss_at_step_10k"] = loss_at_step_10k
        torch.save(ts, training_state_path)

    if wandb.run is not None:
        aliases = [f"step{step}"]
        if is_final:
            aliases.append("latest")
        artifact = wandb.Artifact("ldm-ckpt", type="model")
        artifact.add_file(str(safetensors_path))
        artifact.add_file(str(config_out))
        if training_state_path.exists():
            artifact.add_file(str(training_state_path))
        wandb.log_artifact(artifact, aliases=aliases)


def _log_per_class_loss(
    unet: torch.nn.Module,
    scheduler: DDPMScheduler,
    vae_encode_fn,
    batch: tuple[torch.Tensor, torch.Tensor],
    scale_factor: float,
    device: torch.device,
    step: int,
) -> dict[str, float]:
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
    return per_class


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


def _build_compose_grid_image(
    cells: dict[tuple[str, float], list],
    step: int,
) -> "Image.Image":
    """Labeled 2D grid for the compose artifact.

    Rows  = (anchor, weight) combos in _COMPOSE_ANCHORS × _COMPOSE_WEIGHTS order.
    Cols  = seeds in _COMPOSE_SEEDS order.
    Left margin shows the row label; top margin shows seed values.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    row_order = [(a, w) for a in _COMPOSE_ANCHORS for w in _COMPOSE_WEIGHTS]
    row_labels = {
        ("null",   1.0): "∅-anchor  w=1.0",
        ("null",   2.0): "∅-anchor  w=2.0",
        ("normal", 1.0): "healthy-anchor  w=1.0",
        ("normal", 2.0): "healthy-anchor  w=2.0",
    }
    n_rows = len(row_order)
    n_cols = len(_COMPOSE_SEEDS)
    H, W = cells[row_order[0]][0].shape

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = font_sm = ImageFont.load_default()

    LABEL_W = 220
    HEADER_H = 32
    SEP = 3

    canvas_w = LABEL_W + W * n_cols
    canvas_h = HEADER_H + (H + SEP) * n_rows
    canvas = Image.new("L", (canvas_w, canvas_h), color=230)
    draw = ImageDraw.Draw(canvas)

    # Title bar
    draw.rectangle([(0, 0), (canvas_w, HEADER_H - 1)], fill=50)
    draw.text((LABEL_W + 4, 6),
              f"Cardio+Effusion co-morbid  (PoE)   step={step}",
              fill=220, font=font_sm)

    # Column headers: seed values
    for j, seed in enumerate(_COMPOSE_SEEDS):
        x = LABEL_W + j * W + W // 2 - 30
        draw.text((x, 8), f"seed {seed}", fill=220, font=font_sm)

    # Rows
    for i, key in enumerate(row_order):
        y_top = HEADER_H + i * (H + SEP)

        # Row label strip (dark background)
        draw.rectangle([(0, y_top), (LABEL_W - 1, y_top + H - 1)], fill=50)
        label = row_labels.get(key, str(key))
        draw.text((6, y_top + H // 2 - 10), label, fill=220, font=font)

        # Image cells
        for j, img_np in enumerate(cells[key]):
            canvas.paste(Image.fromarray(img_np, "L"), (LABEL_W + j * W, y_top))

        # Separator
        if i < n_rows - 1:
            draw.rectangle([(0, y_top + H), (canvas_w, y_top + H + SEP - 1)], fill=80)

    return canvas


def _log_compose_grid(
    unet: torch.nn.Module,
    ddim_scheduler: DDIMScheduler,
    vae,
    step: int,
    device: torch.device,
    null_token_idx: int,
) -> None:
    """Run cfg_compose for all anchor × weight combos; log one W&B panel per combo + artifact.

    W&B layout — 4 panels grouped under 'compose/':
        compose/null_anchor/w1.0   — 4 seeds, ∅-anchor, w=1.0
        compose/null_anchor/w2.0   — 4 seeds, ∅-anchor, w=2.0
        compose/healthy_anchor/w1.0 — 4 seeds, healthy-anchor, w=1.0
        compose/healthy_anchor/w2.0 — 4 seeds, healthy-anchor, w=2.0

    Artifact: labeled 4×4 grid PNG (rows=anchor×weight, cols=seeds).
    Skips gracefully if cfg_compose is not yet implemented (plan 09 dependency).
    """
    from src.inference.cfg import cfg_compose
    import numpy as np
    from PIL import Image

    total_samples = len(_COMPOSE_ANCHORS) * len(_COMPOSE_WEIGHTS) * len(_COMPOSE_SEEDS)
    _console.log(
        f"[dim]{_now()}[/dim]  [cyan]PoE compose[/cyan]"
        f"  ε_a + w·(ε_cardio − ε_a) + w·(ε_effusion − ε_a)"
        f"  [dim]{len(_COMPOSE_ANCHORS)} anchors × {len(_COMPOSE_WEIGHTS)} weights"
        f" × {len(_COMPOSE_SEEDS)} seeds = {total_samples} samples[/dim]"
    )

    # groups[(anchor, w)] = list of wandb.Image (one per seed)
    groups: dict[tuple[str, float], list[wandb.Image]] = {}
    cells: dict[tuple[str, float], list] = {}

    for anchor in _COMPOSE_ANCHORS:
        anchor_disp = _ANCHOR_DISPLAY[anchor]
        for w in _COMPOSE_WEIGHTS:
            formula = f"ε_a + {w}·(ε_cardio−ε_a) + {w}·(ε_effusion−ε_a)"
            wandb_imgs: list[wandb.Image] = []
            cell_arrays: list = []

            for seed in _COMPOSE_SEEDS:
                _console.log(
                    f"  [dim]anchor=[/dim][yellow]{anchor}[/yellow]"
                    f"  [dim]w=[/dim][cyan]{w}[/cyan]"
                    f"  [dim]seed={seed}[/dim]"
                )
                g = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn(1, 4, 128, 128, generator=g, device=device)
                try:
                    z0 = cfg_compose(
                        unet, noise, w, ddim_scheduler,
                        anchor=anchor, null_token_idx=null_token_idx,
                    )
                except NotImplementedError:
                    _console.log("  [dim]cfg_compose not yet implemented — skipping[/dim]")
                    return

                decoded = vae.decode(z0)
                if hasattr(decoded, "sample"):
                    decoded = decoded.sample
                img_np = decoded[0, 0].float().clamp(-1, 1).cpu().numpy()
                img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

                caption = (
                    f"Co-morbid CXR: Cardiomegaly + Effusion (PoE) | "
                    f"Anchor: {anchor_disp} | "
                    f"Formula: {formula} | "
                    f"seed={seed} | step={step}"
                )
                wandb_imgs.append(wandb.Image(Image.fromarray(img_np, "L"), caption=caption))
                cell_arrays.append(img_np)

            groups[(anchor, w)] = wandb_imgs
            cells[(anchor, w)] = cell_arrays

    if not groups:
        return

    # One W&B panel per (anchor × weight) — supervisor sees 4 named panels
    panel_key_prefix = {"null": "null_anchor", "normal": "healthy_anchor"}
    log_dict: dict[str, list[wandb.Image]] = {
        f"compose/{panel_key_prefix[anchor]}/w{w:.1f}": imgs
        for (anchor, w), imgs in groups.items()
    }
    wandb.log(log_dict, step=step)

    # Artifact: labeled 4-row × 4-col grid PNG
    with tempfile.TemporaryDirectory() as td:
        grid_pil = _build_compose_grid_image(cells, step)
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
    p.add_argument("--vae-decode-ckpt", default=None,
                   help="VAE checkpoint for fp32 decode-only at grid steps (latent-cache mode). "
                        "Loaded in fp32 on CPU to avoid bf16 black-pixel artifacts in the decoder.")
    p.add_argument("--resume", default=None,
                   help="W&B run ID to resume. Continues the same W&B run AND loads the latest "
                        "local checkpoint from ckpt_dir (model weights + optimizer + LR scheduler).")
    p.add_argument("--cfg-w", type=float, default=1.0, help="CFG weight for monitor grid")
    p.add_argument("--run-name", "--run_name", default=None, dest="run_name", help="Override W&B run name")
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

    run_name = args.run_name or _make_run_name(cfg)

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

    # --- Optimiser + LR schedule ----------------------------------------------
    optimizer = torch.optim.AdamW(
        list(unet.parameters()),
        lr=cfg.get("lr", 1e-4),
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    grad_accum = cfg.get("grad_accum", 4)
    max_steps = cfg.get("max_steps", 100_000)
    lr_warmup_steps = cfg.get("lr_warmup_steps", 500)

    def _lr_lambda(current_step: int) -> float:
        if current_step < lr_warmup_steps:
            return current_step / max(1, lr_warmup_steps)
        progress = (current_step - lr_warmup_steps) / max(1, max_steps - lr_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    ckpt_every = cfg.get("ckpt_every", 10_000)
    ckpt_dir = Path(cfg.get("ckpt_dir", "ckpts/ldm"))

    # --- Resume: load checkpoint before touching W&B --------------------------
    step = 0
    loss_at_step_10k: float | None = None

    if args.resume:
        ckpt_path, resume_step = _find_latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            _console.log(
                f"[yellow]⚠ --resume passed but no checkpoint found in {ckpt_dir}. "
                f"Starting from step 0.[/yellow]"
            )
        else:
            # Load model weights
            raw = safetensors_load(ckpt_path, device=str(device))
            unet_state = {k: v for k, v in raw.items() if not k.startswith("class_embed.")}
            embed_state = {k[len("class_embed."):]: v for k, v in raw.items() if k.startswith("class_embed.")}
            unet.unet.load_state_dict(unet_state)
            unet.class_embed.load_state_dict(embed_state)

            # Load training state (optimizer, LR scheduler, step counter)
            ts_path = ckpt_dir / f"training_state_step{resume_step:07d}.pt"
            if ts_path.exists():
                ts = torch.load(ts_path, map_location=device)
                optimizer.load_state_dict(ts["optimizer"])
                if "lr_scheduler" in ts:
                    lr_scheduler.load_state_dict(ts["lr_scheduler"])
                if "loss_at_step_10k" in ts:
                    loss_at_step_10k = ts["loss_at_step_10k"]
                step = ts.get("step", resume_step)
                _console.log(
                    f"[dim]Resumed step [bold]{step:,}[/bold] — "
                    f"optimizer + LR scheduler state restored[/dim]"
                )
            else:
                # Checkpoint pre-dates training-state saving; optimizer cold-starts.
                # Advance the LR scheduler to the correct position so the cosine
                # schedule is correct even without optimizer momentum.
                for _ in range(resume_step):
                    lr_scheduler.step()
                step = resume_step
                _console.log(
                    f"[yellow]No training_state found for step {resume_step} — "
                    f"optimizer cold-starts (brief loss spike expected). "
                    f"LR scheduler fast-forwarded to step {resume_step}.[/yellow]"
                )
            _console.log(f"[dim]Checkpoint: {ckpt_path.name}[/dim]")

    # --- Banner (after resume detection so step is known) ---------------------
    _print_banner(cfg, run_name, device, max_steps, resume_step=step)

    # --- W&B init -------------------------------------------------------------
    wandb.init(
        project=cfg["wandb"]["project"],
        name=run_name,
        config={k: v for k, v in cfg.items() if k != "wandb"},
        resume="allow" if args.resume else None,
        id=args.resume,
    )
    _console.log(f"[dim]W&B[/dim] run [cyan]{wandb.run.name}[/cyan] → [link={wandb.run.url}]{wandb.run.url}[/link]")

    config_artifact = wandb.Artifact("ldm-config", type="config")
    config_artifact.add_file(str(config_path))
    wandb.log_artifact(config_artifact)

    # --- Data -----------------------------------------------------------------
    latent_cache = args.latent_cache or cfg.get("latent_cache")
    vae_ckpt_path = args.vae_ckpt or cfg.get("vae_ckpt")
    vae_model = None
    _vae_decode_only = False  # True when vae_model is fp32 CPU-resident (latent-cache mode)

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

        # Decode-only VAE for recon_grid — loaded in fp32 to avoid bf16 black-pixel
        # artifacts in the decoder. Lives on CPU between grid steps to save VRAM.
        vae_decode_ckpt_path = args.vae_decode_ckpt or cfg.get("vae_decode_ckpt")
        if vae_decode_ckpt_path:
            from vae.model import VAE
            from vae.config import DEFAULT_CONFIG as vae_cfg_obj
            vae_model = VAE(vae_cfg_obj).float()  # fp32 — bf16 causes black pixels in decoder
            key_used = _load_vae_state(vae_model, vae_decode_ckpt_path, map_location="cpu")
            vae_model.eval()
            for p in vae_model.parameters():
                p.requires_grad_(False)
            _vae_decode_only = True
            _console.log(f"[dim]VAE (decode-only, fp32) loaded from key=[cyan]{key_used}[/cyan] on CPU — moves to GPU at grid steps[/dim]")

    elif vae_ckpt_path:
        from vae.model import VAE
        from vae.config import DEFAULT_CONFIG as vae_cfg_obj
        scale_factor = torch.load(
            cfg.get("scale_factor_path", "data/latents/scale_factor.pt"), map_location=device
        ).float()
        vae_model = VAE(vae_cfg_obj).to(device)
        _load_vae_state(vae_model, vae_ckpt_path, map_location=device)
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
                return vae_model.encode(x)

    else:
        raise ValueError(
            "Provide either --latent-cache or --vae-ckpt. "
            "See plans/single-disease-ldm/plans/07-training-loop.md."
        )

    # --- Training loop --------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)

    steps_per_epoch = len(train_loader)
    total_epochs = max(1, max_steps // steps_per_epoch)
    _console.log(
        f"[dim]Epoch length:[/dim] {steps_per_epoch:,} steps/epoch"
        f"  ({total_epochs} epochs total)"
    )

    def _infinite_loader():
        while True:
            yield from train_loader

    loader_iter = _infinite_loader()
    progress = _make_progress()

    with progress:
        task_id = progress.add_task(
            "train",
            total=max_steps,
            completed=step,
            loss=float("nan"),
            lr=optimizer.param_groups[0]["lr"],
            epoch=step // steps_per_epoch + 1,
            total_epochs=total_epochs,
        )

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
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            current_lr = optimizer.param_groups[0]["lr"]
            epoch = step // steps_per_epoch + 1
            progress.update(
                task_id, advance=1,
                loss=accum_loss, lr=current_lr,
                epoch=epoch, total_epochs=total_epochs,
            )

            # --- per-step W&B logging -----------------------------------------
            log_dict: dict[str, float] = {
                "train/loss": accum_loss,
                "train/lr": current_lr,
            }
            grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            if step % 100 == 0:
                log_dict["train/grad_norm"] = grad_norm_val
            wandb.log(log_dict, step=step)

            # per-class loss + terminal line every 100 steps
            if step % 100 == 0:
                per_class = _log_per_class_loss(
                    unet, scheduler, vae_encode_fn,
                    (x, pre_dropout_labels), scale_factor.item(), device, step,
                )
                cls_parts = "  ".join(
                    f"[dim]{CLASS_NAMES[int(k[-1])]}[/dim]=[green]{v:.4f}[/green]"
                    for k, v in per_class.items()
                )
                _console.log(
                    f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                    f"  loss=[bold green]{accum_loss:.4f}[/bold green]"
                    f"  gnorm=[cyan]{grad_norm_val:.3f}[/cyan]"
                    f"  {cls_parts}"
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
                val_loss = sum(val_losses) / len(val_losses)
                wandb.log({"val/loss": val_loss}, step=step)
                _console.log(
                    f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                    f"  [bold magenta]VAL[/bold magenta]"
                    f"  loss=[bold magenta]{val_loss:.4f}[/bold magenta]"
                )

                # recon grid every 1 000 steps
                if vae_model is not None:
                    _console.log(
                        f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                        f"  [dim]recon grid  4×3 = 12 samples, 50 DDIM steps each[/dim]"
                    )
                    t0 = time.perf_counter()
                    if _vae_decode_only:
                        vae_model.to(device)
                    _log_recon_grid(unet, monitor, ddim_scheduler, vae_model, step, args.cfg_w, null_token_idx)
                    if _vae_decode_only:
                        vae_model.to("cpu")
                        torch.cuda.empty_cache()
                    _console.log(
                        f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                        f"  [dim]recon grid → W&B  [{time.perf_counter()-t0:.1f}s][/dim]"
                    )

            # compose grid every 5 000 steps (both anchors, both weights)
            if step % 5000 == 0 and vae_model is not None:
                t0 = time.perf_counter()
                if _vae_decode_only:
                    vae_model.to(device)
                _log_compose_grid(unet, ddim_scheduler, vae_model, step, device, null_token_idx)
                if _vae_decode_only:
                    vae_model.to("cpu")
                    torch.cuda.empty_cache()
                _console.log(
                    f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                    f"  [cyan]PoE compose done[/cyan]  → W&B  [dim][{time.perf_counter()-t0:.1f}s][/dim]"
                )

            # checkpoint every ckpt_every steps
            if step % ckpt_every == 0:
                _save_checkpoint(
                    unet.unet, unet.class_embed, cfg, ckpt_dir, step, config_path,
                    optimizer=optimizer, lr_scheduler=lr_scheduler,
                    loss_at_step_10k=loss_at_step_10k,
                    is_final=(step == max_steps),
                )
                _console.log(
                    f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                    f"  [bold yellow]CKPT[/bold yellow]"
                    f"  [dim]→ {ckpt_dir}/model_step{step:07d}.safetensors[/dim]"
                )

            # kill criteria
            if step == 10_000:
                loss_at_step_10k = accum_loss
                _console.log(
                    f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                    f"  [dim]kill-criteria reference loss set: {accum_loss:.4f}[/dim]"
                )
            if loss_at_step_10k is not None and step > 10_000 and accum_loss > 2.0 * loss_at_step_10k:
                msg = (
                    f"Loss divergence at step {step}: {accum_loss:.4f} > "
                    f"2× step-10k value {loss_at_step_10k:.4f}"
                )
                wandb.alert(title="LDM training diverged", text=msg, level=wandb.AlertLevel.ERROR)
                _console.log(f"[bold red]KILL CRITERIA MET:[/bold red] {msg}")
                wandb.finish(exit_code=1)
                sys.exit(1)

    # --- final checkpoint + finish --------------------------------------------
    _save_checkpoint(
        unet.unet, unet.class_embed, cfg, ckpt_dir, step, config_path,
        optimizer=optimizer, lr_scheduler=lr_scheduler,
        loss_at_step_10k=loss_at_step_10k,
        is_final=True,
    )
    wandb.finish()
    _console.print(f"\n[bold green]Training complete[/bold green] — {step:,} steps.")


if __name__ == "__main__":
    main()
