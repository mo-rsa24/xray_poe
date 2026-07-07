"""VAE-v2 fine-tune — LPIPS + PatchGAN on the combined border-visible diet.

Resumes from vae_step0025000.pt and trains for additional steps with:
  - L1 reconstruction (λ=1.0)
  - LPIPS perceptual loss (net='alex', λ=0.1)          [--lpips]
  - PatchGAN adversarial loss (70×70 RF, hinge loss)    [--discriminator]
  - KL divergence (β=1e-6, fixed — no annealing)

Note on LPIPS backbone: The architecture decision (docs/vae_v2_architecture_decision.md §5)
identifies RadImageNet-LPIPS as the principled lever for grayscale CXR. The standard
lpips package's AlexNet backbone (used here) is out-of-domain on grayscale but still
provides useful structural priors and makes the dry-run verifiable without additional
dependencies. Swap to RadImageNet when the production run starts.

Usage (dry-run, verifies loss terms):
  python scripts/train_vae_v2.py --dry-run --lpips --discriminator

Usage (full fine-tune, adapt paths as needed):
  python scripts/train_vae_v2.py \\
    --resume ckpts/vae/vae_step0025000.pt \\
    --manifest data/vae_v2_diet/train_manifest.csv \\
    --config configs/vae_v2.yaml \\
    --lpips --discriminator \\
    --disc_warmup_start 1000 --disc_warmup_steps 1000 \\
    --max_steps 75000 --save_every 5000 \\
    --wandb_run exp_vae_v2_finetune \\
    --output_dir ckpts/vae_v2/
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ensure project root is on sys.path when run as `python scripts/train_vae_v2.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich import box
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

_console = Console(highlight=False)

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from vae.config import VAEConfig
from vae.losses import kl_divergence
from vae.model import VAE
from vae.train import EMA, _device, _grad_norm, _wandb_init, _wandb_log, _wandb_log_images


# ---------------------------------------------------------------------------
# PatchGAN discriminator — 70×70 receptive field (taming-transformers style)
# ---------------------------------------------------------------------------

class PatchGAN(nn.Module):
    """5-layer 70×70 PatchGAN using hinge loss.

    Receptive field: 4 + 4*2 + 4*4 + 4*8 = 70 px (at stride-2 layers 1–3,
    stride-1 layers 4–5 with padding-1 kernels of size 4).
    """

    def __init__(self, in_channels: int = 1, ndf: int = 64):
        super().__init__()
        def _block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 4, stride=stride, padding=1, bias=False),
                nn.GroupNorm(min(32, cout), cout),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            # layer 1: no norm (standard PatchGAN convention)
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _block(ndf,       ndf * 2, stride=2),
            _block(ndf * 2,   ndf * 4, stride=2),
            _block(ndf * 4,   ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),  # patch logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss_d(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Hinge discriminator loss."""
        logit_real = self(real.detach())
        logit_fake = self(fake.detach())
        return (F.relu(1.0 - logit_real) + F.relu(1.0 + logit_fake)).mean()

    def loss_g(self, fake: torch.Tensor) -> torch.Tensor:
        """Hinge generator adversarial loss (lower = fooling the disc more)."""
        return -self(fake).mean()


# ---------------------------------------------------------------------------
# LPIPS wrapper — grayscale → 3ch, optional pad-then-trim for MaxPool safety
# ---------------------------------------------------------------------------

class GrayscaleLPIPS(nn.Module):
    """Wraps lpips.LPIPS for single-channel [-1,1] CXR images.

    The MaxPool2d layers in AlexNet/VGG can produce boundary artifacts when
    spatial dimensions are not multiples of 32. We pad to the next multiple of
    32 before computing LPIPS and trim the feature maps after the backbone (the
    lpips library returns a scalar so no trim is needed for the final scalar loss,
    but the input padding prevents index-out-of-bounds in non-default spatial mode).
    """

    def __init__(self, net: str = "alex"):
        super().__init__()
        import lpips as _lpips
        self.fn = _lpips.LPIPS(net=net, verbose=False)
        for p in self.fn.parameters():
            p.requires_grad_(False)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        ph = (32 - h % 32) % 32
        pw = (32 - w % 32) % 32
        # symmetric padding so boundary effect is equal on both sides
        return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode="reflect"), (ph, pw)

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LPIPS expects 3-channel [-1,1] input
        recon_3 = recon.expand(-1, 3, -1, -1)
        target_3 = target.expand(-1, 3, -1, -1)
        recon_p, _ = self._pad(recon_3)
        target_p, _ = self._pad(target_3)
        return self.fn(recon_p, target_p).mean()


# ---------------------------------------------------------------------------
# Manifest dataset
# ---------------------------------------------------------------------------

class ManifestDataset(torch.utils.data.Dataset):
    """Reads image_path column from train_manifest.csv and applies augmentation."""

    def __init__(self, manifest_path: str, res: int, augment: bool = True, aug_cfg: dict | None = None):
        df = pd.read_csv(manifest_path)
        assert "image_path" in df.columns, "train_manifest.csv missing image_path column"
        self.paths = df["image_path"].tolist()
        self.res = res

        if augment and aug_cfg:
            rc = aug_cfg.get("random_crop", {})
            tfm_list = [
                transforms.RandomResizedCrop(
                    size=res,
                    scale=tuple(rc.get("scale", [0.95, 1.0])),
                    ratio=tuple(rc.get("ratio", [0.99, 1.01])),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.RandomHorizontalFlip(p=aug_cfg.get("hflip_prob", 0.5)
                                                if aug_cfg.get("hflip", True) else 0.0),
            ]
        else:
            tfm_list = [transforms.Resize((res, res), transforms.InterpolationMode.LANCZOS)]

        self.tfm = transforms.Compose([
            *tfm_list,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("L")
        return self.tfm(img)


def _make_loader(manifest_path: str, res: int, batch: int, num_workers: int,
                 val_fraction: float, aug_cfg: dict | None, seed: int = 42):
    """Split manifest into train/val loaders."""
    import torch.utils.data as _tud

    ds_train = ManifestDataset(manifest_path, res, augment=True, aug_cfg=aug_cfg)
    ds_val   = ManifestDataset(manifest_path, res, augment=False, aug_cfg=None)

    n = len(ds_train)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val

    gen = torch.Generator().manual_seed(seed)
    idx_train, idx_val = _tud.random_split(range(n), [n_train, n_val], generator=gen)

    train_dl = _tud.DataLoader(
        _tud.Subset(ds_train, idx_train.indices),
        batch_size=batch, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_dl = _tud.DataLoader(
        _tud.Subset(ds_val, idx_val.indices),
        batch_size=batch, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train_dl, val_dl, len(idx_train.indices), len(idx_val.indices), list(idx_val.indices)


# ---------------------------------------------------------------------------
# Disc weight schedule
# ---------------------------------------------------------------------------

def _disc_weight(step: int, warmup_start: int, warmup_steps: int, lambda_disc: float) -> float:
    if step < warmup_start:
        return 0.0
    if warmup_steps <= 0:
        return lambda_disc
    frac = min(1.0, (step - warmup_start) / warmup_steps)
    return lambda_disc * frac


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save(path: str, vae: VAE, opt_g: torch.optim.Optimizer,
          disc: Optional[PatchGAN], opt_d: Optional[torch.optim.Optimizer],
          ema: Optional[EMA], step: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"model": vae.state_dict(), "opt_g": opt_g.state_dict(), "step": step}
    if disc is not None:
        payload["disc"] = disc.state_dict()
    if opt_d is not None:
        payload["opt_d"] = opt_d.state_dict()
    if ema is not None:
        payload["ema"] = ema.shadow
    torch.save(payload, path)
    _console.log(f"[dim]ckpt → {path}[/dim]")


def _load(path: str, vae: VAE, opt_g: torch.optim.Optimizer,
          disc: Optional[PatchGAN], opt_d: Optional[torch.optim.Optimizer],
          ema: Optional[EMA]) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    vae.load_state_dict(ckpt.get("model", ckpt), strict=False)
    if opt_g is not None and "opt_g" in ckpt:
        opt_g.load_state_dict(ckpt["opt_g"])
    if disc is not None and "disc" in ckpt:
        disc.load_state_dict(ckpt["disc"])
    if opt_d is not None and "opt_d" in ckpt:
        opt_d.load_state_dict(ckpt["opt_d"])
    if ema is not None and "ema" in ckpt:
        # checkpoint saved with map_location="cpu"; move shadow to match vae's device
        dev = next(vae.parameters()).device
        ema.shadow = {k: v.to(dev) for k, v in ckpt["ema"].items()}
    return ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# VAE eval hook — CTR delta + PSPNet overlays + presence + recon panels
# ---------------------------------------------------------------------------

# Lazy instrument caches (CPU-resident, same pattern as train_ldm.py)
_VAE_SEG: object = None
_VAE_CLF: object = None


def _vae_load_instruments():
    global _VAE_SEG, _VAE_CLF
    if _VAE_SEG is None:
        try:
            import scripts.ctr_overlay as C
            _VAE_SEG = C._load_seg("cpu")
        except Exception as e:
            _console.log(f"[dim][eval][/dim] PSPNet load failed ({e}) — overlay panels disabled")
            _VAE_SEG = "FAILED"
    if _VAE_CLF is None:
        try:
            from metrics import presence_classifier as P
            _VAE_CLF = P.load_model("cpu")
        except Exception as e:
            _console.log(f"[dim][eval][/dim] presence classifier unavailable ({e}) — presence/* = nan")
            _VAE_CLF = "FAILED"
    seg = None if _VAE_SEG == "FAILED" else _VAE_SEG
    clf = None if _VAE_CLF == "FAILED" else _VAE_CLF
    return seg, clf


def _make_recon_panel(originals: torch.Tensor, recons: torch.Tensor,
                      label_name: str, n: int = 4):
    """Side-by-side matplotlib panel: top row = input, bottom row = reconstruction."""
    try:
        import matplotlib.pyplot as plt
        import wandb as _wb
        n = min(n, originals.shape[0])
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
        to_img = lambda t: ((t[0].detach().cpu().float() + 1) / 2).clamp(0, 1).numpy()
        for i in range(n):
            axes[0, i].imshow(to_img(originals[i]), cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title("input", fontsize=7)
            axes[0, i].axis("off")
            axes[1, i].imshow(to_img(recons[i]), cmap="gray", vmin=0, vmax=1)
            axes[1, i].set_title("recon", fontsize=7)
            axes[1, i].axis("off")
        fig.suptitle(label_name, fontsize=10)
        fig.tight_layout()
        panel = _wb.Image(fig, caption=label_name)
        plt.close(fig)
        return panel
    except Exception:
        return None


def _vae_eval_hook(
    vae: VAE,
    manifest_path: str,
    val_indices: list,
    res: int,
    device: str,
    step: int,
    wb,
    n_per_class: int = 16,
) -> Optional[float]:
    """Full VAE quality eval every 5k steps.

    For each class (normal/cardio/effusion), samples n_per_class images from the
    held-out val split, encodes+decodes through the VAE, then logs:
      panel/{class}_burned   — PSPNet colored overlay on the reconstruction
      panel/{class}_recon    — side-by-side input vs reconstruction
      ctr_med/{class}        — median CTR of the reconstructions
      border_frac/{class}    — fraction with traceable cardiac border
      presence/cardio        — classifier on reconstructed cardio images
      presence/effusion      — classifier on reconstructed effusion images
      presence/normal_fp     — false-positive rate on normal reconstructions
      val/delta_ctr_median   — median |ΔCTR| across all classes (regression gate)

    Non-blocking: all exceptions are caught; returns None on total failure.
    """
    import numpy as np
    import traceback as _tb

    try:
        seg_pair, clf = _vae_load_instruments()

        manifest_df = pd.read_csv(manifest_path)
        val_df = manifest_df.iloc[val_indices].reset_index(drop=True)

        # No-aug loader for sampling: raw PIL → normalised tensor
        load_tfm = transforms.Compose([
            transforms.Resize((res, res), transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        CLASS_MAP = {0: "normal", 1: "cardio", 2: "effusion"}
        metrics: dict = {}
        wb_imgs: dict = {}
        all_delta_ctrs: list = []

        vae.eval()

        for label, label_name in CLASS_MAP.items():
            subset = val_df[val_df["label"] == label]
            if subset.empty:
                continue
            sample = subset.sample(min(n_per_class, len(subset)), random_state=step)

            tensors = []
            for _, row in sample.iterrows():
                img = Image.open(row["image_path"]).convert("L")
                tensors.append(load_tfm(img))
            if not tensors:
                continue

            x = torch.stack(tensors).to(device)
            with torch.no_grad():
                mu, _ = vae.encode_moments(x)
                recon = vae.decode(mu)

            # recon panel
            panel = _make_recon_panel(x, recon, label_name)
            if panel is not None:
                wb_imgs[f"panel/{label_name}_recon"] = panel

            # PSPNet scoring + burned overlays
            ctr_recons: list = []
            border_flags: list = []
            gray_recons: list = []
            burned_panels = []

            if seg_pair is not None:
                from metrics.eval_checkpoint import (
                    _decoded_to_gray512, score_decoded_array,
                    OverlayResult, render_burned, _caption,
                )
                import scripts.ctr_overlay as C
                seg_model, seg_idx = seg_pair

                for i in range(x.shape[0]):
                    orig_gray  = _decoded_to_gray512(x[i])
                    recon_gray = _decoded_to_gray512(recon[i])
                    gray_recons.append(recon_gray)

                    m_orig  = score_decoded_array(seg_model, seg_idx, orig_gray,  "cpu")
                    m_recon = score_decoded_array(seg_model, seg_idx, recon_gray, "cpu",
                                                  with_masks=True)

                    if m_orig.get("ok") and m_recon.get("ok"):
                        all_delta_ctrs.append(abs(m_orig["ctr"] - m_recon["ctr"]))
                        ctr_recons.append(m_recon["ctr"])
                        border_flags.append(bool(m_recon.get("border_visible")))

                    # burned overlay (first 4 per class)
                    if i < 4:
                        msk = m_recon.get("_masks") or {}
                        o = OverlayResult(
                            stem=f"{label_name}_{i:03d}",
                            base=recon_gray,
                            ctr=m_recon["ctr"],
                            heart_conf=m_recon.get("heart_conf", 0.0),
                            opacity=m_recon.get("peri_cardiac_opacity", float("nan")),
                            border_visible=bool(m_recon.get("border_visible")),
                            cp_left=m_recon.get("cp_left", {}),
                            cp_right=m_recon.get("cp_right", {}),
                            heart_line=m_recon.get("heart_line"),
                            thorax_line=m_recon.get("thorax_line"),
                            heart_mask=msk.get(C.HEART),
                            llung_mask=msk.get(C.LLUNG),
                            rlung_mask=msk.get(C.RLUNG),
                            ok=bool(m_recon.get("ok")),
                        )
                        try:
                            import wandb as _wb
                            rgb = render_burned(None, o)
                            burned_panels.append(_wb.Image(rgb, caption=_caption(o)))
                        except Exception:
                            pass
            else:
                # seg unavailable: still collect recon grays for presence
                from metrics.eval_checkpoint import _decoded_to_gray512
                for i in range(x.shape[0]):
                    gray_recons.append(_decoded_to_gray512(recon[i]))

            if ctr_recons:
                metrics[f"ctr_med/{label_name}"] = float(np.median(ctr_recons))
            if border_flags:
                metrics[f"border_frac/{label_name}"] = float(np.mean(border_flags))
            if burned_panels:
                wb_imgs[f"panel/{label_name}_burned"] = burned_panels

            # presence classifier on reconstructions
            if clf is not None and gray_recons:
                try:
                    from metrics.eval_checkpoint import _predict_probs_arrays
                    model, cidx, eidx, ckpt = clf
                    probs = _predict_probs_arrays(model, gray_recons, cidx, eidx, "cpu")
                    thr_c = float(ckpt.get("youden_threshold_cardio", 0.5))
                    thr_e = float(ckpt.get("youden_threshold_effusion", 0.5))
                    if label_name == "cardio":
                        metrics["presence/cardio"] = float((probs[:, 0] > thr_c).mean())
                    elif label_name == "effusion":
                        metrics["presence/effusion"] = float((probs[:, 1] > thr_e).mean())
                    elif label_name == "normal":
                        # false-positive rate: normals mis-classified as any disease
                        metrics["presence/normal_fp"] = float(
                            ((probs[:, 0] > thr_c) | (probs[:, 1] > thr_e)).mean())
                except Exception as e:
                    _console.log(f"[dim][eval][/dim] presence failed for {label_name}: {e}")

        vae.train()

        # aggregate delta CTR
        delta_ctr = float(np.median(all_delta_ctrs)) if all_delta_ctrs else None
        if delta_ctr is not None:
            metrics["val/delta_ctr_median"] = delta_ctr
            flag = " ⚠ REGRESSION" if delta_ctr > 0.015 else ""
            _console.log(f"  VAE eval step {step}: delta_ctr_median=[cyan]{delta_ctr:.4f}[/cyan]{flag}")
            if delta_ctr > 0.015:
                metrics["val/delta_ctr_regression"] = 1.0

        # log scalars + image panels to W&B
        if wb is not None:
            if metrics:
                _wandb_log(wb, metrics, step)
            if wb_imgs:
                try:
                    import wandb as _wb
                    _wb.log(wb_imgs, step=step)
                except Exception as e:
                    _console.log(f"[dim][eval][/dim] W&B image log failed: {e}")

        return delta_ctr

    except Exception as e:
        _console.log(f"[bold red][vae_eval_hook][/bold red] failed at step {step}: {e}")
        _tb.print_exc()
        return None


# ---------------------------------------------------------------------------
# Rich UI helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _make_progress(start_step: int, max_steps: int) -> Progress:
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold magenta]VAE-v2[/bold magenta]"),
        MofNCompleteColumn(),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("  [yellow]l1[/yellow]=[green]{task.fields[l1]:.4f}[/green]"),
        TextColumn("  [yellow]lpips[/yellow]=[cyan]{task.fields[lpips]:.4f}[/cyan]"),
        TextColumn("  [yellow]disc[/yellow]=[cyan]{task.fields[disc]:.4f}[/cyan]"),
        TimeElapsedColumn(),
        TextColumn("[dim]eta[/dim]"),
        TimeRemainingColumn(),
        console=_console,
        refresh_per_second=4,
        transient=False,
    )


def _print_banner(args, cfg_yaml: dict, start_step: int, max_steps: int,
                  n_train: int, n_val: int, device: str) -> None:
    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False, show_edge=False)
    table.add_column(style="bold cyan", no_wrap=True, min_width=18)
    table.add_column(style="white")
    table.add_row("Run", f"[bold]{args.wandb_run or 'vae_v2_finetune'}[/bold]")
    table.add_row("Device", f"{device}  [dim]fp32 (bf16 disabled for decoder)[/dim]")
    table.add_row("Steps", f"{start_step:,} → {max_steps:,}"
                  + (f" [dim](resuming from {args.resume})[/dim]" if args.resume else ""))
    table.add_row("Batch", str(cfg_yaml.get("batch", args.batch or 4)))
    table.add_row("Diet", f"{n_train:,} train / {n_val:,} val")
    table.add_row("LR", str(cfg_yaml.get("lr", args.lr or 5e-5)))
    table.add_row("LPIPS", "[green]✓ AlexNet λ=0.1[/green]" if args.lpips else "[dim]✗[/dim]")
    table.add_row("PatchGAN", f"[green]✓ warmup {args.disc_warmup_start}→"
                  f"{args.disc_warmup_start + args.disc_warmup_steps}[/green]"
                  if args.discriminator else "[dim]✗[/dim]")
    table.add_row("Eval every", f"{args.eval_every:,} steps  ({args.eval_n_per_class}×3 imgs)")
    table.add_row("Save every", str(cfg_yaml.get("ckpt_every", args.save_every or 5000)))
    table.add_row("Warm start", args.resume or "[dim]none[/dim]")
    _console.print(Panel(table,
                         title="[bold magenta] VAE-v2 Fine-Tune [/bold magenta]",
                         border_style="magenta"))


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(args) -> None:
    import yaml

    # dry-run forces CPU: avoids CUDA OOM from attention at 512² with uninitialised weights
    device = "cpu" if args.dry_run else _device(args.device)

    # load YAML config if provided
    cfg_yaml: dict = {}
    if args.config and Path(args.config).exists():
        cfg_yaml = yaml.safe_load(open(args.config))

    res          = args.res or cfg_yaml.get("res", 512)
    batch        = args.batch or cfg_yaml.get("batch", 4)
    lr           = args.lr or cfg_yaml.get("lr", 5e-5)
    weight_decay = args.weight_decay or cfg_yaml.get("weight_decay", 1e-2)
    num_workers  = args.num_workers or cfg_yaml.get("num_workers", 4)
    val_fraction = args.val_fraction or cfg_yaml.get("val_fraction", 0.05)
    aug_cfg      = cfg_yaml.get("augmentation")
    max_steps    = args.max_steps or cfg_yaml.get("steps", 75000)
    save_every   = args.save_every or cfg_yaml.get("ckpt_every", 5000)
    output_dir   = Path(args.output_dir or cfg_yaml.get("ckpt_dir", "ckpts/vae_v2"))
    lambda_recon = 1.0
    lambda_perc  = cfg_yaml.get("perceptual_weight", 0.1) if args.lpips else 0.0
    lambda_disc  = cfg_yaml.get("discriminator_weight", 0.5) if args.discriminator else 0.0
    manifest     = args.manifest or cfg_yaml.get("train_manifest")

    # build VAE
    vae_cfg = VAEConfig(input_resolution=res, use_checkpoint=args.grad_checkpoint)
    vae = VAE(vae_cfg).to(device)
    opt_g = torch.optim.AdamW(vae.parameters(), lr=lr, betas=(0.9, 0.999),
                               weight_decay=weight_decay)
    ema = EMA(vae, decay=cfg_yaml.get("ema_decay", 0.999)) if cfg_yaml.get("ema", True) else None

    # build discriminator
    disc: Optional[PatchGAN] = None
    opt_d: Optional[torch.optim.Optimizer] = None
    if args.discriminator:
        disc = PatchGAN(in_channels=1).to(device)
        # disc LR 10× generator LR: pretrained VAE already produces near-perfect
        # reconstructions (recon_l1≈0.025), so the disc needs a higher LR to keep up.
        disc_lr = lr * 10
        opt_d = torch.optim.AdamW(disc.parameters(), lr=disc_lr, betas=(0.5, 0.9))
        _console.log(f"PatchGAN discriminator: {sum(p.numel() for p in disc.parameters())/1e6:.1f}M params  "
                     f"disc_lr={disc_lr:.2e} (10× gen lr={lr:.2e})")

    # build LPIPS
    lpips_fn: Optional[GrayscaleLPIPS] = None
    if args.lpips:
        lpips_fn = GrayscaleLPIPS(net="alex").to(device)
        _console.log("LPIPS (AlexNet): loaded")

    # resume
    start_step = 0
    if args.resume:
        start_step = _load(args.resume, vae, opt_g, disc, opt_d, ema)
        _console.log(f"resumed from [cyan]{args.resume}[/cyan] at step {start_step}")

    # VAE decoder must run fp32: bf16 GroupNorm produces black/collapsed reconstructions.
    # This matches vae/train.py which always forces fp32 for the VAE codec.
    use_amp   = False
    amp_dtype = torch.bfloat16   # unused, kept for the autocast call signature
    _console.log(f"device:[cyan]{device}[/cyan]  precision:[bold]fp32[/bold] "
                 f"(bf16 disabled for VAE decoder)  "
                 f"lpips:{args.lpips}  disc:{args.discriminator}")

    # ── dry-run: 2 fake steps, print loss dict, exit ──────────────────────────
    if args.dry_run:
        # Use 128² for dry-run: avoids the MONAI mid-block attention OOM at 512²
        # (attention cost is O(L²) and MONAI's non-local attn uses unbounded RAM at 512²)
        dry_res = 128
        dry_vae_cfg = VAEConfig(input_resolution=dry_res, use_checkpoint=False)
        vae = VAE(dry_vae_cfg).to(device)
        if disc:
            disc = disc.to(device)
        if lpips_fn:
            lpips_fn = lpips_fn.to(device)
        print("\n[dry-run] running 2 synthetic steps to verify loss terms (res=128)…")
        vae.train()
        if disc:
            disc.train()
        for step in range(1, 3):
            x = torch.randn(2, 1, dry_res, dry_res, device=device)
            mu, sigma = vae.encode_moments(x)
            z = mu
            recon = vae.decode(z)

            kl  = kl_divergence(mu, sigma)
            l1  = F.l1_loss(recon, x)
            terms: dict[str, float] = {"recon": l1.item(), "kl": kl.item()}

            if lpips_fn is not None:
                perc = lpips_fn(recon.detach().float(), x.float())
                terms["lpips"] = perc.item()

            dw = _disc_weight(start_step + step,
                              args.disc_warmup_start, args.disc_warmup_steps, lambda_disc)
            terms["disc_weight"] = dw
            if disc is not None:
                adv = disc.loss_g(recon.detach())
                terms["disc"] = adv.item()
                d_loss = disc.loss_d(x, recon)
                terms["disc_d"] = d_loss.item()

            total = lambda_recon * l1 + vae_cfg.kl_weight * kl
            if lpips_fn:
                total = total + lambda_perc * perc
            if disc:
                total = total + dw * adv
            terms["total"] = total.item()

            print(f"  step {step}: " + "  ".join(f"{k}={v:.5f}" for k, v in terms.items()))

        assert "recon" in terms and "kl" in terms, "missing base loss keys"
        if args.lpips:
            assert "lpips" in terms, "missing lpips loss key"
        if args.discriminator:
            assert "disc" in terms, "missing disc loss key"
        print("\n[dry-run] PASS — all requested loss keys present ✓")
        return

    # ── data ──────────────────────────────────────────────────────────────────
    assert manifest, "provide --manifest or set train_manifest in config"
    train_dl, val_dl, n_train, n_val, val_indices = _make_loader(
        manifest, res, batch, num_workers, val_fraction, aug_cfg)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wb_cfg  = vars(args) | {"res": res, "batch": batch, "steps": max_steps}
    wb_proj = cfg_yaml.get("wandb_project", "paper3-vae-v2")
    wb = _wandb_init(wb_proj, wb_cfg) if wb_proj else None
    if wb and args.wandb_run:
        try:
            wb.run.name = args.wandb_run
        except Exception:
            pass

    # ── banner ────────────────────────────────────────────────────────────────
    _print_banner(args, cfg_yaml, start_step, max_steps, n_train, n_val, device)
    if wb:
        _console.log(f"[dim]W&B[/dim] run [cyan]{wb.run.name}[/cyan]"
                     f" → [link={wb.run.url}]{wb.run.url}[/link]")

    # ── loop ──────────────────────────────────────────────────────────────────
    vae.train()
    if disc:
        disc.train()

    it = iter(train_dl)
    def _next():
        nonlocal it
        try:
            return next(it).to(device)
        except StopIteration:
            it = iter(train_dl)
            return next(it).to(device)

    t0, seen = time.time(), 0
    ctr_eval_every = args.eval_every
    steps_per_epoch = max(1, n_train // batch)
    total_epochs    = max(1, max_steps // steps_per_epoch)

    # running smoothed values for the progress bar
    _l1_smooth   = float("nan")
    _lpips_smooth = float("nan")
    _disc_smooth  = float("nan")
    _alpha = 0.05  # EMA smoothing

    progress = _make_progress(start_step, max_steps)
    with progress:
        task_id = progress.add_task(
            "train",
            total=max_steps,
            completed=start_step,
            l1=float("nan"),
            lpips=float("nan"),
            disc=float("nan"),
        )

        for step in range(start_step + 1, max_steps + 1):
            x = _next()

            # ── discriminator step ────────────────────────────────────────────
            d_loss_val = float("nan")
            if disc is not None:
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    mu_d, _ = vae.encode_moments(x)
                    fake_d  = vae.decode(mu_d)
                d_loss = disc.loss_d(x, fake_d)
                d_loss.backward()
                opt_d.step()
                d_loss_val = d_loss.item()

            # ── generator step ────────────────────────────────────────────────
            opt_g.zero_grad(set_to_none=True)
            mu, sigma = vae.encode_moments(x)
            z     = mu
            recon = vae.decode(z)

            kl   = kl_divergence(mu.float(), sigma.float())
            l1   = F.l1_loss(recon.float(), x.float())
            total = lambda_recon * l1 + vae_cfg.kl_weight * kl

            perc = None
            if lpips_fn is not None:
                perc = lpips_fn(recon.float(), x.float())
                total = total + lambda_perc * perc

            dw = _disc_weight(step, args.disc_warmup_start, args.disc_warmup_steps, lambda_disc)
            adv = None
            if disc is not None and dw > 0:
                adv = disc.loss_g(recon.float())
                total = total + dw * adv

            total.backward()
            gn = _grad_norm(vae)
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt_g.step()
            if ema:
                ema.update(vae)
            seen += x.shape[0]

            # ── smooth values for progress bar ────────────────────────────────
            _l1_smooth = l1.item() if step == start_step + 1 else (
                _alpha * l1.item() + (1 - _alpha) * _l1_smooth)
            if perc is not None:
                _lpips_smooth = perc.item() if step == start_step + 1 else (
                    _alpha * perc.item() + (1 - _alpha) * (_lpips_smooth or perc.item()))
            if not (d_loss_val != d_loss_val):  # not nan
                _disc_smooth = d_loss_val if step == start_step + 1 else (
                    _alpha * d_loss_val + (1 - _alpha) * (_disc_smooth or d_loss_val))

            progress.update(task_id, advance=1,
                            l1=_l1_smooth,
                            lpips=_lpips_smooth,
                            disc=_disc_smooth)

            # ── periodic W&B + console line ───────────────────────────────────
            if step % args.log_every == 0:
                img_s = seen / max(time.time() - t0, 1e-6)
                metrics = {
                    "loss/total":    total.item(),
                    "loss/recon_l1": l1.item(),
                    "loss/kl":       kl.item(),
                    "train/disc_weight": dw,
                    "train/grad_norm":   gn,
                    "train/img_s":       img_s,
                }
                if perc is not None:
                    metrics["loss/lpips"] = perc.item()
                if disc is not None:
                    metrics["loss/disc_d"] = d_loss_val
                if adv is not None:
                    metrics["loss/disc_g"] = adv.item()
                _wandb_log(wb, metrics, step)

                epoch = step // steps_per_epoch + 1
                parts = (f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                         f"  [dim]ep[/dim] {epoch}/{total_epochs}"
                         f"  l1=[green]{l1.item():.4f}[/green]"
                         f"  kl=[cyan]{kl.item():.1f}[/cyan]")
                if perc is not None:
                    parts += f"  lpips=[cyan]{perc.item():.4f}[/cyan]"
                if adv is not None:
                    parts += f"  adv=[yellow]{adv.item():.4f}[/yellow]  dw={dw:.3f}"
                parts += f"  [dim]{img_s:.1f} img/s[/dim]"
                _console.log(parts)

            # ── CTR eval ──────────────────────────────────────────────────────
            if ctr_eval_every and step % ctr_eval_every == 0:
                _console.log(f"[dim]{_now()}[/dim]  [bold]CTR eval[/bold] step {step}…")
                _vae_eval_hook(vae, manifest, val_indices, res, device, step, wb,
                               n_per_class=args.eval_n_per_class)

            # ── checkpoint ────────────────────────────────────────────────────
            if save_every and step % save_every == 0:
                ckpt_path = output_dir / f"vae_v2_step{step:07d}.pt"
                _save(str(ckpt_path), vae, opt_g, disc, opt_d, ema, step)
                _console.log(f"[dim]checkpoint → {ckpt_path.name}[/dim]")

    # ── final checkpoint ──────────────────────────────────────────────────────
    _save(str(output_dir / "vae_v2_final.pt"), vae, opt_g, disc, opt_d, ema, max_steps)
    elapsed = time.time() - t0
    _console.log(f"\n[bold green]done[/bold green]  "
                 f"{seen:,} imgs  {elapsed:.0f}s  {seen/elapsed:.1f} img/s")
    if wb:
        wb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VAE-v2 LPIPS+PatchGAN fine-tune")

    p.add_argument("--dry-run", action="store_true",
                   help="2 synthetic steps, print loss dict, exit")
    p.add_argument("--config", default="configs/vae_v2.yaml",
                   help="YAML config (values override defaults; CLI overrides YAML)")
    p.add_argument("--manifest", default=None,
                   help="path to train_manifest.csv (overrides config)")
    p.add_argument("--resume", default=None,
                   help="checkpoint to warm-start from")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None)

    # loss flags
    p.add_argument("--lpips", action="store_true",
                   help="enable LPIPS perceptual loss (AlexNet, λ=0.1)")
    p.add_argument("--discriminator", action="store_true",
                   help="enable PatchGAN adversarial loss")
    p.add_argument("--disc_warmup_start", type=int, default=1000,
                   help="step at which disc loss starts affecting generator")
    p.add_argument("--disc_warmup_steps", type=int, default=1000,
                   help="steps over which disc weight ramps 0→λ_disc")

    # architecture / hardware
    p.add_argument("--res", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--fp32", action="store_true")

    # optimisation
    p.add_argument("--max_steps", "--max-steps", dest="max_steps", type=int, default=None,
                   help="total training steps (including pre-resume steps)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=None)
    p.add_argument("--save_every", "--save-every", dest="save_every", type=int, default=None)

    # data
    p.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    p.add_argument("--val-fraction", dest="val_fraction", type=float, default=None)

    # logging
    p.add_argument("--log-every", dest="log_every", type=int, default=100)
    p.add_argument("--wandb_run", default=None,
                   help="W&B run display name")
    p.add_argument("--eval_every", type=int, default=5000,
                   help="run full VAE eval (CTR+presence+panels) every N steps (0=off)")
    p.add_argument("--eval_n_per_class", type=int, default=16,
                   help="images per class (normal/cardio/effusion) sampled at each eval")

    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
