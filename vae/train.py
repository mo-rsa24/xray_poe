"""VAE training loop — overfit-sanity, noise-data, and real-data runs.

  # local synthetic gates
  python -m vae.train --overfit --batch 4 --steps 400 --lr 1e-4
  python -m vae.train --data noise --res 512 --batch 1 --grad-checkpoint --steps 25

  # RunPod real-data train (driven by configs/vae.yaml via train_vae.sh)
  python -m vae.train --data real --data-dir /workspace/Paper3/data/nih/images \
      --csv /workspace/Paper3/data/nih/Data_Entry_2017.csv \
      --res 512 --batch 8 --grad-checkpoint --steps 150000 \
      --ckpt-every 5000 --ckpt-dir ckpts/ \
      --wandb-project paper3-vae --log-images-every 500

Overfit gate notes:
  - Uses deterministic decode(μ) by default: tests codec capacity, not sampling noise.
  - Runs fp32 (bf16 + MS-SSIM diverged at small scale in testing).
  - Real-data overfit (--overfit --data real) draws the first --batch images from the
    training split — a real-wiring gate before the full run.

σ-drift note (see plans/vae/profiling-notes.md):
  - With kl_weight=1e-6 σ can drift large in short runs; reconstruction quality is
    always reported on decode(μ), not a random sample.
  - Monitor z_sigma in W&B; if it blows up on real data, a brief KL warm-up is the lever.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .config import VAEConfig
from .data import fixed_overfit_batch, make_splits, noise_loader, real_cxr_loader
from .losses import LossTerms, kl_divergence, reconstruction_loss
from .model import VAE


# ---------------------------------------------------------------------------
# W&B — optional; gracefully absent when not installed / not needed
# ---------------------------------------------------------------------------

def _wandb_init(project: str, config: dict) -> Optional[object]:
    try:
        import wandb
        run = wandb.init(project=project, config=config, resume="allow")
        print(f"W&B run: {run.url}")
        return wandb
    except Exception as e:
        print(f"[warn] W&B unavailable ({e}); logging to stdout only")
        return None


def _wandb_log(wb, metrics: dict, step: int) -> None:
    if wb is not None:
        wb.log(metrics, step=step)


def _wandb_log_images(wb, tag: str, x: torch.Tensor, recon: torch.Tensor, step: int, n: int = 4) -> None:
    if wb is None:
        return
    try:
        import wandb
        n = min(n, x.shape[0])
        to_img = lambda t: ((t[0, 0].detach().cpu().float() + 1) / 2).clamp(0, 1).numpy()
        panels = []
        for i in range(n):
            panels.append(wandb.Image(to_img(x[i:i+1]), caption=f"input {i}"))
            panels.append(wandb.Image(to_img(recon[i:i+1]), caption=f"recon {i}"))
        wb.log({tag: panels}, step=step)
    except Exception as e:
        print(f"[warn] W&B image log failed: {e}")


def _wandb_log_latent_manifold(wb, model: VAE, loader, device: str, step: int, n_batches: int = 4) -> None:
    """Log a 2-D PCA projection of the latent means as a scatter plot."""
    if wb is None:
        return
    try:
        import wandb
        import numpy as np

        model.eval()
        zs = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                x = batch.to(device)
                mu, _ = model.encode_moments(x)
                zs.append(mu.flatten(1).cpu().float().numpy())
        model.train()
        Z = np.concatenate(zs, axis=0)
        # 2-D PCA via SVD (no sklearn needed)
        Z_c = Z - Z.mean(0)
        _, _, Vt = np.linalg.svd(Z_c, full_matrices=False)
        proj = Z_c @ Vt[:2].T
        table = wandb.Table(columns=["pc1", "pc2"],
                            data=[[float(r[0]), float(r[1])] for r in proj])
        wb.log({"latent/manifold": wandb.plot.scatter(table, "pc1", "pc2",
                title="Latent PCA")}, step=step)
    except Exception as e:
        print(f"[warn] latent manifold log failed: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm(2).item() ** 2
    return total ** 0.5


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def save_checkpoint(path: str, model: VAE, opt, step: int, ema: Optional[EMA] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
    if ema is not None:
        payload["ema"] = ema.shadow
    torch.save(payload, path)
    print(f"checkpoint → {path}")


def load_checkpoint(path: str, model: VAE, opt=None, ema: Optional[EMA] = None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    if ema is not None and "ema" in ckpt:
        ema.shadow = ckpt["ema"]
    return ckpt.get("step", 0)


def _save_figure(x: torch.Tensor, recon: torch.Tensor, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure")
        return
    n = min(4, x.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
    to_img = lambda t: ((t[0, 0].detach().cpu().float() + 1) / 2).clamp(0, 1)
    for i in range(n):
        axes[0, i].imshow(to_img(x[i:i+1]), cmap="gray"); axes[0, i].set_title("in"); axes[0, i].axis("off")
        axes[1, i].imshow(to_img(recon[i:i+1]), cmap="gray"); axes[1, i].set_title("out"); axes[1, i].axis("off")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args) -> None:
    device = _device(args.device)
    cfg = VAEConfig(input_resolution=args.res, use_checkpoint=args.grad_checkpoint)
    model = VAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                             weight_decay=args.weight_decay)
    ema = EMA(model, args.ema_decay) if args.ema else None

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, opt, ema)
        print(f"resumed from {args.resume} at step {start_step}")

    # precision: overfit always fp32 (bf16 + MS-SSIM diverged at small scale)
    use_amp = device == "cuda" and not args.overfit and not args.fp32
    amp_dtype = torch.bfloat16
    print(f"device: {device}  precision: {'bf16' if use_amp else 'fp32'}")

    # W&B
    wb_config = vars(args)
    wb = _wandb_init(args.wandb_project, wb_config) if args.wandb_project else None

    # ---- data source -------------------------------------------------------
    val_loader = None  # set for real-data runs

    if args.overfit:
        if args.data == "real":
            # real-data overfit gate: first --batch images from the training split
            train_paths, _ = make_splits(args.csv, args.data_dir,
                                          val_fraction=args.val_fraction)
            overfit_paths = train_paths[:args.batch]
            ds_loader = real_cxr_loader(overfit_paths, res=args.res, batch=args.batch,
                                        num_workers=0, split="train")
            raw_batch = next(iter(ds_loader)).to(device)
        else:
            raw_batch = fixed_overfit_batch(batch=args.batch, res=args.res).to(device)
        get_batch = lambda: raw_batch
        print(f"[overfit] fixed batch {tuple(raw_batch.shape)} on {device}")

    elif args.data == "real":
        train_paths, val_paths = make_splits(args.csv, args.data_dir,
                                              val_fraction=args.val_fraction)
        print(f"[real] train={len(train_paths)} val={len(val_paths)} images")
        train_dl = real_cxr_loader(train_paths, res=args.res, batch=args.batch,
                                    num_workers=args.num_workers, split="train")
        val_loader = real_cxr_loader(val_paths, res=args.res, batch=args.batch,
                                      num_workers=args.num_workers, split="val")
        it = iter(train_dl)

        def get_batch():
            nonlocal it
            try:
                return next(it).to(device)
            except StopIteration:
                it = iter(train_dl)
                return next(it).to(device)

    else:  # noise
        train_dl = noise_loader(res=args.res, batch=args.batch)
        it = iter(train_dl)

        def get_batch():
            nonlocal it
            try:
                return next(it).to(device)
            except StopIteration:
                it = iter(train_dl)
                return next(it).to(device)
        print(f"[noise] res={args.res} batch={args.batch} on {device}")

    # ---- recon path --------------------------------------------------------
    deterministic = args.deterministic if args.deterministic is not None else args.overfit
    print(f"recon path: {'deterministic decode(μ)' if deterministic else 'sampled z'}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- training loop -----------------------------------------------------
    model.train()
    t0, seen, recent_recon = time.time(), 0, []
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    for step in range(start_step + 1, start_step + args.steps + 1):
        x = get_batch()
        opt.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            mu, sigma = model.encode_moments(x)
            z = mu if deterministic else model.net.sampling(mu, sigma)
            recon = model.decode(z)
            recon_l = reconstruction_loss(recon.float(), x, cfg)
            kl = kl_divergence(mu, sigma)
            loss = LossTerms(total=recon_l + cfg.kl_weight * kl, recon=recon_l, kl=kl)

        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss.total).backward()
            scaler.unscale_(opt)
            gn = _grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.total.backward()
            gn = _grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if ema:
            ema.update(model)
        seen += x.shape[0]
        recent_recon.append(loss.recon.item())

        # ---- logging -------------------------------------------------------
        if step % args.log_every == 0 or step == start_step + 1:
            recent_mean = sum(recent_recon[-args.log_every:]) / len(recent_recon[-args.log_every:])
            sigma_mean = sigma.detach().float().mean().item()
            print(f"step {step:>6}  total {loss.total.item():.5f}  "
                  f"recon {recent_mean:.5f}  kl {loss.kl.item():.1f}  "
                  f"gnorm {gn:.3f}  σ̄ {sigma_mean:.3f}")
            _wandb_log(wb, {
                "loss/total": loss.total.item(),
                "loss/recon": recent_mean,
                "loss/kl": loss.kl.item(),
                "train/grad_norm": gn,
                "train/z_sigma_mean": sigma_mean,
                "train/img_s": seen / max(time.time() - t0, 1e-6),
            }, step=step)

        # ---- recon images to W&B ------------------------------------------
        if args.log_images_every and step % args.log_images_every == 0:
            model.eval()
            with torch.no_grad():
                recon_vis = model.reconstruct(x)
            _wandb_log_images(wb, "recon/train", x, recon_vis, step)
            model.train()

        # ---- latent manifold (val set) ------------------------------------
        if args.manifold_every and val_loader and step % args.manifold_every == 0:
            _wandb_log_latent_manifold(wb, model, val_loader, device, step)

        # ---- periodic checkpoint ------------------------------------------
        if args.ckpt_every and step % args.ckpt_every == 0:
            ckpt_path = str(Path(args.ckpt_dir) / f"vae_step{step:07d}.pt")
            save_checkpoint(ckpt_path, model, opt, step, ema)

    # ---- end of run --------------------------------------------------------
    elapsed = time.time() - t0
    img_s = seen / elapsed if elapsed > 0 else float("nan")
    print(f"\n{seen} imgs in {elapsed:.1f}s → {img_s:.1f} img/s")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak VRAM: {peak:.2f} GB")
        _wandb_log(wb, {"train/peak_vram_gb": peak, "train/img_s": img_s},
                   step=start_step + args.steps)

    # overfit figure
    if args.overfit:
        model.eval()
        with torch.no_grad():
            recon_final = model.reconstruct(get_batch())
        _save_figure(get_batch(), recon_final, args.figure)
        _wandb_log_images(wb, "recon/overfit_final", get_batch(), recon_final,
                          step=start_step + args.steps)

    # final checkpoint
    if args.ckpt:
        save_checkpoint(args.ckpt, model, opt, start_step + args.steps, ema)

    if wb is not None:
        wb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VAE training")

    # mode
    p.add_argument("--overfit", action="store_true",
                   help="overfit a fixed tiny batch (correctness gate)")
    p.add_argument("--data", choices=["noise", "real"], default="noise")

    # real-data paths (required when --data real)
    p.add_argument("--data-dir", default="data/nih/images",
                   help="directory of NIH PNG images")
    p.add_argument("--csv", default="data/nih/Data_Entry_2017.csv",
                   help="NIH Data_Entry_2017.csv with labels + view positions")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)

    # architecture / hardware
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--fp32", action="store_true",
                   help="disable bf16 autocast (always on for --overfit)")

    # optimisation
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)

    # recon path
    p.add_argument("--deterministic", action="store_const", const=True, default=None,
                   help="decode(μ) instead of sample (default ON for --overfit)")
    p.add_argument("--sampled", dest="deterministic", action="store_const", const=False)

    # checkpointing
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--ckpt", default=None, help="final checkpoint path")
    p.add_argument("--ckpt-dir", default="ckpts", help="directory for periodic checkpoints")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="save a checkpoint every N steps (0 = disabled)")

    # logging
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--log-images-every", type=int, default=500,
                   help="log recon images to W&B every N steps (0 = disabled)")
    p.add_argument("--manifold-every", type=int, default=5000,
                   help="log latent manifold PCA to W&B every N steps (0 = disabled)")
    p.add_argument("--wandb-project", default="",
                   help="W&B project name; empty string disables W&B")
    p.add_argument("--figure", default="figures/vae_overfit.png")

    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
