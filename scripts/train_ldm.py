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
    recon/grid              — labeled 4×3 overview every grid_every steps (default 5 000); Artifact ldm-recon-grid:step{N}
    recon/{class}           — per-class gallery (4 seeds) every grid_every steps
    recon/cfg_ablation      — 3×3 grid (w=0/1/2 × class) every grid_every steps, 20 DDIM steps
    eval/* (CTR/yield)      — quality dashboard every grid_every steps (32 imgs/finding × 25 DDIM)
    compose/null_anchor/w{w}    — wandb.Image every compose_every steps (default 15 000); 4 panels (anchor×weight), 4 seeds each; Artifact ldm-compose-grid:step{N}
    compose/healthy_anchor/w{w} —

Checkpoints (every ckpt_every steps):
    model_step{N}.safetensors   — model weights (UNet + class_embed)
    training_state_step{N}.pt   — optimizer state, LR scheduler state, step counter,
                                   kill-criteria reference loss; everything needed for
                                   exact resume
    config_step{N}.yaml         — config snapshot
    All three files are uploaded together as W&B Artifact ldm-ckpt:step{N}.

Resume behaviour:
    --resume <wandb-run-id> loads the latest local checkpoint and restarts the
    training loop exactly at the saved step with the optimizer's Adam
    momentum/variance and cosine LR position intact.  By default it starts a
    FRESH W&B run logging from the checkpoint step (monotonic logging even when
    the source run was killed past the last checkpoint, with lineage recorded in
    the run config); pass --wandb-continue to re-enter the same run instead.

Kill criteria:
    EMA(train/loss) > 2× EMA-at-step-10k for _DIVERGENCE_PATIENCE consecutive
    steps → W&B alert + sys.exit(1).  Smoothed + persistent so ordinary per-step
    DDPM loss noise cannot trip it (a single noisy spike no longer kills the run).
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
# Recipe: Weight-EMA + min-SNR-γ (collapse-prevention / consistency levers)
# ---------------------------------------------------------------------------

class _WeightEMA:
    """Exponential moving average of trainable weights (Karras/EDM-style).

    Maintains a shadow copy updated each optimiser step. ``store_and_copy_to`` /
    ``restore`` let eval + checkpoint run on the SMOOTH EMA weights and then put the
    raw training weights back. The EMA weights are what we release for inference
    (the per-seed consistency win); the raw weights keep training. Keyed by
    ``model.named_parameters()`` so it spans both unet.* and class_embed.*.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int | None = None) -> None:
        # Decay WARMUP: min(decay, (step+1)/(step+10)). Early on the EMA tracks the
        # raw weights closely (so eval panels are meaningful from step ~1k instead of
        # being contaminated by the random init until ~40-50k); it ramps to the full
        # `decay` as training matures. Standard practice (Rombach/SD, EDM).
        d = self.decay
        if step is not None:
            d = min(d, (step + 1) / (step + 10))
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def store_and_copy_to(self, model: torch.nn.Module) -> None:
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}


def _min_snr_weight(alphas_cumprod: torch.Tensor, t: torch.Tensor,
                    gamma: float) -> torch.Tensor:
    """Per-sample min-SNR-γ loss weight for ε-prediction (Hang et al. 2023).

    SNR(t) = ᾱ_t / (1−ᾱ_t);  weight = min(SNR, γ) / SNR.  Down-weights the very
    low-noise (high-SNR) steps that otherwise dominate the gradient and let one
    class/feature (here cardio) sharpen while another (effusion) starves.
    """
    abar = alphas_cumprod.to(t.device)[t].clamp(1e-8, 1.0 - 1e-8)
    snr = abar / (1.0 - abar)
    return snr.clamp(max=gamma) / snr


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

# ── frozen generation-quality eval spec (metrics/eval_checkpoint dashboard) ──────
# Fixed seeds so the CTR/yield dashboard is comparable step-to-step. 64 each for the
# scalar metrics; a 12-image (3 findings × 4 seeds) subset for the overlay panels.
# Reuses _COMPOSE_SEEDS as the panel seeds where sensible.
_EVAL_N_PER_FINDING = 32         # monitoring CTR/yield mean is stable at 32; halves dashboard cost
_EVAL_BASE_SEED_NORMAL = 600_000
_EVAL_BASE_SEED_CARDIO = 700_000
_EVAL_BASE_SEED_EFFUSION = 800_000
_EVAL_BASE_SEED_POE = 900_000

# ── trajectory diagnostic probes (compose_every cadence) ─────────────────────
# Denoising accuracy probe: take a real latent, add noise to t, predict ε, derive
# ẑ₀, decode, compare vs original.  Localises where in the noise schedule the model
# struggles (low SSIM at t=200 = fine-detail failure; low SSIM at t=800 = coarse
# anatomy failure).  PoE composition probe: same but with the composed ε — checks
# whether cardio signature survives when effusion conditioning is added.
_PROBE_T_LEVELS = (200, 400, 600, 800)   # noise timesteps to probe
_PROBE_SEED_BASE = 999_999               # fixed noise seed for reproducibility
_PROBE_LATENTS: dict = {}                # cls → (1,4,128,128) unscaled CPU tensor (mean of 5)
_HPROBE_LATENTS: dict = {}               # cls → list[(1,4,128,128)] unscaled CPU tensors (up to 50)
_EVAL_PANEL_SEEDS = _COMPOSE_SEEDS              # 4 seeds → 4 panels per finding (12 total)
_EVAL_POE_W_CARDIO = 2.0
_EVAL_POE_W_EFFUSION = 1.0
_EVAL_DDIM_STEPS = 25            # monitor-grade sampling; final figures regen at full DDIM

# Lazily-built, CPU-resident instrument handles (PSPNet seg + presence classifier).
# Loaded once on first eval; moved to GPU only for the brief inference, then back —
# mirrors the _vae_decode_only CPU-offload pattern.
_EVAL_SEG = None            # (PSPNet, idx) or "FAILED"
_EVAL_CLASSIFIER = None     # (model, cardio_idx, effusion_idx, ckpt) or "FAILED"


def _eval_load_instruments():
    """Lazy-load PSPNet (+ presence classifier) once, CPU-resident. Returns
    (seg, classifier); either may be None if its load failed (eval degrades but
    never raises)."""
    global _EVAL_SEG, _EVAL_CLASSIFIER
    if _EVAL_SEG is None:
        try:
            import scripts.ctr_overlay as C
            _EVAL_SEG = C._load_seg("cpu")            # CPU-resident; .to(device) at eval
        except Exception as exc:
            _console.log(f"[yellow]eval: PSPNet load failed ({exc}) — dashboard disabled[/yellow]")
            _EVAL_SEG = "FAILED"
    if _EVAL_CLASSIFIER is None:
        try:
            from metrics import presence_classifier as P
            _EVAL_CLASSIFIER = P.load_model("cpu")
        except Exception as exc:
            _console.log(f"[dim]eval: presence classifier unavailable ({exc}) — presence/* = nan[/dim]")
            _EVAL_CLASSIFIER = "FAILED"
    seg = None if _EVAL_SEG == "FAILED" else _EVAL_SEG
    clf = None if _EVAL_CLASSIFIER == "FAILED" else _EVAL_CLASSIFIER
    return seg, clf


def _load_probe_latents(val_dir) -> None:
    """Load probe latents from val split into _PROBE_LATENTS and _HPROBE_LATENTS.

    _PROBE_LATENTS  : mean of up to 5 latents per class → (1,4,128,128) per class.
                      Used by denoising/PoE single-step probes.
    _HPROBE_LATENTS : list of up to 50 latents per class → list[(1,4,128,128)].
                      Used by h-space linear probe and latent arithmetic.
    """
    global _PROBE_LATENTS, _HPROBE_LATENTS
    val_path = Path(val_dir)
    if not val_path.is_dir():
        return
    N_PROBE, N_HPROBE = 5, 50
    by_class: dict = {0: [], 1: [], 2: []}
    for f in sorted(val_path.glob("*.pt")):
        if all(len(v) >= N_HPROBE for v in by_class.values()):
            break
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            l = int(d.get("label", -1))
            if l in by_class and len(by_class[l]) < N_HPROBE:
                by_class[l].append(d["z"].float().unsqueeze(0))   # (1,4,128,128)
        except Exception:
            continue
    # _PROBE_LATENTS: mean of up to N_PROBE films per class
    _PROBE_LATENTS = {}
    for k, zlist in by_class.items():
        if zlist:
            stack = torch.stack([z.squeeze(0) for z in zlist[:N_PROBE]])
            _PROBE_LATENTS[k] = stack.mean(0).unsqueeze(0)
    # _HPROBE_LATENTS: full list for h-space probe
    _HPROBE_LATENTS = {k: v for k, v in by_class.items() if v}
    _console.log(
        f"[dim]Probe buffers:[/dim] {len(_PROBE_LATENTS)} probe classes, "
        f"{sum(len(v) for v in _HPROBE_LATENTS.values())} hprobe latents "
        f"from {val_path}"
    )


def _decode_latent(vae, z_scaled: torch.Tensor, scale_factor: float) -> "np.ndarray":
    """Decode a (1,4,128,128) scaled latent → uint8 (512,512) grayscale."""
    import numpy as np
    with torch.no_grad():
        out = vae.decode(z_scaled / scale_factor)
        if hasattr(out, "sample"):
            out = out.sample
    img = out[0, 0].float().clamp(-1, 1).cpu()
    return ((img + 1) / 2 * 255).numpy().astype(np.uint8)


def _side_by_side_overlay(gray_a: "np.ndarray", gray_b: "np.ndarray",
                           label_a: str, label_b: str,
                           seg_model, idx, device: "torch.device") -> "np.ndarray":
    """PSPNet-overlay both images and return a (512,1024,3) side-by-side RGB panel."""
    import numpy as np
    import metrics.eval_checkpoint as EC

    def _overlay(g):
        m = EC.score_decoded_array(seg_model, idx, g, str(device), with_masks=True)
        masks = m.get("_masks") or {}
        import scripts.ctr_overlay as C
        o = EC.OverlayResult(
            stem=label_a, base=g, ctr=m["ctr"],
            heart_conf=m.get("heart_conf", 0.0),
            opacity=m.get("peri_cardiac_opacity", float("nan")),
            border_visible=bool(m.get("border_visible")),
            cp_left=m.get("cp_left", {}), cp_right=m.get("cp_right", {}),
            heart_line=m.get("heart_line"), thorax_line=m.get("thorax_line"),
            heart_mask=masks.get(C.HEART),
            llung_mask=masks.get(C.LLUNG),
            rlung_mask=masks.get(C.RLUNG),
        )
        return EC.render_burned(None, o), m["ctr"]

    burned_a, ctr_a = _overlay(gray_a)
    burned_b, ctr_b = _overlay(gray_b)
    sep = np.full((512, 4, 3), 80, dtype=np.uint8)   # thin grey separator
    return np.concatenate([burned_a, sep, burned_b], axis=1), ctr_a, ctr_b


def _log_denoise_probe(
    unet, vae, scheduler, scale_factor: float, device: "torch.device", step: int,
    seg
) -> None:
    """Single-step denoising accuracy probe at t ∈ _PROBE_T_LEVELS.

    For each (t, class) pair with a probe latent:
      z₀ (real)  →  z_t = add_noise(z₀, ε, t)  →  ε̂ = unet(z_t, t, class)
      →  ẑ₀ = (z_t − √(1−ᾱ_t) · ε̂) / √(ᾱ_t)  →  decode  →  PSPNet overlay.

    W&B (per t, per class):
      diag/denoise/ssim_t{T}_cls{C}   — SSIM(real, ẑ₀) in [0,1]
      diag/denoise/dctr_t{T}_cls{C}   — |CTR_real − CTR_ẑ₀|
    W&B (per t):
      diag/denoise/panel_t{T}         — (3 classes × 2) side-by-side PSPNet RGB panel
    """
    import numpy as np
    from skimage.metrics import structural_similarity as ssim_fn

    if not _PROBE_LATENTS:
        return

    seg_model, seg_idx = seg
    CLASS_DISPLAY = {0: "normal", 1: "cardio", 2: "effusion"}
    metrics: dict = {}
    ssim_cardio_by_t: dict = {}   # returned for SNR-vs-gap plot

    unet.eval()
    try:
        for t_val in _PROBE_T_LEVELS:
            t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)
            alpha_bar = scheduler.alphas_cumprod.to(device)[t_val]
            sqrt_alpha = alpha_bar.sqrt()
            sqrt_one_minus = (1.0 - alpha_bar).sqrt()

            row_imgs = []   # one (512,1028,3) per class for the panel
            for cls_idx in [0, 1, 2]:
                if cls_idx not in _PROBE_LATENTS:
                    row_imgs.append(None)
                    continue
                z0_raw = _PROBE_LATENTS[cls_idx].to(device)   # (1,4,128,128) unscaled
                z0 = z0_raw * scale_factor
                g = torch.Generator(device=device).manual_seed(
                    _PROBE_SEED_BASE + cls_idx * 1000 + t_val)
                noise = torch.randn_like(z0, generator=g)
                z_t = scheduler.add_noise(z0, noise, t_tensor)
                label = torch.tensor([cls_idx], device=device, dtype=torch.long)
                with torch.no_grad():
                    eps_pred = unet(z_t, t_tensor.expand(1), label)
                z0_pred = (z_t - sqrt_one_minus * eps_pred) / sqrt_alpha

                gray_real = _decode_latent(vae, z0, scale_factor)
                gray_pred = _decode_latent(vae, z0_pred, scale_factor)

                ssim_val = ssim_fn(gray_real.astype(np.float32) / 255,
                                   gray_pred.astype(np.float32) / 255,
                                   data_range=1.0)
                metrics[f"diag/denoise/ssim_t{t_val}_cls{cls_idx}"] = float(ssim_val)
                if cls_idx == 1:
                    ssim_cardio_by_t[t_val] = float(ssim_val)

                panel_row, ctr_real, ctr_pred = _side_by_side_overlay(
                    gray_real, gray_pred,
                    f"real_{CLASS_DISPLAY[cls_idx]}  CTR={ctr_real:.3f}" if False else "real",
                    f"ẑ₀@t={t_val}",
                    seg_model, seg_idx, device,
                )
                if np.isfinite(ctr_real) and np.isfinite(ctr_pred):
                    metrics[f"diag/denoise/dctr_t{t_val}_cls{cls_idx}"] = abs(ctr_real - ctr_pred)
                row_imgs.append(panel_row)

            # Stack valid rows into one panel
            valid = [r for r in row_imgs if r is not None]
            if valid:
                panel = np.concatenate(valid, axis=0)   # (N×512, 1028, 3)
                wandb.log({f"diag/denoise/panel_t{t_val}": wandb.Image(panel)}, step=step)

        wandb.log(metrics, step=step)
    finally:
        unet.train()
    return ssim_cardio_by_t


def _log_poe_compose_probe(
    unet, vae, scheduler, scale_factor: float, device: "torch.device", step: int,
    seg, null_token_idx: int
) -> None:
    """PoE composition probe: does cardio signature survive PoE denoising at each t?

    Takes the real cardio probe latent, noises to t, then applies one step of:
      a) cardio-only CFG  → ẑ₀_cardio
      b) PoE (cardio+effusion) → ẑ₀_poe

    Decodes all three (real, cardio-denoised, PoE-denoised) and logs PSPNet overlays.
    The key scalar: does CTR in ẑ₀_poe stay close to CTR in ẑ₀_cardio?
    If CTR collapses in PoE at high t → early trajectory failure (gross anatomy).
    If CTR collapses only at low t → late trajectory failure (fine rendering).

    W&B (per t):
      diag/poe_compose/cardio_ctr_t{T}    — CTR of cardio-only ẑ₀
      diag/poe_compose/poe_cardio_ctr_t{T} — CTR of PoE ẑ₀ (key: does it hold?)
      diag/poe_compose/panel_t{T}          — 3-panel PSPNet overlay (real|cardio|PoE)
    """
    import numpy as np

    if 1 not in _PROBE_LATENTS:
        return {}   # need real cardio latent

    seg_model, seg_idx = seg
    w_c, w_e = _EVAL_POE_W_CARDIO, _EVAL_POE_W_EFFUSION
    metrics: dict = {}
    ssim_poe_by_t: dict = {}   # returned for SNR-vs-gap plot

    unet.eval()
    try:
        z0_raw = _PROBE_LATENTS[1].to(device)   # (1,4,128,128) real cardio, unscaled
        z0 = z0_raw * scale_factor
        gray_real = _decode_latent(vae, z0, scale_factor)

        for t_val in _PROBE_T_LEVELS:
            t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)
            alpha_bar = scheduler.alphas_cumprod.to(device)[t_val]
            sqrt_alpha = alpha_bar.sqrt()
            sqrt_one_minus = (1.0 - alpha_bar).sqrt()

            g = torch.Generator(device=device).manual_seed(_PROBE_SEED_BASE + 10_000 + t_val)
            noise = torch.randn_like(z0, generator=g)
            z_t = scheduler.add_noise(z0, noise, t_tensor)

            lbl_null = torch.tensor([null_token_idx], device=device, dtype=torch.long)
            lbl_c = torch.tensor([1], device=device, dtype=torch.long)
            lbl_e = torch.tensor([2], device=device, dtype=torch.long)
            with torch.no_grad():
                eps_null = unet(z_t, t_tensor, lbl_null)
                eps_c = unet(z_t, t_tensor, lbl_c)
                eps_e = unet(z_t, t_tensor, lbl_e)

            eps_cardio_cond = eps_null + 2.0 * (eps_c - eps_null)     # w_c=2.0
            eps_poe = eps_null + w_c * (eps_c - eps_null) + w_e * (eps_e - eps_null)

            z0_pred_c = (z_t - sqrt_one_minus * eps_cardio_cond) / sqrt_alpha
            z0_pred_poe = (z_t - sqrt_one_minus * eps_poe) / sqrt_alpha

            gray_c = _decode_latent(vae, z0_pred_c, scale_factor)
            gray_poe = _decode_latent(vae, z0_pred_poe, scale_factor)

            # PoE SSIM vs real cardio (for SNR-vs-gap plot)
            from skimage.metrics import structural_similarity as ssim_fn
            ssim_poe_val = ssim_fn(gray_real.astype(np.float32) / 255,
                                   gray_poe.astype(np.float32) / 255,
                                   data_range=1.0)
            metrics[f"diag/denoise/ssim_t{t_val}_poe"] = float(ssim_poe_val)
            ssim_poe_by_t[t_val] = float(ssim_poe_val)

            # PSPNet score all three
            def _score_and_overlay(g, label):
                import metrics.eval_checkpoint as EC
                import scripts.ctr_overlay as C
                m = EC.score_decoded_array(seg_model, seg_idx, g, str(device), with_masks=True)
                masks = m.get("_masks") or {}
                o = EC.OverlayResult(
                    stem=label, base=g, ctr=m["ctr"],
                    heart_conf=m.get("heart_conf", 0.0),
                    opacity=m.get("peri_cardiac_opacity", float("nan")),
                    border_visible=bool(m.get("border_visible")),
                    cp_left=m.get("cp_left", {}), cp_right=m.get("cp_right", {}),
                    heart_line=m.get("heart_line"), thorax_line=m.get("thorax_line"),
                    heart_mask=masks.get(C.HEART),
                    llung_mask=masks.get(C.LLUNG),
                    rlung_mask=masks.get(C.RLUNG),
                )
                return EC.render_burned(None, o), m["ctr"]

            burned_real, ctr_real = _score_and_overlay(gray_real, "real_cardio")
            burned_c, ctr_c = _score_and_overlay(gray_c, f"cardio@t={t_val}")
            burned_poe, ctr_poe = _score_and_overlay(gray_poe, f"PoE@t={t_val}")

            if np.isfinite(ctr_c):
                metrics[f"diag/poe_compose/cardio_ctr_t{t_val}"] = ctr_c
            if np.isfinite(ctr_poe):
                metrics[f"diag/poe_compose/poe_cardio_ctr_t{t_val}"] = ctr_poe

            sep = np.full((512, 4, 3), 80, dtype=np.uint8)
            panel = np.concatenate([burned_real, sep, burned_c, sep, burned_poe], axis=1)
            wandb.log({f"diag/poe_compose/panel_t{t_val}": wandb.Image(panel)}, step=step)

        wandb.log(metrics, step=step)
    finally:
        unet.train()
    return ssim_poe_by_t


def _log_hspace_probe(
    unet, scheduler, scale_factor: float, device: "torch.device", step: int
) -> None:
    """Linear probe on UNet bottleneck (middle_block) features to track class separation.

    Runs 150 UNet forward passes (50 val latents × 3 classes) at t=500, hooks the
    middle_block output (B,128,16,16) → 32,768-dim feature per sample, then fits a
    LogisticRegression to predict class label.

    W&B (every 25 000 steps — expensive; ~150 forward passes):
      eval/hspace_probe_acc          — 3-class accuracy (normal/cardio/effusion)
      eval/hspace_probe_acc_cardio   — binary cardio-vs-rest accuracy
      eval/hspace_probe_acc_effusion — binary effusion-vs-rest accuracy

    Interpretation: if binary cardio accuracy stays low even as CTR improves, the UNet
    is generating plausible CTR values via a non-class-conditional path — a flag for
    the disentanglement framing. If high from step 25k onward: UNet has cleanly
    separated cardio in h-space, and composition failure lives in the guidance math.
    """
    if not _HPROBE_LATENTS:
        return

    T_PROBE = 500
    t_tensor = torch.tensor([T_PROBE], device=device, dtype=torch.long)

    feats, ys = [], []
    buf: dict = {}
    handle = unet.unet.middle_block.register_forward_hook(
        lambda m, inp, out: buf.__setitem__("h", out.detach().float().cpu())
    )
    unet.eval()
    try:
        for cls_idx, zlist in _HPROBE_LATENTS.items():
            for z_raw in zlist:
                z0 = z_raw.to(device) * scale_factor
                seed = _PROBE_SEED_BASE + cls_idx * 1000 + len(feats)
                g = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn(1, 4, 128, 128, device=device, generator=g)
                z_t = scheduler.add_noise(z0, noise, t_tensor)
                label = torch.tensor([cls_idx], device=device, dtype=torch.long)
                with torch.no_grad():
                    unet(z_t, t_tensor, label)
                if "h" in buf:
                    feats.append(buf.pop("h").numpy().reshape(-1))
                    ys.append(cls_idx)
    finally:
        handle.remove()
        unet.train()

    if len(feats) < 10:
        return

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        X, y = np.array(feats), np.array(ys)
        clf = LogisticRegression(C=0.1, max_iter=200, random_state=42)
        clf.fit(X, y)
        metrics: dict = {"eval/hspace_probe_acc": float(clf.score(X, y))}
        for ci, cname in [(1, "cardio"), (2, "effusion")]:
            yb = (y == ci).astype(int)
            cb = LogisticRegression(C=0.1, max_iter=200, random_state=42)
            cb.fit(X, yb)
            metrics[f"eval/hspace_probe_acc_{cname}"] = float(cb.score(X, yb))
        wandb.log(metrics, step=step)
        _console.log(
            f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
            f"  h-space probe: "
            f"acc={metrics['eval/hspace_probe_acc']:.3f}  "
            f"cardio={metrics.get('eval/hspace_probe_acc_cardio', float('nan')):.3f}  "
            f"effusion={metrics.get('eval/hspace_probe_acc_effusion', float('nan')):.3f}"
        )
    except Exception as exc:
        _console.log(f"[dim]h-space probe fitting failed: {exc}[/dim]")


def _log_latent_arith(step: int) -> None:
    """Latent arithmetic symmetry check (every grid_every steps).

    Tests whether z_c + z_e − z_n is equidistant from the cardio and effusion
    latent centroids.  Uses frozen VAE val latents so the values are constant
    during training — the flat line confirms the VAE geometry is stable, and the
    asymmetry (cos_c vs cos_e) at step 0 tells you which direction PoE's additive
    assumption is biased.

    W&B:
      eval/latent_arith_cos_c  — cos(z_c + z_e − z_n,  z_cardio_centroid)
      eval/latent_arith_cos_e  — cos(z_c + z_e − z_n,  z_effusion_centroid)

    Interpretation: symmetric (cos_c ≈ cos_e) → latent geometry supports PoE.
    cos_e >> cos_c → arithmetic vector is pulled toward effusion space —
    cardio is geometrically subordinate and PoE will over-generate effusion.
    """
    if not _HPROBE_LATENTS or not all(k in _HPROBE_LATENTS for k in [0, 1, 2]):
        return

    def _centroid(cls_idx: int) -> "torch.Tensor":
        zlist = _HPROBE_LATENTS[cls_idx]
        return torch.stack([z.squeeze(0) for z in zlist]).mean(0).flatten()

    z_n = _centroid(0)
    z_c = _centroid(1)
    z_e = _centroid(2)
    z_arith = z_c + z_e - z_n

    cos_c = float(torch.nn.functional.cosine_similarity(
        z_arith.unsqueeze(0), z_c.unsqueeze(0)
    ).item())
    cos_e = float(torch.nn.functional.cosine_similarity(
        z_arith.unsqueeze(0), z_e.unsqueeze(0)
    ).item())
    wandb.log({"eval/latent_arith_cos_c": cos_c, "eval/latent_arith_cos_e": cos_e},
              step=step)


def _log_snr_gap_plot(
    ssim_cardio: dict, ssim_poe: dict, scheduler, step: int
) -> None:
    """Plot composability gap (SSIM_cardio − SSIM_poe) vs log-SNR(t) at compose_every.

    Both SSIM dicts are keyed by t_val and measure single-step denoising accuracy on
    the same real cardio probe latent: ssim_cardio uses cardio-only guidance;
    ssim_poe uses PoE guidance.  The gap = how much effusion conditioning hurts cardio
    reconstruction, as a function of noise level.

    log-SNR(t) = log(ᾱ_t / (1−ᾱ_t)).  Low SNR (left) = early denoising = global
    structure.  High SNR (right) = late denoising = fine texture / borders.

    W&B (artifact + scalar per t):
      diag/snr_gap_plot                   — matplotlib figure
      diag/snr_gap_t{T}                   — scalar gap at each t (for custom W&B expressions)
    """
    import numpy as np

    ts = sorted(t for t in _PROBE_T_LEVELS if t in ssim_cardio and t in ssim_poe)
    if len(ts) < 2:
        return

    alphas = scheduler.alphas_cumprod.cpu().float().numpy()
    log_snrs, gaps = [], []
    gap_metrics: dict = {}
    for t in ts:
        ab = float(np.clip(alphas[t], 1e-6, 1 - 1e-6))
        log_snrs.append(float(np.log(ab / (1.0 - ab))))
        gap = float(ssim_cardio[t]) - float(ssim_poe[t])
        gaps.append(gap)
        gap_metrics[f"diag/snr_gap_t{t}"] = gap
    wandb.log(gap_metrics, step=step)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(log_snrs, gaps, "o-", color="steelblue", linewidth=2, markersize=6)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.fill_between(log_snrs, gaps, 0,
                        where=[g > 0 for g in gaps], alpha=0.15, color="steelblue")
        ax.set_xlabel("log-SNR(t)  [← low SNR / high t = global structure]")
        ax.set_ylabel("SSIM gap  (cardio-only − PoE)")
        ax.set_title(f"Composability gap vs noise level   step={step}")
        for lsnr, gap, t in zip(log_snrs, gaps, ts):
            ax.annotate(f"t={t}", (lsnr, gap),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.tight_layout()
        wandb.log({"diag/snr_gap_plot": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as exc:
        _console.log(f"[dim]SNR-gap plot failed: {exc}[/dim]")


def _eval_gen_decode(unet, vae, ddim_scheduler, null_token_idx, device,
                     mode: str, n: int, base_seed: int, scale_factor: float = 1.0):
    """Generate ``n`` latents (single-disease or PoE), decode, and return a list of
    decoded (512,512) float images in [0,1]. Uses fixed per-image seeds so the
    dashboard batch is frozen across steps. CUDA only briefly; mirrors the grid hooks.

    z0 is in the SCALED latent space (native × scale_factor); it MUST be divided by
    scale_factor before decode (matches generate._decode_latents). Omitting this was a
    bug that was dormant for v1 (sf≈1) but 5× over-scales v2 μ generations (sf≈5.03).
    """
    from src.inference.cfg import cfg_single, cfg_compose
    decoded_imgs: list = []
    for i in range(n):
        g = torch.Generator(device=device).manual_seed(base_seed + i)
        noise = torch.randn(1, 4, 128, 128, generator=g, device=device)
        with torch.no_grad():
            if mode == "normal":
                z0 = cfg_single(unet, noise, 0, 1.0, ddim_scheduler,
                                steps=_EVAL_DDIM_STEPS, anchor="null",
                                null_token_idx=null_token_idx)
            elif mode == "cardio":
                z0 = cfg_single(unet, noise, 1, 1.0, ddim_scheduler,
                                steps=_EVAL_DDIM_STEPS, anchor="null",
                                null_token_idx=null_token_idx)
            elif mode == "effusion":
                z0 = cfg_single(unet, noise, 2, 1.0, ddim_scheduler,
                                steps=_EVAL_DDIM_STEPS, anchor="null",
                                null_token_idx=null_token_idx)
            else:  # poe
                z0 = cfg_compose(unet, noise, _EVAL_POE_W_CARDIO, ddim_scheduler,
                                 steps=_EVAL_DDIM_STEPS, anchor="null",
                                 null_token_idx=null_token_idx,
                                 w_effusion=_EVAL_POE_W_EFFUSION)
            decoded = vae.decode(z0 / scale_factor)
            if hasattr(decoded, "sample"):
                decoded = decoded.sample
            img = decoded[0, 0].float().clamp(-1, 1).cpu()
            decoded_imgs.append((img + 1) / 2)            # → [0,1]
    return decoded_imgs


def _run_quality_dashboard(unet, vae, ddim_scheduler, null_token_idx, device, step,
                           panel: bool, scale_factor: float = 1.0):
    """Generate the frozen eval batch, score it via metrics.eval_checkpoint, and log
    namespaced scalars (+ overlay panels when ``panel``). Fully self-contained and
    GUARDED by its caller; raises into the caller only so the caller can stamp
    eval/instrument_ok=0.
    """
    import metrics.eval_checkpoint as EC

    seg, classifier = _eval_load_instruments()
    if seg is None:
        wandb.log({"eval/instrument_ok": 0}, step=step)
        return

    seg_model, idx = seg
    seg_model.to(device)
    if classifier is not None:
        classifier[0].to(device)
    try:
        # 32 / finding for the scalar dashboard (including normal as negative class for AUC)
        imgs, labels = [], []
        for mode, lab, base in (
            ("normal", EC.LABEL_NORMAL, _EVAL_BASE_SEED_NORMAL),
            ("cardio", EC.LABEL_CARDIO, _EVAL_BASE_SEED_CARDIO),
            ("effusion", EC.LABEL_EFFUSION, _EVAL_BASE_SEED_EFFUSION),
            ("poe", EC.LABEL_POE, _EVAL_BASE_SEED_POE),
        ):
            batch = _eval_gen_decode(unet, vae, ddim_scheduler, null_token_idx,
                                     device, mode, _EVAL_N_PER_FINDING, base,
                                     scale_factor=scale_factor)
            imgs.extend(batch)
            labels.extend([lab] * len(batch))

        metrics, overlays = EC.evaluate_checkpoint(imgs, labels, (seg_model, idx),
                                                   classifier=classifier)
        metrics["ref/cardio_real"] = EC.CARDIO_REAL_CTR
        metrics["ref/threshold"] = EC.CTR_THRESHOLD
        wandb.log(metrics, step=step)

        if panel:
            _log_quality_panels(EC, unet, vae, ddim_scheduler, null_token_idx,
                                device, step, (seg_model, idx), classifier,
                                scale_factor=scale_factor)
    finally:
        seg_model.to("cpu")
        if classifier is not None:
            classifier[0].to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _log_quality_panels(EC, unet, vae, ddim_scheduler, null_token_idx, device, step,
                        seg, classifier, scale_factor: float = 1.0):
    """Render burned + layered overlay panels for the fixed 12 (3 findings × 4 seeds)."""
    seg_model, idx = seg
    for mode, key in (("normal", "normal"), ("cardio", "cardio"), ("effusion", "effusion"), ("poe", "poe")):
        # reuse the panel seeds (4 per finding); seeds chosen to overlap _COMPOSE_SEEDS
        imgs, labels = [], []
        for s in _EVAL_PANEL_SEEDS:
            g = torch.Generator(device=device).manual_seed(s)
            noise = torch.randn(1, 4, 128, 128, generator=g, device=device)
            from src.inference.cfg import cfg_single, cfg_compose
            with torch.no_grad():
                if mode == "cardio":
                    z0 = cfg_single(unet, noise, 1, 1.0, ddim_scheduler,
                                    steps=_EVAL_DDIM_STEPS, anchor="null",
                                    null_token_idx=null_token_idx)
                elif mode == "effusion":
                    z0 = cfg_single(unet, noise, 2, 1.0, ddim_scheduler,
                                    steps=_EVAL_DDIM_STEPS, anchor="null",
                                    null_token_idx=null_token_idx)
                else:
                    z0 = cfg_compose(unet, noise, _EVAL_POE_W_CARDIO, ddim_scheduler,
                                     steps=_EVAL_DDIM_STEPS, anchor="null",
                                     null_token_idx=null_token_idx,
                                     w_effusion=_EVAL_POE_W_EFFUSION)
                decoded = vae.decode(z0 / scale_factor)
                if hasattr(decoded, "sample"):
                    decoded = decoded.sample
                img = (decoded[0, 0].float().clamp(-1, 1).cpu() + 1) / 2
            imgs.append(img)
            labels.append({"normal": EC.LABEL_NORMAL, "cardio": EC.LABEL_CARDIO,
                           "effusion": EC.LABEL_EFFUSION, "poe": EC.LABEL_POE}[mode])
        _, overlays = EC.evaluate_checkpoint(imgs, labels, (seg_model, idx),
                                             classifier=None)
        burned = [wandb.Image(EC.render_burned(None, o), caption=o.stem) for o in overlays]
        layered = [EC.render_layered(None, o) for o in overlays]
        wandb.log({f"panel/{key}_burned": burned,
                   f"panel/{key}_layered": layered}, step=step)
        # OPEN GRAFT (deferred): live CTR×CP scatter with shaded Gate-3 pass region.
        # gate3_yield._scatter already draws exactly this for the offline path; slot a
        # per-step matplotlib scatter of (ctr, cp_max_angle) for the PoE overlays here
        # (axvspan CTR>0.5, axhspan CP>80°) and log it as panel/poe_scatter. Not built.


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
    ema_suffix: bool = False,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ema_suffix: write ONLY the EMA weights file (model_ema_step*.safetensors) —
    # this is the smooth, inference/deployment checkpoint. The raw-weights call
    # (ema_suffix=False) writes the config + training_state + W&B artifact needed
    # for exact resume, so the EMA call skips all of that.
    if ema_suffix:
        ema_path = ckpt_dir / f"model_ema_step{step:07d}.safetensors"
        state = {
            **unet.state_dict(),
            **{f"class_embed.{k}": v for k, v in class_embed.state_dict().items()},
        }
        save_file(state, ema_path)
        return

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


_CFG_ABLATION_WEIGHTS = [0.0, 1.0, 2.0]
_CFG_ABLATION_STEPS = 20   # fast diagnostic — not for publication quality
_CFG_ABLATION_ROW_LABELS = {
    0.0: "w=0  unconditional",
    1.0: "w=1  standard CFG",
    2.0: "w=2  amplified",
}


def _log_cfg_ablation(
    unet: torch.nn.Module,
    monitor: MonitorBatch,
    ddim_scheduler: DDIMScheduler,
    vae,
    step: int,
    null_token_idx: int,
    device: torch.device,
    scale_factor: float = 1.0,
) -> None:
    """3×3 CFG weight ablation grid logged under 'recon/cfg_ablation'.

    Rows = w∈{0,1,2}  (unconditional / standard / amplified).
    Cols = no_finding / cardiomegaly / effusion.
    Seed = monitor.seeds[0] (42). DDIM steps = 20 (diagnostic speed).
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    CLASS_INDICES = [0, 1, 2]
    CLASS_DISPLAY = ["NO FINDING", "CARDIOMEGALY", "EFFUSION"]

    ddim_scheduler.set_timesteps(_CFG_ABLATION_STEPS)
    was_training = unet.training
    unet.eval()
    label_null = torch.tensor([null_token_idx], device=device)

    cells: list[list] = []
    with torch.no_grad():
        for w in _CFG_ABLATION_WEIGHTS:
            row: list = []
            for cls_i, cls_idx in enumerate(CLASS_INDICES):
                z = monitor.noise_for(0, cls_i).clone()  # seed 42, same slot as recon grid
                label_cond = torch.tensor([cls_idx], device=device)
                for t in ddim_scheduler.timesteps:
                    t_batch = torch.tensor([t], device=device)
                    eps_cond = unet(z, t_batch, label_cond)
                    if w != 1.0:
                        eps_uncond = unet(z, t_batch, label_null)
                        eps = eps_uncond + w * (eps_cond - eps_uncond)
                    else:
                        eps = eps_cond
                    z = ddim_scheduler.step(eps, t, z).prev_sample
                decoded = vae.decode(z / scale_factor)
                if hasattr(decoded, "sample"):
                    decoded = decoded.sample
                img_np = decoded[0, 0].float().clamp(-1, 1).cpu().numpy()
                img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                row.append(img_np)
            cells.append(row)

    if was_training:
        unet.train()

    # Build labeled 3×3 grid
    H_px, W_px = cells[0][0].shape
    LABEL_W = 160
    HEADER_H = 30
    SEP = 2

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    canvas_w = LABEL_W + W_px * 3
    canvas_h = HEADER_H + (H_px + SEP) * 3
    canvas = Image.new("L", (canvas_w, canvas_h), color=30)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([(0, 0), (canvas_w, HEADER_H - 1)], fill=20)
    for ci, header in enumerate(CLASS_DISPLAY):
        x = LABEL_W + ci * W_px + W_px // 2 - len(header) * 4
        draw.text((x, 7), header, fill=215, font=font)

    for wi, (w_val, row_imgs) in enumerate(zip(_CFG_ABLATION_WEIGHTS, cells)):
        y_top = HEADER_H + wi * (H_px + SEP)
        draw.rectangle([(0, y_top), (LABEL_W - 1, y_top + H_px - 1)], fill=20)
        draw.text((4, y_top + H_px // 2 - 8), _CFG_ABLATION_ROW_LABELS[w_val], fill=180, font=font)
        for ci, img_np in enumerate(row_imgs):
            canvas.paste(Image.fromarray(img_np, "L"), (LABEL_W + ci * W_px, y_top))
        if wi < 2:
            draw.rectangle([(0, y_top + H_px), (canvas_w, y_top + H_px + SEP - 1)], fill=60)

    caption = (
        f"CFG ablation | seed={monitor.seeds[0]} | "
        f"{_CFG_ABLATION_STEPS} DDIM steps | step={step}"
    )
    wandb.log({"recon/cfg_ablation": wandb.Image(canvas, caption=caption)}, step=step)


def _log_recon_grid(
    unet: torch.nn.Module,
    monitor: MonitorBatch,
    ddim_scheduler: DDIMScheduler,
    vae,
    step: int,
    cfg_w: float,
    null_token_idx: int,
    scale_factor: float = 1.0,
) -> None:
    """Decode the 4×3 monitor grid; log labeled overview + per-class panels + Artifact."""
    from PIL import Image as _PILImage

    grid_pil, cells = monitor.decode_grid(
        unet=unet,
        ddim_scheduler=ddim_scheduler,
        vae=vae,
        cfg_weight=cfg_w,
        null_token_idx=null_token_idx,
        scale_factor=scale_factor,
    )

    # Per-class panels: 4 seeds each — supervisor can pin 3 panels side-by-side
    log_dict: dict = {}
    for cls_i, cls_name in enumerate(["no_finding", "cardiomegaly", "effusion"]):
        log_dict[f"recon/{cls_name}"] = [
            wandb.Image(
                _PILImage.fromarray(cells[seed_i][cls_i], "L"),
                caption=f"seed={monitor.seeds[seed_i]} | w={cfg_w} | step={step}",
            )
            for seed_i in range(len(monitor.seeds))
        ]
    log_dict["recon/grid"] = wandb.Image(grid_pil, caption=f"step={step} | w={cfg_w}")
    wandb.log(log_dict, step=step)

    # artifact: labeled overview grid
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
    scale_factor: float = 1.0,
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

                decoded = vae.decode(z0 / scale_factor)
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
                   help="W&B run ID to resume. Loads the latest local checkpoint from ckpt_dir "
                        "(model weights + optimizer + LR scheduler) and, by default, starts a "
                        "FRESH W&B run logging from the checkpoint step (lineage recorded in "
                        "config) so logging stays monotonic.")
    p.add_argument("--wandb-continue", action="store_true",
                   help="With --resume, re-enter the SAME W&B run instead of starting a fresh "
                        "one. Only safe when the run did not log past the checkpoint step (else "
                        "W&B drops the re-logged steps as non-monotonic). Default is a fresh run.")
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
    # TF32 — free ~10-20% on matmuls/convs; orthogonal to bf16 autocast, no effect on
    # training dynamics (the params/grads stay bf16/fp32; this only speeds fp32 matmul paths).
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = cfg.get("bf16", False) and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    # Heavy-eval cadence (recon grid + CFG ablation + CTR/yield dashboard). Default 5 000
    # so the 192-img × DDIM sampling never dominates wall-time. MUST be a multiple of 1 000
    # (the val-loss cadence it is nested under). val/loss itself stays at every 1 000.
    grid_every = cfg.get("grid_every", 5000)
    # Compose grid (both anchors × weights) + overlay-panel dashboard — the heaviest hook.
    # Independent top-level guard, so any value works; default 15 000 (3× grid_every).
    compose_every = cfg.get("compose_every", 15000)

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
    loss_at_step_10k: float | None = None   # reference EMA captured at step 10k (kill guard)
    loss_ema: float | None = None           # running EMA of train loss (smooths per-step noise)
    divergence_streak = 0                   # consecutive steps with EMA above 2× reference
    _LOSS_EMA_BETA = 0.99                   # ~100-step smoothing window
    _DIVERGENCE_PATIENCE = 200              # sustained breach length before we call it divergence

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

    # --- Recipe: Weight-EMA + min-SNR (init AFTER resume so EMA seeds from the
    #     correct starting weights, fresh or resumed) ----------------------------
    use_ema = bool(cfg.get("use_ema", False))
    ema = _WeightEMA(unet, decay=float(cfg.get("ema_decay", 0.9999))) if use_ema else None
    use_min_snr = bool(cfg.get("use_min_snr", False))
    min_snr_gamma = float(cfg.get("min_snr_gamma", 5.0))
    if use_ema or use_min_snr:
        _console.log(
            f"[dim]recipe:[/dim] "
            f"EMA={'decay '+str(cfg.get('ema_decay', 0.9999)) if use_ema else 'off'}  |  "
            f"min-SNR={'γ='+str(min_snr_gamma) if use_min_snr else 'off'}"
        )

    # --- Banner (after resume detection so step is known) ---------------------
    _print_banner(cfg, run_name, device, max_steps, resume_step=step)

    # --- W&B init -------------------------------------------------------------
    # On --resume we load the LOCAL checkpoint (step N) but the source W&B run may
    # have logged PAST N before dying (e.g. killed at 10 295 with last ckpt at
    # 10 000). Re-entering that run would re-log 10 001–10 295 and W&B drops them
    # as non-monotonic. Forking would be ideal but is gated behind a private
    # preview on this account, so by default we start a FRESH run that logs from
    # step N onward (monotonic, no drops) and records its lineage in the config.
    wandb_config = {k: v for k, v in cfg.items() if k != "wandb"}
    wandb_kwargs: dict = {
        "project": cfg["wandb"]["project"],
        "name": run_name,
        "config": wandb_config,
    }
    if args.resume:
        if args.wandb_continue:
            wandb_kwargs["resume"] = "allow"
            wandb_kwargs["id"] = args.resume
        else:
            wandb_kwargs["name"] = f"{run_name}_cont{step}"
            wandb_config["resumed_from_run"] = args.resume
            wandb_config["resumed_from_step"] = step
    wandb.init(**wandb_kwargs)
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
        # Load probe latents for trajectory diagnostics (silently no-ops if val absent)
        _load_probe_latents(Path(latent_cache) / "val")
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg.get("batch_size", 4),
            sampler=train_ds.make_sampler(
                effusion_weight=cfg.get("effusion_weight", 2.0),
                balance_classes=cfg.get("balance_classes", False),
                class_mult=cfg.get("class_mult"),
            ),
            num_workers=cfg.get("num_workers", 8), pin_memory=True, drop_last=True,
            persistent_workers=True, prefetch_factor=4,
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
            sampler=train_ds.make_sampler(
                effusion_weight=cfg.get("effusion_weight", 2.0),
                balance_classes=cfg.get("balance_classes", False),
                class_mult=cfg.get("class_mult"),
            ),
            num_workers=cfg.get("num_workers", 8), pin_memory=True, drop_last=True,
            persistent_workers=True, prefetch_factor=4,
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
                    if use_min_snr:
                        # per-sample MSE × min-SNR-γ weight (Hang et al. 2023)
                        per_sample = F.mse_loss(
                            eps_pred.float(), noise.float(), reduction="none"
                        ).mean(dim=[1, 2, 3])
                        w = _min_snr_weight(scheduler.alphas_cumprod, t, min_snr_gamma)
                        loss = (w * per_sample).mean() / grad_accum
                    else:
                        loss = F.mse_loss(eps_pred.float(), noise.float()) / grad_accum

                loss.backward()
                accum_loss += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(unet, step)
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
                    (x, pre_dropout_labels), scale_factor, device, step,
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

                # recon grid + CFG ablation + quality dashboard every grid_every steps
                # (default 5 000; nested under the 1 000-step val guard, so grid_every must
                # be a multiple of 1 000). The heavy 192-img × DDIM sampling lives here.
                if vae_model is not None and step % grid_every == 0:
                    _console.log(
                        f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                        f"  [dim]recon grid  4×3 = 12 samples, 50 DDIM steps"
                        f"  +  CFG ablation  3×3, {_CFG_ABLATION_STEPS} steps[/dim]"
                    )
                    t0 = time.perf_counter()
                    if ema is not None:
                        ema.store_and_copy_to(unet)   # eval on EMA (deployment) weights
                    if _vae_decode_only:
                        vae_model.to(device)
                    _log_recon_grid(unet, monitor, ddim_scheduler, vae_model, step, args.cfg_w, null_token_idx, scale_factor=scale_factor)
                    _log_cfg_ablation(unet, monitor, ddim_scheduler, vae_model, step, null_token_idx, device, scale_factor=scale_factor)

                    # generation-quality dashboard (CTR/CP/yield scalars). GUARDED:
                    # the instrument must NEVER raise into the train loop — on any
                    # failure stamp eval/instrument_ok=0, warn, and continue.
                    try:
                        _run_quality_dashboard(
                            unet, vae_model, ddim_scheduler, null_token_idx, device,
                            step, panel=False, scale_factor=scale_factor,
                        )
                    except Exception as exc:
                        wandb.log({"eval/instrument_ok": 0}, step=step)
                        _console.log(
                            f"[yellow]{_now()}  step {step}  quality dashboard FAILED "
                            f"({type(exc).__name__}: {exc}) — eval/instrument_ok=0, continuing[/yellow]"
                        )

                    # latent arithmetic symmetry — cheap (no model forward passes)
                    try:
                        _log_latent_arith(step)
                    except Exception as exc:
                        _console.log(f"[dim]latent arith probe failed: {exc}[/dim]")

                    # h-space linear probe — 150 UNet forward passes; every 25k steps only
                    if step % 25_000 == 0:
                        try:
                            _log_hspace_probe(
                                unet, scheduler, scale_factor, device, step
                            )
                        except Exception as exc:
                            _console.log(f"[dim]h-space probe failed: {exc}[/dim]")

                    if _vae_decode_only:
                        vae_model.to("cpu")
                        torch.cuda.empty_cache()
                    if ema is not None:
                        ema.restore(unet)             # back to raw training weights
                    _console.log(
                        f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                        f"  [dim]recon → W&B  [{time.perf_counter()-t0:.1f}s][/dim]"
                    )

            # compose grid + overlay-panel dashboard every compose_every steps (default 15 000)
            if step % compose_every == 0 and vae_model is not None:
                t0 = time.perf_counter()
                if ema is not None:
                    ema.store_and_copy_to(unet)   # compose/probes on EMA weights
                if _vae_decode_only:
                    vae_model.to(device)
                _log_compose_grid(unet, ddim_scheduler, vae_model, step, device, null_token_idx, scale_factor=scale_factor)

                # burned + layered overlay panels for the fixed 12. GUARDED — same
                # contract as the scalar dashboard: never raise into the train loop.
                try:
                    _run_quality_dashboard(
                        unet, vae_model, ddim_scheduler, null_token_idx, device,
                        step, panel=True, scale_factor=scale_factor,
                    )
                except Exception as exc:
                    wandb.log({"eval/instrument_ok": 0}, step=step)
                    _console.log(
                        f"[yellow]{_now()}  step {step}  quality panels FAILED "
                        f"({type(exc).__name__}: {exc}) — eval/instrument_ok=0, continuing[/yellow]"
                    )

                # trajectory diagnostic probes (denoise accuracy + PoE composition)
                if _PROBE_LATENTS and vae_model is not None:
                    seg_for_probe, _ = _eval_load_instruments()
                    if seg_for_probe is not None:
                        seg_m, seg_i = seg_for_probe
                        seg_m.to(device)
                        try:
                            ssim_c = _log_denoise_probe(
                                unet, vae_model, scheduler, scale_factor,
                                device, step, (seg_m, seg_i),
                            )
                            ssim_poe = _log_poe_compose_probe(
                                unet, vae_model, scheduler, scale_factor,
                                device, step, (seg_m, seg_i), null_token_idx,
                            )
                            if ssim_c and ssim_poe:
                                _log_snr_gap_plot(ssim_c, ssim_poe, scheduler, step)
                        except Exception as exc:
                            _console.log(
                                f"[yellow]{_now()}  step {step}  trajectory probe FAILED "
                                f"({type(exc).__name__}: {exc}) — continuing[/yellow]"
                            )
                        finally:
                            seg_m.to("cpu")

                if _vae_decode_only:
                    vae_model.to("cpu")
                    torch.cuda.empty_cache()
                if ema is not None:
                    ema.restore(unet)             # back to raw training weights
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
                # EMA weights checkpoint (inference/deployment) — swap in, save, restore
                if ema is not None:
                    ema.store_and_copy_to(unet)
                    try:
                        _save_checkpoint(
                            unet.unet, unet.class_embed, cfg, ckpt_dir, step,
                            config_path, ema_suffix=True,
                        )
                    finally:
                        ema.restore(unet)
                _console.log(
                    f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                    f"  [bold yellow]CKPT[/bold yellow]"
                    f"  [dim]→ {ckpt_dir}/model_step{step:07d}.safetensors"
                    f"{'  (+ model_ema_step*)' if ema is not None else ''}[/dim]"
                )

            # --- divergence guard (EMA-smoothed + persistent) -----------------
            # Per-step DDPM loss is intrinsically noisy — each batch denoises a
            # randomly sampled timestep — so a single-step spike is NOT
            # divergence. Smooth with an EMA, anchor the reference to the EMA at
            # step 10k, and only kill when the smoothed loss stays above 2× that
            # reference for _DIVERGENCE_PATIENCE consecutive steps.
            loss_ema = (
                accum_loss if loss_ema is None
                else _LOSS_EMA_BETA * loss_ema + (1.0 - _LOSS_EMA_BETA) * accum_loss
            )
            # Arm at the first warmed step at/after 10k — works for a fresh run
            # (arms exactly at 10 000) and for a --resume from the 10k checkpoint
            # (whose state predates reference capture, so it arms just after).
            if loss_at_step_10k is None and step >= 10_000 and loss_ema is not None:
                loss_at_step_10k = loss_ema
                _console.log(
                    f"[dim]{_now()}[/dim]  step [bold]{step:>6}[/bold]"
                    f"  [dim]kill-criteria reference EMA set: {loss_ema:.4f}[/dim]"
                )
            if loss_at_step_10k is not None and step > 10_000:
                divergence_streak = divergence_streak + 1 if loss_ema > 2.0 * loss_at_step_10k else 0
                if divergence_streak >= _DIVERGENCE_PATIENCE:
                    msg = (
                        f"Loss divergence at step {step}: smoothed loss {loss_ema:.4f} > "
                        f"2× step-10k EMA {loss_at_step_10k:.4f} for "
                        f"{_DIVERGENCE_PATIENCE} consecutive steps"
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
    if ema is not None:
        ema.store_and_copy_to(unet)
        try:
            _save_checkpoint(unet.unet, unet.class_embed, cfg, ckpt_dir, step,
                             config_path, ema_suffix=True)
        finally:
            ema.restore(unet)
    wandb.finish()
    _console.print(f"\n[bold green]Training complete[/bold green] — {step:,} steps.")


if __name__ == "__main__":
    main()
