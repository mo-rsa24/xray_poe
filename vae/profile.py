"""Plan 05 — profiling: the content-independent resource numbers.

Measures peak VRAM (``torch.cuda.max_memory_allocated``) and steady-state throughput
(img/s) for a given res/batch/precision, and (``--sweep``) walks batch size up to the
largest that fits — the VRAM-tier selector. Valid on noise tensors because peak VRAM
and throughput depend only on shape, dtype, batch, architecture, and optimizer — not
on data content.

  python -m vae.profile --res 512 --batch 8 --precision bf16
  python -m vae.profile --res 512 --precision bf16 --sweep

A log line is appended to ``logs/vae_profile.log`` so the budget calculator (plan 06)
can read a measured img/s.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .config import VAEConfig
from .losses import vae_loss
from .model import VAE

LOG_PATH = "logs/vae_profile.log"


def _amp_dtype(precision: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def profile_config(
    res: int, batch: int, precision: str, steps: int = 20, warmup: int = 5,
    grad_checkpoint: bool = False, log: bool = True,
) -> dict:
    if not torch.cuda.is_available():
        return _cpu_smoke(res, batch, precision, steps)

    device = "cuda"
    cfg = VAEConfig(input_resolution=res, use_checkpoint=grad_checkpoint)
    model = VAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dtype = _amp_dtype(precision)
    use_amp = precision != "fp32"

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = None
    for step in range(warmup + steps):
        if step == warmup:
            torch.cuda.synchronize(); t0 = time.time()
        x = torch.randn(batch, 1, res, res, device=device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
            recon, mu, sigma = model(x)
            loss = vae_loss(recon.float(), x, mu, sigma, cfg)
        loss.total.backward()
        opt.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    img_s = (steps * batch) / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    result = {"res": res, "batch": batch, "precision": precision,
              "grad_checkpoint": grad_checkpoint, "peak_vram_gb": round(peak_gb, 2),
              "img_s": round(img_s, 1), "device": torch.cuda.get_device_name(0)}
    print(f"res {res} batch {batch} {precision}{' +ckpt' if grad_checkpoint else ''}: "
          f"peak VRAM {peak_gb:.2f} GB | {img_s:.1f} img/s")
    if log:
        _append_log(result)
    del model, opt
    torch.cuda.empty_cache()
    return result


def _cpu_smoke(res, batch, precision, steps) -> dict:
    print("no CUDA — running a CPU timing smoke (VRAM not measurable on CPU)")
    cfg = VAEConfig(input_resolution=res)
    model = VAE(cfg)
    x = torch.randn(batch, 1, res, res)
    t0 = time.time()
    for _ in range(steps):
        recon, mu, sigma = model(x)
        loss = vae_loss(recon, x, mu, sigma, cfg)
        loss.total.backward()
        model.zero_grad(set_to_none=True)
    img_s = (steps * batch) / (time.time() - t0)
    print(f"CPU smoke: {img_s:.2f} img/s (peak VRAM deferred to GPU)")
    return {"res": res, "batch": batch, "precision": precision, "peak_vram_gb": None,
            "img_s": round(img_s, 2), "device": "cpu"}


def sweep(res: int, precision: str, batches=(1, 2, 4, 8, 16), grad_checkpoint: bool = False) -> list[dict]:
    print(f"\nbatch-size sweep @ res {res} {precision}"
          f"{' +grad-ckpt' if grad_checkpoint else ''}:")
    results, max_fit = [], None
    for b in batches:
        try:
            r = profile_config(res, b, precision, grad_checkpoint=grad_checkpoint, log=False)
            results.append(r)
            max_fit = b
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"res {res} batch {b}: OOM — stop sweep")
                torch.cuda.empty_cache()
                break
            raise
    print(f"max batch that fits: {max_fit}")
    return results


def _append_log(result: dict) -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{result}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VAE profiling (peak VRAM + img/s)")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--sweep", action="store_true", help="walk batch size to the max that fits")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.sweep:
        sweep(args.res, args.precision, grad_checkpoint=args.grad_checkpoint)
    else:
        profile_config(args.res, args.batch, args.precision, steps=args.steps,
                       grad_checkpoint=args.grad_checkpoint)
