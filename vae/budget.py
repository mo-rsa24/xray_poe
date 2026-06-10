"""Plan 06 — GPU-hours + dollar calculator.

Turns the measured *rate* (img/s, from profiling) plus assumed run size (epochs, N)
and a price ($/GPU-hr) into a cost forecast. Being explicit about which inputs are
measured vs assumed keeps the forecast honest — the output flags each one.

  hours = epochs · N / img_per_s / 3600
  cost  = hours · rate · contingency

  python -m vae.budget --img-s 120 --epochs 150 --n 50000 --rate 0.79

If ``--img-s`` is omitted, the most recent measured img/s is read from the profiling
log (logs/vae_profile.log).
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

PROFILE_LOG = "logs/vae_profile.log"


def latest_measured_img_s(log_path: str = PROFILE_LOG) -> float | None:
    p = Path(log_path)
    if not p.exists():
        return None
    last = None
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = ast.literal_eval(line)
            if d.get("img_s"):
                last = float(d["img_s"])
        except (ValueError, SyntaxError):
            continue
    return last


def compute_budget(img_s: float, epochs: int, n: int, rate: float, contingency: float = 1.2) -> dict:
    hours = epochs * n / img_s / 3600.0
    cost = hours * rate * contingency
    return {"gpu_hours": hours, "cost_usd": cost, "img_s": img_s, "epochs": epochs,
            "n": n, "rate": rate, "contingency": contingency}


def main(args) -> None:
    measured = args.img_s is None
    img_s = args.img_s
    if img_s is None:
        img_s = latest_measured_img_s()
        if img_s is None:
            raise SystemExit("No --img-s given and no profiling log found. Run vae.profile first.")
    b = compute_budget(img_s, args.epochs, args.n, args.rate, args.contingency)

    src = "measured (profiling log)" if measured else "assumed (--img-s)"
    print(f"\n≈ {b['gpu_hours']:.1f} GPU-hours, ≈ ${b['cost_usd']:.0f}"
          f"  (incl. {args.contingency:g}× contingency)\n")
    print("inputs:")
    print(f"  img/s        {img_s:>10.1f}   [{src}]")
    print(f"  epochs       {args.epochs:>10}   [assumed]")
    print(f"  N (images)   {args.n:>10}   [assumed]")
    print(f"  $/GPU-hr     {args.rate:>10.2f}   [assumed]")
    print(f"  contingency  {args.contingency:>10.2f}   [assumed]")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VAE training-cost calculator")
    p.add_argument("--img-s", type=float, default=None, help="measured throughput; else read from log")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--n", type=int, required=True, help="number of training images")
    p.add_argument("--rate", type=float, required=True, help="$ per GPU-hour")
    p.add_argument("--contingency", type=float, default=1.2)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
