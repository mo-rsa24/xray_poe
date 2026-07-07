"""Live training velocity + ETA from the W&B run (for when tmux is detached).

Polls the run's logged global_step, computes steps/min over a recent window, and
prints percent-done + ETA to max_steps. Loops until the run finishes.

    python scripts/monitor_progress.py                                   # default run, loop
    python scripts/monitor_progress.py --once                            # single read
    python scripts/monitor_progress.py --name exp5_retrain_s01_20260625 --max-steps 100000
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta

import wandb


def _fmt(sec: float) -> str:
    if sec == float("inf") or sec != sec:
        return "?"
    return str(timedelta(seconds=int(sec)))


def _resolve(api, project, name, run_path):
    if run_path:
        return run_path
    runs = list(api.runs(project, filters={"display_name": name}, order="-created_at"))
    if not runs:
        return None
    # prefer a RUNNING run with this name (newest); else newest of any state
    running = [r for r in runs if r.state == "running"]
    pick = running[0] if running else runs[0]
    return f"{pick.entity}/{pick.project}/{pick.id}"


def _velocity(run, window):
    """steps/sec over the last `window` logged points, from history timestamps."""
    h = run.history(keys=["_step", "_timestamp"], samples=window, pandas=True)
    h = h.dropna().sort_values("_step")
    if len(h) < 2:
        return None
    d_step = float(h["_step"].iloc[-1] - h["_step"].iloc[0])
    d_t = float(h["_timestamp"].iloc[-1] - h["_timestamp"].iloc[0])
    return d_step / d_t if d_t > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="prime_lab/paper3-ldm")
    ap.add_argument("--name", default="exp5_retrain_s01_20260625")
    ap.add_argument("--run-path", default=None, help="entity/project/run_id (skips name lookup)")
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--window", type=int, default=400, help="recent logged points for velocity")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    api = wandb.Api()
    path = _resolve(api, args.project, args.name, args.run_path)
    if not path:
        print(f"no W&B run named '{args.name}' in {args.project} yet (started? give it a few steps)")
        return
    print(f"watching {path} → max_steps {args.max_steps}")

    while True:
        run = api.run(path)                      # re-fetch for fresh summary
        step = run.summary.get("global_step") or run.summary.get("_step")
        if step is None:
            print("  no global_step logged yet…")
        else:
            step = int(step)
            vel = _velocity(run, args.window)     # steps/sec
            if vel and vel > 0:
                rem = (args.max_steps - step) / vel
                loss = run.summary.get("train/loss")
                lt = f" · loss {loss:.4f}" if isinstance(loss, (int, float)) else ""
                print(f"  step {step:>6}/{args.max_steps} ({step/args.max_steps:5.1%}) · "
                      f"{vel*60:5.1f} steps/min · ETA {_fmt(rem)} · {run.state}{lt}")
            else:
                print(f"  step {step:>6}/{args.max_steps} ({step/args.max_steps:5.1%}) · "
                      f"(measuring velocity…) · {run.state}")
        if args.once or run.state == "finished" or (step and step >= args.max_steps):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
