"""Checkpoint-watcher polling loop (Sprint S4 watcher, task 2).

Watch a directory for new LDM checkpoints and run scripts/eval_checkpoint.py on each
one as it appears, so Track-B metrics re-score Track-A's saves with no manual step.
Deliberately simple: a poll loop with a persisted "seen" set, not an MLOps framework.

Launch it once, in the background, when the S2 continuation training starts:

    python scripts/watch_checkpoints.py --dir ckpts --n 500 &

It picks up checkpoints already in --dir (unless --skip-existing) plus every new one.
A checkpoint is evaluated once; the seen-set is persisted to --state so a restart
does not re-evaluate. Each eval is a subprocess inheriting THIS interpreter, so run
the watcher with the jaxstack python and the whole chain uses it.

Stability guard: a checkpoint is only evaluated once its file size has been stable
for one poll interval, so a half-written .safetensors mid-save is not picked up.

CLI:
    python scripts/watch_checkpoints.py --dir ckpts --interval 60 --n 500
    python scripts/watch_checkpoints.py --dir ckpts --once          # drain & exit
    python scripts/watch_checkpoints.py --dir ckpts --once --skip-existing --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_seen(state: Path) -> set[str]:
    if state.is_file():
        return {ln.strip() for ln in state.read_text().splitlines() if ln.strip()}
    return set()


def _mark_seen(state: Path, name: str) -> None:
    state.parent.mkdir(parents=True, exist_ok=True)
    with state.open("a") as f:
        f.write(name + "\n")


def _eval_passthrough(args: argparse.Namespace) -> list[str]:
    """Args forwarded verbatim to eval_checkpoint.py for every checkpoint."""
    out = ["--vae-ckpt", args.vae_ckpt, "--real", args.real, "--n", str(args.n),
           "--w", str(args.w), "--out-root", args.out_root, "--gen-out", args.gen_out,
           "--rfid-gate", args.rfid_gate]
    if args.dry_run:
        out.append("--dry-run")
    return out


def _evaluate(ckpt: Path, args: argparse.Namespace) -> None:
    eval_script = Path(args.eval) if Path(args.eval).is_absolute() else ROOT / args.eval
    cmd = [sys.executable, str(eval_script), "--ckpt", str(ckpt), *_eval_passthrough(args)]
    print(f"\n=== evaluating {ckpt.name} ===")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        # A FAILED gate is exit 1 from the stamp but eval_checkpoint swallows it, so a
        # non-zero here means the eval itself broke. Don't mark seen → retry next poll.
        print(f"  [warn] eval exited {r.returncode} for {ckpt.name}; will retry next poll")
        raise RuntimeError(ckpt.name)


def _stable(ckpt: Path, sizes: dict[str, int]) -> bool:
    """True once the file size is unchanged since the previous poll (write finished)."""
    cur = ckpt.stat().st_size
    prev = sizes.get(ckpt.name)
    sizes[ckpt.name] = cur
    return prev is not None and prev == cur


def main() -> None:
    p = argparse.ArgumentParser(
        description="Poll a dir for new LDM checkpoints and eval each one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dir", default="ckpts", help="Directory to watch")
    p.add_argument("--pattern", default="model_step*.safetensors", help="Checkpoint glob")
    p.add_argument("--eval", default="scripts/eval_checkpoint.py", help="Per-ckpt eval script")
    p.add_argument("--interval", type=float, default=60.0, help="Poll seconds")
    p.add_argument("--once", action="store_true", help="Drain current checkpoints then exit")
    p.add_argument("--skip-existing", action="store_true",
                   help="Mark checkpoints present at startup as seen (don't eval them)")
    p.add_argument("--state", default="results/ckpt_eval/.seen", help="Persisted seen-set file")
    # forwarded to eval_checkpoint.py
    p.add_argument("--vae-ckpt", default="ckpts/vae_step0025000.pt")
    p.add_argument("--real", default="data/nih/images")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--out-root", default="results/ckpt_eval")
    p.add_argument("--gen-out", default="outputs/ckpt_eval")
    p.add_argument("--rfid-gate", default="results/rfid_gate.json")
    p.add_argument("--dry-run", action="store_true", help="Forward --dry-run to each eval")
    args = p.parse_args()

    watch_dir = Path(args.dir) if Path(args.dir).is_absolute() else ROOT / args.dir
    state = Path(args.state) if Path(args.state).is_absolute() else ROOT / args.state
    seen = _load_seen(state)

    if args.skip_existing:
        for c in watch_dir.glob(args.pattern):
            if c.name not in seen:
                seen.add(c.name)
                _mark_seen(state, c.name)
        print(f"[skip-existing] marked {len(seen)} present checkpoint(s) as seen")

    print(f"watching {watch_dir}/{args.pattern}  interval={args.interval}s  "
          f"{'(once)' if args.once else '(continuous; Ctrl-C to stop)'}")
    sizes: dict[str, int] = {}
    try:
        while True:
            # sort by step so checkpoints evaluate in training order
            ckpts = sorted(watch_dir.glob(args.pattern), key=lambda c: c.name)
            for c in ckpts:
                if c.name in seen:
                    continue
                if not (args.once or _stable(c, sizes)):
                    print(f"  [pending] {c.name} (waiting for size to settle)")
                    continue
                try:
                    _evaluate(c, args)
                    seen.add(c.name)
                    _mark_seen(state, c.name)
                except RuntimeError:
                    pass  # eval broke; leave unseen to retry
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
