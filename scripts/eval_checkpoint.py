"""Per-checkpoint eval orchestrator (Sprint S4 watcher, task 1).

Given one LDM checkpoint, run the eval pipeline and record the result keyed by step:

    checkpoint → PoE compose samples → domain-FID (gen-FID) → re-stamp the rFID gate
              → append a row to the watcher's eval index.

It is a thin orchestrator: every heavy step is a subprocess call to an
already-tested script (scripts/generate.py, metrics.fid, scripts/stamp_rfid_gate.py),
so this file imports no torch and can be unit-tested with --dry-run. The subprocesses
inherit THIS interpreter (sys.executable), so launch the watcher with the jaxstack
python and the whole chain uses it.

What it does today (the critical path: gen-FID stamps S1 and is the N2 headline):
    1. generate ~N PoE-composed both-disease images from the checkpoint
    2. domain-FID(real both, composed) in the xrv space  → results/ckpt_eval/<step>/fid.json
    3. re-stamp the rFID gate against that gen-FID         → updates results/rfid_gate.json
    4. append [step, gen-FID, rFID, gate] to results/ckpt_eval/index.md

Deferred hooks (scaffolded, not yet wired — the plan's full pipeline): treatment
C2ST (metrics.c2st), both-present rate (metrics.presence_classifier), Grad-CAM grid
(scripts/grad_cam.py), and W&B logging. See the TODO block in run_eval().

CLI:
    python scripts/eval_checkpoint.py --ckpt ckpts/model_step0040000.safetensors
    python scripts/eval_checkpoint.py --ckpt ckpts/<step>.safetensors --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step_of(ckpt: Path) -> str:
    """'model_step0040000.safetensors' → '0040000' (falls back to the stem)."""
    m = re.search(r"step0*?(\d+)", ckpt.stem)
    return m.group(1).zfill(7) if m else ckpt.stem


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def run_eval(args: argparse.Namespace) -> int:
    ckpt = Path(args.ckpt) if Path(args.ckpt).is_absolute() else ROOT / args.ckpt
    if not ckpt.is_file():
        raise SystemExit(f"--ckpt: no such file: {ckpt}")
    step = step_of(ckpt)
    w_str = f"{args.w:.1f}".replace(".", "p")        # matches generate.py's naming

    gen_out = (ROOT / args.gen_out) / step           # outputs/ckpt_eval/<step>
    composed = gen_out / "compose" / f"w{w_str}"      # where generate.py drops the PNGs
    eval_dir = (ROOT / args.out_root) / step          # results/ckpt_eval/<step>
    fid_json = eval_dir / "fid.json"
    gate_json = ROOT / args.rfid_gate
    index_md = (ROOT / args.out_root) / "index.md"
    py = sys.executable

    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1) PoE compose, 2) gen-FID, 3) re-stamp the rFID gate — as subprocesses.
    steps: list[tuple[str, list[str]]] = [
        ("compose", [
            py, "scripts/generate.py", "--compose", "--w-sweep", str(args.w),
            "--n", str(args.n), "--seed", str(args.seed), "--steps", str(args.steps),
            "--gen-batch-size", str(args.gen_batch_size),
            "--ckpt", str(ckpt), "--vae-ckpt", args.vae_ckpt,
            "--scale-factor", args.scale_factor, "-o", str(gen_out),
        ]),
        ("gen-FID", [
            py, "-m", "metrics.fid", "--embed", "xrv", "--kid",
            "--a", args.real, "--b", str(composed), "--n", str(args.n),
            "--out", str(fid_json),
        ]),
        ("stamp-rFID", [
            py, "scripts/stamp_rfid_gate.py", "--gate", str(gate_json),
            "--target-from", str(fid_json),
        ]),
    ]

    if args.dry_run:
        print(f"[dry-run] step={step}  composed→{composed}")
        for name, cmd in steps:
            print(f"[dry-run] {name}: {' '.join(cmd)}")
        print(f"[dry-run] would append a row to {index_md}")
        return 0

    for name, cmd in steps:
        print(f"\n── [{step}] {name} ──")
        r = subprocess.run(cmd, cwd=ROOT)
        # stamp-rFID returns 1 on a FAILED gate — that is a valid, recorded outcome,
        # not an orchestration error, so don't abort the eval on it.
        if r.returncode != 0 and name != "stamp-rFID":
            print(f"  [error] {name} failed (exit {r.returncode}) — aborting eval for {step}")
            return r.returncode

    # TODO(S4 full pipeline): add the remaining per-checkpoint metrics as subprocess
    # steps, each writing into eval_dir, then surface them in the index row below:
    #   • treatment C2ST     python -m metrics.c2st  --a <real both> --b <composed> ...
    #   • both-present rate  python -m metrics.presence_classifier --images <composed> ...
    #   • Grad-CAM grid      python scripts/grad_cam.py --images <composed> -o <eval_dir>
    #   • W&B logging        log {step, gen_fid, both_present, ...} to the run
    # They are independent of the gen-FID critical path, so they can land incrementally.

    fid = _read_json(fid_json)
    gate = _read_json(gate_json)
    gen_fid = fid.get("fid", {}).get("fid")
    gen_kid = fid.get("kid", {}).get("kid")
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step": step,
        "gen_fid": gen_fid,
        "gen_kid": gen_kid,
        "rfid": gate.get("rfid"),
        "gate": gate.get("pass"),
        "ratio": gate.get("ratio_to_target"),
    }
    (eval_dir / "summary.json").write_text(json.dumps(row, indent=2))
    _append_index(index_md, row)

    def _fmt(x, n=3):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"
    print(f"\n── [{step}] DONE  gen-FID={_fmt(gen_fid)}  rFID={_fmt(row['rfid'])}  "
          f"gate={'PASS' if row['gate'] else 'FAIL' if row['gate'] is False else '—'}")
    print(f"   → {eval_dir/'summary.json'}   index → {index_md}")
    return 0


def _append_index(index_md: Path, row: dict) -> None:
    """Append one markdown table row to the watcher's eval index (header on first write)."""
    index_md.parent.mkdir(parents=True, exist_ok=True)
    header = ("| time | step | gen-FID | gen-KID | rFID | gate |\n"
              "|------|------|---------|---------|------|------|\n")
    if not index_md.is_file():
        index_md.write_text("# Checkpoint eval index (S4 watcher)\n\n" + header)

    def cell(x, n=4):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"
    gate = "✅ pass" if row["gate"] else "❌ fail" if row["gate"] is False else "—"
    line = (f"| {row['time']} | {row['step']} | {cell(row['gen_fid'])} | "
            f"{cell(row['gen_kid'], 5)} | {cell(row['rfid'])} | {gate} |\n")
    with index_md.open("a") as f:
        f.write(line)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-checkpoint eval: compose → gen-FID → re-stamp rFID gate → log.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="LDM checkpoint (.safetensors)")
    p.add_argument("--vae-ckpt", default="ckpts/vae_step0025000.pt")
    p.add_argument("--real", default="data/nih/images", help="Real both-disease set")
    p.add_argument("--scale-factor", default="data/latents/scale_factor.pt")
    p.add_argument("--n", type=int, default=500, help="Composed samples to generate/score")
    p.add_argument("--w", type=float, default=1.0, help="PoE compose CFG weight")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen-batch-size", type=int, default=4)
    p.add_argument("--gen-out", default="outputs/ckpt_eval", help="Root for composed PNGs")
    p.add_argument("--out-root", default="results/ckpt_eval", help="Root for eval JSON + index")
    p.add_argument("--rfid-gate", default="results/rfid_gate.json", help="Cached rFID gate")
    p.add_argument("--dry-run", action="store_true", help="Print the commands; run nothing")
    args = p.parse_args()
    sys.exit(run_eval(args))


if __name__ == "__main__":
    main()
