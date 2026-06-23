"""Re-stamp the rFID gate verdict from a cached rFID + a fresh gen-FID — no GPU.

rFID is a property of the VAE codec alone (vae_step0025000); it does NOT change
when the LDM checkpoint changes — only the target gen-FID does. So once
``scripts/rfid_gate.py`` has measured and cached rFID in results/rfid_gate.json,
the pass/fail verdict for any new gen-FID is pure arithmetic: no reconstruction, no
feature extraction, no torch. This is the loop-closing step the S4 checkpoint-watcher
calls after each S6 eval — recomputing the full rFID per checkpoint would be wasted
GPU since the number is fixed.

Pass: rFID <= target_gen_fid * margin  (the same rule and margin rfid_gate.py used;
margin defaults to whatever is recorded in the cached gate JSON).

CLI:
    # restamp from the S6 gen-FID result
    python scripts/stamp_rfid_gate.py --gate results/rfid_gate.json \\
        --target-from results/fid.json

    # restamp against an explicit target
    python scripts/stamp_rfid_gate.py --target-fid 12.0

Exit code is 1 on a FAILED gate (so a watcher/CI can branch to the fine-tune task),
0 on pass, 2 if the target is missing (nothing to stamp).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def gate_verdict(rfid: float, target_fid: float, margin: float) -> dict:
    """The single source of truth for the rFID pass/fail rule (rFID <= margin·target)."""
    threshold = target_fid * margin
    passed = rfid <= threshold
    return {
        "target_fid": target_fid,
        "margin": margin,
        "threshold": threshold,
        "ratio_to_target": rfid / target_fid if target_fid else None,
        "pass": bool(passed),
        "verdict": (
            f"VAE certified — ships as-is (rFID {rfid:.3f} ≪ target {target_fid:.3f})"
            if passed else
            f"GATE FAILED (rFID {rfid:.3f} > {margin:.2f}×target {target_fid:.3f} "
            f"= {threshold:.3f}) — run the conditional fine-tune"
        ),
    }


def resolve_target(path: str, key: str) -> float:
    """Read a gen-FID scalar from a JSON at a dotted `key`.

    Tolerates the metrics.fid layout ({"fid": {"fid": X, ...}}) and a flat
    {"fid": X}: if the key lands on the fid sub-dict, descend to its scalar.
    """
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--target-from: no such file: {p}")
    cur = json.loads(p.read_text())
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            cur = None
            break
    if isinstance(cur, dict) and isinstance(cur.get("fid"), (int, float)):
        cur = cur["fid"]
    if not isinstance(cur, (int, float)):
        raise SystemExit(f"--target-from: could not resolve a numeric FID from {p} "
                         f"at key {key!r} (got {cur!r})")
    return float(cur)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Re-stamp the rFID gate from a cached rFID + a fresh gen-FID (no GPU).",
    )
    p.add_argument("--gate", default="results/rfid_gate.json",
                   help="Cached rFID gate JSON written by rfid_gate.py")
    p.add_argument("--target-fid", type=float, default=None, dest="target_fid",
                   help="Target gen-FID supplied inline")
    p.add_argument("--target-from", default=None, dest="target_from",
                   help="Read the target gen-FID from this JSON (e.g. results/fid.json)")
    p.add_argument("--target-key", default="fid", dest="target_key",
                   help="Dotted key into --target-from for the gen-FID (default 'fid')")
    p.add_argument("--margin", type=float, default=None,
                   help="Override the pass margin (default: the margin in the gate JSON, else 1/3)")
    p.add_argument("--out", default=None, help="Where to write the stamped JSON (default: in place)")
    args = p.parse_args()

    gate_path = Path(args.gate)
    if not gate_path.is_file():
        raise SystemExit(f"--gate: no cached gate JSON at {gate_path} — run rfid_gate.py first")
    gate = json.loads(gate_path.read_text())

    rfid = gate.get("rfid")
    if not isinstance(rfid, (int, float)):
        raise SystemExit(f"{gate_path} has no numeric 'rfid' to stamp (got {rfid!r})")

    target_fid = args.target_fid
    target_src = "inline"
    if target_fid is None and args.target_from is not None:
        target_fid = resolve_target(args.target_from, args.target_key)
        target_src = f"{args.target_from}:{args.target_key}"

    if target_fid is None:
        print("No target gen-FID supplied (--target-fid / --target-from) — nothing to stamp.")
        print(f"   cached rFID = {rfid:.4f}   (verdict stays indeterminate)")
        sys.exit(2)

    margin = args.margin if args.margin is not None else gate.get("margin", 1.0 / 3.0)
    v = gate_verdict(rfid, target_fid, margin)
    gate.update(v)
    gate["target_src"] = target_src

    out_path = Path(args.out) if args.out else gate_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gate, indent=2))

    print(f"── rFID gate re-stamp (vae_step {gate.get('vae_step')}, n={gate.get('n')})")
    print(f"   rFID       = {rfid:.4f}")
    print(f"   target-FID = {target_fid:.4f}   (from {target_src})")
    print(f"   threshold  = {v['threshold']:.4f}   ({margin:.3f}×target)")
    print(f"\n   {v['verdict']}")
    print(f"\nStamped → {out_path}")

    sys.exit(0 if v["pass"] else 1)


if __name__ == "__main__":
    main()
