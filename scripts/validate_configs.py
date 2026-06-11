"""Validate all LDM YAML configs and print a summary line per file."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config

REQUIRED_KEYS = {
    "lr", "batch_size", "grad_accum", "model_channels", "num_train_timesteps",
    "cfg_dropout_p", "bf16", "max_steps", "ckpt_every", "seed",
    "effusion_weight", "no_finding_cap", "wandb",
}

CONFIGS = [
    "configs/ldm_full.yaml",
    "configs/ldm_ablation_cfg_p00.yaml",
    "configs/ldm_ablation_cfg_p30.yaml",
    "configs/ldm_debug.yaml",
]

root = Path(__file__).parent.parent
failures = 0

for rel in CONFIGS:
    path = root / rel
    try:
        cfg = load_config(path)
        missing = REQUIRED_KEYS - cfg.keys()
        if missing:
            print(f"{rel:<45} FAIL — missing keys: {sorted(missing)}")
            failures += 1
        else:
            extra = ""
            if "cfg_dropout_p" in cfg:
                extra = f"cfg_dropout_p={cfg['cfg_dropout_p']}, "
            key_count = len(cfg)
            print(f"{rel:<45} OK — {extra}{key_count} keys present")
    except Exception as e:
        print(f"{rel:<45} ERROR — {e}")
        failures += 1

sys.exit(failures)
