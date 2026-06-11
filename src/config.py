"""Load and resolve LDM YAML configs with _base_ inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_config(path: str | Path) -> dict[str, Any]:
    """Return fully-merged config dict, resolving _base_ chains.

    Child keys override parent keys; _base_ itself is stripped from the result.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base_name = cfg.pop("_base_", None)
    if base_name is not None:
        base_path = path.parent / base_name
        base_cfg = load_config(base_path)   # recurse for multi-level chains
        base_cfg.update(cfg)                # child wins on conflict
        cfg = base_cfg

    return cfg
