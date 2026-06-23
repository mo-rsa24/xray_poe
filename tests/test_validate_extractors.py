"""Unit tests for extractor validation (plan 06-05).

Numeric-core only — (N, 2) extractor-output arrays in, no images, no torch, no
checkpoint, so these run on any CPU. They pin the agreement verdict: matched
distributions agree (2-D AUC ~0.5), shifted ones disagree (AUC above threshold), and
the figure writer produces a file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metrics.validate_extractors import compare_extractor_dists, make_figure


def _hb(seed: int, heart=(0.55, 0.15), blunt=(0.06, 0.05), n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(*heart, n), rng.normal(*blunt, n)])


# ── agreement verdict ────────────────────────────────────────────────────────────

def test_matched_distributions_agree():
    real = _hb(0)
    gen = _hb(1)                          # same params, different draw
    res = compare_extractor_dists(real, gen, threshold=0.60)
    assert res["auc_2d"] <= 0.60
    assert res["agree"] is True
    assert not res["clears_floor"]


def test_shifted_distributions_disagree():
    real = _hb(2)
    gen = _hb(3, heart=(0.75, 0.15), blunt=(0.20, 0.05))   # both scalars shifted
    res = compare_extractor_dists(real, gen, threshold=0.60)
    assert res["auc_2d"] > 0.60
    assert res["agree"] is False


def test_per_scalar_stats_present():
    res = compare_extractor_dists(_hb(4), _hb(5), threshold=0.60)
    for s in ("heart_size", "blunting"):
        ps = res["per_scalar"][s]
        assert {"real_mean", "gen_mean", "ks_stat", "ks_p", "auc"} <= set(ps)


def test_nonfinite_rows_dropped():
    real = _hb(6)
    real[0] = [np.nan, np.nan]
    res = compare_extractor_dists(real, _hb(7), threshold=0.60)
    assert res["n_real"] == len(real) - 1


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        compare_extractor_dists(np.zeros((10, 3)), np.zeros((10, 2)))


# ── figure ───────────────────────────────────────────────────────────────────────

def test_make_figure_writes_file(tmp_path: Path):
    real, gen = _hb(8), _hb(9)
    res = compare_extractor_dists(real, gen, threshold=0.60)
    out = tmp_path / "val.png"
    make_figure(real, gen, res, out)
    assert out.exists() and out.stat().st_size > 0
