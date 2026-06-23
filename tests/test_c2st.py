"""Unit tests for the two-sample metrics (plan 06: C2ST + MMD).

Numeric-core only — feature arrays in, no images, no torch, no checkpoints, so these
run on any CPU. They encode the spec's sanity check: ~0.5 / MMD~0 on identical
distributions, ~1.0 / MMD large on obviously different ones.
"""
from __future__ import annotations

import numpy as np
import pytest

from metrics.c2st import c2st_auc, mmd_rbf


def _two_sets(seed: int, shift: float, d: int = 32, n: int = 400):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, d))
    b = rng.standard_normal((n, d)) + shift
    return a, b


# ── C2ST ─────────────────────────────────────────────────────────────────────

def test_c2st_identical_is_near_half():
    a, b = _two_sets(seed=0, shift=0.0)
    res = c2st_auc(a, b, seed=0)
    assert res["auc"] < 0.60                 # indistinguishable
    assert not res["clears_floor"]           # within the real-vs-real null floor


def test_c2st_shifted_is_near_one():
    a, b = _two_sets(seed=1, shift=3.0)
    res = c2st_auc(a, b, seed=0)
    assert res["auc"] > 0.95                 # trivially separable
    assert res["clears_floor"]


def test_c2st_auc_is_folded_above_half():
    # AUC is folded to [0.5, 1]; swapping A and B must give the same magnitude.
    a, b = _two_sets(seed=2, shift=2.0)
    assert c2st_auc(a, b, seed=0)["auc"] == pytest.approx(
        c2st_auc(b, a, seed=0)["auc"], abs=0.05)


def test_c2st_identical_sets_are_degenerate_half():
    # The spec's `real_a/ real_a/` smoke test: the same set twice has no
    # distributional gap, so the overlap guard collapses it to AUC 0.5.
    a, _ = _two_sets(seed=5, shift=0.0)
    res = c2st_auc(a, a.copy(), seed=0)
    assert res["auc"] == 0.5
    assert res.get("degenerate") is True
    assert res["n_overlap"] == 2 * len(a)


def test_c2st_dim_mismatch_raises():
    with pytest.raises(ValueError):
        c2st_auc(np.zeros((10, 4)), np.zeros((10, 5)))


# ── MMD ──────────────────────────────────────────────────────────────────────

def test_mmd_identical_is_near_zero():
    a, b = _two_sets(seed=3, shift=0.0)
    res = mmd_rbf(a, b, seed=0)
    assert abs(res["mmd2"]) < 0.02           # ~0 under H0 (unbiased, can be slightly <0)
    assert res["p_value"] > 0.05             # not distinguishable


def test_mmd_shifted_is_large_and_significant():
    a, b = _two_sets(seed=4, shift=3.0)
    res = mmd_rbf(a, b, seed=0)
    assert res["mmd2"] > 0.1
    assert res["p_value"] < 0.05             # distinguishable


def test_mmd_dim_mismatch_raises():
    with pytest.raises(ValueError):
        mmd_rbf(np.zeros((10, 4)), np.zeros((10, 5)))
