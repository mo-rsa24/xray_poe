"""Unit tests for the realism distances (plan 06: FID + KID).

Numeric-core only — feature arrays in, no images, no torch, no checkpoints, so these
run on any CPU. They encode the spec's sanity check: FID ~0 / KID ~0 on identical
distributions (the real-vs-real noise floor), FID large / KID large on obviously
different ones.
"""
from __future__ import annotations

import numpy as np
import pytest

from metrics.fid import fid_from_features, frechet_distance, kid_poly


def _two_sets(seed: int, shift: float, d: int = 32, n: int = 2000):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, d))
    b = rng.standard_normal((n, d)) + shift
    return a, b


# ── FID ──────────────────────────────────────────────────────────────────────

def test_fid_identical_is_near_zero():
    a, b = _two_sets(seed=0, shift=0.0)
    res = fid_from_features(a, b)
    assert res["fid"] < 1.0                  # ~0 noise floor (finite-sample bias only)


def test_fid_shifted_is_large():
    a, b = _two_sets(seed=1, shift=3.0)
    assert fid_from_features(a, b)["fid"] > 50.0


def test_fid_grows_with_separation():
    base, _ = _two_sets(seed=2, shift=0.0)
    rng = np.random.default_rng(7)
    small = base + 0.5 + 0.0 * rng.standard_normal(base.shape)
    large = base + 3.0
    f_small = fid_from_features(base, small)["fid"]
    f_large = fid_from_features(base, large)["fid"]
    assert f_large > f_small > 0.0           # monotone in the mean gap


def test_fid_is_symmetric():
    a, b = _two_sets(seed=3, shift=2.0)
    assert fid_from_features(a, b)["fid"] == pytest.approx(
        fid_from_features(b, a)["fid"], rel=1e-6)


def test_fid_self_is_zero():
    a, _ = _two_sets(seed=4, shift=0.0)
    # identical mean and covariance → Fréchet distance is exactly 0 (up to clamp)
    assert fid_from_features(a, a.copy())["fid"] == pytest.approx(0.0, abs=1e-6)


def test_fid_dim_mismatch_raises():
    with pytest.raises(ValueError):
        fid_from_features(np.zeros((10, 4)), np.zeros((10, 5)))


def test_frechet_distance_closed_form():
    # Two 1-D Gaussians: FID = (mu1-mu2)² + (s1 + s2 - 2·sqrt(s1·s2)) = Δmu² + (√s1-√s2)².
    mu1, s1 = np.array([0.0]), np.array([[4.0]])
    mu2, s2 = np.array([3.0]), np.array([[1.0]])
    expected = 3.0 ** 2 + (2.0 - 1.0) ** 2
    assert frechet_distance(mu1, s1, mu2, s2) == pytest.approx(expected, abs=1e-6)


# ── KID ──────────────────────────────────────────────────────────────────────

def test_kid_identical_is_near_zero():
    a, b = _two_sets(seed=5, shift=0.0)
    res = kid_poly(a, b, n_subsets=50, subset_size=500, seed=0)
    assert abs(res["kid"]) < 0.01            # unbiased → ~0 under H0, even at small N


def test_kid_shifted_is_large():
    a, b = _two_sets(seed=6, shift=3.0)
    res = kid_poly(a, b, n_subsets=50, subset_size=500, seed=0)
    assert res["kid"] > 0.5
    assert res["kid_std"] >= 0.0


def test_kid_reports_an_honest_noise_band():
    # The across-subset std is the small-N noise floor; it must be present and finite.
    a, b = _two_sets(seed=8, shift=0.0)
    res = kid_poly(a, b, n_subsets=30, subset_size=300, seed=0)
    assert len(res["kid_subsets"]) == 30
    assert res["subset_size"] == 300
    assert np.isfinite(res["kid_std"])


def test_kid_dim_mismatch_raises():
    with pytest.raises(ValueError):
        kid_poly(np.zeros((10, 4)), np.zeros((10, 5)))
