"""Unit tests for the real-vs-real floor (plan 06-03).

Numeric-core only — feature arrays in, no images, no torch, no checkpoint, so these
run on any CPU. They pin the floor's defining behaviour: halves of one distribution
sit at the no-difference floor (AUC ~0.5 within the null bound, FID ~0, KID ~0), the
split-bootstrap CI brackets the point estimate, and small N raises the power flag.
"""
from __future__ import annotations

import numpy as np

from metrics.floor import floor_from_features, split_halves


def _same_dist(n: int, d: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


# ── split ──────────────────────────────────────────────────────────────────────

def test_split_is_disjoint_and_even_odd():
    feats = np.arange(20).reshape(10, 2)
    a, b = split_halves(feats)
    assert len(a) == 5 and len(b) == 5
    assert np.array_equal(a, feats[0::2])
    assert np.array_equal(b, feats[1::2])


# ── floor on identical halves ────────────────────────────────────────────────────

def test_floor_identical_sits_at_no_difference():
    f = floor_from_features(_same_dist(2000), n_boot=80, kid_subset_size=400, seed=0)
    assert not f["c2st"]["clears_floor"]          # AUC within the analytic null bound
    assert f["mmd"]["p_value"] > 0.05             # indistinguishable
    assert f["fid"]["fid"] < 1.0                  # ~0 (finite-sample floor)
    assert abs(f["kid"]["kid"]) < 0.01            # unbiased → ~0


def test_floor_fid_point_inside_split_bootstrap_ci():
    # The split-bootstrap (fresh disjoint halves) must bracket the point estimate —
    # the resample-with-replacement bug pushed the FID point *below* its own CI.
    f = floor_from_features(_same_dist(2000), n_boot=80, kid_subset_size=400, seed=1)
    lo, hi = f["fid"]["ci95"]
    assert lo <= f["fid"]["fid"] <= hi


def test_floor_power_flag_trips_for_small_n():
    small = floor_from_features(_same_dist(120), n_boot=40, kid_subset_size=40,
                               min_n=200, seed=2)
    assert small["power_flag"] is True            # 60/half < min_n=200
    big = floor_from_features(_same_dist(1000), n_boot=40, kid_subset_size=200,
                             min_n=200, seed=2)
    assert big["power_flag"] is False             # 500/half >= 200


def test_floor_reports_both_halves_and_bounds():
    f = floor_from_features(_same_dist(800), n_boot=40, kid_subset_size=200, seed=3)
    assert f["n_a"] == 400 and f["n_b"] == 400
    for key in ("mmd", "fid", "kid"):
        assert "upper95" in f[key] and "ci95" in f[key]
