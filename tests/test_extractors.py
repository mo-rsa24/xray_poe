"""Unit tests for the heart-size & blunting extractors (plan 06-04).

Numeric-core only — synthetic CAM arrays in, no images, no torch, no checkpoint, so
these run on any CPU. They pin the geometry of the two scalars: heart_size is the
cardio activation width measured *within the cardiac band* (the fix for the
diffuse-normal failure), and blunting is the effusion activation confined to the
bottom-lateral costophrenic quadrants.
"""
from __future__ import annotations

import numpy as np
import pytest

from metrics.extractors import blunting_from_cam, heart_size_from_cam
from metrics.grad_cam_utils import bbox_from_cam

H = W = 512


def _zeros() -> np.ndarray:
    return np.zeros((H, W), dtype=np.float32)


# ── heart_size ─────────────────────────────────────────────────────────────────

def test_heart_size_wider_band_gives_larger_value():
    wide = _zeros();  wide[200:300, 100:400] = 1.0     # cols span 300/512
    narrow = _zeros(); narrow[200:300, 240:270] = 1.0  # cols span 30/512
    hw = heart_size_from_cam(wide)
    hn = heart_size_from_cam(narrow)
    assert hw > hn
    assert hw == pytest.approx(300 / W, abs=1 / W)
    assert hn == pytest.approx(30 / W, abs=1 / W)


def test_heart_size_ignores_activation_outside_cardiac_band():
    # Hot region only in the apex (rows above the 0.35 band top) → not counted.
    apex = _zeros(); apex[0:50, 100:400] = 1.0
    assert heart_size_from_cam(apex) == 0.0


def test_heart_size_zero_when_no_activation():
    assert heart_size_from_cam(_zeros()) == 0.0


# ── blunting ───────────────────────────────────────────────────────────────────

def test_blunting_high_in_costophrenic_corners():
    cam = _zeros()
    cam[int(H * 0.6):, :int(W * 0.3)] = 1.0     # bottom-left
    cam[int(H * 0.6):, int(W * 0.7):] = 1.0     # bottom-right
    # the ROI is exactly those corners → mean activation ~1.0
    assert blunting_from_cam(cam) == pytest.approx(1.0, abs=1e-6)


def test_blunting_low_for_central_activation():
    cam = _zeros(); cam[220:300, 220:300] = 1.0   # centre, outside the basal ROI
    assert blunting_from_cam(cam) == pytest.approx(0.0, abs=1e-6)


def test_blunting_zero_when_no_activation():
    assert blunting_from_cam(_zeros()) == 0.0


# ── bbox helper (shared Grad-CAM util) ───────────────────────────────────────────

def test_bbox_from_cam_bounds_thresholded_region():
    cam = _zeros(); cam[100:200, 150:250] = 1.0
    x0, y0, x1, y1 = bbox_from_cam(cam, frac=0.5)
    assert (x0, y0, x1, y1) == (150, 100, 249, 199)


def test_bbox_from_cam_none_when_empty():
    cam = _zeros(); cam[:] = -1.0   # nothing >= frac*max with a non-positive max
    assert bbox_from_cam(cam) is None
