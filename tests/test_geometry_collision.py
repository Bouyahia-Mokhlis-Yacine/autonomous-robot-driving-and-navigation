"""Tests for the SAT-based rectangle collision helpers in common.geometry."""

from __future__ import annotations

import math

import pytest

from common.geometry import polygons_intersect, rectangle_corners


def test_rectangle_corners_axis_aligned() -> None:
    corners = rectangle_corners(0.0, 0.0, 2.0, 1.0, 0.0)
    xs = sorted(c[0] for c in corners)
    ys = sorted(c[1] for c in corners)
    assert xs == pytest.approx([-1.0, -1.0, 1.0, 1.0])
    assert ys == pytest.approx([-0.5, -0.5, 0.5, 0.5])


def test_rectangle_corners_rotated_90_degrees() -> None:
    # A 2 (length) x 1 (width) rectangle rotated 90 deg swaps its extents.
    corners = rectangle_corners(0.0, 0.0, 2.0, 1.0, math.pi / 2.0)
    xs = sorted(round(c[0], 9) for c in corners)
    ys = sorted(round(c[1], 9) for c in corners)
    assert xs == pytest.approx([-0.5, -0.5, 0.5, 0.5])
    assert ys == pytest.approx([-1.0, -1.0, 1.0, 1.0])


def test_overlapping_axis_aligned_rectangles_intersect() -> None:
    a = rectangle_corners(0.0, 0.0, 2.0, 2.0, 0.0)
    b = rectangle_corners(1.0, 0.0, 2.0, 2.0, 0.0)
    assert polygons_intersect(a, b) is True


def test_separated_rectangles_do_not_intersect() -> None:
    a = rectangle_corners(0.0, 0.0, 1.0, 1.0, 0.0)
    b = rectangle_corners(10.0, 10.0, 1.0, 1.0, 0.0)
    assert polygons_intersect(a, b) is False


def test_rotated_rectangle_corner_grazing_axis_aligned_box() -> None:
    # A 45-degree-rotated square whose corner pokes into an axis-aligned box.
    rotated = rectangle_corners(1.5, 0.0, math.sqrt(2), math.sqrt(2), math.pi / 4.0)
    box = rectangle_corners(0.0, 0.0, 2.0, 2.0, 0.0)
    assert polygons_intersect(rotated, box) is True

    far_rotated = rectangle_corners(5.0, 0.0, math.sqrt(2), math.sqrt(2), math.pi / 4.0)
    assert polygons_intersect(far_rotated, box) is False
