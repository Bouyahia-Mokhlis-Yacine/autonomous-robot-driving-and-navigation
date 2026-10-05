"""Phase 1 tests: slam.scan_preprocessing (LaserScan -> local Cartesian
points, split into hits vs. max-range bearings, invalid ranges dropped)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.messages import LaserScan
from slam.scan_preprocessing import PreprocessedScan, preprocess_scan

RANGE_MIN = 0.15
RANGE_MAX = 8.0


def make_scan(ranges: list[float], angle_min: float = 0.0, angle_increment: float = 0.0) -> LaserScan:
    return LaserScan(
        timestamp=0.0,
        angle_min=angle_min,
        angle_increment=angle_increment,
        ranges=ranges,
        range_min=RANGE_MIN,
        range_max=RANGE_MAX,
    )


def test_known_single_ray_cartesian_conversion() -> None:
    scan = make_scan([2.0], angle_min=0.0, angle_increment=1.0)
    result = preprocess_scan(scan)
    assert result.hit_points_local.shape == (1, 2)
    assert result.hit_points_local[0, 0] == pytest.approx(2.0)
    assert result.hit_points_local[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert result.max_range_bearings.shape == (0,)


def test_multiple_rays_at_known_angles() -> None:
    # angles: -pi/2 (down/-Y), 0 (+X), +pi/2 (up/+Y), all at range 1.0
    scan = make_scan([1.0, 1.0, 1.0], angle_min=-math.pi / 2.0, angle_increment=math.pi / 2.0)
    result = preprocess_scan(scan)
    assert result.hit_points_local.shape == (3, 2)
    np.testing.assert_allclose(result.hit_points_local[0], [0.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(result.hit_points_local[1], [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(result.hit_points_local[2], [0.0, 1.0], atol=1e-9)


def test_max_range_rays_excluded_from_hit_points() -> None:
    scan = make_scan([1.0, RANGE_MAX, 2.0, RANGE_MAX], angle_min=0.0, angle_increment=0.5)
    result = preprocess_scan(scan)
    # Only indices 0 and 2 are real hits.
    assert result.hit_points_local.shape == (2, 2)
    ranges_recovered = np.linalg.norm(result.hit_points_local, axis=1)
    np.testing.assert_allclose(sorted(ranges_recovered), [1.0, 2.0])


def test_max_range_bearings_preserved() -> None:
    scan = make_scan([1.0, RANGE_MAX, 2.0, RANGE_MAX], angle_min=0.0, angle_increment=0.5)
    result = preprocess_scan(scan)
    # Indices 1 and 3 -> angles 0.5 and 1.5.
    assert result.max_range_bearings.shape == (2,)
    np.testing.assert_allclose(sorted(result.max_range_bearings), [0.5, 1.5])


def test_invalid_ranges_excluded() -> None:
    too_close = RANGE_MIN / 2.0
    scan = make_scan([too_close, 1.0, too_close, RANGE_MAX], angle_min=0.0, angle_increment=1.0)
    result = preprocess_scan(scan)
    # Only index 1 is a real hit; index 3 is max-range; indices 0 and 2
    # (below range_min) appear in neither output.
    assert result.hit_points_local.shape == (1, 2)
    assert result.max_range_bearings.shape == (1,)


def test_empty_scan() -> None:
    scan = make_scan([], angle_min=0.0, angle_increment=0.0)
    result = preprocess_scan(scan)
    assert result.hit_points_local.shape == (0, 2)
    assert result.max_range_bearings.shape == (0,)


def test_no_hit_scan_all_max_range() -> None:
    scan = make_scan([RANGE_MAX] * 5, angle_min=0.0, angle_increment=0.1)
    result = preprocess_scan(scan)
    assert result.hit_points_local.shape == (0, 2)
    assert result.max_range_bearings.shape == (5,)


def test_angle_convention_zero_is_plus_x_increasing_is_counterclockwise() -> None:
    scan = make_scan([1.0, 1.0, 1.0], angle_min=0.0, angle_increment=math.pi / 2.0)
    result = preprocess_scan(scan)
    # angle 0 -> +X, angle pi/2 -> +Y, angle pi -> -X (counterclockwise sweep).
    np.testing.assert_allclose(result.hit_points_local[0], [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(result.hit_points_local[1], [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(result.hit_points_local[2], [-1.0, 0.0], atol=1e-9)


def test_output_shapes_and_types() -> None:
    scan = make_scan([1.0, RANGE_MAX, RANGE_MIN / 2.0, 2.0], angle_min=0.0, angle_increment=0.3)
    result = preprocess_scan(scan)
    assert isinstance(result, PreprocessedScan)
    assert isinstance(result.hit_points_local, np.ndarray)
    assert isinstance(result.max_range_bearings, np.ndarray)
    assert result.hit_points_local.dtype == np.float64
    assert result.max_range_bearings.dtype == np.float64
    assert result.hit_points_local.ndim == 2
    assert result.hit_points_local.shape[1] == 2
    assert result.max_range_bearings.ndim == 1


def test_realistic_synthetic_361_ray_scan() -> None:
    angle_min = -math.pi
    angle_increment = math.radians(1.0)
    num_rays = 361
    ranges = []
    expected_hits = 0
    expected_max_range = 0
    expected_invalid = 0
    for i in range(num_rays):
        # A repeating pattern exercising all three classifications.
        m = i % 3
        if m == 0:
            ranges.append(3.0 + 0.1 * math.sin(i * 0.05))  # real hit
            expected_hits += 1
        elif m == 1:
            ranges.append(RANGE_MAX)  # no detection
            expected_max_range += 1
        else:
            ranges.append(RANGE_MIN / 2.0)  # invalid, too close
            expected_invalid += 1

    scan = make_scan(ranges, angle_min=angle_min, angle_increment=angle_increment)
    result = preprocess_scan(scan)

    assert result.hit_points_local.shape == (expected_hits, 2)
    assert result.max_range_bearings.shape == (expected_max_range,)
    assert expected_hits + expected_max_range + expected_invalid == num_rays

    # Spot-check: every hit point's recovered range is within the real-hit band.
    recovered_ranges = np.linalg.norm(result.hit_points_local, axis=1)
    assert np.all(recovered_ranges >= RANGE_MIN)
    assert np.all(recovered_ranges < RANGE_MAX)

    # Spot-check: every max-range bearing falls within the scan's angular span.
    assert np.all(result.max_range_bearings >= angle_min - 1e-9)
    assert np.all(result.max_range_bearings <= angle_min + (num_rays - 1) * angle_increment + 1e-9)
