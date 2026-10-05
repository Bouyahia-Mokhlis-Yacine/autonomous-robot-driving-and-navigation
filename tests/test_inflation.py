"""Phase 5 tests: obstacle inflation building a planning map from the raw grid."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.messages import OccupancyGrid
from mapping.inflation import compute_inflation_radius, inflate_from_footprint, inflate_occupancy_grid


def make_grid(size: int = 11, resolution: float = 1.0) -> OccupancyGrid:
    cells = np.full((size, size), OccupancyGrid.FREE, dtype=np.int8)
    return OccupancyGrid(resolution=resolution, width=size, height=size, origin_x=0.0, origin_y=0.0, cells=cells)


def test_compute_inflation_radius() -> None:
    radius = compute_inflation_radius(robot_width=0.4, robot_length=0.5, safety_margin=0.1)
    expected = math.hypot(0.4, 0.5) / 2.0 + 0.1
    assert radius == pytest.approx(expected)


def test_compute_inflation_radius_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        compute_inflation_radius(0.0, 0.5, 0.1)
    with pytest.raises(ValueError):
        compute_inflation_radius(0.4, 0.5, -0.1)


def test_zero_radius_returns_unchanged_copy() -> None:
    grid = make_grid()
    grid.cells[5, 5] = OccupancyGrid.OCCUPIED

    inflated = inflate_occupancy_grid(grid, inflation_radius=0.0)

    assert np.array_equal(inflated.cells, grid.cells)
    assert inflated.cells is not grid.cells  # a copy, not the same array


def test_inflation_does_not_mutate_input_grid() -> None:
    grid = make_grid()
    grid.cells[5, 5] = OccupancyGrid.OCCUPIED
    original = grid.cells.copy()

    inflate_occupancy_grid(grid, inflation_radius=1.5)

    assert np.array_equal(grid.cells, original)


def test_inflation_marks_exact_neighborhood_for_single_obstacle() -> None:
    grid = make_grid(size=11, resolution=1.0)
    grid.cells[5, 5] = OccupancyGrid.OCCUPIED

    inflated = inflate_occupancy_grid(grid, inflation_radius=1.5)

    expected_occupied = {(r, c) for r in range(4, 7) for c in range(4, 7)}
    actual_occupied = {
        (r, c)
        for r in range(inflated.height)
        for c in range(inflated.width)
        if inflated.cells[r, c] == OccupancyGrid.OCCUPIED
    }
    assert actual_occupied == expected_occupied


def test_inflation_grows_with_larger_radius() -> None:
    grid = make_grid(size=21, resolution=1.0)
    grid.cells[10, 10] = OccupancyGrid.OCCUPIED

    small = inflate_occupancy_grid(grid, inflation_radius=1.0)
    large = inflate_occupancy_grid(grid, inflation_radius=3.0)

    small_count = int(np.sum(small.cells == OccupancyGrid.OCCUPIED))
    large_count = int(np.sum(large.cells == OccupancyGrid.OCCUPIED))
    assert large_count > small_count


def test_inflation_near_grid_edge_does_not_crash_or_wrap() -> None:
    grid = make_grid(size=5, resolution=1.0)
    grid.cells[0, 0] = OccupancyGrid.OCCUPIED

    inflated = inflate_occupancy_grid(grid, inflation_radius=2.0)  # radius exceeds grid bounds

    assert inflated.cells.shape == (5, 5)
    # No wraparound: the far corner must remain untouched.
    assert inflated.cells[4, 4] == OccupancyGrid.FREE


def test_inflation_preserves_far_away_unknown_cells() -> None:
    size = 21
    cells = np.full((size, size), OccupancyGrid.UNKNOWN, dtype=np.int8)
    cells[10, 10] = OccupancyGrid.OCCUPIED
    grid = OccupancyGrid(resolution=1.0, width=size, height=size, origin_x=0.0, origin_y=0.0, cells=cells)

    inflated = inflate_occupancy_grid(grid, inflation_radius=1.0)

    assert inflated.cells[0, 0] == OccupancyGrid.UNKNOWN


def test_inflate_from_footprint_matches_manual_computation() -> None:
    grid = make_grid()
    grid.cells[5, 5] = OccupancyGrid.OCCUPIED
    radius = compute_inflation_radius(0.4, 0.5, 0.1)

    via_wrapper = inflate_from_footprint(grid, robot_width=0.4, robot_length=0.5, safety_margin=0.1)
    via_manual = inflate_occupancy_grid(grid, inflation_radius=radius)

    assert np.array_equal(via_wrapper.cells, via_manual.cells)


def test_inflation_rejects_negative_radius() -> None:
    grid = make_grid()
    with pytest.raises(ValueError):
        inflate_occupancy_grid(grid, inflation_radius=-1.0)
