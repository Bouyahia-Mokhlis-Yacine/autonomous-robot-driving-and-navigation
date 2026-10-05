"""Phase 14 tests: path-clearance measurement (simulation.path_clearance).

Tests 1-4 (synthetic, fast). Tests 5-6 (real Phase 12/13 scenario data,
existing-behavior-unchanged) live in tests/test_phase14_path_diagnosis.py
alongside the rest of the expensive scenario instrumentation.
"""

from __future__ import annotations

import math

import pytest

from common.messages import OccupancyGrid, Path, PathPoint
from mapping.occupancy_grid import OccupancyGridMap
from simulation.path_clearance import measure_path_clearance

F, O = OccupancyGrid.FREE, OccupancyGrid.OCCUPIED


def make_grid(size: float = 10.0, resolution: float = 0.1) -> OccupancyGridMap:
    return OccupancyGridMap(resolution=resolution, width_m=size, height_m=size)


# --- Test 1: clearance on a simple known grid (no obstacle nearby) ---------

def test_clearance_with_no_nearby_obstacle_reports_search_radius() -> None:
    grid_map = make_grid()  # every cell defaults to UNKNOWN -- no OCCUPIED anywhere
    path = Path(points=[PathPoint(x=5.0, y=5.0)])

    result = measure_path_clearance(path, grid_map, search_radius_m=1.5)

    assert result.min_clearance_m == pytest.approx(1.5)
    assert result.mean_clearance_m == pytest.approx(1.5)
    assert result.num_samples == 1


# --- Test 2: clearance for a straight path parallel to a known wall --------

def test_clearance_for_straight_path_matches_known_distance() -> None:
    grid_map = make_grid(size=10.0, resolution=0.1)
    # A wall at x=6.0, spanning the full y-range -- one column of OCCUPIED cells.
    wall_col = grid_map.world_to_grid(6.0, 5.0)[1]
    for row in range(grid_map.grid.height):
        grid_map.set_cell(row, wall_col, O)

    # A straight path at x=5.5, running parallel to the wall -- constant
    # distance 0.5m (minus half a cell's worth of discretization) to it.
    path = Path(points=[PathPoint(x=5.5, y=2.0), PathPoint(x=5.5, y=8.0)])
    result = measure_path_clearance(path, grid_map, sample_step_m=0.1, search_radius_m=2.0)

    assert result.min_clearance_m == pytest.approx(0.5, abs=0.06)
    assert result.mean_clearance_m == pytest.approx(0.5, abs=0.06)
    assert result.path_length_m == pytest.approx(6.0)


# --- Test 3: clearance near (and through) an occupied cell -----------------

def test_clearance_detects_close_approach_to_occupied_cell() -> None:
    grid_map = make_grid(size=10.0, resolution=0.1)
    obstacle_row, obstacle_col = grid_map.world_to_grid(5.0, 5.0)
    grid_map.set_cell(obstacle_row, obstacle_col, O)

    # A path that passes directly through the obstacle's cell -- distance
    # to the nearest occupied cell center must reach (near) zero somewhere
    # along the path.
    path = Path(points=[PathPoint(x=4.5, y=5.0), PathPoint(x=5.5, y=5.0)])
    result = measure_path_clearance(path, grid_map, sample_step_m=0.05, search_radius_m=2.0)

    assert result.min_clearance_m < 0.06  # samples right at/near the obstacle center
    assert result.min_clearance_grid == (obstacle_row, obstacle_col)


def test_clearance_min_location_matches_known_closest_point() -> None:
    grid_map = make_grid(size=10.0, resolution=0.1)
    obstacle_row, obstacle_col = grid_map.world_to_grid(7.0, 5.0)
    grid_map.set_cell(obstacle_row, obstacle_col, O)
    obstacle_center = grid_map.grid_to_world(obstacle_row, obstacle_col)

    # A single straight segment whose closest approach (x=6.0) is
    # unambiguously ~1.0m from the obstacle's cell center.
    path = Path(points=[PathPoint(x=4.0, y=obstacle_center[1]), PathPoint(x=6.0, y=obstacle_center[1])])
    result = measure_path_clearance(path, grid_map, sample_step_m=0.1, search_radius_m=3.0)

    expected = obstacle_center[0] - 6.0
    assert result.min_clearance_m == pytest.approx(expected, abs=0.02)
    assert result.min_clearance_segment_index == 0
    assert result.min_clearance_world[0] == pytest.approx(6.0)


# --- Test 4: deterministic --------------------------------------------------

def test_clearance_measurement_is_deterministic() -> None:
    grid_map = make_grid(size=10.0, resolution=0.1)
    grid_map.set_cell(*grid_map.world_to_grid(5.0, 5.0), O)
    path = Path(points=[PathPoint(x=3.0, y=3.0), PathPoint(x=7.0, y=7.0)])

    result_a = measure_path_clearance(path, grid_map, sample_step_m=0.05)
    result_b = measure_path_clearance(path, grid_map, sample_step_m=0.05)

    assert result_a == result_b


def test_empty_path_reports_infinite_clearance() -> None:
    grid_map = make_grid()
    result = measure_path_clearance(Path(points=[]), grid_map)
    assert math.isinf(result.min_clearance_m)
    assert result.num_samples == 0
