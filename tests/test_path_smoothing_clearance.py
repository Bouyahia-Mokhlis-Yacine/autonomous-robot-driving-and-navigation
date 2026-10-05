"""Phase 15 tests: clearance-aware line-of-sight / path smoothing.

Fast, synthetic tests only -- no full closed-loop simulation needed
(that's covered separately in tests/test_phase15_clearance_scenario.py,
which reuses the Phase 12/13/14 deterministic scenario).
"""

from __future__ import annotations

from common.messages import OccupancyGrid, Path, PathPoint
from mapping.occupancy_grid import OccupancyGridMap
from planning.path_smoothing import has_line_of_sight, process_path, simplify_line_of_sight
from simulation.path_clearance import measure_path_clearance

F, U, O = OccupancyGrid.FREE, OccupancyGrid.UNKNOWN, OccupancyGrid.OCCUPIED


def make_grid(size: float = 10.0, resolution: float = 0.1) -> OccupancyGridMap:
    return OccupancyGridMap(resolution=resolution, width_m=size, height_m=size)


# --- Test 1: zero clearance is exactly the pre-Phase-15 behavior -----------

def test_zero_clearance_matches_default_los_behavior() -> None:
    grid_map = make_grid()
    grid_map.set_cell(*grid_map.world_to_grid(5.0, 5.1), O)  # just above the line below

    a, b = PathPoint(x=3.0, y=5.0), PathPoint(x=7.0, y=5.0)
    assert has_line_of_sight(grid_map, a, b) == has_line_of_sight(grid_map, a, b, clearance_m=0.0)


# --- Test 2 & 3: a near-miss segment is rejected only once clearance requires it

def test_segment_passing_within_clearance_of_occupied_cell_is_rejected() -> None:
    grid_map = make_grid(resolution=0.1)
    obstacle_cell = grid_map.world_to_grid(5.0, 5.5)  # ~0.5m from the y=5.0 line
    grid_map.set_cell(*obstacle_cell, O)

    a, b = PathPoint(x=3.0, y=5.0), PathPoint(x=7.0, y=5.0)

    # The segment does not touch the obstacle's cell at all (no collision).
    assert has_line_of_sight(grid_map, a, b, clearance_m=0.0) is True
    # A margin comfortably larger than the ~0.5m gap must reject it.
    assert has_line_of_sight(grid_map, a, b, clearance_m=0.8) is False
    # A margin comfortably smaller than the gap must still accept it.
    assert has_line_of_sight(grid_map, a, b, clearance_m=0.2) is True


# --- Test 4 & 5: clearance is metric, consistent across resolutions --------

def test_clearance_is_metric_not_cell_count() -> None:
    for resolution in (0.05, 0.1, 0.25):
        grid_map = make_grid(size=10.0, resolution=resolution)
        grid_map.set_cell(*grid_map.world_to_grid(5.0, 5.5), O)  # ~0.5m from the y=5.0 line

        a, b = PathPoint(x=3.0, y=5.0), PathPoint(x=7.0, y=5.0)
        # A margin comfortably below/above the ~0.5m gap must behave
        # identically regardless of grid resolution -- proving the
        # margin is interpreted in meters, not grid cells (at 0.25m
        # resolution, 0.2m would be less than one cell if this were
        # cell-counted instead).
        assert has_line_of_sight(grid_map, a, b, clearance_m=0.2) is True
        assert has_line_of_sight(grid_map, a, b, clearance_m=0.8) is False


# --- Test 6 & 7: LOS simplification respects clearance, falls back safely --

def test_simplify_line_of_sight_rejects_a_previously_accepted_tight_shortcut() -> None:
    grid_map = make_grid(resolution=0.1)
    # A corner obstacle the path must round, close enough that a direct
    # shortcut from the first to the last point would graze it.
    for row, col in [(50, c) for c in range(55, 70)]:
        grid_map.set_cell(row, col, O)

    waypoints = Path(points=[
        PathPoint(x=5.0, y=4.0), PathPoint(x=5.0, y=4.9), PathPoint(x=7.0, y=4.9), PathPoint(x=7.0, y=6.0),
    ])

    without_clearance = simplify_line_of_sight(waypoints, grid_map, clearance_m=0.0)
    with_clearance = simplify_line_of_sight(waypoints, grid_map, clearance_m=0.3)

    # Adding clearance must never produce MORE waypoints removed than
    # without it -- it can only keep the path closer to (or equal to)
    # the original, safer waypoint sequence.
    assert len(with_clearance.points) >= len(without_clearance.points)

    # Test 7: the result remains a valid, endpoint-preserving, collision-free path.
    assert with_clearance.points[0] == waypoints.points[0]
    assert with_clearance.points[-1] == waypoints.points[-1]
    for a, b in zip(with_clearance.points, with_clearance.points[1:]):
        assert has_line_of_sight(grid_map, a, b, clearance_m=0.0)  # still genuinely collision-free


# --- Test 8: increasing clearance never decreases achieved clearance -------

def test_increasing_clearance_config_increases_or_maintains_path_clearance() -> None:
    grid_map = make_grid(resolution=0.05)
    for row, col in [(50, c) for c in range(52, 70)]:
        grid_map.set_cell(row, col, O)

    raw = Path(points=[PathPoint(x=4.5, y=4.0), PathPoint(x=4.5, y=4.9), PathPoint(x=7.0, y=4.9), PathPoint(x=7.0, y=6.0)])

    path_low = process_path(raw, grid_map, simplify=True, smooth=False, clearance_m=0.0)
    path_high = process_path(raw, grid_map, simplify=True, smooth=False, clearance_m=0.3)

    clearance_low = measure_path_clearance(path_low, grid_map, sample_step_m=0.02)
    clearance_high = measure_path_clearance(path_high, grid_map, sample_step_m=0.02)

    assert clearance_high.min_clearance_m >= clearance_low.min_clearance_m - 1e-9


# --- Test 9: existing collision checking remains intact ---------------------

def test_collision_checking_still_absolute_regardless_of_clearance() -> None:
    grid_map = make_grid(resolution=0.1)
    grid_map.set_cell(*grid_map.world_to_grid(5.0, 5.0), O)
    a, b = PathPoint(x=4.0, y=5.0), PathPoint(x=6.0, y=5.0)  # straight through the obstacle

    for clearance in (0.0, 0.1, 0.5):
        assert has_line_of_sight(grid_map, a, b, clearance_m=clearance) is False


# --- Test 10: UNKNOWN cells remain freely traversable (unaffected) ---------

def test_unknown_cells_never_trigger_clearance_rejection() -> None:
    grid_map = make_grid()  # every cell defaults to UNKNOWN, nothing OCCUPIED anywhere
    a, b = PathPoint(x=1.0, y=1.0), PathPoint(x=9.0, y=9.0)
    assert has_line_of_sight(grid_map, a, b, clearance_m=0.5) is True
