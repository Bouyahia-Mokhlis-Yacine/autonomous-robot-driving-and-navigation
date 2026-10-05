"""Phase 6 tests: grid<->world path conversion and the GlobalPlanner wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from common.messages import Goal, OccupancyGrid, Pose2D
from mapping.inflation import inflate_occupancy_grid
from mapping.occupancy_grid import OccupancyGridMap
from planning.global_planner import GlobalPlanner
from planning.path import grid_path_to_world_path


def test_grid_path_to_world_path_uses_cell_centers() -> None:
    grid_map = OccupancyGridMap(resolution=0.5, width_m=5.0, height_m=5.0, origin_x=0.0, origin_y=0.0)
    path = grid_path_to_world_path(grid_map, [(0, 0), (0, 1), (1, 1)])

    assert len(path.points) == 3
    assert path.points[0].x == pytest.approx(0.25)
    assert path.points[0].y == pytest.approx(0.25)
    assert path.points[1].x == pytest.approx(0.75)
    assert path.points[1].y == pytest.approx(0.25)
    assert path.points[2].x == pytest.approx(0.75)
    assert path.points[2].y == pytest.approx(0.75)


def test_grid_path_to_world_path_empty_list() -> None:
    grid_map = OccupancyGridMap(resolution=0.5, width_m=5.0, height_m=5.0)
    path = grid_path_to_world_path(grid_map, [])
    assert path.is_empty()


def test_global_planner_direct_path_in_empty_room() -> None:
    """Scenario A: empty room -> direct navigation."""
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    planner = GlobalPlanner(allow_diagonal=True)

    path = planner.plan(grid_map, start=Pose2D(x=0.5, y=0.5, theta=0.0), goal=Goal(x=8.5, y=8.5))

    assert path is not None
    assert not path.is_empty()
    assert path.points[0].x == pytest.approx(0.5)
    assert path.points[-1].x == pytest.approx(8.5)


def test_global_planner_routes_around_wall() -> None:
    """Scenario B: wall -> route around obstacle."""
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    # A wall spanning the whole width except a gap at column 8, at row 5.
    for col in range(10):
        if col != 8:
            grid_map.set_cell(5, col, OccupancyGrid.OCCUPIED)

    planner = GlobalPlanner(allow_diagonal=True)
    path = planner.plan(grid_map, start=Pose2D(x=0.5, y=0.5, theta=0.0), goal=Goal(x=0.5, y=9.5))

    assert path is not None
    passed_through_gap = any(
        grid_map.world_to_grid(p.x, p.y) == (5, 8) for p in path.points
    )
    assert passed_through_gap


def test_global_planner_rejects_occupied_goal() -> None:
    """Scenario E: invalid/unreachable goal -> reject safely (no exception)."""
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    grid_map.set_cell(*grid_map.world_to_grid(8.5, 8.5), OccupancyGrid.OCCUPIED)

    planner = GlobalPlanner(allow_diagonal=True)
    path = planner.plan(grid_map, start=Pose2D(x=0.5, y=0.5, theta=0.0), goal=Goal(x=8.5, y=8.5))

    assert path is None


def test_global_planner_rejects_fully_blocked_route() -> None:
    """Scenario F: system blocked -> no path (None), not an exception."""
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    for col in range(10):
        grid_map.set_cell(5, col, OccupancyGrid.OCCUPIED)  # solid wall, no gap

    planner = GlobalPlanner(allow_diagonal=True)
    path = planner.plan(grid_map, start=Pose2D(x=0.5, y=0.5, theta=0.0), goal=Goal(x=0.5, y=9.5))

    assert path is None


def test_global_planner_narrow_passage_rejected_after_inflation() -> None:
    """Scenario C: narrow passage smaller than the safety footprint is blocked
    once the raw grid is inflated -- GlobalPlanner itself is inflation-agnostic,
    it just plans over whatever grid it is given."""
    grid_map = OccupancyGridMap(resolution=0.1, width_m=10.0, height_m=10.0)
    for col in range(100):
        if 49 <= col <= 51:  # a 0.3 m gap, narrower than a realistic robot + margin
            continue
        grid_map.set_cell(50, col, OccupancyGrid.OCCUPIED)

    inflated = inflate_occupancy_grid(grid_map.grid, inflation_radius=0.3)
    inflated_map = OccupancyGridMap(resolution=0.1, width_m=10.0, height_m=10.0)
    inflated_map.grid = inflated

    planner = GlobalPlanner(allow_diagonal=True)
    path = planner.plan(inflated_map, start=Pose2D(x=0.5, y=0.5, theta=0.0), goal=Goal(x=0.5, y=9.5))

    assert path is None  # the inflated gap is fully closed
