"""Phase 9 tests: goal validation."""

from __future__ import annotations

from common.messages import Goal, OccupancyGrid
from mapping.occupancy_grid import OccupancyGridMap
from navigation.goal_manager import validate_goal


def test_valid_goal_on_unexplored_cell() -> None:
    # Unknown (unexplored) cells are treated as traversable, matching A*'s
    # own passability rule -- only OCCUPIED blocks.
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    ok, reason = validate_goal(grid_map, Goal(x=5.0, y=5.0))
    assert ok is True
    assert reason == ""


def test_valid_goal_in_free_space() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    grid_map.set_cell(*grid_map.world_to_grid(5.0, 5.0), OccupancyGrid.FREE)
    ok, reason = validate_goal(grid_map, Goal(x=5.0, y=5.0))
    assert ok is True
    assert reason == ""


def test_goal_out_of_bounds_rejected() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    ok, reason = validate_goal(grid_map, Goal(x=100.0, y=100.0))
    assert ok is False
    assert "bounds" in reason


def test_goal_on_occupied_cell_rejected() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    grid_map.set_cell(*grid_map.world_to_grid(5.0, 5.0), OccupancyGrid.OCCUPIED)
    ok, reason = validate_goal(grid_map, Goal(x=5.0, y=5.0))
    assert ok is False
    assert "occupied" in reason


