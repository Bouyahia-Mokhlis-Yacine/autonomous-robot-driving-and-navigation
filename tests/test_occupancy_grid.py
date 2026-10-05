"""Phase 3 tests: occupancy grid transforms, cell access, scan integration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.config import LidarConfig
from common.messages import LaserScan, OccupancyGrid, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from simulation.environment import Environment
from simulation.obstacles import RectangleObstacle
from simulation.simulated_lidar import SimulatedLidar


def test_rejects_invalid_construction_params() -> None:
    with pytest.raises(ValueError):
        OccupancyGridMap(resolution=0.0, width_m=5.0, height_m=5.0)
    with pytest.raises(ValueError):
        OccupancyGridMap(resolution=0.5, width_m=0.0, height_m=5.0)


def test_grid_dimensions_computed_from_resolution() -> None:
    grid_map = OccupancyGridMap(resolution=0.5, width_m=5.0, height_m=2.5)
    assert grid_map.grid.width == 10
    assert grid_map.grid.height == 5
    assert np.all(grid_map.grid.cells == OccupancyGrid.UNKNOWN)


def test_world_to_grid_and_back_at_origin() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    row, col = grid_map.world_to_grid(2.3, 4.7)
    assert (row, col) == (4, 2)
    x, y = grid_map.grid_to_world(4, 2)
    assert x == pytest.approx(2.5)
    assert y == pytest.approx(4.5)


def test_world_to_grid_with_nonzero_origin() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0, origin_x=-5.0, origin_y=-5.0)
    row, col = grid_map.world_to_grid(0.0, 0.0)
    assert (row, col) == (5, 5)


def test_get_set_cell_and_bounds_checking() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=4.0, height_m=4.0)
    grid_map.set_cell(1, 2, OccupancyGrid.OCCUPIED)
    assert grid_map.get_cell(1, 2) == OccupancyGrid.OCCUPIED

    with pytest.raises(IndexError):
        grid_map.get_cell(100, 100)
    with pytest.raises(IndexError):
        grid_map.set_cell(-1, 0, OccupancyGrid.FREE)


def make_scan(range_value: float, range_max: float = 10.0) -> LaserScan:
    return LaserScan(
        timestamp=0.0,
        angle_min=0.0,
        angle_increment=1.0,  # single-ray scan; increment value is irrelevant
        ranges=[range_value],
        range_min=0.1,
        range_max=range_max,
    )


def test_update_from_scan_marks_traversed_free_and_endpoint_occupied() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    scan = make_scan(range_value=3.0, range_max=10.0)

    grid_map.update_from_scan(pose, scan)

    assert grid_map.get_cell(5, 5) == OccupancyGrid.FREE  # origin cell
    assert grid_map.get_cell(5, 6) == OccupancyGrid.FREE
    assert grid_map.get_cell(5, 7) == OccupancyGrid.FREE
    assert grid_map.get_cell(5, 8) == OccupancyGrid.OCCUPIED  # hit endpoint
    assert grid_map.get_cell(5, 9) == OccupancyGrid.UNKNOWN  # beyond the hit, untouched


def test_update_from_scan_max_range_ray_marks_free_not_occupied() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    pose = Pose2D(x=1.0, y=1.0, theta=0.0)
    scan = make_scan(range_value=5.0, range_max=5.0)  # no-detection: range == range_max

    grid_map.update_from_scan(pose, scan)

    assert grid_map.get_cell(1, 6) == OccupancyGrid.FREE


def test_update_from_scan_ignores_cells_outside_grid() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=5.0)
    pose = Pose2D(x=4.0, y=2.0, theta=0.0)
    scan = make_scan(range_value=8.0, range_max=10.0)  # endpoint far outside the grid

    grid_map.update_from_scan(pose, scan)  # must not raise

    assert grid_map.get_cell(2, 4) == OccupancyGrid.FREE  # in-bounds portion still updated


def test_unrelated_cells_remain_unknown_after_scan() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    scan = make_scan(range_value=3.0, range_max=10.0)

    grid_map.update_from_scan(pose, scan)

    assert grid_map.get_cell(0, 0) == OccupancyGrid.UNKNOWN
    assert grid_map.get_cell(9, 9) == OccupancyGrid.UNKNOWN


def test_full_room_scan_produces_free_occupied_and_unknown_regions() -> None:
    """Integration: Environment + SimulatedLidar + OccupancyGridMap.

    Demonstrates the Phase 3 acceptance criterion: a simulated room scan
    produces a map with free areas, an obstacle, and unknown areas.
    """
    env = Environment(
        width=10.0,
        height=10.0,
        robot_length=0.4,
        robot_width=0.4,
        robot_pose=Pose2D(x=5.0, y=5.0, theta=0.0),
    )
    env.add_static_obstacle(RectangleObstacle(x=8.0, y=5.0, width=1.0, height=1.0))

    lidar = SimulatedLidar(
        config=LidarConfig(
            range_min=0.1,
            range_max=15.0,
            angle_min=-math.pi,
            angle_max=math.pi,
            angular_resolution=math.radians(2.0),
            scan_rate=10.0,
        )
    )
    scan = lidar.scan(env, env.robot_pose, timestamp=0.0)

    grid_map = OccupancyGridMap(resolution=0.1, width_m=10.0, height_m=10.0)
    grid_map.update_from_scan(env.robot_pose, scan)

    cells = grid_map.grid.cells
    counts = {
        "free": int(np.sum(cells == OccupancyGrid.FREE)),
        "occupied": int(np.sum(cells == OccupancyGrid.OCCUPIED)),
        "unknown": int(np.sum(cells == OccupancyGrid.UNKNOWN)),
    }
    assert counts["free"] > 0
    assert counts["occupied"] > 0
    assert counts["unknown"] > 0

    # The obstacle's near face (x=7.5, facing the robot at x=5) is what
    # LiDAR actually observes -- rays can't see through it to its center.
    front_row, front_col = grid_map.world_to_grid(7.5, 5.0)
    neighborhood_occupied = any(
        grid_map.in_bounds(r, c) and grid_map.get_cell(r, c) == OccupancyGrid.OCCUPIED
        for r in range(front_row - 2, front_row + 3)
        for c in range(front_col - 2, front_col + 3)
    )
    assert neighborhood_occupied

    # The robot's own cell (open space) should be free.
    robot_row, robot_col = grid_map.world_to_grid(5.0, 5.0)
    assert grid_map.get_cell(robot_row, robot_col) == OccupancyGrid.FREE
