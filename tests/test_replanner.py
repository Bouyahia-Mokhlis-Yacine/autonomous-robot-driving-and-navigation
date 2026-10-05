"""Phase 9/10 tests: path-obstruction detection, deviation, obstacle-ahead,
dynamic obstacle layer, planning-map combination, and the Replanner
trigger aggregator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.config import LidarConfig
from common.messages import LaserScan, OccupancyGrid, Path, PathPoint, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from navigation.replanner import (
    DynamicObstacleLayer,
    Replanner,
    ReplannerConfig,
    combine_planning_map,
    obstacle_ahead,
    path_deviation,
    path_is_blocked,
)
from simulation.environment import Environment
from simulation.obstacles import RectangleObstacle
from simulation.simulated_lidar import SimulatedLidar


def test_empty_path_is_never_blocked() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    assert path_is_blocked(Path(points=[]), grid_map) is False


def test_clear_path_is_not_blocked() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    path = Path(points=[PathPoint(0.5, 0.5), PathPoint(5.5, 5.5), PathPoint(9.5, 9.5)])
    assert path_is_blocked(path, grid_map) is False


def test_path_through_new_obstacle_is_blocked() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    path = Path(points=[PathPoint(0.5, 0.5), PathPoint(5.5, 0.5), PathPoint(9.5, 0.5)])
    grid_map.set_cell(0, 5, OccupancyGrid.OCCUPIED)  # sits exactly on the middle waypoint
    assert path_is_blocked(path, grid_map) is True


def test_path_crossing_obstacle_between_waypoints_is_blocked() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    path = Path(points=[PathPoint(0.5, 0.5), PathPoint(9.5, 0.5)])
    grid_map.set_cell(0, 5, OccupancyGrid.OCCUPIED)  # between the two waypoints, not on either
    assert path_is_blocked(path, grid_map) is True


def test_path_out_of_grid_bounds_is_blocked() -> None:
    grid_map = OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)
    path = Path(points=[PathPoint(0.5, 0.5), PathPoint(50.0, 50.0)])
    assert path_is_blocked(path, grid_map) is True


# ---------------------------------------------------------------- deviation ---


def test_path_deviation_zero_on_path() -> None:
    path = Path(points=[PathPoint(0.0, 0.0), PathPoint(10.0, 0.0)])
    assert path_deviation(Pose2D(x=5.0, y=0.0, theta=0.0), path) == pytest.approx(0.0)


def test_path_deviation_measures_perpendicular_distance() -> None:
    path = Path(points=[PathPoint(0.0, 0.0), PathPoint(10.0, 0.0)])
    assert path_deviation(Pose2D(x=5.0, y=3.0, theta=0.0), path) == pytest.approx(3.0)


def test_path_deviation_empty_path_is_zero() -> None:
    assert path_deviation(Pose2D(x=1.0, y=1.0, theta=0.0), Path(points=[])) == pytest.approx(0.0)


def test_path_deviation_single_point_path() -> None:
    path = Path(points=[PathPoint(3.0, 4.0)])
    assert path_deviation(Pose2D(x=0.0, y=0.0, theta=0.0), path) == pytest.approx(5.0)


# ------------------------------------------------------------ obstacle ahead ---


def test_obstacle_ahead_detects_occupied_cell_in_heading_direction() -> None:
    grid_map = OccupancyGridMap(resolution=0.5, width_m=10.0, height_m=10.0)
    grid_map.set_cell(*grid_map.world_to_grid(5.5, 5.0), OccupancyGrid.OCCUPIED)
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    assert obstacle_ahead(pose, grid_map, lookahead=1.0) is True


def test_obstacle_ahead_clear_when_nothing_within_lookahead() -> None:
    grid_map = OccupancyGridMap(resolution=0.5, width_m=10.0, height_m=10.0)
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    assert obstacle_ahead(pose, grid_map, lookahead=0.3) is False


def test_obstacle_ahead_ignores_obstacle_behind_heading() -> None:
    grid_map = OccupancyGridMap(resolution=0.5, width_m=10.0, height_m=10.0)
    grid_map.set_cell(*grid_map.world_to_grid(4.0, 5.0), OccupancyGrid.OCCUPIED)  # behind
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)  # facing +x
    assert obstacle_ahead(pose, grid_map, lookahead=1.0) is False


def test_obstacle_ahead_map_boundary_counts_as_obstacle() -> None:
    grid_map = OccupancyGridMap(resolution=0.5, width_m=10.0, height_m=10.0)
    pose = Pose2D(x=9.8, y=5.0, theta=0.0)  # near the right wall, facing it
    assert obstacle_ahead(pose, grid_map, lookahead=1.0) is True


# ---------------------------------------------------------- dynamic layer ---


def test_dynamic_obstacle_layer_reflects_only_latest_scan() -> None:
    env = Environment(width=10.0, height=10.0, robot_length=0.4, robot_width=0.4, robot_pose=Pose2D(5.0, 5.0, 0.0))
    obstacle = RectangleObstacle(x=6.0, y=5.0, width=0.5, height=0.5)
    env.add_static_obstacle(obstacle)  # "static" here just means fixed for this test's purposes
    lidar = SimulatedLidar(
        config=LidarConfig(
            range_min=0.1, range_max=10.0, angle_min=-math.pi, angle_max=math.pi,
            angular_resolution=math.radians(2.0), scan_rate=10.0,
        )
    )
    layer = DynamicObstacleLayer(resolution=0.1, width_m=10.0, height_m=10.0)

    scan = lidar.scan(env, env.robot_pose, timestamp=0.0)
    layer.update(env.robot_pose, scan)
    grid = layer.get_grid()
    assert np.any(grid.cells == OccupancyGrid.OCCUPIED)

    # Obstacle moves away; a fresh scan + update must not remember the old detection.
    env.static_obstacles.clear()
    scan2 = lidar.scan(env, env.robot_pose, timestamp=1.0)
    layer.update(env.robot_pose, scan2)
    grid2 = layer.get_grid()
    # near the obstacle's old near face
    row = int(math.floor((5.0 - grid2.origin_y) / grid2.resolution))
    col = int(math.floor((5.75 - grid2.origin_x) / grid2.resolution))
    assert grid2.cells[row, col] != OccupancyGrid.OCCUPIED


# ---------------------------------------------------------- map combination ---


def test_combine_planning_map_occupied_wins() -> None:
    static = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=5.0)
    dynamic = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=5.0)
    static.set_cell(1, 1, OccupancyGrid.FREE)
    dynamic.set_cell(1, 1, OccupancyGrid.OCCUPIED)

    combined = combine_planning_map(static.grid, dynamic.grid)
    assert combined.cells[1, 1] == OccupancyGrid.OCCUPIED


def test_combine_planning_map_free_beats_unknown() -> None:
    static = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=5.0)
    dynamic = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=5.0)
    static.set_cell(2, 2, OccupancyGrid.FREE)
    # dynamic cell stays UNKNOWN

    combined = combine_planning_map(static.grid, dynamic.grid)
    assert combined.cells[2, 2] == OccupancyGrid.FREE


def test_combine_planning_map_both_unknown_stays_unknown() -> None:
    static = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=5.0)
    dynamic = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=5.0)
    combined = combine_planning_map(static.grid, dynamic.grid)
    assert np.all(combined.cells == OccupancyGrid.UNKNOWN)


def test_combine_planning_map_static_map_not_mutated() -> None:
    static = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=5.0)
    dynamic = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=5.0)
    dynamic.set_cell(0, 0, OccupancyGrid.OCCUPIED)
    original = static.grid.cells.copy()

    combine_planning_map(static.grid, dynamic.grid)

    assert np.array_equal(static.grid.cells, original)  # dynamic obstacle never leaks into the static map


def test_combine_planning_map_rejects_mismatched_grids() -> None:
    static = OccupancyGridMap(resolution=1.0, width_m=5.0, height_m=5.0)
    dynamic = OccupancyGridMap(resolution=0.5, width_m=5.0, height_m=5.0)
    with pytest.raises(ValueError):
        combine_planning_map(static.grid, dynamic.grid)


# ---------------------------------------------------------------- Replanner ---


def make_open_grid() -> OccupancyGridMap:
    return OccupancyGridMap(resolution=1.0, width_m=10.0, height_m=10.0)


def test_replanner_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ReplannerConfig(replan_interval=0.0, deviation_threshold=1.0)
    with pytest.raises(ValueError):
        ReplannerConfig(replan_interval=1.0, deviation_threshold=0.0)


def test_replanner_no_trigger_when_everything_nominal() -> None:
    replanner = Replanner(ReplannerConfig(replan_interval=10.0, deviation_threshold=5.0))
    grid_map = make_open_grid()
    path = Path(points=[PathPoint(0.5, 0.5), PathPoint(9.5, 0.5)])
    should_replan, reason = replanner.should_replan(Pose2D(0.5, 0.5, 0.0), path, grid_map, dt=0.1)
    assert should_replan is False
    assert reason == ""


def test_replanner_triggers_on_blocked_path() -> None:
    replanner = Replanner(ReplannerConfig(replan_interval=10.0, deviation_threshold=5.0))
    grid_map = make_open_grid()
    grid_map.set_cell(0, 5, OccupancyGrid.OCCUPIED)
    path = Path(points=[PathPoint(0.5, 0.5), PathPoint(9.5, 0.5)])
    should_replan, reason = replanner.should_replan(Pose2D(0.5, 0.5, 0.0), path, grid_map, dt=0.1)
    assert should_replan is True
    assert "blocked" in reason


def test_replanner_triggers_on_deviation() -> None:
    replanner = Replanner(ReplannerConfig(replan_interval=10.0, deviation_threshold=1.0))
    grid_map = make_open_grid()
    path = Path(points=[PathPoint(0.5, 0.5), PathPoint(9.5, 0.5)])
    should_replan, reason = replanner.should_replan(Pose2D(0.5, 5.0, 0.0), path, grid_map, dt=0.1)
    assert should_replan is True
    assert "deviated" in reason


def test_replanner_triggers_periodically() -> None:
    replanner = Replanner(ReplannerConfig(replan_interval=0.5, deviation_threshold=5.0))
    grid_map = make_open_grid()
    path = Path(points=[PathPoint(0.5, 0.5), PathPoint(9.5, 0.5)])
    pose = Pose2D(0.5, 0.5, 0.0)

    should_replan, _ = replanner.should_replan(pose, path, grid_map, dt=0.3)
    assert should_replan is False

    should_replan, reason = replanner.should_replan(pose, path, grid_map, dt=0.3)  # total 0.6s >= 0.5s
    assert should_replan is True
    assert "periodic" in reason


def test_replanner_reset_timer_restarts_the_period() -> None:
    replanner = Replanner(ReplannerConfig(replan_interval=0.5, deviation_threshold=5.0))
    grid_map = make_open_grid()
    path = Path(points=[PathPoint(0.5, 0.5), PathPoint(9.5, 0.5)])
    pose = Pose2D(0.5, 0.5, 0.0)

    replanner.should_replan(pose, path, grid_map, dt=0.4)
    replanner.reset_timer()
    should_replan, _ = replanner.should_replan(pose, path, grid_map, dt=0.4)
    assert should_replan is False
