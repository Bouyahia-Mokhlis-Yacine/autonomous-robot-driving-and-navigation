"""Phase 0 tests: package imports, core dataclasses, config loading, logging."""

from __future__ import annotations

import numpy as np
import pytest

from common.config import load_navigation_config, load_robot_config
from common.geometry import angle_diff, clamp, distance, normalize, wrap_angle
from common.logger import get_logger
from common.messages import (
    Goal,
    LaserScan,
    OccupancyGrid,
    Path,
    PathPoint,
    Pose2D,
    VelocityCommand,
)


def test_laser_scan_construction_and_validation() -> None:
    scan = LaserScan(
        timestamp=1.0,
        angle_min=-1.0,
        angle_increment=0.5,
        ranges=[1.0, 2.0, 3.0],
        range_min=0.1,
        range_max=5.0,
    )
    assert scan.num_rays == 3
    assert scan.angle_at(0) == pytest.approx(-1.0)
    assert scan.angle_at(2) == pytest.approx(0.0)


def test_laser_scan_rejects_invalid_range_bounds() -> None:
    with pytest.raises(ValueError):
        LaserScan(
            timestamp=0.0,
            angle_min=0.0,
            angle_increment=0.1,
            ranges=[1.0],
            range_min=5.0,
            range_max=1.0,
        )


def test_pose2d_as_tuple() -> None:
    pose = Pose2D(x=1.0, y=2.0, theta=0.5, timestamp=0.0)
    assert pose.as_tuple() == (1.0, 2.0, 0.5)


def test_goal_and_path_point() -> None:
    goal = Goal(x=3.0, y=4.0)
    point = PathPoint(x=goal.x, y=goal.y)
    assert point.x == 3.0 and point.y == 4.0


def test_path_container() -> None:
    path = Path()
    assert path.is_empty()
    path.points.append(PathPoint(0.0, 0.0))
    path.points.append(PathPoint(1.0, 0.0))
    assert len(path) == 2
    assert not path.is_empty()


def test_velocity_command() -> None:
    cmd = VelocityCommand(linear=0.5, angular=-0.2)
    assert cmd.linear == 0.5
    assert cmd.angular == -0.2


def test_occupancy_grid_shape_validation() -> None:
    cells = np.full((10, 20), OccupancyGrid.UNKNOWN, dtype=np.int8)
    grid = OccupancyGrid(
        resolution=0.05, width=20, height=10, origin_x=0.0, origin_y=0.0, cells=cells
    )
    assert grid.cells.shape == (10, 20)

    with pytest.raises(ValueError):
        OccupancyGrid(
            resolution=0.05,
            width=20,
            height=10,
            origin_x=0.0,
            origin_y=0.0,
            cells=np.zeros((5, 5), dtype=np.int8),
        )


def test_geometry_helpers() -> None:
    assert distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)
    assert normalize(0.0, 0.0) == (0.0, 0.0)
    nx, ny = normalize(3.0, 4.0)
    assert nx == pytest.approx(0.6)
    assert ny == pytest.approx(0.8)
    assert clamp(5.0, 0.0, 1.0) == 1.0
    assert clamp(-5.0, 0.0, 1.0) == 0.0
    assert wrap_angle(3.5 * 3.141592653589793) == pytest.approx(
        -0.5 * 3.141592653589793
    )
    assert angle_diff(0.1, -0.1) == pytest.approx(0.2)


def test_load_robot_config() -> None:
    config = load_robot_config()
    assert config.robot.wheel_base == pytest.approx(0.30)
    assert config.safety.inflation_radius == pytest.approx(0.25)
    assert config.limits.max_linear_velocity == pytest.approx(0.50)


def test_load_navigation_config() -> None:
    config = load_navigation_config()
    assert config.lidar.range_max == pytest.approx(8.0)
    assert config.mapping.resolution == pytest.approx(0.05)
    assert config.planning.allow_diagonal is True


def test_get_logger_returns_namespaced_logger() -> None:
    logger = get_logger("tests.phase0")
    assert logger.name == "autonomous_navigation.tests.phase0"
