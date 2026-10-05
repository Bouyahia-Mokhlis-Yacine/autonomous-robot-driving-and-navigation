"""Phase 2 tests: ray-casting simulated LiDAR against an Environment."""

from __future__ import annotations

import math

import pytest

from common.config import LidarConfig
from common.messages import Pose2D
from simulation.environment import Environment
from simulation.obstacles import RectangleObstacle
from simulation.simulated_lidar import SimulatedLidar


def make_lidar(**overrides) -> SimulatedLidar:
    defaults = dict(
        range_min=0.1,
        range_max=20.0,
        angle_min=0.0,
        angle_max=math.pi / 2.0,
        angular_resolution=math.pi / 2.0,
        scan_rate=10.0,
    )
    defaults.update(overrides)
    return SimulatedLidar(config=LidarConfig(**defaults))


def make_room(width: float = 10.0, height: float = 10.0, pose: Pose2D | None = None) -> Environment:
    return Environment(
        width=width,
        height=height,
        robot_length=0.5,
        robot_width=0.4,
        robot_pose=pose if pose is not None else Pose2D(x=width / 2.0, y=height / 2.0, theta=0.0),
    )


def test_lidar_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        make_lidar(range_min=-1.0)
    with pytest.raises(ValueError):
        make_lidar(range_max=0.05)  # <= range_min
    with pytest.raises(ValueError):
        make_lidar(angular_resolution=0.0)
    with pytest.raises(ValueError):
        make_lidar(angle_max=0.0, angle_min=0.0)
    with pytest.raises(ValueError):
        make_lidar(scan_rate=0.0)


def test_ray_in_empty_room_hits_far_wall() -> None:
    env = make_room(width=10.0, height=10.0)
    lidar = make_lidar()  # rays at 0 rad and pi/2 rad, robot facing +x (theta=0)
    scan = lidar.scan(env, env.robot_pose, timestamp=1.0)

    # ray 0 -> +x direction: wall at x=10, robot at x=5 -> distance 5
    assert scan.ranges[0] == pytest.approx(5.0)
    # ray 1 (pi/2) -> +y direction: wall at y=10, robot at y=5 -> distance 5
    assert scan.ranges[1] == pytest.approx(5.0)


def test_ray_hits_obstacle_before_wall() -> None:
    env = make_room(width=10.0, height=10.0)
    env.add_static_obstacle(RectangleObstacle(x=7.0, y=5.0, width=1.0, height=1.0))
    lidar = make_lidar()
    scan = lidar.scan(env, env.robot_pose, timestamp=0.0)

    assert scan.ranges[0] == pytest.approx(1.5)  # obstacle near edge at x=6.5


def test_obstacle_behind_ray_does_not_affect_hit() -> None:
    env = make_room(width=10.0, height=10.0)
    env.add_static_obstacle(RectangleObstacle(x=3.0, y=5.0, width=1.0, height=1.0))
    lidar = make_lidar()
    scan = lidar.scan(env, env.robot_pose, timestamp=0.0)

    # ray 0 points +x, away from the obstacle at x=3 -> should still hit the far wall
    assert scan.ranges[0] == pytest.approx(5.0)


def test_range_beyond_range_max_reports_range_max() -> None:
    env = make_room(width=200.0, height=200.0, pose=Pose2D(x=100.0, y=100.0, theta=0.0))
    lidar = make_lidar(range_max=2.0, range_min=0.1)
    scan = lidar.scan(env, env.robot_pose, timestamp=0.0)

    assert scan.ranges[0] == pytest.approx(2.0)
    assert scan.ranges[0] == scan.range_max


def test_scan_metadata_matches_config() -> None:
    lidar = make_lidar(
        angle_min=-math.pi, angle_max=math.pi, angular_resolution=math.pi / 4.0
    )
    env = make_room()
    scan = lidar.scan(env, env.robot_pose, timestamp=5.5)

    assert scan.timestamp == 5.5
    assert scan.angle_min == pytest.approx(-math.pi)
    assert scan.angle_increment == pytest.approx(math.pi / 4.0)
    expected_num_rays = int(round((math.pi - (-math.pi)) / (math.pi / 4.0))) + 1
    assert scan.num_rays == expected_num_rays
    assert len(scan.ranges) == expected_num_rays
    assert scan.range_min == pytest.approx(0.1)
    assert scan.range_max == pytest.approx(20.0)


def test_scan_respects_robot_heading() -> None:
    env = make_room(width=10.0, height=10.0, pose=Pose2D(x=5.0, y=5.0, theta=math.pi / 2.0))
    lidar = make_lidar()  # local rays at 0 and pi/2
    scan = lidar.scan(env, env.robot_pose, timestamp=0.0)

    # Robot faces +y (theta=pi/2): local ray 0 -> world +y -> wall at y=10, distance 5
    assert scan.ranges[0] == pytest.approx(5.0)
    # local ray pi/2 -> world angle pi -> -x direction -> wall at x=0, distance 5
    assert scan.ranges[1] == pytest.approx(5.0)


def test_symmetric_room_scan_matches_geometry() -> None:
    """Demo/visualization-style check: a square room's scan is symmetric."""
    env = make_room(width=10.0, height=10.0)
    lidar = make_lidar(
        angle_min=0.0, angle_max=3.0 * math.pi / 2.0, angular_resolution=math.pi / 2.0
    )
    scan = lidar.scan(env, env.robot_pose, timestamp=0.0)

    assert scan.num_rays == 4
    for r in scan.ranges:
        assert r == pytest.approx(5.0)


def test_asymmetric_room_scan_matches_geometry() -> None:
    """A non-square room must produce different ranges per cardinal direction."""
    env = make_room(width=20.0, height=6.0, pose=Pose2D(x=10.0, y=3.0, theta=0.0))
    lidar = make_lidar(
        angle_min=0.0, angle_max=3.0 * math.pi / 2.0, angular_resolution=math.pi / 2.0
    )
    scan = lidar.scan(env, env.robot_pose, timestamp=0.0)

    # order: +x, +y, -x, -y
    assert scan.ranges[0] == pytest.approx(10.0)  # to right wall
    assert scan.ranges[1] == pytest.approx(3.0)   # to top wall
    assert scan.ranges[2] == pytest.approx(10.0)  # to left wall
    assert scan.ranges[3] == pytest.approx(3.0)   # to bottom wall
