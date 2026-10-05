"""Phase 4 tests: SLAM.update/get_map contract, and a full sim integration
where the robot moves around a room while SLAM tracks pose and builds a map.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.config import LidarConfig
from common.messages import OccupancyGrid, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from simulation.environment import Environment
from simulation.obstacles import RectangleObstacle
from simulation.simulated_lidar import SimulatedLidar
from slam.backend import GroundTruthBackend
from slam.slam import SLAM


def build_scene() -> tuple[Environment, SimulatedLidar]:
    env = Environment(
        width=10.0,
        height=10.0,
        robot_length=0.4,
        robot_width=0.4,
        robot_pose=Pose2D(x=2.0, y=2.0, theta=0.0),
    )
    env.add_static_obstacle(RectangleObstacle(x=7.0, y=7.0, width=1.0, height=1.0))
    lidar = SimulatedLidar(
        config=LidarConfig(
            range_min=0.1,
            range_max=15.0,
            angle_min=-math.pi,
            angle_max=math.pi,
            angular_resolution=math.radians(3.0),
            scan_rate=10.0,
        )
    )
    return env, lidar


def test_get_pose_is_none_before_first_update() -> None:
    grid_map = OccupancyGridMap(resolution=0.1, width_m=10.0, height_m=10.0)
    env, _ = build_scene()
    backend = GroundTruthBackend(pose_provider=lambda: env.robot_pose)
    slam = SLAM(backend=backend, occupancy_grid_map=grid_map)

    assert slam.get_pose() is None


def test_update_returns_backend_pose_and_updates_map() -> None:
    grid_map = OccupancyGridMap(resolution=0.1, width_m=10.0, height_m=10.0)
    env, lidar = build_scene()
    backend = GroundTruthBackend(pose_provider=lambda: env.robot_pose)
    slam = SLAM(backend=backend, occupancy_grid_map=grid_map)

    scan = lidar.scan(env, env.robot_pose, timestamp=0.0)
    pose = slam.update(scan)

    assert pose == env.robot_pose
    assert slam.get_pose() == env.robot_pose
    assert np.any(slam.get_map().cells != OccupancyGrid.UNKNOWN)


def test_get_map_reflects_grid_message() -> None:
    grid_map = OccupancyGridMap(resolution=0.1, width_m=10.0, height_m=10.0)
    env, lidar = build_scene()
    backend = GroundTruthBackend(pose_provider=lambda: env.robot_pose)
    slam = SLAM(backend=backend, occupancy_grid_map=grid_map)

    scan = lidar.scan(env, env.robot_pose, timestamp=0.0)
    slam.update(scan)

    assert slam.get_map() is grid_map.grid


def test_robot_trajectory_and_growing_map_over_multiple_steps() -> None:
    """Acceptance criterion: the robot moves around a room, SLAM estimates
    its trajectory (via the ground-truth backend) while the map grows."""
    grid_map = OccupancyGridMap(resolution=0.1, width_m=10.0, height_m=10.0)
    env, lidar = build_scene()
    backend = GroundTruthBackend(pose_provider=lambda: env.robot_pose)
    slam = SLAM(backend=backend, occupancy_grid_map=grid_map)

    waypoints = [(3.0, 2.0), (4.0, 2.0), (4.0, 4.0), (5.0, 5.0)]
    trajectory: list[Pose2D] = []
    known_cell_counts: list[int] = []

    for wx, wy in waypoints:
        moved = env.try_move_robot(Pose2D(x=wx, y=wy, theta=0.0))
        assert moved  # all waypoints are collision-free in this scene

        scan = lidar.scan(env, env.robot_pose, timestamp=float(len(trajectory)))
        pose = slam.update(scan)
        trajectory.append(pose)

        known = int(np.sum(grid_map.grid.cells != OccupancyGrid.UNKNOWN))
        known_cell_counts.append(known)

    # The reported trajectory matches the true simulator path exactly
    # (ground-truth backend), demonstrating pose tracking across steps.
    for pose, (wx, wy) in zip(trajectory, waypoints):
        assert pose.x == pytest.approx(wx)
        assert pose.y == pytest.approx(wy)

    # The map accumulates knowledge over time rather than resetting.
    assert known_cell_counts == sorted(known_cell_counts)
    assert known_cell_counts[-1] > known_cell_counts[0]

    # The known obstacle is present in the final map near its true location.
    row, col = grid_map.world_to_grid(6.5, 6.5)  # obstacle's near corner (facing the robot)
    neighborhood_occupied = any(
        grid_map.in_bounds(r, c) and grid_map.get_cell(r, c) == OccupancyGrid.OCCUPIED
        for r in range(row - 3, row + 4)
        for c in range(col - 3, col + 4)
    )
    assert neighborhood_occupied
