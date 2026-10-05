"""LiDAR mounted off the footprint centre (robot.yaml robot.lidar_x/lidar_y;
8 cm toward the back on the real robot, 2026-09-30): SLAM tracks the sensor,
navigation and safety use the footprint centre."""

from __future__ import annotations

import math

import pytest

from common.config import load_robot_config
from common.geometry import offset_pose
from common.messages import LaserScan, Pose2D
from control.path_follower import MotionMode
from navigation.motion_safety import MotionSafety, MotionSafetyConfig
from navigation.world_model_bridge import apply_accepted_pose, robot_pose_from_lidar
from robot.sensor_only_robot import SensorOnlyRobot
from slam.motion_model import CommandHistory, MountedMotionPrior, mounted_prior
from slam.scan_matching_pipeline import PipelineResult
from common.messages import VelocityCommand

MOUNT = (-0.08, 0.0)


def test_config_has_the_measured_mount() -> None:
    r = load_robot_config().robot
    assert (r.lidar_x, r.lidar_y) == MOUNT


def test_offset_pose_round_trip_and_direction() -> None:
    robot = Pose2D(1.0, 2.0, math.pi / 2)                  # facing +y
    lidar = offset_pose(robot, *MOUNT)
    assert (lidar.x, lidar.y, lidar.theta) == pytest.approx((1.0, 1.92, math.pi / 2))   # 8 cm behind = -y
    back = robot_pose_from_lidar(lidar, MOUNT)
    assert (back.x, back.y) == pytest.approx((1.0, 2.0))


def test_robot_receives_its_footprint_centre_not_the_lidar_pose() -> None:
    robot = SensorOnlyRobot(initial_pose=Pose2D(0.0, 0.0, 0.0))
    result = PipelineResult(pose=Pose2D(0.5, 0.0, 0.0), accepted=True, localized=True,
                            consecutive_failures=0, registration_result=None, rejection_reason=None)
    assert apply_accepted_pose(robot, result, MOUNT)
    assert (robot.get_pose().x, robot.get_pose().y) == pytest.approx((0.58, 0.0))   # centre is 8 cm ahead
    rejected = PipelineResult(pose=Pose2D(9.0, 9.0, 0.0), accepted=False, localized=True,
                              consecutive_failures=1, registration_result=None, rejection_reason="x")
    assert not apply_accepted_pose(robot, rejected, MOUNT)
    assert robot.get_pose().x == pytest.approx(0.58)


def test_turn_in_place_swings_the_rear_mounted_lidar_sideways() -> None:
    history = CommandHistory(0.5, 1.5)
    history.record(0.0, VelocityCommand(0.0, 1.0))
    prior = MountedMotionPrior(history, *MOUNT)
    dx, dy, dth = prior.predict_motion(0.0, math.pi / 2)     # body turns 90 deg left in place
    assert dth == pytest.approx(math.pi / 2)
    # Sensor at (-0.08, 0) swings to (0, -0.08): moved +0.08 x, -0.08 y (sensor frame at t0).
    assert (dx, dy) == pytest.approx((0.08, -0.08), abs=1e-6)
    assert mounted_prior(history, 0.0, 0.0) is history and mounted_prior(None, *MOUNT) is None


def _scan(*sensor_points) -> LaserScan:
    ranges = [8.0] * 361
    for x, y in sensor_points:
        ranges[round((math.atan2(y, x) + math.pi) / math.radians(1.0))] = math.hypot(x, y)
    return LaserScan(timestamp=0.0, angle_min=-math.pi, angle_increment=math.radians(1.0),
                     ranges=ranges, range_min=0.15, range_max=8.0)


@pytest.mark.parametrize("mount_x, front_blocked, rear_blocked", [(0.0, False, True), (-0.08, True, False)])
def test_safety_zones_are_measured_from_the_footprint_not_the_sensor(mount_x, front_blocked, rear_blocked) -> None:
    # Footprint 0.5 m long -> front zone 0.25..0.55 m ahead of the CENTRE.
    # Sensor sees something 0.60 m ahead and 0.50 m behind (two bins each).
    cfg = MotionSafetyConfig(robot_length=0.5, robot_width=0.4, stop_distance=0.3, lidar_x=mount_x)
    scan = _scan((0.60, 0.0), (0.60, 0.02), (-0.50, 0.0), (-0.50, 0.02))
    s = MotionSafety(cfg, scan_provider=lambda: scan)
    assert (not s.is_clear(MotionMode.FORWARD, Pose2D(0, 0, 0), None)) == front_blocked
    assert (not s.is_clear(MotionMode.REVERSE, Pose2D(0, 0, 0), None)) == rear_blocked
