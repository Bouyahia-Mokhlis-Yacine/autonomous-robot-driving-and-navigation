"""Tests for robot.sensor_only_robot.SensorOnlyRobot."""

from __future__ import annotations

from common.messages import Pose2D, VelocityCommand
from robot.sensor_only_robot import SensorOnlyRobot


def test_get_pose_returns_initial_pose() -> None:
    pose = Pose2D(x=1.0, y=2.0, theta=0.5, timestamp=0.0)
    robot = SensorOnlyRobot(initial_pose=pose)
    assert robot.get_pose() == pose


def test_set_pose_updates_get_pose() -> None:
    robot = SensorOnlyRobot(initial_pose=Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0))
    new_pose = Pose2D(x=3.0, y=4.0, theta=1.0, timestamp=5.0)
    robot.set_pose(new_pose)
    assert robot.get_pose() == new_pose


def test_set_velocity_and_stop_do_not_raise_and_do_not_affect_pose() -> None:
    pose = Pose2D(x=1.0, y=1.0, theta=0.0, timestamp=0.0)
    robot = SensorOnlyRobot(initial_pose=pose)
    robot.set_velocity(VelocityCommand(linear=0.5, angular=0.2))
    assert robot.last_command == VelocityCommand(linear=0.5, angular=0.2)
    assert robot.get_pose() == pose

    robot.stop()
    assert robot.last_command == VelocityCommand(linear=0.0, angular=0.0)
    assert robot.get_pose() == pose
