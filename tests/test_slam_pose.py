"""Tests for slam.pose.integrate_pose (dead-reckoning kinematics)."""

from __future__ import annotations

import math

import pytest

from common.messages import Pose2D, VelocityCommand
from slam.pose import integrate_pose


def test_straight_line_motion() -> None:
    pose = Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0)
    cmd = VelocityCommand(linear=1.0, angular=0.0)
    new_pose = integrate_pose(pose, cmd, dt=2.0)

    assert new_pose.x == pytest.approx(2.0)
    assert new_pose.y == pytest.approx(0.0)
    assert new_pose.theta == pytest.approx(0.0)
    assert new_pose.timestamp == pytest.approx(2.0)


def test_pure_rotation_in_place() -> None:
    pose = Pose2D(x=1.0, y=1.0, theta=0.0, timestamp=0.0)
    cmd = VelocityCommand(linear=0.0, angular=math.pi / 2.0)
    new_pose = integrate_pose(pose, cmd, dt=1.0)

    assert new_pose.x == pytest.approx(1.0)
    assert new_pose.y == pytest.approx(1.0)
    assert new_pose.theta == pytest.approx(math.pi / 2.0)


def test_motion_along_existing_heading() -> None:
    pose = Pose2D(x=0.0, y=0.0, theta=math.pi / 2.0, timestamp=0.0)
    cmd = VelocityCommand(linear=1.0, angular=0.0)
    new_pose = integrate_pose(pose, cmd, dt=1.0)

    assert new_pose.x == pytest.approx(0.0, abs=1e-9)
    assert new_pose.y == pytest.approx(1.0)


def test_zero_dt_is_a_no_op() -> None:
    pose = Pose2D(x=3.0, y=4.0, theta=1.0, timestamp=5.0)
    cmd = VelocityCommand(linear=1.0, angular=1.0)
    new_pose = integrate_pose(pose, cmd, dt=0.0)

    assert new_pose.x == pytest.approx(3.0)
    assert new_pose.y == pytest.approx(4.0)
    assert new_pose.theta == pytest.approx(1.0)
    assert new_pose.timestamp == pytest.approx(5.0)


def test_negative_dt_rejected() -> None:
    pose = Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0)
    cmd = VelocityCommand(linear=1.0, angular=0.0)
    with pytest.raises(ValueError):
        integrate_pose(pose, cmd, dt=-1.0)


def test_theta_wraps_past_pi() -> None:
    pose = Pose2D(x=0.0, y=0.0, theta=3.0, timestamp=0.0)
    cmd = VelocityCommand(linear=0.0, angular=1.0)
    new_pose = integrate_pose(pose, cmd, dt=1.0)

    assert -math.pi < new_pose.theta <= math.pi
