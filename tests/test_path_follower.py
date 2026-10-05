"""Phase 8 tests: PathFollower controller (pose+path -> VelocityCommand)."""

from __future__ import annotations

import math

import pytest

from common.messages import Path, PathPoint, Pose2D
from control.path_follower import PathFollower, PathFollowerConfig


def make_follower(**overrides) -> PathFollower:
    defaults = dict(
        max_linear_velocity=1.0,
        max_angular_velocity=2.0,
        max_linear_acceleration=10.0,  # high, so most tests aren't accel-limited
        max_angular_acceleration=10.0,
        goal_tolerance=0.1,
        heading_gain=2.0,
    )
    defaults.update(overrides)
    return PathFollower(PathFollowerConfig(**defaults))


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        PathFollowerConfig(1.0, 2.0, 1.0, 1.0, goal_tolerance=0.0, heading_gain=1.0)
    with pytest.raises(ValueError):
        PathFollowerConfig(0.0, 2.0, 1.0, 1.0, goal_tolerance=0.1, heading_gain=1.0)


def test_is_goal_reached_empty_path() -> None:
    follower = make_follower()
    assert follower.is_goal_reached(Pose2D(0, 0, 0), Path(points=[])) is True


def test_is_goal_reached_within_and_outside_tolerance() -> None:
    follower = make_follower(goal_tolerance=0.5)
    path = Path(points=[PathPoint(5.0, 5.0)])
    assert follower.is_goal_reached(Pose2D(x=5.1, y=5.0, theta=0.0), path) is True
    assert follower.is_goal_reached(Pose2D(x=0.0, y=0.0, theta=0.0), path) is False


def test_compute_command_on_empty_path_is_zero() -> None:
    follower = make_follower()
    cmd = follower.compute_command(Pose2D(0, 0, 0), Path(points=[]), dt=0.1)
    assert cmd.linear == 0.0
    assert cmd.angular == 0.0


def test_compute_command_zero_once_goal_reached() -> None:
    follower = make_follower(goal_tolerance=0.5)
    path = Path(points=[PathPoint(0.1, 0.0)])
    cmd = follower.compute_command(Pose2D(x=0.0, y=0.0, theta=0.0), path, dt=0.1)
    assert cmd.linear == pytest.approx(0.0)
    assert cmd.angular == pytest.approx(0.0)


def test_facing_target_drives_forward_with_no_turn() -> None:
    follower = make_follower()
    path = Path(points=[PathPoint(10.0, 0.0)])
    cmd = follower.compute_command(Pose2D(x=0.0, y=0.0, theta=0.0), path, dt=0.1)
    assert cmd.linear > 0.0
    assert cmd.angular == pytest.approx(0.0, abs=1e-9)


def test_facing_away_from_target_turns_without_driving_backward() -> None:
    follower = make_follower()
    path = Path(points=[PathPoint(-10.0, 0.0)])  # target directly behind
    cmd = follower.compute_command(Pose2D(x=0.0, y=0.0, theta=0.0), path, dt=0.1)
    assert cmd.linear == pytest.approx(0.0, abs=1e-9)  # cos(pi) clamped to 0, never negative
    assert abs(cmd.angular) > 0.0  # turning to face the target


def test_target_to_the_left_turns_left() -> None:
    follower = make_follower()
    path = Path(points=[PathPoint(0.0, 10.0)])  # target at +90 degrees
    cmd = follower.compute_command(Pose2D(x=0.0, y=0.0, theta=0.0), path, dt=0.1)
    assert cmd.angular > 0.0


def test_velocity_never_exceeds_configured_limits() -> None:
    follower = make_follower(max_linear_velocity=0.5, max_angular_velocity=1.0, heading_gain=100.0)
    path = Path(points=[PathPoint(0.0, 10.0)])
    cmd = follower.compute_command(Pose2D(x=0.0, y=0.0, theta=0.0), path, dt=1.0)
    assert abs(cmd.linear) <= 0.5 + 1e-9
    assert abs(cmd.angular) <= 1.0 + 1e-9


def test_acceleration_limits_ramp_velocity_gradually() -> None:
    follower = make_follower(max_linear_acceleration=0.2, max_angular_acceleration=0.2)
    path = Path(points=[PathPoint(10.0, 0.0)])
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)

    cmd1 = follower.compute_command(pose, path, dt=0.1)
    assert cmd1.linear == pytest.approx(0.02, abs=1e-9)  # 0.2 m/s^2 * 0.1s

    cmd2 = follower.compute_command(pose, path, dt=0.1)
    assert cmd2.linear == pytest.approx(0.04, abs=1e-9)


def test_reset_clears_ramp_and_local_planner_state() -> None:
    follower = make_follower(max_linear_acceleration=0.2, max_angular_acceleration=0.2)
    path = Path(points=[PathPoint(10.0, 0.0)])
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    follower.compute_command(pose, path, dt=1.0)  # ramps last_command.linear up

    follower.reset()
    cmd = follower.compute_command(pose, path, dt=0.1)
    assert cmd.linear == pytest.approx(0.02, abs=1e-9)  # ramp restarted from 0


def test_negative_dt_rejected() -> None:
    follower = make_follower()
    with pytest.raises(ValueError):
        follower.compute_command(Pose2D(0, 0, 0), Path(points=[PathPoint(1, 0)]), dt=-0.1)


def test_robot_converges_toward_goal_over_simulated_steps() -> None:
    """Acceptance-style check: repeatedly applying compute_command via simple
    Euler integration drives the robot to the goal within tolerance, without
    wild oscillation (bounded angular command magnitude throughout)."""
    follower = make_follower(
        max_linear_velocity=1.0, max_angular_velocity=2.0,
        max_linear_acceleration=2.0, max_angular_acceleration=4.0,
        goal_tolerance=0.1, heading_gain=2.0,
    )
    path = Path(points=[PathPoint(5.0, 3.0)])
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    dt = 0.05

    for _ in range(2000):
        if follower.is_goal_reached(pose, path):
            break
        cmd = follower.compute_command(pose, path, dt)
        assert abs(cmd.angular) <= 2.0 + 1e-6
        assert abs(cmd.linear) <= 1.0 + 1e-6
        pose = Pose2D(
            x=pose.x + cmd.linear * math.cos(pose.theta) * dt,
            y=pose.y + cmd.linear * math.sin(pose.theta) * dt,
            theta=pose.theta + cmd.angular * dt,
        )

    assert follower.is_goal_reached(pose, path)
