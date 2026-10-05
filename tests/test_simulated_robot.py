"""Phase 8 tests: SimulatedRobot (RobotInterface impl) -- clamping, watchdog,
motion integration, and collision safety."""

from __future__ import annotations

import pytest

from common.messages import Pose2D, VelocityCommand
from robot.simulated_robot import SimulatedRobot
from simulation.environment import Environment
from simulation.obstacles import RectangleObstacle


def make_robot(**overrides) -> tuple[SimulatedRobot, Environment]:
    env = Environment(
        width=20.0, height=20.0, robot_length=0.4, robot_width=0.4,
        robot_pose=Pose2D(x=5.0, y=5.0, theta=0.0),
    )
    defaults = dict(max_linear_velocity=1.0, max_angular_velocity=2.0, command_timeout=0.5)
    defaults.update(overrides)
    return SimulatedRobot(environment=env, **defaults), env


def test_rejects_invalid_construction_params() -> None:
    env = Environment(width=5.0, height=5.0, robot_length=0.4, robot_width=0.4, robot_pose=Pose2D(1, 1, 0))
    with pytest.raises(ValueError):
        SimulatedRobot(env, max_linear_velocity=0.0, max_angular_velocity=1.0, command_timeout=0.5)
    with pytest.raises(ValueError):
        SimulatedRobot(env, max_linear_velocity=1.0, max_angular_velocity=1.0, command_timeout=0.0)


def test_get_pose_reflects_environment() -> None:
    robot, env = make_robot()
    assert robot.get_pose() == env.robot_pose


def test_set_velocity_clamps_to_limits() -> None:
    robot, env = make_robot(max_linear_velocity=0.5, max_angular_velocity=1.0, command_timeout=5.0)
    robot.set_velocity(VelocityCommand(linear=10.0, angular=-10.0))
    robot.step(dt=1.0)
    # After 1s at the clamped max (0.5 m/s, theta unchanged), x should advance by ~0.5.
    assert env.robot_pose.x == pytest.approx(5.5, abs=1e-6)


def test_stop_zeros_velocity() -> None:
    robot, env = make_robot()
    robot.set_velocity(VelocityCommand(linear=1.0, angular=0.0))
    robot.stop()
    robot.step(dt=1.0)
    assert env.robot_pose.x == pytest.approx(5.0)
    assert env.robot_pose.y == pytest.approx(5.0)


def test_motion_integrates_commanded_velocity() -> None:
    robot, env = make_robot(command_timeout=5.0)
    robot.set_velocity(VelocityCommand(linear=0.5, angular=0.0))
    robot.step(dt=2.0)
    assert env.robot_pose.x == pytest.approx(6.0)  # 5.0 + 0.5*2.0


def test_watchdog_stops_robot_after_timeout() -> None:
    robot, env = make_robot(command_timeout=0.3)
    robot.set_velocity(VelocityCommand(linear=1.0, angular=0.0))

    robot.step(dt=0.2)  # within timeout: moves
    x_after_first_step = env.robot_pose.x
    assert x_after_first_step > 5.0

    robot.step(dt=0.5)  # exceeds command_timeout since last set_velocity: watchdog stops it
    x_after_watchdog = env.robot_pose.x
    assert x_after_watchdog == pytest.approx(x_after_first_step, abs=1e-9)


def test_watchdog_resets_on_new_command() -> None:
    robot, env = make_robot(command_timeout=0.3)
    robot.set_velocity(VelocityCommand(linear=1.0, angular=0.0))
    robot.step(dt=0.2)
    robot.set_velocity(VelocityCommand(linear=1.0, angular=0.0))  # refresh watchdog
    robot.step(dt=0.2)  # still within timeout of the refreshed command
    assert env.robot_pose.x > 5.2  # kept moving instead of stopping


def test_collision_blocks_motion() -> None:
    # Realistic control-loop step size (10-30 Hz per project timing
    # guidance). SimulatedRobot checks collision once per step against the
    # candidate end pose, not a continuous swept path -- a single huge dt
    # could in principle tunnel through a thin obstacle, but that is not
    # how step() is meant to be driven, so this test uses a realistic rate.
    env = Environment(
        width=20.0, height=20.0, robot_length=0.4, robot_width=0.4,
        robot_pose=Pose2D(x=5.0, y=5.0, theta=0.0),
    )
    env.add_static_obstacle(RectangleObstacle(x=6.0, y=5.0, width=1.0, height=1.0))
    robot = SimulatedRobot(env, max_linear_velocity=1.0, max_angular_velocity=1.0, command_timeout=20.0)

    robot.set_velocity(VelocityCommand(linear=1.0, angular=0.0))
    for _ in range(200):  # 200 * 0.05s = 10s of simulated time, in realistic steps
        robot.step(dt=0.05)

    assert env.robot_pose.x < 5.5  # stopped at the obstacle's near face, never crossed it


def test_negative_dt_rejected() -> None:
    robot, _ = make_robot()
    with pytest.raises(ValueError):
        robot.step(dt=-0.1)
