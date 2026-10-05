"""Phase 1 tests: 2D world, static/dynamic obstacles, robot footprint, collisions."""

from __future__ import annotations

import math

import pytest

from common.messages import Pose2D
from simulation.environment import Environment
from simulation.obstacles import DynamicObstacle, RectangleObstacle
from simulation.simulator import Simulator


def make_environment(**overrides) -> Environment:
    defaults = dict(
        width=10.0,
        height=10.0,
        robot_length=0.5,
        robot_width=0.4,
        robot_pose=Pose2D(x=1.0, y=1.0, theta=0.0),
    )
    defaults.update(overrides)
    return Environment(**defaults)


def test_environment_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError):
        make_environment(width=0.0)
    with pytest.raises(ValueError):
        make_environment(robot_length=-1.0)


def test_robot_in_open_space_has_no_collision() -> None:
    env = make_environment()
    assert env.check_collision() is False


def test_robot_out_of_bounds_collides() -> None:
    env = make_environment(robot_pose=Pose2D(x=-1.0, y=5.0, theta=0.0))
    assert env.check_collision() is True

    env2 = make_environment()
    assert env2.check_collision(Pose2D(x=100.0, y=5.0, theta=0.0)) is True


def test_static_obstacle_blocks_overlapping_pose() -> None:
    env = make_environment(robot_pose=Pose2D(x=5.0, y=5.0, theta=0.0))
    env.add_static_obstacle(RectangleObstacle(x=5.0, y=5.0, width=1.0, height=1.0))
    assert env.check_collision() is True


def test_static_obstacle_does_not_block_far_pose() -> None:
    env = make_environment(robot_pose=Pose2D(x=1.0, y=1.0, theta=0.0))
    env.add_static_obstacle(RectangleObstacle(x=8.0, y=8.0, width=1.0, height=1.0))
    assert env.check_collision() is False


def test_rotated_robot_footprint_can_collide_where_unrotated_would_not() -> None:
    # A long, narrow robot: axis-aligned it clears a nearby obstacle in y,
    # but rotated 90 degrees its length now extends into y and collides.
    obstacle = RectangleObstacle(x=5.0, y=5.6, width=0.4, height=0.4)

    env_unrotated = make_environment(
        robot_length=1.0,
        robot_width=0.2,
        robot_pose=Pose2D(x=5.0, y=5.0, theta=0.0),
    )
    env_unrotated.add_static_obstacle(obstacle)
    assert env_unrotated.check_collision() is False

    env_rotated = make_environment(
        robot_length=1.0,
        robot_width=0.2,
        robot_pose=Pose2D(x=5.0, y=5.0, theta=math.pi / 2.0),
    )
    env_rotated.add_static_obstacle(
        RectangleObstacle(x=5.0, y=5.6, width=0.4, height=0.4)
    )
    assert env_rotated.check_collision() is True


def test_try_move_robot_succeeds_when_free() -> None:
    env = make_environment()
    ok = env.try_move_robot(Pose2D(x=2.0, y=2.0, theta=0.0))
    assert ok is True
    assert env.robot_pose.x == 2.0
    assert env.robot_pose.y == 2.0


def test_try_move_robot_rejected_on_collision_and_pose_unchanged() -> None:
    env = make_environment(robot_pose=Pose2D(x=1.0, y=1.0, theta=0.0))
    env.add_static_obstacle(RectangleObstacle(x=5.0, y=5.0, width=1.0, height=1.0))

    ok = env.try_move_robot(Pose2D(x=5.0, y=5.0, theta=0.0))
    assert ok is False
    assert env.robot_pose.x == 1.0
    assert env.robot_pose.y == 1.0


def test_dynamic_obstacle_moves_over_time() -> None:
    env = make_environment()
    obstacle = DynamicObstacle(x=1.0, y=1.0, width=0.5, height=0.5, velocity_x=1.0, velocity_y=0.0)
    env.add_dynamic_obstacle(obstacle)

    env.step(dt=1.0)
    assert obstacle.x == pytest.approx(2.0)
    assert obstacle.y == pytest.approx(1.0)


def test_dynamic_obstacle_bounces_off_world_wall() -> None:
    env = make_environment(width=5.0, height=5.0)
    obstacle = DynamicObstacle(x=4.7, y=2.5, width=0.5, height=0.5, velocity_x=1.0, velocity_y=0.0)
    env.add_dynamic_obstacle(obstacle)

    env.step(dt=1.0)
    # Obstacle's right edge is clamped to the wall and velocity reflects.
    assert obstacle.x == pytest.approx(4.75)
    assert obstacle.velocity_x == pytest.approx(-1.0)


def test_all_obstacles_combines_static_and_dynamic() -> None:
    env = make_environment()
    env.add_static_obstacle(RectangleObstacle(x=2.0, y=2.0, width=1.0, height=1.0))
    env.add_dynamic_obstacle(
        DynamicObstacle(x=3.0, y=3.0, width=1.0, height=1.0, velocity_x=0.0, velocity_y=0.0)
    )
    assert len(env.all_obstacles()) == 2


def test_simulator_advances_dynamic_obstacles_and_clock() -> None:
    env = make_environment()
    obstacle = DynamicObstacle(x=1.0, y=1.0, width=0.5, height=0.5, velocity_x=0.5, velocity_y=0.0)
    env.add_dynamic_obstacle(obstacle)
    sim = Simulator(environment=env)

    sim.run(duration=2.0, dt=0.1)

    assert sim.time == pytest.approx(2.0)
    assert obstacle.x == pytest.approx(2.0)


def test_simulator_does_not_move_robot_or_expose_planning_api() -> None:
    # The simulator/environment must not contain planning logic: there is
    # no path/goal-planning API on Environment or Simulator.
    env = make_environment()
    assert not hasattr(env, "plan_path")
    assert not hasattr(env, "goal")
