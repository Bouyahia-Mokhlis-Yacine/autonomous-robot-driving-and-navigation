"""Reactive obstacle-stop livelock guard under realistic pose jitter.

After a reactive stop, if the check still trips and path-following has not
run since that stop (the robot has only been commanded zero), navigation
falls through to path-following once so the robot can turn away -- then
stops again if the check still trips. This used to be detected by comparing
the pose with the stop pose exactly (< 1e-6), which a real, stationary,
noisily-localized robot never satisfies: it stopped on every tick and never
steered (realistic stress run E_blackout_recovery, seed 7)."""

from __future__ import annotations

import pytest

from common.messages import Goal, OccupancyGrid, Pose2D, VelocityCommand
from navigation.navigation_manager import NavigationState
from tests.test_navigation_manager import make_grid, make_manager

ZERO = VelocityCommand(linear=0.0, angular=0.0)


def setup(jitter: float):
    """Robot at (1.25, 1.25) facing +x, a real obstacle cell ~0.5 m ahead on
    that heading, goal straight up (+y) -- a valid path that the heading ray
    nevertheless clips until the robot turns. Positions are cell centres of
    the 0.5 m test grid, so mm-level jitter never changes which cells the
    ray crosses -- only the pose value."""
    manager, robot = make_manager(pose=Pose2D(1.25, 1.25, 0.0), obstacle_stop_distance=1.0)
    grid = make_grid()
    grid.set_cell(*grid.world_to_grid(1.75, 1.25), OccupancyGrid.OCCUPIED)
    manager.set_goal(Goal(1.25, 8.25), grid)
    assert manager.state is NavigationState.MOVING

    ticks = []

    def tick(k: int):
        # Stationary robot, but the ESTIMATED pose jitters by a few mm,
        # alternating sign, like realistic scan matching (never exactly equal).
        if robot.last_command == ZERO:
            robot.pose = Pose2D(1.25 + jitter * (-1) ** k, 1.25 - jitter * (-1) ** k, 0.001 * jitter * (-1) ** k)
        cmd = manager.update(grid, 0.1)
        ticks.append(cmd)
        return cmd

    return manager, robot, grid, tick


@pytest.mark.parametrize("jitter", [0.0, 0.003])
def test_follows_the_path_after_a_reactive_stop_despite_pose_jitter(jitter: float) -> None:
    manager, robot, grid, tick = setup(jitter)
    assert tick(0) == ZERO                        # reactive stop fires
    assert manager._reactive_stop_awaiting_follow
    cmd = tick(1)                                 # still tripping, nothing has moved: follow once
    assert cmd != ZERO
    assert cmd.angular > 0.0                      # turning toward the path (+y), not driving at the obstacle
    assert manager.get_local_target() is not None
    assert not manager._reactive_stop_awaiting_follow


def test_old_exact_pose_guard_would_have_stopped_every_tick() -> None:
    """Documents the bug: with jitter, the pose is never exactly the stop
    pose, so an equality test never engages. The flag does not depend on it."""
    manager, robot, grid, tick = setup(0.003)
    stops = sum(tick(k) == ZERO for k in range(20))
    assert stops <= 11                            # alternates stop / follow, never 20/20


@pytest.mark.parametrize("jitter", [0.0, 0.003])
def test_a_persisting_obstacle_still_stops_the_robot_again(jitter: float) -> None:
    """Falling through once must not become permission to keep driving:
    the very next trip after a follow tick stops again."""
    manager, robot, grid, tick = setup(jitter)
    tick(0)
    tick(1)                                       # the one follow tick
    assert tick(2) == ZERO                        # check still trips -> genuine stop again
    assert manager._reactive_stop_awaiting_follow


def test_stop_follow_alternation_even_if_the_robot_does_not_move() -> None:
    """While the check keeps tripping: stop, follow once, stop, ... -- also
    when the follow tick did not actually move the robot (this fake robot
    never moves; think "physically blocked"). The old exact-pose guard kept
    falling through and commanding motion on every tick in that case."""
    manager, robot, grid, tick = setup(0.0)
    pattern = [tick(k) == ZERO for k in range(6)]
    assert pattern == [True, False, True, False, True, False]


def test_a_new_obstacle_stops_a_moving_robot_immediately() -> None:
    """Genuine reactive stops are not weakened: a robot that has been
    following its path stops on the first tick an obstacle appears ahead."""
    manager, robot = make_manager(pose=Pose2D(1.0, 1.0, 1.5708), obstacle_stop_distance=1.0)
    grid = make_grid()
    manager.set_goal(Goal(1.0, 8.0), grid)
    for _ in range(3):
        assert manager.update(grid, 0.1) != ZERO  # driving
        robot.integrate(0.1)
    assert not manager._reactive_stop_awaiting_follow
    p = robot.get_pose()
    grid.set_cell(*grid.world_to_grid(p.x, p.y + 0.6), OccupancyGrid.OCCUPIED)  # obstacle appears ahead
    assert manager.update(grid, 0.1) == ZERO
    assert manager._reactive_stop_awaiting_follow


def test_new_goal_and_cancel_reset_the_guard() -> None:
    manager, robot, grid, tick = setup(0.0)
    tick(0)
    assert manager._reactive_stop_awaiting_follow
    manager.set_goal(Goal(1.25, 8.25), grid)
    assert not manager._reactive_stop_awaiting_follow
    tick(1)
    manager.cancel_goal()
    assert not manager._reactive_stop_awaiting_follow


def test_blackout_seed7_stress_run_now_reaches_the_goal() -> None:
    """The realistic stress run that crawled for 45 s at the doorway."""
    import localization_stress_test as stress
    from simulation.sensor_simulation_config import load_sensor_simulation_config

    scenario = next(s for s in stress.SCENARIOS if s.name == "E_blackout_recovery")
    result = stress.run_scenario(scenario, load_sensor_simulation_config(mode="realistic", seed=7))
    assert result.goal_reached
    assert not result.collision
    assert result.loss_events >= 1                    # the blackout still stopped the robot
    assert result.distance_driven_while_not_localized < 0.1
    assert result.sim_time < 40.0                     # was still MOVING at 60 s
