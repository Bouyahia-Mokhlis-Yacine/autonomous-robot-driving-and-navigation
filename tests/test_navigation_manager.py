"""Phase 9/10 tests: NavigationManager state machine, including Phase 10's
obstacle-ahead safety stop and periodic/deviation replan triggers."""

from __future__ import annotations

import pytest

from common.messages import Goal, OccupancyGrid, Pose2D, VelocityCommand
from control.path_follower import PathFollower, PathFollowerConfig
from mapping.occupancy_grid import OccupancyGridMap
from navigation.navigation_manager import NavigationManager, NavigationState
from navigation.replanner import Replanner, ReplannerConfig
from planning.global_planner import GlobalPlanner
from slam.pose import integrate_pose


class FakeRobot:
    """A minimal RobotInterface test double: no physics of its own, the
    test drives ``pose`` directly (or via ``integrate`` for convergence
    tests) so NavigationManager's state logic is tested in isolation."""

    def __init__(self, pose: Pose2D) -> None:
        self.pose = pose
        self.last_command = VelocityCommand(linear=0.0, angular=0.0)
        self.stop_calls = 0

    def set_velocity(self, cmd: VelocityCommand) -> None:
        self.last_command = cmd

    def stop(self) -> None:
        self.last_command = VelocityCommand(linear=0.0, angular=0.0)
        self.stop_calls += 1

    def get_pose(self) -> Pose2D:
        return self.pose

    def integrate(self, dt: float) -> None:
        self.pose = integrate_pose(self.pose, self.last_command, dt)


def make_manager(
    pose: Pose2D = Pose2D(x=1.0, y=1.0, theta=0.0),
    replan_interval: float = 100.0,
    deviation_threshold: float = 100.0,
    obstacle_stop_distance: float = 0.05,
) -> tuple[NavigationManager, FakeRobot]:
    """Defaults keep the periodic timer and deviation trigger effectively
    disabled (huge thresholds) and the obstacle-ahead lookahead tiny, so
    existing Phase 9-style tests aren't perturbed by Phase 10 triggers
    unless a test explicitly tightens them."""
    robot = FakeRobot(pose)
    follower = PathFollower(
        PathFollowerConfig(
            max_linear_velocity=1.0,
            max_angular_velocity=2.0,
            max_linear_acceleration=5.0,
            max_angular_acceleration=5.0,
            goal_tolerance=0.15,
            heading_gain=2.0,
        )
    )
    planner = GlobalPlanner(allow_diagonal=True)
    replanner = Replanner(ReplannerConfig(replan_interval=replan_interval, deviation_threshold=deviation_threshold))
    manager = NavigationManager(
        global_planner=planner,
        path_follower=follower,
        robot=robot,
        replanner=replanner,
        obstacle_stop_distance=obstacle_stop_distance,
    )
    return manager, robot


def make_grid(size: float = 10.0, resolution: float = 0.5) -> OccupancyGridMap:
    return OccupancyGridMap(resolution=resolution, width_m=size, height_m=size)


def test_initial_state_is_idle() -> None:
    manager, _ = make_manager()
    assert manager.state is NavigationState.IDLE
    assert manager.get_goal() is None
    assert manager.get_path() is None


def test_update_while_idle_stops_robot_and_returns_zero() -> None:
    manager, robot = make_manager()
    cmd = manager.update(make_grid(), dt=0.1)
    assert cmd == VelocityCommand(linear=0.0, angular=0.0)
    assert robot.stop_calls >= 1


def test_set_goal_valid_reaches_moving_with_a_path() -> None:
    manager, _ = make_manager()
    grid_map = make_grid()
    manager.set_goal(Goal(x=8.0, y=8.0), grid_map)
    assert manager.state is NavigationState.MOVING
    assert manager.get_path() is not None
    assert not manager.get_path().is_empty()
    assert manager.get_goal() == Goal(x=8.0, y=8.0)


def test_set_goal_on_occupied_cell_rejected_to_blocked() -> None:
    manager, robot = make_manager()
    grid_map = make_grid()
    grid_map.set_cell(*grid_map.world_to_grid(8.0, 8.0), OccupancyGrid.OCCUPIED)

    manager.set_goal(Goal(x=8.0, y=8.0), grid_map)

    assert manager.state is NavigationState.BLOCKED
    assert manager.get_path() is None
    assert robot.stop_calls >= 1


def test_set_goal_unreachable_goes_to_blocked() -> None:
    manager, _ = make_manager(pose=Pose2D(x=1.0, y=1.0, theta=0.0))
    grid_map = make_grid()
    # Fully enclose the goal cell so it is free but unreachable.
    gr, gc = grid_map.world_to_grid(8.0, 8.0)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            grid_map.set_cell(gr + dr, gc + dc, OccupancyGrid.OCCUPIED)

    manager.set_goal(Goal(x=8.0, y=8.0), grid_map)

    assert manager.state is NavigationState.BLOCKED
    assert manager.get_path() is None


def test_goal_reached_transition() -> None:
    manager, robot = make_manager(pose=Pose2D(x=1.0, y=1.0, theta=0.0))
    grid_map = make_grid()
    manager.set_goal(Goal(x=1.05, y=1.0), grid_map)  # essentially already there
    assert manager.state is NavigationState.MOVING

    manager.update(grid_map, dt=0.1)
    assert manager.state is NavigationState.GOAL_REACHED
    assert robot.stop_calls >= 1


def test_update_after_goal_reached_stays_and_returns_zero() -> None:
    manager, robot = make_manager(pose=Pose2D(x=1.0, y=1.0, theta=0.0))
    grid_map = make_grid()
    manager.set_goal(Goal(x=1.05, y=1.0), grid_map)
    manager.update(grid_map, dt=0.1)
    assert manager.state is NavigationState.GOAL_REACHED

    cmd = manager.update(grid_map, dt=0.1)
    assert cmd == VelocityCommand(linear=0.0, angular=0.0)
    assert manager.state is NavigationState.GOAL_REACHED


def test_replanning_when_path_becomes_blocked() -> None:
    manager, robot = make_manager(pose=Pose2D(x=1.0, y=1.0, theta=0.0))
    grid_map = make_grid(size=10.0, resolution=1.0)
    manager.set_goal(Goal(x=8.0, y=1.0), grid_map)
    assert manager.state is NavigationState.MOVING
    original_path = manager.get_path()

    # Block the straight-line route after the fact (row 1 == world y=1.0
    # at this 1.0 m/cell resolution, matching the robot's/goal's row).
    grid_map.set_cell(1, 5, OccupancyGrid.OCCUPIED)

    cmd = manager.update(grid_map, dt=0.1)

    assert manager.state is NavigationState.MOVING  # successfully replanned
    assert manager.get_path() is not original_path
    assert manager.get_path() is not None


def test_replanning_fails_goes_to_blocked() -> None:
    manager, robot = make_manager(pose=Pose2D(x=0.5, y=0.5, theta=0.0))
    grid_map = make_grid(size=5.0, resolution=1.0)
    manager.set_goal(Goal(x=3.5, y=0.5), grid_map)
    assert manager.state is NavigationState.MOVING

    # A solid full-height wall (no gap) completely separates robot from goal.
    for row in range(5):
        grid_map.set_cell(row, 2, OccupancyGrid.OCCUPIED)

    cmd = manager.update(grid_map, dt=0.1)

    assert manager.state is NavigationState.BLOCKED
    assert cmd == VelocityCommand(linear=0.0, angular=0.0)
    assert robot.stop_calls >= 1


def test_emergency_stop_from_moving() -> None:
    manager, robot = make_manager()
    grid_map = make_grid()
    manager.set_goal(Goal(x=8.0, y=8.0), grid_map)
    assert manager.state is NavigationState.MOVING

    manager.emergency_stop()
    assert manager.state is NavigationState.EMERGENCY_STOP
    assert robot.stop_calls >= 1

    cmd = manager.update(grid_map, dt=0.1)
    assert cmd == VelocityCommand(linear=0.0, angular=0.0)
    assert manager.state is NavigationState.EMERGENCY_STOP


def test_set_goal_ignored_during_emergency_stop() -> None:
    manager, _ = make_manager()
    grid_map = make_grid()
    manager.emergency_stop()

    manager.set_goal(Goal(x=8.0, y=8.0), grid_map)

    assert manager.state is NavigationState.EMERGENCY_STOP
    assert manager.get_goal() is None


def test_clear_emergency_stop_returns_to_idle() -> None:
    manager, _ = make_manager()
    manager.emergency_stop()
    manager.clear_emergency_stop()
    assert manager.state is NavigationState.IDLE


def test_cancel_goal_returns_to_idle_and_clears_state() -> None:
    manager, robot = make_manager()
    grid_map = make_grid()
    manager.set_goal(Goal(x=8.0, y=8.0), grid_map)
    assert manager.state is NavigationState.MOVING

    manager.cancel_goal()

    assert manager.state is NavigationState.IDLE
    assert manager.get_goal() is None
    assert manager.get_path() is None
    assert robot.stop_calls >= 1


def test_full_convergence_to_goal_reached_via_update_loop() -> None:
    """Acceptance-style: goal -> plan -> move -> goal reached, end to end."""
    manager, robot = make_manager(pose=Pose2D(x=1.0, y=1.0, theta=0.0))
    grid_map = make_grid(size=10.0, resolution=0.5)
    manager.set_goal(Goal(x=7.0, y=6.0), grid_map)
    assert manager.state is NavigationState.MOVING

    dt = 0.05
    for _ in range(4000):
        if manager.state is NavigationState.GOAL_REACHED:
            break
        manager.update(grid_map, dt)
        robot.integrate(dt)

    assert manager.state is NavigationState.GOAL_REACHED


# --------------------------------------------------------- Phase 10 triggers ---


def test_obstacle_ahead_forces_immediate_stop_and_replan() -> None:
    manager, robot = make_manager(pose=Pose2D(x=1.0, y=1.0, theta=0.0), obstacle_stop_distance=1.0)
    grid_map = make_grid(size=10.0, resolution=0.5)
    manager.set_goal(Goal(x=8.0, y=1.0), grid_map)
    assert manager.state is NavigationState.MOVING
    original_path = manager.get_path()
    stop_calls_before = robot.stop_calls

    # An obstacle appears directly ahead, within obstacle_stop_distance.
    grid_map.set_cell(*grid_map.world_to_grid(1.4, 1.0), OccupancyGrid.OCCUPIED)

    cmd = manager.update(grid_map, dt=0.1)

    # This tick always stops, regardless of whether the ensuing replan succeeds.
    assert cmd == VelocityCommand(linear=0.0, angular=0.0)
    assert robot.stop_calls > stop_calls_before
    assert manager.state in (NavigationState.MOVING, NavigationState.BLOCKED)
    if manager.state is NavigationState.MOVING:
        assert manager.get_path() is not original_path  # a fresh plan was committed


def test_periodic_replan_triggers_without_any_obstruction() -> None:
    manager, robot = make_manager(
        pose=Pose2D(x=1.0, y=1.0, theta=0.0),
        replan_interval=0.2,
        deviation_threshold=100.0,
        obstacle_stop_distance=0.01,
    )
    grid_map = make_grid(size=10.0, resolution=1.0)
    manager.set_goal(Goal(x=8.0, y=1.0), grid_map)
    assert manager.state is NavigationState.MOVING
    original_path = manager.get_path()

    manager.update(grid_map, dt=0.3)  # exceeds the 0.2s replan_interval

    assert manager.state is NavigationState.MOVING  # map is unchanged, so the replan just succeeds again
    assert manager.get_path() is not original_path  # a fresh Path object was committed


def test_deviation_triggers_replan_from_new_pose() -> None:
    manager, robot = make_manager(
        pose=Pose2D(x=1.0, y=1.0, theta=0.0),
        replan_interval=100.0,
        deviation_threshold=0.5,
        obstacle_stop_distance=0.01,
    )
    grid_map = make_grid(size=10.0, resolution=1.0)
    manager.set_goal(Goal(x=8.0, y=1.0), grid_map)
    original_path = manager.get_path()

    robot.pose = Pose2D(x=1.0, y=5.0, theta=0.0)  # drifted 4 m off the planned route
    manager.update(grid_map, dt=0.1)

    assert manager.state is NavigationState.MOVING
    assert manager.get_path() is not original_path
    # The new path starts from where the robot actually is now.
    assert manager.get_path().points[0].y == pytest.approx(5.0, abs=1.0)


def test_no_spurious_replanning_when_thresholds_are_loose() -> None:
    manager, robot = make_manager(pose=Pose2D(x=1.0, y=1.0, theta=0.0))  # all Phase 10 triggers effectively disabled
    grid_map = make_grid(size=10.0, resolution=1.0)
    manager.set_goal(Goal(x=8.0, y=1.0), grid_map)
    original_path = manager.get_path()

    for _ in range(10):
        manager.update(grid_map, dt=0.05)
        robot.integrate(0.05)

    assert manager.state is NavigationState.MOVING
    assert manager.get_path() is original_path  # never replanned
