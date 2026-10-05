"""Phase 13 tests: inflated-start-cell recovery.

Tests 1, 2, 3, 4, 5, 6, 7, 9 are fast, synthetic unit/integration tests
(no full closed-loop simulation). Test 8 (the deterministic Phase 12
scenario reaching the goal) runs the real simulation and, per the same
wall-clock cost explained in tests/test_closed_loop_simulation.py's
module docstring, takes on the order of two to three minutes.
"""

from __future__ import annotations

import inspect

import pytest

from common.messages import Goal, OccupancyGrid, Pose2D, VelocityCommand
from control.path_follower import PathFollower, PathFollowerConfig
from mapping.occupancy_grid import OccupancyGridMap
from navigation.navigation_manager import NavigationManager, NavigationState
from navigation.recovery import (
    RecoveryConfig,
    inflated_start_cell_blocked,
    recovery_direction_is_safe,
    reverse_target_point,
)
from navigation.replanner import Replanner, ReplannerConfig
from planning.global_planner import GlobalPlanner
from slam.pose import integrate_pose

F, U, O = OccupancyGrid.FREE, OccupancyGrid.UNKNOWN, OccupancyGrid.OCCUPIED


class FakeRobot:
    """Same minimal RobotInterface test double as tests/test_navigation_manager.py."""

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


def make_grid(size: float = 10.0, resolution: float = 0.5) -> OccupancyGridMap:
    return OccupancyGridMap(resolution=resolution, width_m=size, height_m=size)


def make_manager(
    pose: Pose2D,
    recovery_config: RecoveryConfig | None = None,
    obstacle_stop_distance: float = 0.05,
) -> tuple[NavigationManager, FakeRobot]:
    robot = FakeRobot(pose)
    follower = PathFollower(PathFollowerConfig(
        max_linear_velocity=1.0, max_angular_velocity=2.0,
        max_linear_acceleration=5.0, max_angular_acceleration=5.0,
        goal_tolerance=0.15, heading_gain=2.0,
    ))
    planner = GlobalPlanner(allow_diagonal=True)
    replanner = Replanner(ReplannerConfig(replan_interval=100.0, deviation_threshold=100.0))
    manager = NavigationManager(
        global_planner=planner, path_follower=follower, robot=robot, replanner=replanner,
        obstacle_stop_distance=obstacle_stop_distance, recovery_config=recovery_config,
    )
    return manager, robot


# --- Test 1: detect inflated start-cell condition ---------------------------

def test_inflated_start_cell_blocked_detects_condition() -> None:
    raw_map = make_grid()
    planning_map = make_grid()
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    row, col = raw_map.world_to_grid(pose.x, pose.y)
    raw_map.set_cell(row, col, F)
    planning_map.set_cell(row, col, O)

    assert inflated_start_cell_blocked(pose, raw_map, planning_map) is True


# --- Test 2: do not trigger when the raw cell is itself occupied/unknown ----

def test_inflated_start_cell_blocked_false_when_raw_also_occupied() -> None:
    raw_map = make_grid()
    planning_map = make_grid()
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    row, col = raw_map.world_to_grid(pose.x, pose.y)
    raw_map.set_cell(row, col, O)  # a genuine, confirmed obstacle -- not the buffer case
    planning_map.set_cell(row, col, O)

    assert inflated_start_cell_blocked(pose, raw_map, planning_map) is False


def test_inflated_start_cell_blocked_false_when_raw_unknown() -> None:
    raw_map = make_grid()
    planning_map = make_grid()
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    row, col = raw_map.world_to_grid(pose.x, pose.y)
    # raw_map cell left UNKNOWN (default) -- not the demonstrated
    # "raw says FREE" condition, so recovery must not claim it is.
    planning_map.set_cell(row, col, O)

    assert inflated_start_cell_blocked(pose, raw_map, planning_map) is False


def test_inflated_start_cell_blocked_false_when_planning_cell_not_occupied() -> None:
    raw_map = make_grid()
    planning_map = make_grid()
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    row, col = raw_map.world_to_grid(pose.x, pose.y)
    raw_map.set_cell(row, col, F)
    planning_map.set_cell(row, col, F)  # not actually blocked in the planning map

    assert inflated_start_cell_blocked(pose, raw_map, planning_map) is False


def test_recovery_direction_is_safe_reflects_raw_occupancy_behind_robot() -> None:
    raw_map = make_grid()
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)  # facing +X, so "behind" is -X

    # Nothing marked occupied behind -- default UNKNOWN counts as safe
    # (has_line_of_sight only rejects OCCUPIED cells).
    assert recovery_direction_is_safe(pose, raw_map, distance=0.3) is True

    target = reverse_target_point(pose, 0.3)
    row, col = raw_map.world_to_grid(target.x, target.y)
    raw_map.set_cell(row, col, O)
    assert recovery_direction_is_safe(pose, raw_map, distance=0.3) is False


# --- Tests 3, 4, 5: bounded recovery, real velocity command, stops ----------

def test_recovery_produces_real_reverse_velocity_and_is_bounded() -> None:
    raw_map = make_grid()
    planning_map = make_grid()
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    row, col = raw_map.world_to_grid(pose.x, pose.y)
    raw_map.set_cell(row, col, F)
    planning_map.set_cell(row, col, O)
    # Leave a wide FREE lane to the goal (once recovery clears the start
    # cell) so the eventual re-plan can succeed -- not the focus of this
    # test, but avoids conflating "recovery is bounded" with "no path".
    for c in range(planning_map.grid.width):
        planning_map.set_cell(row, c, F if c != col else O)

    config = RecoveryConfig(linear_speed=0.1, probe_distance=0.5, max_duration=0.25, max_attempts=1)
    manager, robot = make_manager(pose, recovery_config=config)

    manager.set_goal(Goal(x=8.0, y=5.0), planning_map, raw_map)
    assert manager.state is NavigationState.RECOVERING  # Test 1 (behaviorally): condition detected on set_goal

    # Test 4: RECOVERING actually commands a real, nonzero reverse velocity.
    cmd = manager.update(planning_map, dt=0.1, raw_grid_map=raw_map)
    assert cmd.linear == pytest.approx(-0.1)
    assert cmd.angular == pytest.approx(0.0)
    assert robot.last_command.linear == pytest.approx(-0.1)

    # Test 3: bounded in duration -- max_duration=0.25s, dt=0.1s -> the
    # THIRD update() call (elapsed 0.3s >= 0.25s) must end the maneuver.
    manager.update(planning_map, dt=0.1, raw_grid_map=raw_map)
    cmd = manager.update(planning_map, dt=0.1, raw_grid_map=raw_map)

    # Test 5: the robot is commanded to stop once the bounded interval elapses.
    assert cmd.linear == pytest.approx(0.0)
    assert cmd.angular == pytest.approx(0.0)
    assert robot.stop_calls >= 1

    # Since planning_map's start cell is still OCCUPIED (this test never
    # actually moves the FakeRobot -- see Test 6 for the "map clears"
    # case) and max_attempts=1 is already exhausted, the state machine
    # must not loop back into RECOVERING again -- it either re-attempts
    # a genuine plan (fails, since the cell is still occupied -> BLOCKED)
    # or reports BLOCKED directly. Either way, RECOVERING must not recur.
    assert manager.state is not NavigationState.RECOVERING
    assert manager.state is NavigationState.BLOCKED


def test_recovery_attempts_are_bounded_and_do_not_loop_forever() -> None:
    """If the robot never actually moves (pose never changes -- as in
    this synthetic test), repeated recovery timeouts must still
    terminate at max_attempts, never looping RECOVERING indefinitely."""
    raw_map = make_grid()
    planning_map = make_grid()
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    row, col = raw_map.world_to_grid(pose.x, pose.y)
    raw_map.set_cell(row, col, F)
    planning_map.set_cell(row, col, O)

    config = RecoveryConfig(linear_speed=0.1, probe_distance=0.5, max_duration=0.2, max_attempts=3)
    manager, robot = make_manager(pose, recovery_config=config)
    manager.set_goal(Goal(x=8.0, y=5.0), planning_map, raw_map)

    recovering_entries = 0
    prev_state = manager.state
    if prev_state is NavigationState.RECOVERING:
        recovering_entries += 1
    for _ in range(40):  # far more ticks than could possibly be needed if bounded correctly
        if manager.state is NavigationState.BLOCKED:
            break
        manager.update(planning_map, dt=0.1, raw_grid_map=raw_map)
        if manager.state is NavigationState.RECOVERING and prev_state is not NavigationState.RECOVERING:
            recovering_entries += 1
        prev_state = manager.state
    else:
        pytest.fail("recovery never terminated within 40 ticks -- looks unbounded")

    assert manager.state is NavigationState.BLOCKED
    assert recovering_entries <= config.max_attempts


# --- Test 6: normal planning resumes once the condition clears --------------

def test_normal_planning_resumes_once_start_cell_clears() -> None:
    raw_map = make_grid()
    planning_map = make_grid()
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    row, col = raw_map.world_to_grid(pose.x, pose.y)
    raw_map.set_cell(row, col, F)
    planning_map.set_cell(row, col, O)
    for c in range(planning_map.grid.width):
        if c != col:
            planning_map.set_cell(row, c, F)

    config = RecoveryConfig(linear_speed=0.1, probe_distance=0.5, max_duration=0.2, max_attempts=3)
    manager, robot = make_manager(pose, recovery_config=config)
    manager.set_goal(Goal(x=8.0, y=5.0), planning_map, raw_map)
    assert manager.state is NavigationState.RECOVERING

    # Simulate the recovery maneuver actually having moved the robot (and
    # the world model having been rebuilt) by the time it completes: the
    # start cell is FREE in the planning map now, one cell over.
    new_pose = Pose2D(x=4.5, y=5.0, theta=0.0)
    robot.pose = new_pose
    new_row, new_col = raw_map.world_to_grid(new_pose.x, new_pose.y)
    planning_map.set_cell(new_row, new_col, F)

    manager.update(planning_map, dt=0.1, raw_grid_map=raw_map)
    cmd = manager.update(planning_map, dt=0.1, raw_grid_map=raw_map)  # elapsed 0.2s >= max_duration -> re-plan

    assert manager.state is NavigationState.MOVING
    path = manager.get_path()
    assert path is not None and not path.is_empty()


# --- Test 7: genuine no-path still reaches BLOCKED, without recovery -------

def test_genuine_no_path_reaches_blocked_without_any_recovery_attempt() -> None:
    raw_map = make_grid()
    planning_map = make_grid()
    pose = Pose2D(x=5.0, y=5.0, theta=0.0)
    row, col = raw_map.world_to_grid(pose.x, pose.y)
    raw_map.set_cell(row, col, F)
    planning_map.set_cell(row, col, F)  # start is genuinely FREE in both maps

    # Ring of OCCUPIED cells completely enclosing the goal -- unreachable.
    goal_row, goal_col = planning_map.world_to_grid(9.0, 9.0)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            planning_map.set_cell(goal_row + dr, goal_col + dc, O)
    planning_map.set_cell(goal_row, goal_col, F)
    raw_map.set_cell(goal_row, goal_col, F)

    config = RecoveryConfig(max_attempts=3)
    manager, robot = make_manager(pose, recovery_config=config)
    manager.set_goal(Goal(x=9.0, y=9.0), planning_map, raw_map)

    assert manager.state is NavigationState.BLOCKED
    assert manager.get_path() is None
    # No recovery was ever attempted -- the demonstrated condition
    # (planning OCCUPIED + raw FREE at the robot's OWN cell) never held.
    for _ in range(5):
        manager.update(planning_map, dt=0.1, raw_grid_map=raw_map)
        assert manager.state is not NavigationState.RECOVERING
    assert manager.state is NavigationState.BLOCKED


# --- Test 9: ground-truth isolation ------------------------------------------

def test_recovery_module_never_references_simulator_ground_truth() -> None:
    import navigation.recovery as recovery_module

    source = inspect.getsource(recovery_module)
    assert "simulation.environment" not in source
    assert "simulation.simulated_lidar" not in source
    assert "robot.simulated_robot" not in source
    assert "ground_truth" not in source.lower()


# --- Test 8 (expensive): the deterministic Phase 12 scenario ---------------

def test_phase12_scenario_recovers_and_never_permanently_blocks() -> None:
    """Proves the actual, demonstrated fix: the Phase 12.5 root cause
    (permanent BLOCKED from an occupied-by-inflation start cell) is
    resolved -- recovery fires, produces real bounded motion, and the
    scenario never reaches a terminal BLOCKED state.

    Does NOT assert goal_reached=True. Investigated empirically (see the
    Phase 13 final report): even with a recovery distance well past the
    configured inflation radius, the robot enters a genuine, repeatable
    oscillation -- it backs away safely, but the risk-aware planner's
    taut, corner-hugging route leads it back to nearly the same
    inflation-boundary point near the scenario's doorway, re-triggering
    recovery again. This is a real, DIFFERENT, structural finding
    (planner/path-smoothing route geometry vs. the inflation safety
    margin), out of this phase's explicit scope (no planner/inflation
    changes) -- reported, not silently hidden by weakening this
    assertion to something false.

    Later resolved: instrumentation showed the robot was only
    1.5-3 cm inside the buffer (path-follower chord cutting along the
    boundary-hugging path); ``navigation.recovery.buffer_escape_point``
    now handles that shallow case without RECOVERING -- see
    tests/test_doorway_recovery_regression.py, which asserts goal reached.
    """
    from common.config import load_navigation_config, load_robot_config
    from navigation.navigation_manager import NavigationState
    from simulation.closed_loop import ClosedLoopSimulation
    from simulation.scenario import default_scenario

    robot_cfg = load_robot_config()
    nav_cfg = load_navigation_config()
    scenario = default_scenario(robot_cfg.robot.width, robot_cfg.robot.length)
    sim = ClosedLoopSimulation(
        environment=scenario.environment,
        ground_truth_start=scenario.ground_truth_start,
        goal=scenario.slam_frame_goal,
        nav_cfg=nav_cfg,
        robot_cfg=robot_cfg,
        dt=0.1,
        max_simulation_time=60.0,
    )
    metrics = sim.run()

    assert metrics.collision is False
    assert metrics.final_nav_state != NavigationState.BLOCKED.value
    # Real, substantial ground-truth motion occurred (recovery + normal
    # driving), not a frozen/near-zero-progress run.
    assert metrics.ground_truth_distance > 1.0
    assert metrics.mean_position_error is not None and metrics.mean_position_error < 0.5
