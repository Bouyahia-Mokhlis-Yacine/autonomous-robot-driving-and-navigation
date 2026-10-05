"""Phase 10 tests: navigation.world_model_bridge -- the plumbing
between PersistentWorldModel/ScanMatchingPipeline (Phases 1-8) and the
existing, unmodified navigation stack (GlobalPlanner/NavigationManager/
SensorOnlyRobot).

Deterministic, hardware-free throughout: synthetic scans with
ground-truth poses (no ICP), synthetic PipelineResult instances (no
live registration). Proves composition, not new logic -- every
assertion here is about whether existing, already-tested components
correctly consume this bridge's output, not about re-testing those
components' own internals.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.config import load_robot_config
from common.messages import Goal, LaserScan, OccupancyGrid, Pose2D
from mapping.inflation import compute_inflation_radius
from mapping.occupancy_grid import OccupancyGridMap
from mapping.persistent_world_model import PersistentWorldModel
from navigation.world_model_bridge import apply_accepted_pose, build_planning_map
from planning.global_planner import GlobalPlanner
from robot.sensor_only_robot import SensorOnlyRobot
from slam.scan_matching_pipeline import BootstrapStatus, PipelineResult, RegistrationResult

ROBOT_CONFIG = load_robot_config()


def make_scan(hits: dict[float, float], timestamp: float = 0.0, background: float = 0.0) -> LaserScan:
    """Same convention as tests/test_persistent_world_model.py: background
    defaults to an invalid (no-op) range so only the given bearings
    have any effect."""
    angle_min = -math.pi
    angle_increment = math.radians(1.0)
    num_bins = int(round(2.0 * math.pi / angle_increment)) + 1
    ranges = [background] * num_bins
    for angle_deg, r in hits.items():
        wrapped_deg = ((angle_deg + 180.0) % 360.0) - 180.0  # into (-180, 180], matching angle_min=-pi
        idx = round((math.radians(wrapped_deg) - angle_min) / angle_increment)
        idx = max(0, min(num_bins - 1, idx))
        ranges[idx] = r
    return LaserScan(
        timestamp=timestamp, angle_min=angle_min, angle_increment=angle_increment,
        ranges=ranges, range_min=0.15, range_max=8.0,
    )


def confirm_point(model: PersistentWorldModel, pose: Pose2D, bearing_deg: float, r: float) -> None:
    """Directly (ground-truth pose, no ICP) confirm one obstacle point."""
    for t in (0.0, 0.2, 0.5):
        model.integrate(pose, make_scan({bearing_deg: r}, timestamp=t), now=t)


def make_result(pose: Pose2D, **overrides) -> PipelineResult:
    defaults = dict(
        pose=pose, accepted=True, localized=True, consecutive_failures=0,
        registration_result=None, rejection_reason=None, map_integrated=True,
        bootstrap_status=BootstrapStatus.COMPLETE,
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


ORIGIN_POSE = Pose2D(x=0.0, y=0.0, theta=0.0)


# --- Part A: planning-map bridge --------------------------------------------

def test_build_planning_map_returns_occupancy_grid_map_with_matching_metadata() -> None:
    model = PersistentWorldModel()
    source = model.get_grid()

    planning_map = build_planning_map(model, ROBOT_CONFIG)

    assert isinstance(planning_map, OccupancyGridMap)
    assert planning_map.grid.resolution == pytest.approx(source.resolution)
    assert planning_map.grid.origin_x == pytest.approx(source.origin_x)
    assert planning_map.grid.origin_y == pytest.approx(source.origin_y)
    assert planning_map.grid.width == source.width
    assert planning_map.grid.height == source.height
    assert planning_map.grid.cells.shape == source.cells.shape


def test_planning_map_uses_existing_robot_configuration_for_inflation() -> None:
    model = PersistentWorldModel()
    confirm_point(model, ORIGIN_POSE, 0.0, 3.0)

    planning_map = build_planning_map(model, ROBOT_CONFIG)

    expected_radius = compute_inflation_radius(
        ROBOT_CONFIG.robot.width, ROBOT_CONFIG.robot.length, ROBOT_CONFIG.safety.inflation_radius
    )
    resolution = planning_map.grid.resolution
    # A cell strictly inside the configured radius (floor, not ceil --
    # inflate_occupancy_grid includes a cell iff its distance <=
    # inflation_radius, so ceil(radius/res) cells away can legitimately
    # fall just OUTSIDE the radius) must be occupied...
    inside_cells = int(math.floor(expected_radius / resolution)) - 1
    # ...while a cell clearly beyond the radius must not be (proves the
    # ACTUAL configured radius was used, not some other arbitrary value).
    outside_cells = int(math.ceil(expected_radius / resolution)) + 5
    row, col = planning_map.world_to_grid(3.0, 0.0)

    assert planning_map.get_cell(row, col + inside_cells) == OccupancyGrid.OCCUPIED
    assert planning_map.get_cell(row, col + outside_cells) != OccupancyGrid.OCCUPIED


def test_world_model_grid_is_not_mutated_by_the_bridge() -> None:
    model = PersistentWorldModel()
    confirm_point(model, ORIGIN_POSE, 0.0, 3.0)
    confirm_point(model, ORIGIN_POSE, 90.0, 2.0)

    state_before = model.state.copy()
    pending_before = model.pending_since.copy()
    grid_cells_before = model.get_grid().cells.copy()

    build_planning_map(model, ROBOT_CONFIG)

    np.testing.assert_array_equal(model.state, state_before)
    both_nan = np.isnan(model.pending_since) & np.isnan(pending_before)
    np.testing.assert_array_equal(np.isnan(model.pending_since), np.isnan(pending_before))
    np.testing.assert_array_equal(model.pending_since[~both_nan], pending_before[~both_nan])
    np.testing.assert_array_equal(model.get_grid().cells, grid_cells_before)


def test_planning_map_is_an_independent_copy_not_a_view() -> None:
    model = PersistentWorldModel()
    confirm_point(model, ORIGIN_POSE, 0.0, 3.0)

    planning_map = build_planning_map(model, ROBOT_CONFIG)
    planning_map.grid.cells[0, 0] = OccupancyGrid.OCCUPIED  # mutate the returned map directly

    # The world model's own grid must be unaffected.
    assert model.get_grid().cells[0, 0] != OccupancyGrid.OCCUPIED


def test_confirmed_obstacle_remains_occupied_after_inflation() -> None:
    model = PersistentWorldModel()
    confirm_point(model, ORIGIN_POSE, 0.0, 3.0)
    assert model.get_cell_state(3.0, 0.0).name == "CONFIRMED"

    planning_map = build_planning_map(model, ROBOT_CONFIG)
    row, col = planning_map.world_to_grid(3.0, 0.0)
    assert planning_map.get_cell(row, col) == OccupancyGrid.OCCUPIED


def test_tentative_geometry_is_not_treated_as_occupied_in_planning_map() -> None:
    # A single observation -- tentative, not confirmed (Phase 5's own
    # rule, unchanged and untouched by this bridge).
    model = PersistentWorldModel()
    model.integrate(ORIGIN_POSE, make_scan({0.0: 3.0}, timestamp=0.0), now=0.0)

    planning_map = build_planning_map(model, ROBOT_CONFIG)
    row, col = planning_map.world_to_grid(3.0, 0.0)
    assert planning_map.get_cell(row, col) != OccupancyGrid.OCCUPIED


# --- Part C, test 1: confirmed obstacle + GlobalPlanner ---------------------

def test_global_planner_routes_around_confirmed_obstacle_via_bridge() -> None:
    model = PersistentWorldModel()
    # A short confirmed wall directly between start and goal, with a
    # side gap the planner must find.
    wall_pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    for bearing in (80.0, 85.0, 90.0, 95.0, 100.0):
        confirm_point(model, wall_pose, bearing, 2.0)

    planning_map = build_planning_map(model, ROBOT_CONFIG)
    planner = GlobalPlanner(allow_diagonal=True)

    path = planner.plan(planning_map, start=Pose2D(x=0.0, y=0.0, theta=0.0), goal=Goal(x=0.0, y=4.0))
    assert path is not None
    assert not path.is_empty()
    # The path must not pass through any occupied (inflated) cell.
    for point in path.points:
        row, col = planning_map.world_to_grid(point.x, point.y)
        assert planning_map.get_cell(row, col) != OccupancyGrid.OCCUPIED


def test_global_planner_reports_no_path_when_fully_enclosed() -> None:
    model = PersistentWorldModel()
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    # A confirmed ring of obstacles completely surrounding the goal,
    # inflated enough (via the small ring radius + default inflation)
    # to leave no gap.
    for bearing_deg in range(0, 360, 10):
        confirm_point(model, pose, float(bearing_deg), 1.0)

    planning_map = build_planning_map(model, ROBOT_CONFIG)
    planner = GlobalPlanner(allow_diagonal=True)
    path = planner.plan(planning_map, start=Pose2D(x=0.0, y=0.0, theta=0.0), goal=Goal(x=5.0, y=5.0))
    assert path is None


# --- Part C, test 2: world coordinates preserved (no shift/rotation) -------

def test_world_coordinates_preserved_through_bridge_and_planner() -> None:
    model = PersistentWorldModel()
    confirm_point(model, ORIGIN_POSE, 45.0, 2.0)  # obstacle at world (~1.414, 1.414)

    planning_map = build_planning_map(model, ROBOT_CONFIG)
    obstacle_x = 2.0 * math.cos(math.radians(45.0))
    obstacle_y = 2.0 * math.sin(math.radians(45.0))
    row, col = planning_map.world_to_grid(obstacle_x, obstacle_y)
    assert planning_map.get_cell(row, col) == OccupancyGrid.OCCUPIED

    # A path planned well away from the obstacle bearing must reach
    # its exact requested world-coordinate goal (GlobalPlanner snaps
    # the final waypoint to the exact goal) -- proving no shift or
    # rotation was introduced anywhere in the bridge.
    planner = GlobalPlanner(allow_diagonal=True)
    goal = Goal(x=-3.0, y=0.0)
    path = planner.plan(planning_map, start=Pose2D(x=0.0, y=0.0, theta=0.0), goal=goal)
    assert path is not None
    assert path.points[-1].x == pytest.approx(goal.x)
    assert path.points[-1].y == pytest.approx(goal.y)
    assert path.points[0].x == pytest.approx(0.0, abs=planning_map.grid.resolution)
    assert path.points[0].y == pytest.approx(0.0, abs=planning_map.grid.resolution)


# --- Part C, test 3: source map immutability (explicit, standalone) --------

def test_source_world_model_grid_unchanged_after_bridge_snapshot_comparison() -> None:
    model = PersistentWorldModel()
    confirm_point(model, ORIGIN_POSE, 0.0, 3.0)

    snapshot = model.get_grid().cells.copy()
    build_planning_map(model, ROBOT_CONFIG)
    after = model.get_grid().cells

    np.testing.assert_array_equal(snapshot, after)


# --- Part B / D: pose bridge --------------------------------------------------

def test_accepted_result_updates_robot_pose() -> None:
    robot = SensorOnlyRobot(initial_pose=Pose2D(x=0.0, y=0.0, theta=0.0))
    accepted = make_result(Pose2D(x=1.5, y=-0.5, theta=0.2))

    updated = apply_accepted_pose(robot, accepted)

    assert updated is True
    assert robot.get_pose() == Pose2D(x=1.5, y=-0.5, theta=0.2)


def test_rejected_result_does_not_change_robot_pose() -> None:
    robot = SensorOnlyRobot(initial_pose=Pose2D(x=0.0, y=0.0, theta=0.0))
    apply_accepted_pose(robot, make_result(Pose2D(x=1.0, y=1.0, theta=0.1)))  # establish a known pose
    pose_before = robot.get_pose()

    rejected = make_result(
        Pose2D(x=99.0, y=99.0, theta=3.0),  # even if pose looks like a huge jump
        accepted=False, rejection_reason="insufficient_correspondences", consecutive_failures=1,
    )
    updated = apply_accepted_pose(robot, rejected)

    assert updated is False
    assert robot.get_pose() == pose_before
    assert robot.get_pose().x == pytest.approx(1.0)
    assert robot.get_pose().y == pytest.approx(1.0)


def test_pose_bridge_sequence_accept_then_reject_then_accept() -> None:
    """Integration-style: a short sequence of accepted/rejected results,
    proving the robot's pose only ever reflects the most recent
    ACCEPTED one, never a rejected one, at every step."""
    robot = SensorOnlyRobot(initial_pose=Pose2D(x=0.0, y=0.0, theta=0.0))

    apply_accepted_pose(robot, make_result(Pose2D(x=0.1, y=0.0, theta=0.0)))
    assert robot.get_pose().x == pytest.approx(0.1)

    apply_accepted_pose(robot, make_result(Pose2D(x=50.0, y=50.0, theta=1.0), accepted=False))
    assert robot.get_pose().x == pytest.approx(0.1)  # unchanged by the rejection

    apply_accepted_pose(robot, make_result(Pose2D(x=0.2, y=0.05, theta=0.02)))
    assert robot.get_pose().x == pytest.approx(0.2)
    assert robot.get_pose().y == pytest.approx(0.05)


# --- Part E: minimal end-to-end composition (no live ICP, ground-truth poses) --

def test_end_to_end_world_model_to_navigation_composition() -> None:
    """Proves the full intended composition without live ICP: confirmed
    world-model geometry -> planning map -> GlobalPlanner -> Path, and
    an accepted pipeline-shaped result -> SensorOnlyRobot ->
    NavigationManager reading it back via get_pose() -- using the
    existing, completely unmodified NavigationManager/GlobalPlanner/
    Replanner/SensorOnlyRobot classes."""
    from control.path_follower import PathFollower, PathFollowerConfig
    from navigation.navigation_manager import NavigationManager
    from navigation.replanner import Replanner, ReplannerConfig

    model = PersistentWorldModel()
    confirm_point(model, ORIGIN_POSE, 90.0, 3.0)  # an obstacle off to one side, not blocking the direct route

    robot = SensorOnlyRobot(initial_pose=Pose2D(x=0.0, y=0.0, theta=0.0))
    apply_accepted_pose(robot, make_result(Pose2D(x=0.0, y=0.0, theta=0.0)))

    planning_map = build_planning_map(model, ROBOT_CONFIG)

    follower = PathFollower(PathFollowerConfig(
        max_linear_velocity=ROBOT_CONFIG.limits.max_linear_velocity,
        max_angular_velocity=ROBOT_CONFIG.limits.max_angular_velocity,
        max_linear_acceleration=ROBOT_CONFIG.limits.max_linear_acceleration,
        max_angular_acceleration=ROBOT_CONFIG.limits.max_angular_acceleration,
        goal_tolerance=ROBOT_CONFIG.safety.goal_tolerance,
        heading_gain=ROBOT_CONFIG.control.heading_gain,
        waypoint_tolerance=ROBOT_CONFIG.control.waypoint_tolerance,
    ))
    replanner = Replanner(ReplannerConfig(
        replan_interval=1.0, deviation_threshold=ROBOT_CONFIG.safety.path_deviation_threshold,
    ))
    nav_manager = NavigationManager(
        global_planner=GlobalPlanner(allow_diagonal=True),
        path_follower=follower, robot=robot, replanner=replanner,
        obstacle_stop_distance=ROBOT_CONFIG.safety.emergency_stop_distance,
    )

    nav_manager.set_goal(Goal(x=-3.0, y=0.0), planning_map)

    from navigation.navigation_manager import NavigationState
    assert nav_manager.state == NavigationState.MOVING
    path = nav_manager.get_path()
    assert path is not None and not path.is_empty()

    cmd = nav_manager.update(planning_map, dt=0.1)
    assert cmd is not None  # a real VelocityCommand was produced by the UNMODIFIED existing stack
