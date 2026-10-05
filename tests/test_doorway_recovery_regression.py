"""Regression: the doorway RECOVERING loop.

Mechanism: the planned path runs
along the inflation-buffer boundary around the doorway corner; the path
follower cuts the chord on the inside of that curve, so the robot's centre
sags 1-3.5 cm INTO the 0.25 m safety buffer (real footprint clearance still
~0.27 m). The reactive obstacle-ahead ray then started inside the buffer and
fired; the replan saw an inflated start cell and entered RECOVERING, which
reversed ~0.9 m straight back; the fresh plan hugged the same boundary and
the robot returned to the same spot -- 16 cycles, 190 s, 32.7 m driven for a
3 m route. The fix treats a SHALLOW intrusion as being at the nearest
buffer-free cell, both for the planner start and for the reactive ray;
deep intrusions still get the reverse recovery.
"""

from __future__ import annotations

import math

import pytest

from common.config import load_navigation_config, load_robot_config
from common.messages import Goal, OccupancyGrid, Pose2D, VelocityCommand
from control.path_follower import PathFollower, PathFollowerConfig
from mapping.inflation import compute_inflation_radius, inflate_occupancy_grid
from mapping.occupancy_grid import OccupancyGridMap
from navigation.navigation_manager import NavigationManager, NavigationState
from navigation.recovery import RecoveryConfig, buffer_escape_point, inflated_start_cell_blocked
from navigation.replanner import Replanner, ReplannerConfig
from planning.global_planner import GlobalPlanner

ROBOT_CFG = load_robot_config()
NAV_CFG = load_navigation_config()
INFLATION = compute_inflation_radius(ROBOT_CFG.robot.width, ROBOT_CFG.robot.length, ROBOT_CFG.safety.inflation_radius)


def maps_with_wall(wall_x: float = 3.0) -> tuple[OccupancyGridMap, OccupancyGridMap]:
    """10 x 10 m at 5 cm, all observed FREE, one vertical wall at x = wall_x
    (y in [2, 8]); returns (raw, inflated planning map)."""
    raw = OccupancyGridMap(resolution=0.05, width_m=10.0, height_m=10.0)
    raw.grid.cells[:, :] = OccupancyGrid.FREE
    for y in range(40, 160):
        raw.set_cell(y, int(wall_x / 0.05), OccupancyGrid.OCCUPIED)
    planning = OccupancyGridMap(resolution=0.05, width_m=10.0, height_m=10.0)
    planning.grid = inflate_occupancy_grid(raw.grid, INFLATION)
    return raw, planning


class FakeRobot:
    def __init__(self, pose: Pose2D) -> None:
        self.pose = pose
        self.last_command = VelocityCommand(0.0, 0.0)

    def set_velocity(self, cmd): self.last_command = cmd
    def stop(self): self.last_command = VelocityCommand(0.0, 0.0)
    def get_pose(self): return self.pose


def manager(pose: Pose2D, max_escape: float = 0.25) -> NavigationManager:
    follower = PathFollower(PathFollowerConfig(
        max_linear_velocity=0.5, max_angular_velocity=1.5, max_linear_acceleration=5.0,
        max_angular_acceleration=5.0, goal_tolerance=0.15, heading_gain=2.0, waypoint_tolerance=0.2))
    return NavigationManager(
        global_planner=GlobalPlanner(allow_diagonal=True), path_follower=follower, robot=FakeRobot(pose),
        replanner=Replanner(ReplannerConfig(replan_interval=100.0, deviation_threshold=100.0)),
        obstacle_stop_distance=0.30, recovery_config=RecoveryConfig(max_escape_distance=max_escape),
    )


# ----------------------------------------------------------- escape point --

def test_escape_point_is_the_nearest_buffer_free_cell() -> None:
    raw, planning = maps_with_wall()
    pose = Pose2D(3.0 - INFLATION + 0.03, 5.0, 0.0)  # 3 cm inside the buffer
    assert inflated_start_cell_blocked(pose, raw, planning)
    escape = buffer_escape_point(pose, raw, planning, 0.25)
    assert escape is not None
    r, c = planning.world_to_grid(escape.x, escape.y)
    assert planning.get_cell(r, c) != OccupancyGrid.OCCUPIED
    assert escape.x < pose.x and math.hypot(escape.x - pose.x, escape.y - pose.y) < 0.1  # a few cm, away from the wall


def test_no_escape_for_a_deep_intrusion_or_when_disabled() -> None:
    raw, planning = maps_with_wall()
    deep = Pose2D(3.0 - INFLATION + 0.35, 5.0, 0.0)  # 35 cm in: bounding circle overlaps the wall
    assert buffer_escape_point(deep, raw, planning, 0.25) is None
    shallow = Pose2D(3.0 - INFLATION + 0.03, 5.0, 0.0)
    assert buffer_escape_point(shallow, raw, planning, 0.0) is None


def test_escape_never_crosses_a_raw_obstacle_or_uses_unknown_cells() -> None:
    raw, planning = maps_with_wall()
    pose = Pose2D(3.0 - INFLATION + 0.03, 5.0, 0.0)
    # Make every candidate on the free side UNKNOWN in the raw map: unobserved
    # space is never an escape target.
    for r in range(80, 121):
        for c in range(40, int((3.0 - INFLATION) / 0.05) + 1):
            raw.set_cell(r, c, OccupancyGrid.UNKNOWN)
    assert buffer_escape_point(pose, raw, planning, 0.25) is None


# ---------------------------------------------------- planner / reactive --

def test_shallow_intrusion_plans_onward_instead_of_reversing() -> None:
    raw, planning = maps_with_wall()
    pose = Pose2D(3.0 - INFLATION + 0.03, 5.0, math.pi / 2)  # heading along the wall
    nm = manager(pose)
    nm.set_goal(Goal(1.0, 8.5), planning, raw)
    assert nm.state is NavigationState.MOVING
    first = nm.get_path().points[0]
    assert planning.get_cell(*planning.world_to_grid(first.x, first.y)) != OccupancyGrid.OCCUPIED
    cmd = nm.update(planning, 0.1, raw)
    assert nm.state is NavigationState.MOVING
    assert cmd.linear >= 0.0  # never the reverse maneuver


def test_deep_intrusion_still_uses_the_reverse_recovery() -> None:
    raw, planning = maps_with_wall()
    nm = manager(Pose2D(3.0 - INFLATION + 0.35, 5.0, 0.0))
    nm.set_goal(Goal(1.0, 8.5), planning, raw)
    assert nm.state is NavigationState.RECOVERING


def test_reactive_stop_still_fires_when_heading_deeper_into_the_buffer() -> None:
    """Shallowly inside the buffer, facing the wall: must stop."""
    raw, planning = maps_with_wall()
    pose = Pose2D(3.0 - INFLATION + 0.03, 5.0, 0.0)  # facing +x, the wall
    nm = manager(pose)
    nm.set_goal(Goal(1.0, 8.5), planning, raw)
    cmd = nm.update(planning, 0.1, raw)
    assert cmd == VelocityCommand(0.0, 0.0)
    assert nm.state in (NavigationState.REPLANNING, NavigationState.MOVING)
    assert nm._reactive_stop_awaiting_follow  # the reactive stop fired


def test_reactive_stop_does_not_fire_when_leaving_along_the_buffer() -> None:
    """Shallowly inside the buffer, heading along/away from the wall: the
    ray no longer starts inside the buffer, so no spurious stop."""
    raw, planning = maps_with_wall()
    pose = Pose2D(3.0 - INFLATION + 0.03, 5.0, math.pi / 2 + 0.3)  # along the wall, slightly away
    nm = manager(pose)
    nm.set_goal(Goal(1.0, 8.5), planning, raw)
    nm.update(planning, 0.1, raw)
    assert not nm._reactive_stop_awaiting_follow


# ------------------------------------------------------ scenario regression --

@pytest.mark.parametrize("mode", ["ideal", "realistic"])
def test_doorway_is_passed_without_a_recovery_loop(mode: str) -> None:
    """The original failure scenario. The robot must reach the same
    situation (its own cell inside the buffer near the doorway corner),
    resolve it without RECOVERING, and reach the goal on a direct route."""
    import navigation.navigation_manager as nm_module
    from simulation.closed_loop import ClosedLoopSimulation
    from simulation.scenario import default_scenario
    from simulation.sensor_simulation_config import load_sensor_simulation_config

    sc = default_scenario(ROBOT_CFG.robot.width, ROBOT_CFG.robot.length)
    sim = ClosedLoopSimulation(sc.environment, sc.ground_truth_start, sc.slam_frame_goal, NAV_CFG, ROBOT_CFG,
                               max_simulation_time=60.0,
                               sensor_simulation=load_sensor_simulation_config(mode=mode, seed=42))
    escapes = []
    original = nm_module.buffer_escape_point

    def observed(pose, raw, planning, max_distance):
        point = original(pose, raw, planning, max_distance)
        if point is not None:
            escapes.append((pose.x, pose.y))
        return point

    nm_module.buffer_escape_point = observed
    entered_recovering = 0
    prev = sim.nav_manager.state
    try:
        while sim._sim_time < 60.0 - 1e-9:
            sim.step()
            state = sim.nav_manager.state
            if state is NavigationState.RECOVERING and prev is not NavigationState.RECOVERING:
                entered_recovering += 1
            prev = state
            if sim.metrics.goal_reached or sim.metrics.collision:
                break
    finally:
        nm_module.buffer_escape_point = original

    m = sim.metrics
    assert escapes, "the robot never entered the buffer -- the original situation was not reproduced"
    assert entered_recovering == 0
    assert m.goal_reached and not m.collision
    assert m.ground_truth_distance < 1.5 * m.planned_path_length   # no back-and-forth (was 32.7 m for 3 m)
    assert m.simulation_time < 40.0                                 # was ~190 s (ideal) / ~82 s (realistic)
    assert m.max_position_error < 0.15
