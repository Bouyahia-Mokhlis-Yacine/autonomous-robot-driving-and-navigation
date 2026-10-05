"""Phase 12.5: diagnostic instrumentation for ``ClosedLoopSimulation``.

Debugging/analysis only -- nothing here is read by, or feeds back into,
the navigation stack. Everything is built by wrapping objects
``ClosedLoopSimulation`` already exposes publicly (``sim.global_planner``,
``sim.replanner``, ``sim.nav_manager.update``) from the OUTSIDE, so no
production algorithm (A*, the state machine, replanning) is modified in
any way -- this module only observes calls that were already going to
happen.

Two things are produced by one recorded run:
    - ``trace``: a ``DiagnosticSnapshot`` per tick (Part 2), cheap to log
      or export -- ground truth pose, estimated pose, nav state, command,
      replan reason, path length, local target.
    - ``failure_snapshot``: a detailed ``FailureSnapshot`` (Parts 3-6),
      captured at the EXACT ``GlobalPlanner.plan()`` call that first
      returns ``None`` -- raw and inflated grid statistics, robot/goal
      cell states, and independent A* reachability probes run directly
      against copies of the same grids (current config, UNKNOWN cost
      forced to 1.0, and the raw pre-inflation grid) to isolate whether
      inflation, UNKNOWN weighting, or the raw world-model geometry
      itself is what makes the goal unreachable at that moment.

Phase 14 adds two more, both purely diagnostic (nothing here changes
``planning.path_smoothing``'s behavior -- the pipeline stages below are
re-run with the SAME existing, unmodified functions, just captured for
inspection instead of only being visible to ``NavigationManager``):
    - ``plan_events``: for every successful ``GlobalPlanner.plan()`` call,
      the raw A* path, the line-of-sight-simplified path, and the final
      gradient-smoothed path (exactly what ``planning.path_smoothing.
      process_path(raw_path, grid_map, simplify=True, smooth=True)``
      computes internally, reconstructed stage-by-stage here), each with
      a ``simulation.path_clearance.ClearanceMeasurement``.
    - ``recovery_trigger_events``: one snapshot per NEW Phase 13 recovery
      attempt (detected via ``NavigationManager.get_recovery_attempts()``
      increasing, from the outside -- no new hook needed), with the
      robot's raw/inflated cell state, distance to the nearest raw
      OCCUPIED cell, and the stale path that was active when it triggered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.geometry import distance, wrap_angle
from common.messages import Goal, OccupancyGrid, Path, PathPoint, Pose2D, VelocityCommand
from mapping.inflation import compute_inflation_radius
from mapping.occupancy_grid import OccupancyGridMap
from planning.astar import astar
from planning.path_smoothing import remove_collinear_points, remove_duplicate_points, simplify_line_of_sight, smooth_path
from simulation.closed_loop import ClosedLoopSimulation
from simulation.path_clearance import ClearanceMeasurement, _nearest_occupied_distance_m, measure_path_clearance

Cell = tuple[int, int]

_CELL_STATE_NAMES = {
    OccupancyGrid.UNKNOWN: "UNKNOWN",
    OccupancyGrid.FREE: "FREE",
    OccupancyGrid.OCCUPIED: "OCCUPIED",
}


def cell_state_name(grid: OccupancyGrid, cell: Cell) -> str:
    row, col = cell
    if not (0 <= row < grid.height and 0 <= col < grid.width):
        return "OUT_OF_BOUNDS"
    return _CELL_STATE_NAMES[int(grid.cells[row, col])]


@dataclass(frozen=True)
class GridStats:
    unknown: int
    free: int
    occupied: int
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float

    @staticmethod
    def from_grid(grid: OccupancyGrid) -> "GridStats":
        return GridStats(
            unknown=int(np.count_nonzero(grid.cells == OccupancyGrid.UNKNOWN)),
            free=int(np.count_nonzero(grid.cells == OccupancyGrid.FREE)),
            occupied=int(np.count_nonzero(grid.cells == OccupancyGrid.OCCUPIED)),
            width=grid.width, height=grid.height, resolution=grid.resolution,
            origin_x=grid.origin_x, origin_y=grid.origin_y,
        )

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _reachability_probe(
    grid: OccupancyGrid, start: Cell, goal: Cell, allow_diagonal: bool, unknown_cost_multiplier: float
) -> dict:
    """Run A* directly against ``grid`` and report whether a path exists,
    without going through GlobalPlanner (diagnostic-only, never used by
    the real planner)."""
    for name, cell in (("start", start), ("goal", goal)):
        row, col = cell
        if not (0 <= row < grid.height and 0 <= col < grid.width):
            return {"path_exists": False, "path_length": None, "error": f"{name} cell out of bounds"}
        if grid.cells[row, col] == OccupancyGrid.OCCUPIED:
            return {"path_exists": False, "path_length": None, "error": f"{name} cell is occupied"}

    path = astar(grid, start, goal, allow_diagonal=allow_diagonal, unknown_cost_multiplier=unknown_cost_multiplier)
    return {"path_exists": path is not None, "path_length": (len(path) if path is not None else None), "error": None}


@dataclass(frozen=True)
class FailureSnapshot:
    tick: int
    timestamp: float
    ground_truth_pose: Pose2D
    estimated_pose: Pose2D
    goal: Goal

    robot_cell: Cell
    goal_cell: Cell
    robot_cell_state_raw: str
    robot_cell_state_planning: str
    goal_cell_state_raw: str
    goal_cell_state_planning: str

    raw_grid_stats: GridStats
    planning_grid_stats: GridStats

    inflation_radius: float
    robot_width: float
    robot_length: float
    safety_margin: float
    allow_diagonal: bool
    unknown_cost_multiplier: float

    # Part 5: independent reachability probes at the exact failure moment.
    path_exists_current_config: dict
    path_exists_unknown_cost_1: dict
    path_exists_raw_uninflated: dict

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ground_truth_pose"] = self.ground_truth_pose.__dict__
        d["estimated_pose"] = self.estimated_pose.__dict__
        d["goal"] = self.goal.__dict__
        d["raw_grid_stats"] = self.raw_grid_stats.as_dict()
        d["planning_grid_stats"] = self.planning_grid_stats.as_dict()
        return d


@dataclass
class DiagnosticSnapshot:
    tick: int
    timestamp: float
    ground_truth_pose: Pose2D
    estimated_pose: Pose2D
    goal: Goal | None
    navigation_state: str
    velocity_command: VelocityCommand | None
    replan_reason: str | None
    path_length: int | None
    local_target: PathPoint | None
    planning_attempted: bool
    planning_succeeded: bool | None


@dataclass
class ReplayFrame:
    """One frame for the visual replay (simulation.replay_viewer) --
    deliberately separate from DiagnosticSnapshot: this carries the
    (possibly-shared-by-reference, only-updated-when-rebuilt) grid
    snapshots the GUI needs to actually draw the map, which the plain
    per-tick diagnostic trace does not need and would be wasteful to
    duplicate into."""

    tick: int
    timestamp: float
    ground_truth_pose: Pose2D  # Environment's own (corner-origin) frame -- display/evaluation only
    ground_truth_pose_slam_frame: Pose2D  # same pose, translated into the SLAM frame for display alongside estimated_pose
    estimated_pose: Pose2D
    goal: Goal | None
    navigation_state: str
    path_points: tuple[PathPoint, ...]
    raw_path_points: tuple[PathPoint, ...]  # Phase 14: most recent raw A* path (pre-smoothing), for the replay overlay
    local_target: PathPoint | None
    raw_grid: OccupancyGrid
    planning_grid: OccupancyGrid
    scan_points_world: tuple[tuple[float, float], ...]
    position_error: float
    heading_error: float
    recovery_attempts: int  # Phase 13 -- see NavigationManager.get_recovery_attempts()
    recovery_elapsed: float  # Phase 13 -- see NavigationManager.get_recovery_elapsed()


@dataclass
class PlanEvent:
    """Phase 14: every stage of the real pipeline
    (``planning.path_smoothing.process_path(raw_path, grid_map,
    simplify=True, smooth=True)``, unmodified), captured for one
    successful ``GlobalPlanner.plan()`` call, each with a clearance
    measurement -- see ``simulation.path_clearance``."""

    tick: int
    timestamp: float
    start_pose: Pose2D
    goal: Goal
    raw_path: Path
    los_path: Path
    final_path: Path
    raw_clearance: ClearanceMeasurement
    los_clearance: ClearanceMeasurement
    final_clearance: ClearanceMeasurement


@dataclass
class RecoveryTriggerEvent:
    """Phase 14: one snapshot per NEW Phase 13 RECOVERING attempt --
    detected from the outside via ``NavigationManager.
    get_recovery_attempts()`` increasing, no new hook needed."""

    tick: int
    timestamp: float
    attempt_number: int
    ground_truth_pose_slam_frame: Pose2D
    estimated_pose: Pose2D
    robot_cell: Cell
    raw_cell_state: str
    planning_cell_state: str
    nearest_occupied_distance_raw_m: float
    inflation_radius_m: float
    active_path: Path | None
    active_path_clearance: ClearanceMeasurement | None


@dataclass
class DiagnosticRun:
    trace: list[DiagnosticSnapshot] = field(default_factory=list)
    frames: list[ReplayFrame] = field(default_factory=list)
    failure_snapshot: FailureSnapshot | None = None
    plan_events: list[PlanEvent] = field(default_factory=list)
    recovery_trigger_events: list[RecoveryTriggerEvent] = field(default_factory=list)


def run_with_diagnostics(
    sim: ClosedLoopSimulation,
    stop_ticks_after_failure: int | None = 10,
    record_frames: bool = True,
) -> DiagnosticRun:
    """Run ``sim`` to completion (or ``stop_ticks_after_failure`` ticks
    past the first planner failure, whichever is sooner), recording a
    full diagnostic trace and, on the first ``GlobalPlanner.plan()`` call
    that returns ``None``, a detailed ``FailureSnapshot``.

    Wraps ``sim.global_planner.plan`` and ``sim.replanner.should_replan``
    (both already-public attributes -- see ``ClosedLoopSimulation``) for
    the duration of this call and restores the originals afterward, so
    ``sim`` is left exactly as it would be without this module ever
    having run, other than the recorded observations.
    """
    run = DiagnosticRun()

    original_plan = sim.global_planner.plan
    original_should_replan = sim.replanner.should_replan
    original_update = sim.nav_manager.update

    state = {
        "replan_reason": None, "planning_attempted": False, "planning_succeeded": None, "last_cmd": None,
        "last_raw_path": None,
    }

    def wrapped_plan(grid_map: OccupancyGridMap, start: Pose2D, goal: Goal):
        state["planning_attempted"] = True
        result = original_plan(grid_map, start, goal)
        state["planning_succeeded"] = result is not None
        if result is None and run.failure_snapshot is None:
            run.failure_snapshot = _build_failure_snapshot(sim, grid_map, start, goal)
        if result is not None:
            run.plan_events.append(_build_plan_event(sim, grid_map, start, goal, result))
            state["last_raw_path"] = result
        return result

    def wrapped_should_replan(pose, path, grid_map, dt):
        should, reason = original_should_replan(pose, path, grid_map, dt)
        if should:
            state["replan_reason"] = reason
        return should, reason

    def wrapped_update(grid_map, dt, raw_grid_map=None):
        cmd = original_update(grid_map, dt, raw_grid_map)
        state["last_cmd"] = cmd
        return cmd

    sim.global_planner.plan = wrapped_plan  # type: ignore[method-assign]
    sim.replanner.should_replan = wrapped_should_replan  # type: ignore[method-assign]
    sim.nav_manager.update = wrapped_update  # type: ignore[method-assign]

    ticks_since_failure: int | None = None
    prev_recovery_attempts = sim.nav_manager.get_recovery_attempts()
    try:
        while sim.metrics.simulation_time < sim.max_simulation_time:
            state["replan_reason"] = None
            state["planning_attempted"] = False
            state["planning_succeeded"] = None
            path_before_step = sim.nav_manager.get_path()

            sim.step()

            new_attempts = sim.nav_manager.get_recovery_attempts()
            if new_attempts > prev_recovery_attempts:
                run.recovery_trigger_events.append(
                    _build_recovery_trigger_event(sim, new_attempts, path_before_step)
                )
            prev_recovery_attempts = new_attempts

            path = sim.nav_manager.get_path()
            run.trace.append(DiagnosticSnapshot(
                tick=sim.metrics.ticks,
                timestamp=sim.metrics.simulation_time,
                ground_truth_pose=sim.environment.robot_pose,
                estimated_pose=sim.nav_robot.get_pose(),
                goal=sim.nav_manager.get_goal(),
                navigation_state=sim.nav_manager.state.value,
                velocity_command=state["last_cmd"],
                replan_reason=state["replan_reason"],
                path_length=(len(path.points) if path is not None else None),
                local_target=sim.nav_manager.get_local_target(),
                planning_attempted=state["planning_attempted"],
                planning_succeeded=state["planning_succeeded"],
            ))

            if record_frames:
                run.frames.append(_build_replay_frame(sim, state["last_raw_path"]))

            if sim.metrics.goal_reached or sim.metrics.collision:
                break
            if run.failure_snapshot is not None:
                ticks_since_failure = 0 if ticks_since_failure is None else ticks_since_failure + 1
                if stop_ticks_after_failure is not None and ticks_since_failure >= stop_ticks_after_failure:
                    break
    finally:
        sim.global_planner.plan = original_plan  # type: ignore[method-assign]
        sim.replanner.should_replan = original_should_replan  # type: ignore[method-assign]
        sim.nav_manager.update = original_update  # type: ignore[method-assign]

    return run


def _build_failure_snapshot(
    sim: ClosedLoopSimulation, planning_grid_map: OccupancyGridMap, start: Pose2D, goal: Goal
) -> FailureSnapshot:
    raw_grid_map = sim.world_model.occupancy_grid
    raw_grid = raw_grid_map.grid
    planning_grid = planning_grid_map.grid

    robot_cell = raw_grid_map.world_to_grid(start.x, start.y)
    goal_cell = raw_grid_map.world_to_grid(goal.x, goal.y)

    inflation_radius = compute_inflation_radius(
        sim.robot_cfg.robot.width, sim.robot_cfg.robot.length, sim.robot_cfg.safety.inflation_radius
    )

    allow_diagonal = sim.global_planner.allow_diagonal
    unknown_cost_multiplier = sim.global_planner.unknown_cost_multiplier

    path_current = _reachability_probe(planning_grid, robot_cell, goal_cell, allow_diagonal, unknown_cost_multiplier)
    path_unknown_1 = _reachability_probe(planning_grid, robot_cell, goal_cell, allow_diagonal, 1.0)
    path_raw = _reachability_probe(raw_grid, robot_cell, goal_cell, allow_diagonal, unknown_cost_multiplier)

    return FailureSnapshot(
        tick=sim.metrics.ticks,
        timestamp=sim.metrics.simulation_time,
        ground_truth_pose=sim.environment.robot_pose,
        estimated_pose=start,
        goal=goal,
        robot_cell=robot_cell,
        goal_cell=goal_cell,
        robot_cell_state_raw=cell_state_name(raw_grid, robot_cell),
        robot_cell_state_planning=cell_state_name(planning_grid, robot_cell),
        goal_cell_state_raw=cell_state_name(raw_grid, goal_cell),
        goal_cell_state_planning=cell_state_name(planning_grid, goal_cell),
        raw_grid_stats=GridStats.from_grid(raw_grid),
        planning_grid_stats=GridStats.from_grid(planning_grid),
        inflation_radius=inflation_radius,
        robot_width=sim.robot_cfg.robot.width,
        robot_length=sim.robot_cfg.robot.length,
        safety_margin=sim.robot_cfg.safety.inflation_radius,
        allow_diagonal=allow_diagonal,
        unknown_cost_multiplier=unknown_cost_multiplier,
        path_exists_current_config=path_current,
        path_exists_unknown_cost_1=path_unknown_1,
        path_exists_raw_uninflated=path_raw,
    )


def _build_plan_event(
    sim: ClosedLoopSimulation, grid_map: OccupancyGridMap, start: Pose2D, goal: Goal, raw_path: Path
) -> PlanEvent:
    """Reconstructs exactly what ``planning.path_smoothing.process_path(
    raw_path, grid_map, simplify=True, smooth=True)`` computes internally
    (same functions, same arguments, same grid_map) -- so the caller can
    see each intermediate stage, without process_path itself exposing
    them and without modifying it to do so."""
    deduped = remove_duplicate_points(raw_path)
    collinear_reduced = remove_collinear_points(deduped)
    los_path = simplify_line_of_sight(collinear_reduced, grid_map)
    final_path = smooth_path(los_path, grid_map)

    return PlanEvent(
        tick=sim.metrics.ticks,
        timestamp=sim.metrics.simulation_time,
        start_pose=start,
        goal=goal,
        raw_path=raw_path,
        los_path=los_path,
        final_path=final_path,
        raw_clearance=measure_path_clearance(raw_path, grid_map),
        los_clearance=measure_path_clearance(los_path, grid_map),
        final_clearance=measure_path_clearance(final_path, grid_map),
    )


def _build_recovery_trigger_event(
    sim: ClosedLoopSimulation, attempt_number: int, active_path: Path | None
) -> RecoveryTriggerEvent:
    raw_grid_map = sim.world_model.occupancy_grid
    raw_grid = raw_grid_map.grid
    planning_grid_map = sim.planning_map
    planning_grid = planning_grid_map.grid

    estimated_pose = sim.nav_robot.get_pose()
    robot_cell = raw_grid_map.world_to_grid(estimated_pose.x, estimated_pose.y)
    inflation_radius = compute_inflation_radius(
        sim.robot_cfg.robot.width, sim.robot_cfg.robot.length, sim.robot_cfg.safety.inflation_radius
    )

    nearest_occupied_m = _nearest_occupied_distance_m(
        raw_grid, estimated_pose.x, estimated_pose.y, search_radius_m=max(2.0, inflation_radius * 2.0)
    )

    return RecoveryTriggerEvent(
        tick=sim.metrics.ticks,
        timestamp=sim.metrics.simulation_time,
        attempt_number=attempt_number,
        ground_truth_pose_slam_frame=sim._ground_truth_to_slam(sim.environment.robot_pose),
        estimated_pose=estimated_pose,
        robot_cell=robot_cell,
        raw_cell_state=cell_state_name(raw_grid, robot_cell),
        planning_cell_state=cell_state_name(planning_grid, robot_cell),
        nearest_occupied_distance_raw_m=nearest_occupied_m,
        inflation_radius_m=inflation_radius,
        active_path=active_path,
        active_path_clearance=(
            measure_path_clearance(active_path, planning_grid_map) if active_path is not None else None
        ),
    )


def _build_replay_frame(sim: ClosedLoopSimulation, last_raw_path: Path | None = None) -> ReplayFrame:
    path = sim.nav_manager.get_path()
    expected = sim._ground_truth_to_slam(sim.environment.robot_pose)
    estimated = sim.nav_robot.get_pose()
    return ReplayFrame(
        tick=sim.metrics.ticks,
        timestamp=sim.metrics.simulation_time,
        ground_truth_pose=sim.environment.robot_pose,
        ground_truth_pose_slam_frame=expected,
        estimated_pose=estimated,
        goal=sim.nav_manager.get_goal(),
        navigation_state=sim.nav_manager.state.value,
        path_points=tuple(path.points) if path is not None else (),
        raw_path_points=tuple(last_raw_path.points) if last_raw_path is not None else (),
        local_target=sim.nav_manager.get_local_target(),
        raw_grid=sim.world_model.get_grid(),
        planning_grid=sim.planning_map.grid,
        scan_points_world=(),  # populated by the caller if it wants scan overlays; kept optional/cheap by default
        position_error=distance(expected.x, expected.y, estimated.x, estimated.y),
        heading_error=abs(wrap_angle(expected.theta - estimated.theta)),
        recovery_attempts=sim.nav_manager.get_recovery_attempts(),
        recovery_elapsed=sim.nav_manager.get_recovery_elapsed(),
    )
