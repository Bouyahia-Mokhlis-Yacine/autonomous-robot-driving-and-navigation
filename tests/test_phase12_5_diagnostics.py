"""Phase 12.5 tests: diagnostic instrumentation, reachability probes, and
the visual replay's data contract for the Phase 12 BLOCKED investigation.

Runtime note: ``test_scenario_reproduces_blocked_with_full_diagnostic_evidence``
runs the full deterministic Phase 12 scenario to its actual BLOCKED
transition (~33s of simulated time) -- per the same wall-clock cost
explained in ``tests/test_closed_loop_simulation.py``'s module docstring,
this single test takes on the order of 90 seconds.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from common.config import load_navigation_config, load_robot_config
from common.messages import OccupancyGrid
from simulation.closed_loop import ClosedLoopSimulation
from simulation.diagnostics import _reachability_probe, cell_state_name, run_with_diagnostics
from simulation.scenario import default_scenario

ROBOT_CFG = load_robot_config()
NAV_CFG = load_navigation_config()


def make_grid(rows: list[list[int]], resolution: float = 1.0, origin_x: float = 0.0, origin_y: float = 0.0) -> OccupancyGrid:
    cells = np.array(rows, dtype=np.int8)
    height, width = cells.shape
    return OccupancyGrid(resolution=resolution, width=width, height=height, origin_x=origin_x, origin_y=origin_y, cells=cells)


F, U, O = OccupancyGrid.FREE, OccupancyGrid.UNKNOWN, OccupancyGrid.OCCUPIED


# --- Test 4: robot/goal grid-state inspection ------------------------------

def test_cell_state_name_reports_correct_states() -> None:
    grid = make_grid([[F, O, U]])
    assert cell_state_name(grid, (0, 0)) == "FREE"
    assert cell_state_name(grid, (0, 1)) == "OCCUPIED"
    assert cell_state_name(grid, (0, 2)) == "UNKNOWN"
    assert cell_state_name(grid, (5, 5)) == "OUT_OF_BOUNDS"
    assert cell_state_name(grid, (-1, 0)) == "OUT_OF_BOUNDS"


# --- Test 5: raw-vs-inflated reachability diagnostics -----------------------

def test_reachability_probe_detects_inflation_closing_a_corridor() -> None:
    # A single-row corridor, all FREE: reachable.
    open_corridor = make_grid([[F, F, F, F, F]])
    result = _reachability_probe(open_corridor, (0, 0), (0, 4), allow_diagonal=False, unknown_cost_multiplier=1.0)
    assert result["path_exists"] is True
    assert result["path_length"] == 5

    # The same corridor with its middle cell OCCUPIED (simulating
    # inflation closing the gap): unreachable.
    closed_corridor = make_grid([[F, F, O, F, F]])
    result = _reachability_probe(closed_corridor, (0, 0), (0, 4), allow_diagonal=False, unknown_cost_multiplier=1.0)
    assert result["path_exists"] is False
    assert result["path_length"] is None


def test_reachability_probe_reports_occupied_start_cell_explicitly() -> None:
    """This is exactly the Phase 12 BLOCKED failure mode: the START cell
    itself (the robot's own position) is occupied in the inflated map."""
    grid = make_grid([[O, F, F]])
    result = _reachability_probe(grid, (0, 0), (0, 2), allow_diagonal=False, unknown_cost_multiplier=1.0)
    assert result["path_exists"] is False
    assert result["error"] == "start cell is occupied"


def test_reachability_probe_out_of_bounds_cell() -> None:
    grid = make_grid([[F, F, F]])
    result = _reachability_probe(grid, (5, 5), (0, 2), allow_diagonal=False, unknown_cost_multiplier=1.0)
    assert result["path_exists"] is False
    assert result["error"] == "start cell out of bounds"


# --- Test 6: the visual replay's data contract ------------------------------

def test_replay_viewer_rejects_empty_frame_list() -> None:
    """No tkinter window is ever touched here -- the empty-list guard is
    the first line of __init__, before any Tk call (matches this
    project's convention of never constructing a live Tk window from the
    automated test suite -- see visualization/gui.py's own tests)."""
    from visualization.replay_viewer import ReplayViewer

    with pytest.raises(ValueError):
        ReplayViewer(frames=[])


def test_diagnostics_and_replay_modules_do_not_leak_ground_truth_into_navigation() -> None:
    """Static check: simulation.diagnostics and visualization.replay_viewer
    must never call anything that would push ground truth into the
    navigation stack (set_goal/set_velocity on a real robot interface,
    or NavigationManager construction) -- they only ever READ already-
    recorded/public state for display and analysis."""
    import simulation.diagnostics as diagnostics_module
    import visualization.replay_viewer as replay_viewer_module

    for module in (diagnostics_module, replay_viewer_module):
        source = inspect.getsource(module)
        assert "NavigationManager(" not in source
        assert ".set_pose(" not in source
        assert "robot.set_velocity(" not in source


# --- Tests 1, 2 & 5 (combined, expensive) -----------------------------------
# UPDATED for Phase 13 (see navigation/recovery.py). This test originally
# proved the Phase 12.5 root cause: the scenario permanently entering
# BLOCKED via an inflated-only-occupied start cell, captured as a
# FailureSnapshot at the exact GlobalPlanner.plan() call that returned
# None. Phase 13 added a bounded recovery maneuver that intercepts that
# exact condition INSIDE NavigationManager._plan() -- before
# GlobalPlanner.plan() is ever called for it -- so this diagnostic run no
# longer reaches BLOCKED and no longer captures a FailureSnapshot for this
# scenario. That is the intended, demonstrated fix (see
# tests/test_recovery.py and the Phase 13 final report), not a defect in
# this test or the diagnostic tooling -- both are re-verified below
# against their own, still-valid contracts instead.

def test_scenario_no_longer_reaches_permanent_blocked_after_phase13_recovery() -> None:
    scenario = default_scenario(ROBOT_CFG.robot.width, ROBOT_CFG.robot.length)
    sim = ClosedLoopSimulation(
        environment=scenario.environment,
        ground_truth_start=scenario.ground_truth_start,
        goal=scenario.slam_frame_goal,
        nav_cfg=NAV_CFG,
        robot_cfg=ROBOT_CFG,
        dt=0.1,
        max_simulation_time=45.0,
    )

    run = run_with_diagnostics(sim, stop_ticks_after_failure=3, record_frames=True)

    # The Phase 12.5 finding is fixed: no permanent BLOCKED, and (since
    # recovery intercepts the condition before GlobalPlanner.plan() is
    # ever asked) no failure snapshot is captured for this scenario.
    assert sim.nav_manager.state.value != "BLOCKED"
    assert run.failure_snapshot is None

    # The inflated-start-cell condition is now resolved before it becomes a
    # problem: shallow intrusions plan from the nearest buffer-free cell
    # (navigation.recovery.buffer_escape_point), so this scenario passes the
    # doorway without RECOVERING and reaches the goal (tests/
    # test_doorway_recovery_regression.py covers that fix; tests/
    # test_recovery.py still covers the reverse maneuver for deep intrusions).
    assert sim.metrics.goal_reached

    # The diagnostic trace/frame machinery itself (Phase 12.5's actual
    # deliverable) remains fully valid -- same data-contract checks as
    # before, just no longer tied to a BLOCKED outcome.
    assert len(run.trace) > 0
    assert len(run.frames) > 0
    last_frame = run.frames[-1]
    assert last_frame.raw_grid is not None
    assert last_frame.planning_grid is not None
    assert last_frame.ground_truth_pose_slam_frame is not None
    assert last_frame.estimated_pose is not None

    # Ground truth isolation still holds under instrumentation: the
    # estimated pose actually used by navigation must not equal the raw
    # ground-truth pose (different, offset frames -- see
    # simulation/scenario.py), and localization stayed reasonably
    # accurate throughout, ruling out "localization is the root cause".
    assert (last_frame.estimated_pose.x, last_frame.estimated_pose.y) != (
        last_frame.ground_truth_pose.x, last_frame.ground_truth_pose.y,
    )
    assert last_frame.position_error < 0.5
