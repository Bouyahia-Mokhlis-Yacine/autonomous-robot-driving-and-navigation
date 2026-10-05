"""Phase 14 tests: path-clearance diagnosis against the real Phase 12/13
deterministic scenario (Part 9, items 5-6).

Runtime note: same wall-clock cost pattern as tests/test_recovery.py's
module docstring -- this file's one scenario-level test runs to the
first recovery event (~33s of simulated time), costing on the order of
90-120 seconds of wall-clock time.
"""

from __future__ import annotations

import math

import pytest

from common.config import load_navigation_config, load_robot_config
from simulation.closed_loop import ClosedLoopSimulation
from simulation.diagnostics import run_with_diagnostics
from simulation.scenario import default_scenario

ROBOT_CFG = load_robot_config()
# This module documents the Phase 12.5/14 doorway signature measured with the
# LiDAR at the footprint centre (the geometry it pins -- e.g. nearest raw
# obstacle == inflation radius at the recovery trigger -- is that of the sensor
# pose). Keep that configuration; the real robot's rear mount (robot.yaml,
# 2026-09-30) is covered by tests/test_lidar_mount.py and the scenario suites.
import dataclasses  # noqa: E402
ROBOT_CFG = dataclasses.replace(ROBOT_CFG, robot=dataclasses.replace(ROBOT_CFG.robot, lidar_x=0.0, lidar_y=0.0))
NAV_CFG = load_navigation_config()


def test_phase12_scenario_produces_measurable_clearance_data() -> None:
    """Part 9, test 5: the real deterministic scenario produces real,
    finite, plausible clearance measurements for every pipeline stage,
    and recovery-trigger events carry the expected Phase 12.5 signature
    (planning OCCUPIED, raw FREE, near the configured inflation radius)."""
    scenario = default_scenario(ROBOT_CFG.robot.width, ROBOT_CFG.robot.length)
    sim = ClosedLoopSimulation(
        environment=scenario.environment,
        ground_truth_start=scenario.ground_truth_start,
        goal=scenario.slam_frame_goal,
        nav_cfg=NAV_CFG,
        robot_cfg=ROBOT_CFG,
        dt=0.1,
        max_simulation_time=40.0,
    )
    # This test documents the Phase 12.5 recovery-trigger signature, so it
    # needs the pre-fix behavior that produced those triggers: the shallow
    # buffer escape (navigation.recovery.buffer_escape_point) is disabled
    # here. With it enabled the scenario no longer enters RECOVERING at all
    # -- see tests/test_doorway_recovery_regression.py.
    from navigation.recovery import RecoveryConfig
    sim.nav_manager._recovery_config = RecoveryConfig(max_escape_distance=0.0)

    run = run_with_diagnostics(sim, stop_ticks_after_failure=5, record_frames=False)

    assert len(run.plan_events) > 0
    for event in run.plan_events:
        for clearance in (event.raw_clearance, event.los_clearance, event.final_clearance):
            assert math.isfinite(clearance.min_clearance_m)
            assert clearance.min_clearance_m >= 0.0
            assert clearance.path_length_m > 0.0
            assert clearance.num_samples > 0

    assert len(run.recovery_trigger_events) > 0
    first = run.recovery_trigger_events[0]
    assert first.raw_cell_state == "FREE"
    assert first.planning_cell_state == "OCCUPIED"
    assert first.nearest_occupied_distance_raw_m == pytest.approx(first.inflation_radius_m, abs=0.02)
    assert first.active_path_clearance is not None
    # The active path's own clearance is small -- consistent with the
    # Phase 14 finding that the path itself (not just localization
    # slop) already sits at/near the inflation boundary.
    assert first.active_path_clearance.min_clearance_m < 0.1
