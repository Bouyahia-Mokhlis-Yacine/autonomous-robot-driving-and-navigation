"""Phase 15 tests: the real deterministic Phase 12/13/14 scenario, now
with the configured path-smoothing clearance margin (0.10m) wired
through ``simulation.closed_loop.ClosedLoopSimulation``.

Runtime note: same wall-clock cost pattern as tests/test_recovery.py's
module docstring. The robot reaches the goal at ~t=122s of simulated
time in this scenario (measured), so this test's bounded 150s costs
several minutes of wall-clock time (~6 minutes on the machine this was
developed on). Both numbers are historical: with the vectorized map update
and the doorway buffer-escape fix it now reaches the goal at ~21 s.
"""

from __future__ import annotations

from common.config import load_navigation_config, load_robot_config
from navigation.navigation_manager import NavigationState
from simulation.closed_loop import ClosedLoopSimulation
from simulation.scenario import default_scenario

ROBOT_CFG = load_robot_config()
NAV_CFG = load_navigation_config()


def test_phase12_scenario_with_clearance_margin_reaches_goal() -> None:
    """Do NOT change the scenario, robot config, dt, LiDAR, localization,
    or recovery configuration -- only the newly-wired
    path_clearance_margin (from config/navigation.yaml, 0.10m) differs
    from the Phase 12/13/14 baseline."""
    scenario = default_scenario(ROBOT_CFG.robot.width, ROBOT_CFG.robot.length)
    sim = ClosedLoopSimulation(
        environment=scenario.environment,
        ground_truth_start=scenario.ground_truth_start,
        goal=scenario.slam_frame_goal,
        nav_cfg=NAV_CFG,
        robot_cfg=ROBOT_CFG,
        dt=0.1,
        # Originally measured at ~t=122s (3 doorway recovery cycles). Since
        # the buffer-escape fix (tests/test_doorway_recovery_regression.py)
        # the robot passes the doorway without RECOVERING, at ~t=21s; the
        # original budget is kept so this test still only checks what it
        # was written for (clearance margin wired, goal reached, no collision).
        max_simulation_time=150.0,
    )
    assert sim.nav_manager._path_clearance_margin == NAV_CFG.planning.path_clearance_margin > 0.0

    metrics = sim.run()

    assert metrics.collision is False
    assert metrics.goal_reached is True
    assert metrics.final_nav_state == NavigationState.GOAL_REACHED.value

