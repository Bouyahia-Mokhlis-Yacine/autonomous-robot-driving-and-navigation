"""Localization-loss safety: ``slam.localization_health`` and the
``NavigationManager`` LOCALIZATION_LOST response."""

from __future__ import annotations

import pytest

from common.messages import Goal, Pose2D, VelocityCommand
from control.path_follower import PathFollower, PathFollowerConfig
from navigation.navigation_manager import NavigationManager, NavigationState
from navigation.replanner import Replanner, ReplannerConfig
from planning.global_planner import GlobalPlanner
from slam.localization_health import LocalizationHealth, LocalizationHealthConfig, LocalizationHealthMonitor
from slam.scan_matching_pipeline import BootstrapStatus, PipelineResult
from tests.test_navigation_manager import FakeRobot, make_grid, make_manager

ZERO = VelocityCommand(linear=0.0, angular=0.0)


def result(accepted: bool = True, localized: bool = True, bootstrap=BootstrapStatus.COMPLETE) -> PipelineResult:
    return PipelineResult(
        pose=Pose2D(0.0, 0.0, 0.0), accepted=accepted, localized=localized, consecutive_failures=0,
        registration_result=None, rejection_reason=None if accepted else "test", bootstrap_status=bootstrap,
    )


class Switch:
    """Test double for a localization-health source."""

    def __init__(self, valid: bool = True) -> None:
        self.valid = valid

    def is_localization_valid(self) -> bool:
        return self.valid


def managed(valid: bool = True) -> tuple[NavigationManager, FakeRobot, Switch]:
    manager, robot = make_manager()
    switch = Switch(valid)
    manager._localization = switch  # same object make_manager built, plus a health source
    return manager, robot, switch


# ------------------------------------------------------------------ monitor --

def test_monitor_is_initializing_until_bootstrap_completes() -> None:
    m = LocalizationHealthMonitor()
    m.on_result(result(bootstrap=BootstrapStatus.BOOTSTRAPPING), 0.1)
    assert m.update(0.1) is LocalizationHealth.INITIALIZING
    assert not m.is_localization_valid()
    m.on_result(result(), 0.2)
    assert m.update(0.2) is LocalizationHealth.LOCALIZED
    assert m.is_localization_valid()


def test_monitor_degrades_then_loses_a_stale_pose() -> None:
    cfg = LocalizationHealthConfig(degraded_after_s=0.5, lost_after_s=1.0)
    m = LocalizationHealthMonitor(cfg)
    m.on_result(result(), 1.0)
    assert m.update(1.3) is LocalizationHealth.LOCALIZED
    assert m.update(1.7) is LocalizationHealth.DEGRADED and m.is_localization_valid()
    assert m.update(2.1) is LocalizationHealth.LOST and not m.is_localization_valid()
    assert m.loss_events == 1


def test_monitor_is_lost_as_soon_as_the_pipeline_says_so() -> None:
    m = LocalizationHealthMonitor()
    m.on_result(result(), 1.0)
    m.on_result(result(accepted=False, localized=False), 1.1)
    assert m.update(1.1) is LocalizationHealth.LOST


def test_monitor_needs_consecutive_confirmations_to_recover() -> None:
    m = LocalizationHealthMonitor(LocalizationHealthConfig(recovery_confirmations=2))
    m.on_result(result(), 1.0)
    m.on_result(result(accepted=False, localized=False), 1.1)
    m.on_result(result(), 1.2)
    assert m.update(1.2) is LocalizationHealth.LOST          # one good scan is not enough
    m.on_result(result(accepted=False), 1.3)
    m.on_result(result(), 1.4)
    assert m.update(1.4) is LocalizationHealth.LOST          # the streak was broken
    m.on_result(result(), 1.5)
    assert m.update(1.5) is LocalizationHealth.LOCALIZED
    assert m.loss_events == 1


def test_recovery_after_staleness_only_counts_scans_after_the_loss() -> None:
    m = LocalizationHealthMonitor(LocalizationHealthConfig(degraded_after_s=0.5, lost_after_s=1.0,
                                                           recovery_confirmations=2))
    for t in (0.1, 0.2, 0.3):
        m.on_result(result(), t)
    assert m.update(2.0) is LocalizationHealth.LOST          # stale, no rejection involved
    m.on_result(result(), 2.1)
    assert m.update(2.1) is LocalizationHealth.LOST
    m.on_result(result(), 2.2)
    assert m.update(2.2) is LocalizationHealth.LOCALIZED


# ------------------------------------------------------- navigation response --

def test_robot_stops_when_localization_becomes_invalid() -> None:
    manager, robot, switch = managed()
    grid = make_grid()
    manager.set_goal(Goal(8.0, 8.0), grid)
    assert manager.update(grid, 0.1) != ZERO  # driving normally
    switch.valid = False
    stops_before = robot.stop_calls
    assert manager.update(grid, 0.1) == ZERO
    assert manager.state is NavigationState.LOCALIZATION_LOST
    assert robot.stop_calls > stops_before
    assert robot.last_command == ZERO
    assert manager.get_goal() == Goal(8.0, 8.0)  # the goal is kept, not abandoned


def test_navigation_does_not_use_a_stale_pose_while_lost() -> None:
    """While invalid, a pose that lands exactly on the goal must NOT be
    taken as GOAL_REACHED, and nothing may be commanded."""
    manager, robot, switch = managed()
    grid = make_grid()
    manager.set_goal(Goal(8.0, 8.0), grid)
    switch.valid = False
    robot.pose = Pose2D(8.0, 8.0, 0.0)
    for _ in range(20):
        assert manager.update(grid, 0.1) == ZERO
    assert manager.state is NavigationState.LOCALIZATION_LOST


def test_navigation_resumes_from_the_recovered_pose() -> None:
    manager, robot, switch = managed()
    grid = make_grid()
    manager.set_goal(Goal(8.0, 8.0), grid)
    manager.update(grid, 0.1)
    switch.valid = False
    manager.update(grid, 0.1)
    old_path = manager.get_path()

    robot.pose = Pose2D(2.0, 6.0, 1.0)  # localization comes back somewhere else
    switch.valid = True
    assert manager.update(grid, 0.1) == ZERO            # replans this tick, drives from the next
    assert manager.state is NavigationState.MOVING
    new_path = manager.get_path()
    assert new_path is not old_path
    assert (new_path.points[0].x, new_path.points[0].y) == pytest.approx((2.0, 6.0), abs=0.5)
    assert manager.update(grid, 0.1) != ZERO
    for _ in range(400):
        robot.integrate(0.1)
        manager.update(grid, 0.1)
        if manager.state is NavigationState.GOAL_REACHED:
            break
    assert manager.state is NavigationState.GOAL_REACHED


def test_goal_set_while_lost_is_planned_once_valid() -> None:
    manager, robot, switch = managed(valid=False)
    grid = make_grid()
    manager.set_goal(Goal(8.0, 8.0), grid)
    assert manager.state is NavigationState.LOCALIZATION_LOST
    assert manager.get_path() is None
    switch.valid = True
    manager.update(grid, 0.1)
    assert manager.state is NavigationState.MOVING and manager.get_path() is not None


def test_without_a_health_source_behavior_is_unchanged() -> None:
    manager, robot = make_manager()
    grid = make_grid()
    assert manager.is_localization_valid()
    manager.set_goal(Goal(8.0, 8.0), grid)
    assert manager.state is NavigationState.MOVING
    assert manager.update(grid, 0.1) != ZERO


def test_emergency_stop_and_cancel_still_work_while_lost() -> None:
    manager, robot, switch = managed()
    grid = make_grid()
    manager.set_goal(Goal(8.0, 8.0), grid)
    switch.valid = False
    manager.update(grid, 0.1)
    manager.cancel_goal()
    assert manager.state is NavigationState.IDLE
    manager.set_goal(Goal(8.0, 8.0), grid)
    manager.emergency_stop()
    assert manager.state is NavigationState.EMERGENCY_STOP
