"""Phase 8 tests: LocalPlanner waypoint target selection."""

from __future__ import annotations

from common.messages import Path, PathPoint, Pose2D
from control.local_planner import LocalPlanner


def make_path() -> Path:
    return Path(points=[PathPoint(0, 0), PathPoint(1, 0), PathPoint(2, 0), PathPoint(3, 0)])


def test_select_target_on_empty_path_returns_none() -> None:
    planner = LocalPlanner(waypoint_tolerance=0.2)
    assert planner.select_target(Pose2D(0, 0, 0), Path(points=[])) is None


def test_select_target_starts_at_first_waypoint() -> None:
    planner = LocalPlanner(waypoint_tolerance=0.2)
    target = planner.select_target(Pose2D(x=-1.0, y=0.0, theta=0.0), make_path())
    assert (target.x, target.y) == (0, 0)


def test_select_target_advances_as_waypoints_are_reached() -> None:
    planner = LocalPlanner(waypoint_tolerance=0.2)
    path = make_path()

    target = planner.select_target(Pose2D(x=0.0, y=0.0, theta=0.0), path)
    assert (target.x, target.y) == (1, 0)  # already within tolerance of (0,0)

    target = planner.select_target(Pose2D(x=1.0, y=0.0, theta=0.0), path)
    assert (target.x, target.y) == (2, 0)


def test_select_target_never_advances_past_final_waypoint() -> None:
    planner = LocalPlanner(waypoint_tolerance=0.2)
    path = make_path()
    # Walk the robot sequentially through the path, as it would in practice.
    planner.select_target(Pose2D(x=0.0, y=0.0, theta=0.0), path)
    planner.select_target(Pose2D(x=1.0, y=0.0, theta=0.0), path)
    planner.select_target(Pose2D(x=2.0, y=0.0, theta=0.0), path)
    target = planner.select_target(Pose2D(x=3.0, y=0.0, theta=0.0), path)
    assert (target.x, target.y) == (3, 0)  # stays on the final waypoint, no overflow


def test_select_target_does_not_skip_ahead_when_far_from_current_target() -> None:
    planner = LocalPlanner(waypoint_tolerance=0.2)
    path = make_path()
    # Robot far from everything: should still target the first waypoint.
    target = planner.select_target(Pose2D(x=-5.0, y=-5.0, theta=0.0), path)
    assert (target.x, target.y) == (0, 0)


def test_reset_returns_to_first_waypoint() -> None:
    planner = LocalPlanner(waypoint_tolerance=0.2)
    path = make_path()
    planner.select_target(Pose2D(x=2.0, y=0.0, theta=0.0), path)  # advances index
    planner.reset()
    target = planner.select_target(Pose2D(x=-5.0, y=-5.0, theta=0.0), path)
    assert (target.x, target.y) == (0, 0)


def test_rejects_nonpositive_tolerance() -> None:
    import pytest

    with pytest.raises(ValueError):
        LocalPlanner(waypoint_tolerance=0.0)
