"""Phase 8 tests: visualization.gui's pure, headless-testable pieces
(VisualizationState / build_visualization_state / LatestStateBox) and
the new black-background palette in visualization.map_viewer.

No Tk widgets are created or exercised here -- WorldModelGUI's actual
canvas drawing is verified manually (see the Phase 8 final report),
consistent with how this project has always verified GUI rendering
throughout every previous phase. These tests cover the pure logic
feeding that rendering: coordinate transforms, state snapshotting, the
failed-registration truth model, bootstrap-status passthrough, and the
latest-state (no backlog) handoff.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.messages import LaserScan, Pose2D
from mapping.persistent_world_model import CellState, PersistentWorldModel
from slam.scan_matching_pipeline import BootstrapStatus, PipelineResult
from visualization.gui import LatestStateBox, VisualizationState, build_visualization_state
from visualization.map_viewer import DARK_PALETTE
from common.messages import OccupancyGrid


def make_scan(hits: dict[float, float], timestamp: float = 0.0, background: float = 0.0) -> LaserScan:
    """Same construction pattern as tests/test_persistent_world_model.py:
    background defaults to an invalid (no-op) range so only the given
    bearings have any effect."""
    angle_min = -math.pi
    angle_increment = math.radians(1.0)
    range_min, range_max = 0.15, 8.0
    num_bins = int(round(2.0 * math.pi / angle_increment)) + 1
    ranges = [background] * num_bins
    for angle_deg, r in hits.items():
        idx = round((math.radians(angle_deg) - angle_min) / angle_increment)
        ranges[idx] = r
    return LaserScan(
        timestamp=timestamp, angle_min=angle_min, angle_increment=angle_increment,
        ranges=ranges, range_min=range_min, range_max=range_max,
    )


def make_result(pose: Pose2D, **overrides) -> PipelineResult:
    defaults = dict(
        pose=pose, accepted=True, localized=True, consecutive_failures=0,
        registration_result=None, rejection_reason=None, map_integrated=True,
        bootstrap_status=BootstrapStatus.COMPLETE,
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


# --- A/C. Coordinate transform (world-to-screen reuse + LiDAR transform) ---

def test_current_scan_points_use_documented_world_transform() -> None:
    # local (1, 0) at pose (0,0,0) -> world (1, 0).
    model = PersistentWorldModel()
    scan = make_scan({0.0: 1.0})
    result = make_result(Pose2D(x=0.0, y=0.0, theta=0.0))
    state = build_visualization_state(model, result, scan, trajectory=())
    assert len(state.current_scan_points_world) == 1
    x, y = state.current_scan_points_world[0]
    assert x == pytest.approx(1.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_current_scan_points_rotate_with_accepted_pose() -> None:
    # Same local point, but the accepted pose is rotated 90 degrees ->
    # world (0, 1), per the project's world = R(theta) @ local + [x,y]
    # convention (theta positive = CCW).
    model = PersistentWorldModel()
    scan = make_scan({0.0: 1.0})
    result = make_result(Pose2D(x=0.0, y=0.0, theta=math.pi / 2.0))
    state = build_visualization_state(model, result, scan, trajectory=())
    x, y = state.current_scan_points_world[0]
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(1.0, abs=1e-9)


def test_current_scan_points_translate_with_accepted_pose() -> None:
    model = PersistentWorldModel()
    scan = make_scan({0.0: 1.0})
    result = make_result(Pose2D(x=2.0, y=3.0, theta=0.0))
    state = build_visualization_state(model, result, scan, trajectory=())
    x, y = state.current_scan_points_world[0]
    assert x == pytest.approx(3.0, abs=1e-9)  # 2.0 + 1.0
    assert y == pytest.approx(3.0, abs=1e-9)


# --- D. Unknown/free/occupied rendering (DARK_PALETTE) ----------------------

def test_dark_palette_free_matches_black_background() -> None:
    assert DARK_PALETTE[OccupancyGrid.FREE] == (0, 0, 0)


def test_dark_palette_unknown_distinct_from_free_and_dark() -> None:
    unknown = DARK_PALETTE[OccupancyGrid.UNKNOWN]
    assert unknown != DARK_PALETTE[OccupancyGrid.FREE]
    assert max(unknown) < 100  # "dark", per the design -- clearly not a bright color


def test_dark_palette_occupied_is_high_contrast() -> None:
    occupied = DARK_PALETTE[OccupancyGrid.OCCUPIED]
    unknown = DARK_PALETTE[OccupancyGrid.UNKNOWN]
    free = DARK_PALETTE[OccupancyGrid.FREE]
    assert min(occupied) > 150  # bright, clearly visible against a black canvas
    assert occupied != unknown
    assert occupied != free


def test_default_palette_unaffected_by_dark_palette_addition() -> None:
    # Regression guard: adding DARK_PALETTE must not have touched the
    # existing white-background DEFAULT_PALETTE the old NavigationGUI
    # relies on implicitly (see visualization/map_viewer.py's comment).
    from visualization.map_viewer import DEFAULT_PALETTE
    assert DEFAULT_PALETTE[OccupancyGrid.FREE] == (255, 255, 255)
    assert DEFAULT_PALETTE[OccupancyGrid.OCCUPIED] == (0, 0, 0)


# --- E. Tentative geometry distinguishable from confirmed --------------------

def test_tentative_points_separate_from_confirmed_in_visualization_state() -> None:
    model = PersistentWorldModel()
    pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    # Confirm one point (3 observations spanning >= confirmation_duration).
    for t in (0.0, 0.2, 0.5):
        model.integrate(pose, make_scan({0.0: 2.0}, timestamp=t), now=t)
    # A second, different point observed only once -- tentative.
    model.integrate(pose, make_scan({90.0: 3.0}, timestamp=0.6), now=0.6)

    result = make_result(pose)
    state = build_visualization_state(model, result, make_scan({}, timestamp=0.6), trajectory=())

    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED
    assert model.get_cell_state(0.0, 3.0) == CellState.TENTATIVE

    # The confirmed point must NOT appear in tentative_points_world...
    for x, y in state.tentative_points_world:
        assert math.hypot(x - 2.0, y - 0.0) > 0.1
    # ...and the tentative point must appear there.
    assert any(math.hypot(x - 0.0, y - 3.0) < 0.1 for x, y in state.tentative_points_world)
    # ...and must NOT appear as OCCUPIED in the public grid (Phase 5's
    # own TENTATIVE -> public UNKNOWN mapping, unchanged).
    row, col = model.occupancy_grid.world_to_grid(0.0, 3.0)
    assert state.grid.cells[row, col] == OccupancyGrid.UNKNOWN


# --- F. Failed registration uses the last accepted pose, never a seed -------

def test_failed_registration_state_uses_last_accepted_pose_not_a_seed() -> None:
    model = PersistentWorldModel()
    last_accepted = Pose2D(x=1.0, y=2.0, theta=0.3, timestamp=1.0)
    # A rejected result's .pose is, per PipelineResult's own contract,
    # already frozen at the last accepted pose -- this test locks that
    # contract in from the visualization side too, so a future change
    # to either side can't silently start feeding a seed into the GUI.
    rejected_result = make_result(
        last_accepted, accepted=False, localized=True, consecutive_failures=1,
        rejection_reason="insufficient_correspondences",
    )
    scan = make_scan({0.0: 1.0}, timestamp=1.1)
    state = build_visualization_state(model, rejected_result, scan, trajectory=())
    assert state.robot_pose == last_accepted
    assert state.robot_pose.x == pytest.approx(1.0)
    assert state.robot_pose.y == pytest.approx(2.0)


# --- G. Bootstrap status representation --------------------------------------

@pytest.mark.parametrize("status", [BootstrapStatus.BOOTSTRAPPING, BootstrapStatus.COMPLETE, BootstrapStatus.FAILED])
def test_bootstrap_status_passes_through_to_visualization_state(status: BootstrapStatus) -> None:
    model = PersistentWorldModel()
    result = make_result(Pose2D(0.0, 0.0, 0.0), bootstrap_status=status, localized=(status != BootstrapStatus.FAILED))
    scan = make_scan({})
    state = build_visualization_state(model, result, scan, trajectory=())
    assert state.bootstrap_status == status


# --- H. Latest-state behavior (no backlog) -----------------------------------

def test_latest_state_box_replaces_rather_than_queues() -> None:
    box = LatestStateBox()
    assert box.get() is None

    model = PersistentWorldModel()
    scan = make_scan({})
    state1 = build_visualization_state(model, make_result(Pose2D(0.0, 0.0, 0.0)), scan, trajectory=())
    state2 = build_visualization_state(model, make_result(Pose2D(1.0, 0.0, 0.0)), scan, trajectory=())
    state3 = build_visualization_state(model, make_result(Pose2D(2.0, 0.0, 0.0)), scan, trajectory=())

    box.set(state1)
    box.set(state2)
    box.set(state3)  # simulates 3 scans processed before the GUI ever reads

    latest = box.get()
    assert latest is state3  # only the newest survives -- no queue, nothing accumulated
    assert latest.robot_pose.x == pytest.approx(2.0)


# --- I. No map mutation -------------------------------------------------------

def test_build_visualization_state_does_not_mutate_world_model() -> None:
    model = PersistentWorldModel()
    pose = Pose2D(0.0, 0.0, 0.0)
    for t in (0.0, 0.2, 0.5):
        model.integrate(pose, make_scan({0.0: 2.0}, timestamp=t), now=t)

    state_before = model.state.copy()
    pending_before = model.pending_since.copy()
    grid_before = model.get_grid().cells.copy()

    result = make_result(pose)
    scan = make_scan({45.0: 1.5}, timestamp=0.6)
    build_visualization_state(model, result, scan, trajectory=((0.0, 0.0),))

    np.testing.assert_array_equal(model.state, state_before)
    both_nan = np.isnan(model.pending_since) & np.isnan(pending_before)
    np.testing.assert_array_equal(np.isnan(model.pending_since), np.isnan(pending_before))
    np.testing.assert_array_equal(model.pending_since[~both_nan], pending_before[~both_nan])
    np.testing.assert_array_equal(model.get_grid().cells, grid_before)


def test_build_visualization_state_is_pure_repeated_calls_identical() -> None:
    model = PersistentWorldModel()
    pose = Pose2D(0.0, 0.0, 0.0)
    model.integrate(pose, make_scan({0.0: 2.0}, timestamp=0.0), now=0.0)
    result = make_result(pose)
    scan = make_scan({0.0: 2.0}, timestamp=0.0)

    state_a = build_visualization_state(model, result, scan, trajectory=((0.0, 0.0),))
    state_b = build_visualization_state(model, result, scan, trajectory=((0.0, 0.0),))
    np.testing.assert_array_equal(state_a.tentative_points_world, state_b.tentative_points_world)
    assert state_a.current_scan_points_world == state_b.current_scan_points_world
    assert state_a.robot_pose == state_b.robot_pose
