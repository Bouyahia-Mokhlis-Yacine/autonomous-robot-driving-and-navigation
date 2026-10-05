"""Phase 5 tests: mapping.persistent_world_model.PersistentWorldModel.

Ground-truth poses only -- no ICP, no scan_matching_pipeline -- to
isolate temporal mapping behavior exactly as the approved design
requires.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.messages import LaserScan, OccupancyGrid, Pose2D
from mapping.persistent_world_model import CellState, PersistentWorldModel

RANGE_MIN = 0.15
RANGE_MAX = 8.0
ANGLE_INCREMENT = math.radians(1.0)


def make_scan(hits: dict[float, float], timestamp: float = 0.0, background: float = 0.0) -> LaserScan:
    """Build a LaserScan with ``background`` everywhere except the given
    ``{angle_degrees: range_meters}`` overrides.

    ``background`` defaults to an INVALID range (0.0, below range_min)
    rather than range_max: at 1-degree angular resolution, adjacent
    rays' straight-line paths can pass through the very same grid cell
    as a close-range test point (offset ~= r*sin(1deg), which is under
    one 0.05m cell already at r<~3m) -- an all-max-range background
    would then silently carve that exact test cell free via a
    NEIGHBORING ray, contaminating the test. An invalid background has
    no effect anywhere, so only the explicitly-specified bearings do
    anything. Angles are wrapped into (-180, 180] to match LaserScan's
    own angle_min=-pi convention.
    """
    angle_min = -math.pi
    num_bins = int(round(2.0 * math.pi / ANGLE_INCREMENT)) + 1
    ranges = [background] * num_bins
    for angle_deg, r in hits.items():
        wrapped_deg = ((angle_deg + 180.0) % 360.0) - 180.0
        idx = round((math.radians(wrapped_deg) - angle_min) / ANGLE_INCREMENT)
        idx = max(0, min(num_bins - 1, idx))
        ranges[idx] = r
    return LaserScan(
        timestamp=timestamp, angle_min=angle_min, angle_increment=ANGLE_INCREMENT,
        ranges=ranges, range_min=RANGE_MIN, range_max=RANGE_MAX,
    )


ORIGIN_POSE = Pose2D(x=0.0, y=0.0, theta=0.0)


# --- A. Initial unknown grid ------------------------------------------------

def test_initial_grid_is_all_unknown() -> None:
    model = PersistentWorldModel()
    grid = model.get_grid()
    assert np.all(grid.cells == OccupancyGrid.UNKNOWN)
    assert np.all(model.state == CellState.UNKNOWN)


# --- B. New occupied geometry becomes tentative -----------------------------

def test_new_hit_becomes_tentative_not_confirmed() -> None:
    model = PersistentWorldModel()
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.0), now=0.0)
    assert model.get_cell_state(2.0, 0.0) == CellState.TENTATIVE
    assert model.get_pending_since(2.0, 0.0) == pytest.approx(0.0)
    row, col = model.occupancy_grid.world_to_grid(2.0, 0.0)
    assert model.get_grid().cells[row, col] == OccupancyGrid.UNKNOWN  # not exposed as occupied


# --- C. Confirmation at 0.5 seconds -----------------------------------------

def test_confirmed_after_configured_duration_of_continued_observation() -> None:
    model = PersistentWorldModel()
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.0), now=0.0)
    assert model.get_cell_state(2.0, 0.0) == CellState.TENTATIVE
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.2), now=0.2)
    assert model.get_cell_state(2.0, 0.0) == CellState.TENTATIVE
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.5), now=0.5)
    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED
    row, col = model.occupancy_grid.world_to_grid(2.0, 0.0)
    assert model.get_grid().cells[row, col] == OccupancyGrid.OCCUPIED


# --- D. Confirmation is time-based, not scan-count-based --------------------

def test_confirmation_depends_on_elapsed_time_not_scan_count() -> None:
    model = PersistentWorldModel()
    timestamps = [0.00, 0.07, 0.19, 0.31, 0.49, 0.51]
    for t in timestamps[:-1]:
        model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=t), now=t)
        assert model.get_cell_state(2.0, 0.0) == CellState.TENTATIVE, f"unexpectedly confirmed at t={t}"
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=timestamps[-1]), now=timestamps[-1])
    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED


# --- E. Tentative geometry excluded from confirmed reference ---------------

def test_tentative_not_returned_by_confirmed_reference_query() -> None:
    model = PersistentWorldModel()
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.0), now=0.0)
    points = model.get_confirmed_occupied_points_world((0.0, 0.0), radius=8.0)
    assert points.shape == (0, 2)


# --- F. Tentative geometry occluded before confirmation ---------------------

def test_occluded_before_confirmation_stays_tentative() -> None:
    model = PersistentWorldModel()
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.0), now=0.0)
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.2), now=0.2)
    # A closer obstacle now blocks this exact bearing -- the target cell
    # (at r=2.0) is never traced at all this scan: genuinely occluded.
    model.integrate(ORIGIN_POSE, make_scan({0.0: 1.0}, timestamp=0.3), now=0.3)
    assert model.get_cell_state(2.0, 0.0) == CellState.TENTATIVE
    assert model.get_pending_since(2.0, 0.0) == pytest.approx(0.0)  # unchanged, not reset
    # Time continues to pass while still occluded -- must NOT confirm.
    model.integrate(ORIGIN_POSE, make_scan({0.0: 1.0}, timestamp=1.0), now=1.0)
    assert model.get_cell_state(2.0, 0.0) == CellState.TENTATIVE


# --- G. Tentative geometry reappears -----------------------------------------

def test_reappearance_continues_confirmation_span() -> None:
    model = PersistentWorldModel()
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.0), now=0.0)
    model.integrate(ORIGIN_POSE, make_scan({0.0: 1.0}, timestamp=0.3), now=0.3)  # occluded
    assert model.get_cell_state(2.0, 0.0) == CellState.TENTATIVE
    # Reappears at t=0.6: span from the FIRST sighting (t=0.0) is 0.6s >= 0.5s.
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.6), now=0.6)
    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED


# --- H. Confirmed geometry persists when out of range -----------------------

def test_confirmed_geometry_persists_when_out_of_range() -> None:
    model = PersistentWorldModel()
    near_pose = Pose2D(x=7.0, y=0.0, theta=0.0)
    for t in (0.0, 0.2, 0.5):
        model.integrate(near_pose, make_scan({0.0: 2.0}, timestamp=t), now=t)  # world (9, 0)
    assert model.get_cell_state(9.0, 0.0) == CellState.CONFIRMED

    far_pose = Pose2D(x=-9.0, y=0.0, theta=0.0)  # (9,0) is now 18m away, far beyond range_max
    model.integrate(far_pose, make_scan({0.0: RANGE_MAX}, timestamp=1.0), now=1.0)  # reaches only 8m
    assert model.get_cell_state(9.0, 0.0) == CellState.CONFIRMED


# --- I. Confirmed geometry persists under occlusion --------------------------

def test_confirmed_geometry_persists_under_occlusion() -> None:
    model = PersistentWorldModel()
    for t in (0.0, 0.2, 0.5):
        model.integrate(ORIGIN_POSE, make_scan({0.0: 5.0}, timestamp=t), now=t)  # A at r=5
    assert model.get_cell_state(5.0, 0.0) == CellState.CONFIRMED

    # A closer obstacle B now blocks the same bearing.
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.6), now=0.6)
    assert model.get_cell_state(5.0, 0.0) == CellState.CONFIRMED  # A unaffected


# --- J. Confirmed geometry cleared by direct free-space observation ---------

def test_confirmed_cleared_by_direct_free_observation() -> None:
    model = PersistentWorldModel()
    for t in (0.0, 0.2, 0.5):
        model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=t), now=t)
    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED

    model.integrate(ORIGIN_POSE, make_scan({0.0: RANGE_MAX}, timestamp=0.6), now=0.6)  # clear line of sight, nothing there
    assert model.get_cell_state(2.0, 0.0) == CellState.FREE
    row, col = model.occupancy_grid.world_to_grid(2.0, 0.0)
    assert model.get_grid().cells[row, col] == OccupancyGrid.FREE


# --- K. Obstacle behind another obstacle is NOT cleared ----------------------

def test_obstacle_behind_another_is_not_cleared() -> None:
    model = PersistentWorldModel()
    for t in (0.0, 0.2, 0.5):
        model.integrate(ORIGIN_POSE, make_scan({0.0: 5.0}, timestamp=t), now=t)  # A farther
    for t in (0.6, 0.8, 1.1):
        model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=t), now=t)  # B closer, confirmed too
    assert model.get_cell_state(5.0, 0.0) == CellState.CONFIRMED  # A protected by B
    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED  # B itself confirmed


# --- L. Max-range ray clears visible geometry --------------------------------

def test_max_range_ray_clears_geometry_within_range() -> None:
    model = PersistentWorldModel()
    for t in (0.0, 0.2, 0.5):
        model.integrate(ORIGIN_POSE, make_scan({0.0: 3.0}, timestamp=t), now=t)
    assert model.get_cell_state(3.0, 0.0) == CellState.CONFIRMED
    model.integrate(ORIGIN_POSE, make_scan({0.0: RANGE_MAX}, timestamp=0.6), now=0.6)  # max-range along the same bearing
    assert model.get_cell_state(3.0, 0.0) == CellState.FREE


# --- M. Beyond max range remains untouched -----------------------------------

def test_geometry_beyond_max_range_untouched_by_max_range_ray() -> None:
    model = PersistentWorldModel()
    near_pose = Pose2D(x=6.0, y=0.0, theta=0.0)
    for t in (0.0, 0.2, 0.5):
        model.integrate(near_pose, make_scan({0.0: 1.9}, timestamp=t), now=t)  # world (7.9, 0)
    assert model.get_cell_state(7.9, 0.0) == CellState.CONFIRMED

    far_pose = Pose2D(x=-2.0, y=0.0, theta=0.0)  # (7.9,0) is 9.9m away -- beyond range_max=8.0
    model.integrate(far_pose, make_scan({0.0: RANGE_MAX}, timestamp=1.0), now=1.0)
    assert model.get_cell_state(7.9, 0.0) == CellState.CONFIRMED


# --- N. Invalid ranges --------------------------------------------------------

def test_invalid_ranges_produce_no_observation() -> None:
    model = PersistentWorldModel()
    scan = make_scan({}, timestamp=0.0)
    ranges = list(scan.ranges)
    ranges[0] = RANGE_MIN / 2.0  # too close
    ranges[10] = float("nan")
    ranges[20] = float("inf")
    invalid_scan = LaserScan(
        timestamp=0.0, angle_min=scan.angle_min, angle_increment=scan.angle_increment,
        ranges=ranges, range_min=scan.range_min, range_max=scan.range_max,
    )
    model.integrate(ORIGIN_POSE, invalid_scan, now=0.0)
    assert not np.any(model.state == CellState.TENTATIVE)
    assert not np.any(model.state == CellState.CONFIRMED)
    # The three invalid bearings' own endpoint cells must not have been
    # touched at all (still UNKNOWN, not even FREE).
    for angle_deg, idx in [(0.0, 0), (10.0, 10), (20.0, 20)]:
        angle = scan.angle_min + idx * scan.angle_increment
        x = math.cos(angle) * 3.0  # arbitrary point along that exact bearing
        y = math.sin(angle) * 3.0
        assert model.get_cell_state(x, y) == CellState.UNKNOWN


# --- O. Coordinate transformation --------------------------------------------

def test_nonzero_pose_transforms_hit_into_correct_world_coordinates() -> None:
    model = PersistentWorldModel()
    pose = Pose2D(x=2.0, y=3.0, theta=0.0)
    for t in (0.0, 0.2, 0.5):
        model.integrate(pose, make_scan({0.0: 1.0}, timestamp=t), now=t)  # local (1,0) -> world (3,3)
    assert model.get_cell_state(3.0, 3.0) == CellState.CONFIRMED
    assert model.get_cell_state(1.0, 0.0) == CellState.UNKNOWN  # NOT the untransformed local point


# --- P. World frame remains fixed as the robot moves -------------------------

def test_world_frame_does_not_rotate_with_robot() -> None:
    model = PersistentWorldModel()
    # Observe world point (3, 0) from pose (0,0,0): local bearing 0, r=3.
    model.integrate(ORIGIN_POSE, make_scan({0.0: 3.0}, timestamp=0.0), now=0.0)
    assert model.get_cell_state(3.0, 0.0) == CellState.TENTATIVE

    # Robot rotates in place to face +90 degrees (no translation). The
    # SAME physical point (3,0) is now at local bearing -90 degrees.
    rotated_pose = Pose2D(x=0.0, y=0.0, theta=math.pi / 2.0)
    model.integrate(rotated_pose, make_scan({-90.0: 3.0}, timestamp=0.6), now=0.6)

    # The world-frame location must be the SAME cell, now confirmed --
    # not some other cell the rotation would imply if the world frame
    # incorrectly rotated with the robot.
    assert model.get_cell_state(3.0, 0.0) == CellState.CONFIRMED


# --- Q. Endpoint precedence ----------------------------------------------------

def test_hit_endpoint_is_not_cleared_by_its_own_ray_traversal() -> None:
    model = PersistentWorldModel()
    # A short ray: cells strictly before the endpoint (if any, at this
    # resolution/range) are marked free by the SAME ray that also
    # marks its own endpoint occupied -- the endpoint must win there.
    model.integrate(ORIGIN_POSE, make_scan({0.0: 0.3}, timestamp=0.0), now=0.0)
    assert model.get_cell_state(0.3, 0.0) == CellState.TENTATIVE
    for t in (0.2, 0.5):
        model.integrate(ORIGIN_POSE, make_scan({0.0: 0.3}, timestamp=t), now=t)
    assert model.get_cell_state(0.3, 0.0) == CellState.CONFIRMED


# --- R. Multiple observations coexist independently ---------------------------

def test_multiple_confirmed_regions_coexist_independently() -> None:
    model = PersistentWorldModel()
    for t in (0.0, 0.2, 0.5):
        model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0, 90.0: 3.0, 200.0: 1.5}, timestamp=t), now=t)
    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED
    assert model.get_cell_state(0.0, 3.0) == CellState.CONFIRMED
    x3 = 1.5 * math.cos(math.radians(200.0))
    y3 = 1.5 * math.sin(math.radians(200.0))
    assert model.get_cell_state(x3, y3) == CellState.CONFIRMED


# --- S. Clearing one obstacle does not affect unrelated cells -----------------

def test_clearing_one_obstacle_does_not_affect_another() -> None:
    model = PersistentWorldModel()
    for t in (0.0, 0.2, 0.5):
        model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0, 90.0: 2.0}, timestamp=t), now=t)
    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED
    assert model.get_cell_state(0.0, 2.0) == CellState.CONFIRMED

    # Clear only the first one; the second must be untouched.
    model.integrate(ORIGIN_POSE, make_scan({0.0: RANGE_MAX, 90.0: 2.0}, timestamp=0.6), now=0.6)
    assert model.get_cell_state(2.0, 0.0) == CellState.FREE
    assert model.get_cell_state(0.0, 2.0) == CellState.CONFIRMED


# --- T. Reference query radius --------------------------------------------------

def test_confirmed_reference_query_respects_radius() -> None:
    model = PersistentWorldModel()
    for t in (0.0, 0.2, 0.5):
        model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0, 90.0: 6.0}, timestamp=t), now=t)
    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED
    assert model.get_cell_state(0.0, 6.0) == CellState.CONFIRMED

    close_only = model.get_confirmed_occupied_points_world((0.0, 0.0), radius=3.0)
    assert close_only.shape[0] == 1
    np.testing.assert_allclose(close_only[0], [2.025, 0.025], atol=1e-6)

    both = model.get_confirmed_occupied_points_world((0.0, 0.0), radius=8.0)
    assert both.shape[0] == 2


# --- U. Reference query center --------------------------------------------------

def test_confirmed_reference_query_center_is_world_coordinates() -> None:
    model = PersistentWorldModel()
    for t in (0.0, 0.2, 0.5):
        model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=t), now=t)
    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED

    # Centered far away -- must return nothing despite a generous radius.
    far = model.get_confirmed_occupied_points_world((-5.0, -5.0), radius=2.0)
    assert far.shape == (0, 2)
    # Centered near the actual point -- must return it.
    near = model.get_confirmed_occupied_points_world((2.0, 0.0), radius=0.5)
    assert near.shape[0] == 1


# --- V. Empty/no-hit scan --------------------------------------------------------

def test_empty_scan_invents_nothing() -> None:
    model = PersistentWorldModel()
    empty = LaserScan(timestamp=0.0, angle_min=0.0, angle_increment=0.0, ranges=[], range_min=RANGE_MIN, range_max=RANGE_MAX)
    model.integrate(ORIGIN_POSE, empty, now=0.0)
    assert np.all(model.state == CellState.UNKNOWN)
    assert np.all(model.get_grid().cells == OccupancyGrid.UNKNOWN)


def test_all_max_range_scan_invents_no_occupied_geometry() -> None:
    model = PersistentWorldModel()
    model.integrate(ORIGIN_POSE, make_scan({}, timestamp=0.0, background=RANGE_MAX), now=0.0)  # every ray max-range
    assert not np.any(model.state == CellState.TENTATIVE)
    assert not np.any(model.state == CellState.CONFIRMED)
    assert np.any(model.state == CellState.FREE)  # visible free space was updated


# --- W. Max-range-only scan: visible free space updated, beyond stays unknown --

def test_max_range_only_scan_leaves_beyond_range_unknown() -> None:
    model = PersistentWorldModel()
    model.integrate(ORIGIN_POSE, make_scan({0.0: RANGE_MAX}, timestamp=0.0), now=0.0)
    # Well within range: observed free.
    assert model.get_cell_state(3.0, 0.0) == CellState.FREE
    # Far beyond range_max along the SAME bearing: never touched.
    assert model.get_cell_state(9.5, 0.0) == CellState.UNKNOWN


# =========================================================================
# Internal state-representation tests (section 24)
# =========================================================================

def test_state_transition_unknown_to_tentative_to_confirmed() -> None:
    model = PersistentWorldModel()
    assert model.get_cell_state(2.0, 0.0) == CellState.UNKNOWN
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.0), now=0.0)
    assert model.get_cell_state(2.0, 0.0) == CellState.TENTATIVE
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.5), now=0.5)
    assert model.get_cell_state(2.0, 0.0) == CellState.CONFIRMED


def test_state_transition_tentative_contradicted_becomes_free() -> None:
    model = PersistentWorldModel()
    model.integrate(ORIGIN_POSE, make_scan({0.0: 2.0}, timestamp=0.0), now=0.0)
    assert model.get_cell_state(2.0, 0.0) == CellState.TENTATIVE
    assert model.get_pending_since(2.0, 0.0) is not None
    model.integrate(ORIGIN_POSE, make_scan({0.0: RANGE_MAX}, timestamp=0.1), now=0.1)  # contradicted before confirmation
    assert model.get_cell_state(2.0, 0.0) == CellState.FREE
    assert model.get_pending_since(2.0, 0.0) is None  # discarded


# =========================================================================
# Determinism
# =========================================================================

def test_deterministic_replay() -> None:
    def run() -> PersistentWorldModel:
        m = PersistentWorldModel()
        events = [
            (ORIGIN_POSE, {0.0: 2.0}, 0.0),
            (ORIGIN_POSE, {0.0: 2.0}, 0.2),
            (ORIGIN_POSE, {0.0: 1.0}, 0.3),  # occlusion
            (ORIGIN_POSE, {0.0: 2.0}, 0.6),  # reappear -> confirms
            (ORIGIN_POSE, {90.0: 4.0}, 0.7),
            (ORIGIN_POSE, {}, 1.0),  # clears the 0.0-bearing obstacle
        ]
        for pose, hits, t in events:
            m.integrate(pose, make_scan(hits, timestamp=t), now=t)
        return m

    model_a = run()
    model_b = run()
    np.testing.assert_array_equal(model_a.state, model_b.state)
    np.testing.assert_array_equal(model_a.get_grid().cells, model_b.get_grid().cells)
    a_pending, b_pending = model_a.pending_since, model_b.pending_since
    both_nan = np.isnan(a_pending) & np.isnan(b_pending)
    np.testing.assert_array_equal(np.isnan(a_pending), np.isnan(b_pending))
    np.testing.assert_array_equal(a_pending[~both_nan], b_pending[~both_nan])
