"""Phase 2 + Phase 3 tests: slam.scan_registration (point-to-point ICP,
with Phase 3's trimmed-ICP + angular-sector spread robustness added on
top of the unchanged Phase 2 algorithm and public API).

Uses a fixed asymmetric, three-armed synthetic point cloud (irregular
spacing along each arm, arms at non-grid bearings spanning 3 distinct
45-degree angular sectors) throughout so rotation is always observable
and Phase 3's default min_sectors=3 spread check is naturally satisfied
by realistic geometry, not a special case.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.messages import Pose2D
from slam.scan_registration import RegistrationConfig, _keep_count, _sector_indices, register


def l_shaped_points() -> np.ndarray:
    # Three deliberately IRREGULARLY-spaced arms at non-grid bearings,
    # spanning 3 distinct 45-degree angular sectors (bearings ~10,
    # ~125, ~255 degrees -> sectors 0, 2, 5) -- satisfies Phase 3's
    # default min_sectors=3 with realistic, non-symmetric geometry.
    # Evenly-spaced (grid) points are a known pathological case for
    # plain nearest-neighbor point-to-point ICP: correspondences can
    # alias to an adjacent grid point instead of the true match,
    # converging to a self-consistent but wrong local minimum. Varied
    # spacing removes that periodicity.
    def arm(angle_deg: float, radii: list[float]) -> np.ndarray:
        theta = math.radians(angle_deg)
        return np.array([[r * math.cos(theta), r * math.sin(theta)] for r in radii])

    arm1 = arm(10.0, [0.4, 0.7, 1.1, 1.6, 2.0, 2.5, 2.9])
    arm2 = arm(125.0, [0.35, 0.6, 0.95, 1.3, 1.7])
    arm3 = arm(255.0, [0.3, 0.55, 0.85, 1.2])
    return np.vstack([arm1, arm2, arm3])


def apply_pose(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T + np.array([pose.x, pose.y])


def make_config(**overrides) -> RegistrationConfig:
    return RegistrationConfig(**overrides)


# --- A. Identity -------------------------------------------------------

def test_identity_registration() -> None:
    points = l_shaped_points()
    result = register(points, points, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.rejected_reason is None
    assert result.converged is True
    assert result.pose.x == pytest.approx(0.0, abs=1e-6)
    assert result.pose.y == pytest.approx(0.0, abs=1e-6)
    assert result.pose.theta == pytest.approx(0.0, abs=1e-6)
    assert result.residual == pytest.approx(0.0, abs=1e-6)


# --- B. Pure translation (both signs) -----------------------------------

@pytest.mark.parametrize("dx,dy", [(0.15, 0.1), (-0.15, 0.1), (0.1, -0.2), (-0.15, -0.15)])
def test_pure_translation(dx: float, dy: float) -> None:
    # A pure translation shifts every point by the same amount, so
    # (unlike rotation, where near-origin points shift less) there are
    # no "easy" bootstrap correspondences when starting from a zero
    # seed if the shift exceeds max_correspondence_distance -- these
    # magnitudes stay comfortably inside the default threshold, which
    # is itself representative of real inter-scan motion at the
    # design's assumed 0.1-0.5 m/s / ~11.5 Hz operating point.
    local = l_shaped_points()
    true_pose = Pose2D(x=dx, y=dy, theta=0.0)
    reference = apply_pose(local, true_pose)
    result = register(reference, local, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.rejected_reason is None
    assert result.converged is True
    assert result.pose.x == pytest.approx(dx, abs=1e-3)
    assert result.pose.y == pytest.approx(dy, abs=1e-3)
    assert result.pose.theta == pytest.approx(0.0, abs=1e-3)


# --- C. Pure rotation ----------------------------------------------------

def test_pure_rotation() -> None:
    local = l_shaped_points()
    true_pose = Pose2D(x=0.0, y=0.0, theta=math.radians(20.0))
    reference = apply_pose(local, true_pose)
    result = register(reference, local, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.rejected_reason is None
    assert result.converged is True
    assert result.pose.theta == pytest.approx(math.radians(20.0), abs=1e-3)
    assert result.pose.x == pytest.approx(0.0, abs=1e-3)
    assert result.pose.y == pytest.approx(0.0, abs=1e-3)


# --- D. Combined motion ---------------------------------------------------

def test_combined_translation_and_rotation() -> None:
    local = l_shaped_points()
    true_pose = Pose2D(x=0.15, y=-0.1, theta=math.radians(12.0))
    reference = apply_pose(local, true_pose)
    result = register(reference, local, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.rejected_reason is None, result.rejected_reason
    assert result.converged is True
    assert result.pose.x == pytest.approx(0.15, abs=1e-3)
    assert result.pose.y == pytest.approx(-0.1, abs=1e-3)
    assert result.pose.theta == pytest.approx(math.radians(12.0), abs=1e-3)


# --- E. Seed offset --------------------------------------------------------

def test_converges_from_imperfect_seed() -> None:
    local = l_shaped_points()
    true_pose = Pose2D(x=0.3, y=0.2, theta=math.radians(10.0))
    reference = apply_pose(local, true_pose)
    seed = Pose2D(x=0.2, y=0.1, theta=math.radians(5.0))  # deliberately off, but reasonable
    result = register(reference, local, seed, make_config())
    assert result.rejected_reason is None
    assert result.converged is True
    assert result.pose.x == pytest.approx(0.3, abs=1e-3)
    assert result.pose.y == pytest.approx(0.2, abs=1e-3)
    assert result.pose.theta == pytest.approx(math.radians(10.0), abs=1e-3)


# --- F. Correspondence distance rejection ----------------------------------

def test_correspondence_distance_rejection_excludes_outliers() -> None:
    local = l_shaped_points()
    true_pose = Pose2D(x=0.2, y=0.1, theta=0.0)
    reference = apply_pose(local, true_pose)
    # Outlier scan points that will never land near the reference cloud,
    # regardless of any reasonable pose estimate.
    outliers_local = np.array([[50.0, 50.0], [-50.0, 60.0], [70.0, -40.0]])
    scan = np.vstack([local, outliers_local])
    result = register(reference, scan, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.rejected_reason is None
    assert result.converged is True
    assert result.pose.x == pytest.approx(0.2, abs=1e-3)
    assert result.pose.y == pytest.approx(0.1, abs=1e-3)
    # The outliers must not all have been accepted as correspondences.
    assert result.correspondence_count <= len(local)


# --- G. Insufficient correspondences ----------------------------------------

def test_insufficient_correspondences_when_clouds_are_far_apart() -> None:
    local = l_shaped_points()
    reference = local + np.array([100.0, 100.0])  # far beyond any correspondence distance
    result = register(reference, local, Pose2D(x=0.0, y=0.0, theta=0.0), make_config(max_correspondence_distance=0.3))
    assert result.converged is False
    assert result.rejected_reason == "insufficient_correspondences"
    assert result.pose.x == pytest.approx(0.0)
    assert result.pose.y == pytest.approx(0.0)
    assert result.pose.theta == pytest.approx(0.0)


# --- H. Degenerate geometry ---------------------------------------------------

def test_degenerate_geometry_all_matched_points_coincide() -> None:
    scan = np.array([[1.0, 1.0]] * 5)
    reference = np.array([[1.2, 0.9]] * 5)  # close enough to correspond, but every point identical
    # min_sectors=1 isolates the degenerate-geometry mechanism being
    # tested here from Phase 3's separate spread check (all 5 points
    # share one bearing by construction, which the spread check would
    # otherwise catch first -- see test_insufficient_spatial_spread_*
    # for that mechanism specifically).
    cfg = make_config(max_correspondence_distance=1.0, min_correspondences=2, min_sectors=1)
    result = register(reference, scan, Pose2D(x=0.0, y=0.0, theta=0.0), cfg)
    assert result.converged is False
    assert result.rejected_reason == "degenerate_geometry"


# --- I. Max iterations -----------------------------------------------------

def test_max_iterations_reports_not_converged() -> None:
    local = l_shaped_points()
    # A case that reliably takes several iterations to converge under
    # the default config (verified: 3-4 iterations) -- max_iterations=1
    # must cut it off before it gets there.
    true_pose = Pose2D(x=0.15, y=0.1, theta=0.0)
    reference = apply_pose(local, true_pose)
    cfg = make_config(max_iterations=1)
    result = register(reference, local, Pose2D(x=0.0, y=0.0, theta=0.0), cfg)
    assert result.iterations == 1
    assert result.converged is False
    assert result.rejected_reason == "max_iterations"


# --- J. Determinism ---------------------------------------------------------

def test_determinism_identical_inputs_produce_identical_output() -> None:
    local = l_shaped_points()
    true_pose = Pose2D(x=0.35, y=-0.15, theta=math.radians(12.0))
    reference = apply_pose(local, true_pose)
    seed = Pose2D(x=0.1, y=0.05, theta=math.radians(3.0))
    cfg = make_config()

    result1 = register(reference.copy(), local.copy(), seed, cfg)
    result2 = register(reference.copy(), local.copy(), seed, cfg)

    assert result1.pose.x == result2.pose.x
    assert result1.pose.y == result2.pose.y
    assert result1.pose.theta == result2.pose.theta
    assert result1.converged == result2.converged
    assert result1.residual == result2.residual
    assert result1.correspondence_count == result2.correspondence_count
    assert result1.iterations == result2.iterations
    assert result1.rejected_reason == result2.rejected_reason


# --- K. No mutation ----------------------------------------------------------

def test_inputs_are_not_mutated() -> None:
    local = l_shaped_points()
    true_pose = Pose2D(x=0.2, y=0.1, theta=math.radians(8.0))
    reference = apply_pose(local, true_pose)
    local_before = local.copy()
    reference_before = reference.copy()

    register(reference, local, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())

    np.testing.assert_array_equal(local, local_before)
    np.testing.assert_array_equal(reference, reference_before)


# --- Coordinate-frame convention -------------------------------------------

def test_coordinate_frame_convention_world_equals_R_theta_local_plus_xy() -> None:
    local = l_shaped_points()
    true_pose = Pose2D(x=0.1, y=0.1, theta=math.radians(10.0))
    reference = apply_pose(local, true_pose)
    result = register(reference, local, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.rejected_reason is None

    # The recovered pose, applied via the documented formula
    # world = R(theta) @ local + [x, y], must reproduce the known
    # reference points.
    theta = result.pose.theta
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    predicted_world = local @ R.T + np.array([result.pose.x, result.pose.y])
    np.testing.assert_allclose(predicted_world, reference, atol=1e-3)

    # Explicit convention check: positive theta rotates +X toward +Y
    # (counterclockwise), independent of any registration result.
    quarter_turn = np.array([[math.cos(math.pi / 2), -math.sin(math.pi / 2)],
                              [math.sin(math.pi / 2), math.cos(math.pi / 2)]])
    rotated_x_axis = quarter_turn @ np.array([1.0, 0.0])
    np.testing.assert_allclose(rotated_x_axis, [0.0, 1.0], atol=1e-9)


# =========================================================================
# Phase 3: trimmed ICP + angular-sector spread validation
# =========================================================================

def bearing_point(angle_deg: float, r: float) -> list[float]:
    theta = math.radians(angle_deg)
    return [r * math.cos(theta), r * math.sin(theta)]


# --- A. Trim fraction behavior --------------------------------------------

def test_keep_count_formula() -> None:
    cfg = make_config(trim_fraction=0.8, min_correspondences=3)
    assert _keep_count(16, cfg) == 13  # ceil(0.8*16) = 12.8 -> 13
    assert _keep_count(10, cfg) == 8   # ceil(0.8*10) = 8
    assert _keep_count(4, cfg) == 4    # ceil(0.8*4) = 3.2 -> 4
    assert _keep_count(3, cfg) == 3    # already at the min_correspondences floor
    # The min_correspondences floor overrides a trim_fraction that would
    # otherwise cut below it.
    cfg_floor = make_config(trim_fraction=0.5, min_correspondences=5)
    assert _keep_count(6, cfg_floor) == 5  # ceil(0.5*6)=3, but floored to 5
    # keep_count must never exceed the actual correspondence count.
    assert _keep_count(2, make_config(trim_fraction=1.0, min_correspondences=2)) == 2


def test_trim_fraction_retains_only_best_correspondences() -> None:
    # 16 "clean" correspondences (residual 0, identity transform) plus 3
    # scattered points with a deliberately large-but-in-range residual
    # (~0.18m, under the default 0.3m max_correspondence_distance).
    # raw correspondence_count = 19, keep = ceil(0.8*19) = 16 -- exactly
    # the clean-point count, so trimming must drop all 3 bad ones for
    # the retained set (and thus the fit) to be residual-free.
    local = l_shaped_points()
    reference = local.copy()  # identity: residual 0 for every clean point
    bad_local = np.array([bearing_point(190.0, 1.0), bearing_point(200.0, 1.3), bearing_point(210.0, 0.8)])
    bad_reference = bad_local + np.array([0.15, -0.1])

    scan = np.vstack([local, bad_local])
    full_reference = np.vstack([reference, bad_reference])

    result = register(full_reference, scan, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.rejected_reason is None
    assert result.converged is True
    assert result.correspondence_count == 16
    assert result.residual == pytest.approx(0.0, abs=1e-9)
    assert result.pose.x == pytest.approx(0.0, abs=1e-6)
    assert result.pose.y == pytest.approx(0.0, abs=1e-6)
    assert result.pose.theta == pytest.approx(0.0, abs=1e-6)


# --- B. Trimmed outlier robustness -----------------------------------------

def test_trimmed_registration_close_to_static_transform_despite_outliers() -> None:
    local = l_shaped_points()  # 16 static points
    true_pose = Pose2D(x=0.1, y=0.1, theta=math.radians(10.0))
    static_reference = apply_pose(local, true_pose)

    # 3 scattered outlier/moving-geometry points with reference
    # correspondences inconsistent with the true rigid transform.
    moving_local = np.array([bearing_point(190.0, 1.0), bearing_point(200.0, 1.3), bearing_point(210.0, 0.8)])
    moving_reference = apply_pose(moving_local, true_pose) + np.array([0.15, -0.1])

    scan = np.vstack([local, moving_local])
    reference = np.vstack([static_reference, moving_reference])

    result = register(reference, scan, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.rejected_reason is None
    assert result.converged is True
    assert result.correspondence_count == 16  # the 3 outliers were trimmed away
    assert result.pose.x == pytest.approx(0.1, abs=1e-3)
    assert result.pose.y == pytest.approx(0.1, abs=1e-3)
    assert result.pose.theta == pytest.approx(math.radians(10.0), abs=1e-3)


# --- C. Minimum correspondence protection -----------------------------------

def test_insufficient_correspondences_not_rescued_by_trimming() -> None:
    # Only 2 raw correspondences -- below the default min_correspondences=3.
    # Trimming must never be reached; there is nothing to "trim up to" the
    # floor when the raw set is already too small.
    pts = np.array([[0.5, 0.0], [0.0, 0.5]])
    result = register(pts, pts, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.converged is False
    assert result.rejected_reason == "insufficient_correspondences"
    assert result.correspondence_count == 2


# --- D. Sector calculation ---------------------------------------------------

@pytest.mark.parametrize(
    "angle_deg,expected_sector",
    [(0.0, 0), (44.0, 0), (46.0, 1), (90.0, 2), (134.0, 2), (180.0, 4), (270.0, 6), (315.0, 7), (359.0, 7)],
)
def test_sector_assignment_at_known_bearings(angle_deg: float, expected_sector: int) -> None:
    point = np.array([bearing_point(angle_deg, 1.0)])
    sectors = _sector_indices(point)
    assert sectors[0] == expected_sector


def test_sector_convention_zero_is_plus_x_ccw_increasing() -> None:
    # +X (bearing 0) -> sector 0; +Y (bearing 90, reached by rotating +X
    # counterclockwise) -> sector 2 -- confirms the CCW-increasing
    # convention this project uses everywhere else.
    points = np.array([[1.0, 0.0], [0.0, 1.0]])
    sectors = _sector_indices(points)
    assert sectors[0] == 0
    assert sectors[1] == 2


# --- E. Sufficient spatial spread -------------------------------------------

def test_sufficient_spread_allows_registration_to_proceed() -> None:
    points = np.vstack([
        [bearing_point(10.0, 0.5), bearing_point(10.0, 0.9)],
        [bearing_point(125.0, 0.5), bearing_point(125.0, 0.9)],
        [bearing_point(255.0, 0.5), bearing_point(255.0, 0.9)],
    ])
    result = register(points, points, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.rejected_reason is None
    assert result.converged is True


# --- F. Insufficient spatial spread -----------------------------------------

def test_insufficient_spread_rejects_registration() -> None:
    # Only 2 sectors represented (bearings 10 and 125 degrees) -- below
    # the default min_sectors=3.
    points = np.vstack([
        np.array([bearing_point(10.0, r) for r in [0.4, 0.7, 1.1, 1.6, 2.0]]),
        np.array([bearing_point(125.0, r) for r in [0.35, 0.6, 0.95, 1.3]]),
    ])
    result = register(points, points, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.converged is False
    assert result.rejected_reason == "insufficient_spatial_spread"


# --- G. Rotation wraparound --------------------------------------------------

def test_sector_assignment_across_pi_boundary() -> None:
    # Bearings just below/above +-180 degrees (the atan2 branch cut) must
    # land in the two sectors adjacent to the 180-degree sector boundary,
    # not jump somewhere unrelated due to a wraparound bug.
    points = np.array([bearing_point(179.9, 1.0), bearing_point(-179.9, 1.0)])
    sectors = _sector_indices(points)
    assert sectors[0] == 3   # just under 180 -> sector [135, 180)
    assert sectors[1] == 4   # just over 180 (== -179.9) -> sector [180, 225)

    # Bearings just below/above 0 degrees (the [0, 2*pi) normalization
    # seam) must land in sector 7 and sector 0 respectively, not alias
    # into each other.
    points_zero = np.array([bearing_point(0.01, 1.0), bearing_point(-0.01, 1.0)])
    sectors_zero = _sector_indices(points_zero)
    assert sectors_zero[0] == 0
    assert sectors_zero[1] == 7


# --- H. Spread applied after trimming ---------------------------------------

def test_spread_check_uses_retained_set_not_raw_set() -> None:
    # Raw correspondences span 3 sectors (0, 2, 5), satisfying min_sectors
    # BEFORE trimming. But the 2 points outside sector 0 have a
    # deliberately worse residual than all 12 sector-0 points, and
    # keep_count works out to exactly 12 -- trimming removes precisely
    # the 2 minority-sector points, leaving only sector 0 covered.
    sector0 = np.array([bearing_point(10.0, r) for r in np.linspace(0.3, 1.4, 12)])
    minority = np.array([bearing_point(125.0, 1.0), bearing_point(255.0, 1.0)])

    scan = np.vstack([sector0, minority])
    reference = scan.copy()
    reference[-2:] += np.array([0.2, 0.0])  # push the 2 minority points' residual up, still < 0.3

    result = register(reference, scan, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.converged is False
    assert result.rejected_reason == "insufficient_spatial_spread"
    assert result.correspondence_count == 12  # the retained (post-trim) count, not the raw 14


# --- I. Moving geometry minority ---------------------------------------------

def test_moving_object_cluster_does_not_dominate_pose_estimate() -> None:
    local = l_shaped_points()
    true_pose = Pose2D(x=0.05, y=0.05, theta=math.radians(8.0))
    static_reference = apply_pose(local, true_pose)

    # A small, spatially clustered "object" (its own sector) whose
    # reference correspondence reflects a stale/old position -- e.g. the
    # object moved between when the reference map was built and now.
    cluster_local = np.array([bearing_point(60.0, 0.9), bearing_point(63.0, 1.0), bearing_point(57.0, 0.8)])
    cluster_reference = apply_pose(cluster_local, true_pose) + np.array([0.2, 0.05])

    scan = np.vstack([local, cluster_local])
    reference = np.vstack([static_reference, cluster_reference])

    result = register(reference, scan, Pose2D(x=0.0, y=0.0, theta=0.0), make_config())
    assert result.rejected_reason is None
    assert result.converged is True
    assert result.correspondence_count == 16  # the moving cluster was trimmed away
    assert result.pose.x == pytest.approx(0.05, abs=1e-3)
    assert result.pose.y == pytest.approx(0.05, abs=1e-3)
    assert result.pose.theta == pytest.approx(math.radians(8.0), abs=1e-3)


# --- J. Determinism (robust path) -------------------------------------------

def test_robust_registration_is_deterministic() -> None:
    local = l_shaped_points()
    true_pose = Pose2D(x=0.1, y=0.1, theta=math.radians(10.0))
    static_reference = apply_pose(local, true_pose)
    moving_local = np.array([bearing_point(190.0, 1.0), bearing_point(200.0, 1.3), bearing_point(210.0, 0.8)])
    moving_reference = apply_pose(moving_local, true_pose) + np.array([0.15, -0.1])
    scan = np.vstack([local, moving_local])
    reference = np.vstack([static_reference, moving_reference])
    cfg = make_config()

    result1 = register(reference.copy(), scan.copy(), Pose2D(x=0.0, y=0.0, theta=0.0), cfg)
    result2 = register(reference.copy(), scan.copy(), Pose2D(x=0.0, y=0.0, theta=0.0), cfg)

    assert result1.pose.x == result2.pose.x
    assert result1.pose.y == result2.pose.y
    assert result1.pose.theta == result2.pose.theta
    assert result1.converged == result2.converged
    assert result1.residual == result2.residual
    assert result1.correspondence_count == result2.correspondence_count
    assert result1.iterations == result2.iterations
    assert result1.rejected_reason == result2.rejected_reason


# --- Configuration validation (new Phase 3 fields) --------------------------

@pytest.mark.parametrize("trim_fraction", [0.0, -0.1, 1.1])
def test_invalid_trim_fraction_rejected(trim_fraction: float) -> None:
    with pytest.raises(ValueError):
        make_config(trim_fraction=trim_fraction)


@pytest.mark.parametrize("min_sectors", [0, -1, 9])
def test_invalid_min_sectors_rejected(min_sectors: int) -> None:
    with pytest.raises(ValueError):
        make_config(min_sectors=min_sectors)
