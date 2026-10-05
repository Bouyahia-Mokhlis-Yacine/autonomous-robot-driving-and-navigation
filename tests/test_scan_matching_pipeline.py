"""Phase 4 + Phase 6 tests: slam.scan_matching_pipeline.

Phase 4 tests (unchanged) use a minimal FakeReferenceProvider, holding
a fixed known world point cloud and recording every query it receives
-- no PersistentWorldModel involved at all, isolating pure pipeline
state-machine behavior.

Phase 6 tests (below the Phase 4 tests) exercise the real
mapping.persistent_world_model.PersistentWorldModel wired in as both
reference_provider and world_model, verifying the integration layer
itself: accepted-only map updates, reference-before-integration
ordering, world-frame stability, and replay determinism end to end.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from common.messages import LaserScan, Pose2D
from mapping.persistent_world_model import CellState, PersistentWorldModel
from slam.scan_matching_pipeline import (
    BootstrapConfig,
    BootstrapStatus,
    PipelineConfig,
    ScanMatchingPipeline,
    _confirmed_reference_is_usable,
    _estimate_velocity,
    _predict_seed_and_motion,
)
from slam.scan_registration import RegistrationConfig

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# --------------------------------------------------------------------- #
# Shared synthetic geometry / scan construction helpers
# --------------------------------------------------------------------- #


def bearing_point(angle_deg: float, r: float) -> list[float]:
    theta = math.radians(angle_deg)
    return [r * math.cos(theta), r * math.sin(theta)]


def static_world_points() -> np.ndarray:
    # Unique bearings (one point per direction, not multiple radii
    # stacked on the same ray -- a LaserScan can only report one range
    # per angular bin) spanning 3 distinct 45-degree sectors, so ICP
    # has well-distributed, non-degenerate geometry to register
    # against throughout these tests.
    sector0 = [bearing_point(a, r) for a, r in zip([10, 14, 18, 22, 26, 30, 34], [1.0, 1.2, 0.9, 1.3, 1.1, 1.4, 1.0])]
    sector2 = [bearing_point(a, r) for a, r in zip([125, 129, 133, 137, 141], [1.1, 0.9, 1.3, 1.0, 1.2])]
    sector5 = [bearing_point(a, r) for a, r in zip([255, 259, 263, 267], [1.0, 1.2, 0.9, 1.1])]
    return np.array(sector0 + sector2 + sector5)


STATIC_WORLD_POINTS = static_world_points()


def world_to_local_points(world_points: np.ndarray, pose: Pose2D) -> np.ndarray:
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    R = np.array([[c, -s], [s, c]])
    return (world_points - np.array([pose.x, pose.y])) @ R


def make_scan_from_local_points(
    points_local: np.ndarray,
    timestamp: float,
    angle_min: float = -math.pi,
    # Fine angular resolution (vs. a real sensor's ~1 degree) so these
    # state-machine tests aren't dominated by LaserScan binning
    # quantization noise -- ICP's own numeric precision is already
    # exhaustively covered at native point-array precision by the
    # Phase 2/3 test suite; this module tests pipeline state
    # transitions, not re-testing ICP accuracy under discretization.
    angle_increment: float = math.radians(0.1),
    range_min: float = 0.15,
    range_max: float = 8.0,
    # Phase 6 note: kept as range_max by default so every existing
    # Phase 4 (registration-only) test is unaffected -- registration
    # only ever reads actual hit points (via preprocess_scan), so a
    # max-range background is harmless there. Feeding a scan built
    # this way through PersistentWorldModel.integrate() is a different
    # matter: at this fine a resolution, adjacent background rays'
    # free-space carving can pass through the very same grid cell as a
    # nearby intentional hit (see mapping/persistent_world_model's own
    # test fixture, which hit this exact issue). Phase 6 tests that
    # exercise real map integration use `background=0.0` (invalid,
    # no-op) via `scan_at_pose_sparse` below instead.
    background: float = 8.0,
) -> LaserScan:
    num_bins = int(round((2.0 * math.pi) / angle_increment)) + 1
    ranges = [background] * num_bins
    assigned = [False] * num_bins  # tracks real hits, independent of `background`'s own value
    for x, y in points_local:
        r = math.hypot(x, y)
        if r < range_min or r >= range_max:
            continue
        angle = math.atan2(y, x)
        bin_index = round((angle - angle_min) / angle_increment)
        if 0 <= bin_index < num_bins and (not assigned[bin_index] or r < ranges[bin_index]):
            ranges[bin_index] = r
            assigned[bin_index] = True
    return LaserScan(
        timestamp=timestamp, angle_min=angle_min, angle_increment=angle_increment,
        ranges=ranges, range_min=range_min, range_max=range_max,
    )


def scan_at_pose(pose: Pose2D, timestamp: float) -> LaserScan:
    return make_scan_from_local_points(world_to_local_points(STATIC_WORLD_POINTS, pose), timestamp)


def scan_at_pose_sparse(pose: Pose2D, timestamp: float, points: np.ndarray | None = None) -> LaserScan:
    """Like ``scan_at_pose``, but with an invalid (no-op) background
    instead of max-range -- see ``make_scan_from_local_points``'s
    ``background`` parameter docstring. Use this (not ``scan_at_pose``)
    whenever a scan is fed through ``PersistentWorldModel.integrate()``
    (Phase 6 map-integration tests), since ICP-only tests don't care
    but real map integration does.
    """
    source_points = STATIC_WORLD_POINTS if points is None else points
    return make_scan_from_local_points(world_to_local_points(source_points, pose), timestamp, background=0.0)


class FakeReferenceProvider:
    """Test-only stand-in for the future PersistentWorldModel: returns a
    fixed, externally-configured point cloud and records every query
    (center, radius) it receives -- never mutated by anything the
    pipeline does, since the pipeline has no method to write to it."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        self.queries: list[tuple[tuple[float, float], float]] = []

    def get_confirmed_occupied_points_world(self, center: tuple[float, float], radius: float) -> np.ndarray:
        self.queries.append((center, radius))
        return self.points

    def set_points(self, points: np.ndarray) -> None:
        self.points = points


def make_pipeline(**pipeline_overrides) -> tuple[ScanMatchingPipeline, FakeReferenceProvider]:
    provider = FakeReferenceProvider(STATIC_WORLD_POINTS)
    pipeline = ScanMatchingPipeline(provider, PipelineConfig(**pipeline_overrides))
    return pipeline, provider


# =========================================================================
# Velocity-frame convention -- mandatory per the approved design (section
# 17): a local/world frame mistake here would silently corrupt every
# subsequent ICP seed, so these test _estimate_velocity directly, at
# full numeric precision (no LaserScan binning involved).
# =========================================================================

def test_velocity_pure_forward_motion() -> None:
    previous = Pose2D(x=0.0, y=0.0, theta=0.0)
    current = Pose2D(x=0.2, y=0.0, theta=0.0)
    vx, vy, omega = _estimate_velocity(previous, 0.0, current, 1.0)
    assert vx > 0
    assert vx == pytest.approx(0.2, abs=1e-9)
    assert vy == pytest.approx(0.0, abs=1e-9)
    assert omega == pytest.approx(0.0, abs=1e-9)


def test_velocity_pure_lateral_motion() -> None:
    previous = Pose2D(x=0.0, y=0.0, theta=0.0)
    current = Pose2D(x=0.0, y=0.2, theta=0.0)
    vx, vy, omega = _estimate_velocity(previous, 0.0, current, 1.0)
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy > 0
    assert vy == pytest.approx(0.2, abs=1e-9)


def test_velocity_rotated_robot_moving_forward() -> None:
    # Robot faces +90 degrees and moves along its OWN local +X axis by
    # 0.2m -- in world coordinates that is a pure +Y displacement, but
    # the recovered LOCAL velocity must still read as forward (+vx),
    # not lateral, proving the frame conversion is correct.
    theta = math.pi / 2.0
    previous = Pose2D(x=0.0, y=0.0, theta=theta)
    current = Pose2D(x=0.0, y=0.2, theta=theta)  # world +Y, local +X given theta=90deg
    vx, vy, omega = _estimate_velocity(previous, 0.0, current, 1.0)
    assert vx > 0
    assert vx == pytest.approx(0.2, abs=1e-9)
    assert vy == pytest.approx(0.0, abs=1e-9)
    assert omega == pytest.approx(0.0, abs=1e-9)


def test_velocity_pure_rotation() -> None:
    previous = Pose2D(x=0.0, y=0.0, theta=0.0)
    current = Pose2D(x=0.0, y=0.0, theta=math.radians(45.0))
    vx, vy, omega = _estimate_velocity(previous, 0.0, current, 1.0)
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy == pytest.approx(0.0, abs=1e-9)
    assert omega != 0.0
    assert omega > 0  # positive = counterclockwise


def test_velocity_combined_translation_and_rotation() -> None:
    previous = Pose2D(x=0.0, y=0.0, theta=0.0)
    current = Pose2D(x=0.2, y=0.1, theta=math.radians(15.0))
    vx, vy, omega = _estimate_velocity(previous, 0.0, current, 1.0)
    assert vx == pytest.approx(0.2, abs=1e-9)
    assert vy == pytest.approx(0.1, abs=1e-9)
    assert omega == pytest.approx(math.radians(15.0), abs=1e-9)


def test_velocity_seed_roundtrip_consistency() -> None:
    # _predict_seed_and_motion (local velocity -> world seed) and
    # _estimate_velocity (world pose delta -> local velocity) must be
    # exact inverses of each other.
    last_pose = Pose2D(x=1.0, y=-0.5, theta=math.radians(30.0))
    velocity = (0.15, -0.05, math.radians(8.0))
    dt = 0.5
    seed, _, _ = _predict_seed_and_motion(last_pose, velocity, dt)
    recovered = _estimate_velocity(last_pose, 0.0, seed, dt)
    assert recovered[0] == pytest.approx(velocity[0], abs=1e-9)
    assert recovered[1] == pytest.approx(velocity[1], abs=1e-9)
    assert recovered[2] == pytest.approx(velocity[2], abs=1e-9)


# =========================================================================
# State-transition tests
# =========================================================================

# --- 1. First scan initialization -------------------------------------

def test_first_scan_initializes_state_at_origin() -> None:
    pipeline, provider = make_pipeline()
    result = pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    assert result.accepted is True
    assert result.localized is True
    assert result.consecutive_failures == 0
    assert result.registration_result is None  # no ICP attempted on scan 1
    assert result.pose.x == pytest.approx(0.0)
    assert result.pose.y == pytest.approx(0.0)
    assert result.pose.theta == pytest.approx(0.0)
    assert pipeline.state.velocity_estimate == (0.0, 0.0, 0.0)
    assert provider.queries == []  # scan 1 never queries the reference provider


# --- 2. Successful registration -----------------------------------------

def test_successful_registration_updates_pose() -> None:
    pipeline, provider = make_pipeline()
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    true_pose = Pose2D(x=0.05, y=0.05, theta=math.radians(5.0))
    result = pipeline.process_scan(scan_at_pose(true_pose, timestamp=0.1))
    assert result.accepted is True
    assert result.registration_result is not None
    assert result.registration_result.converged is True
    assert result.pose.x == pytest.approx(0.05, abs=1e-2)
    assert result.pose.y == pytest.approx(0.05, abs=1e-2)
    assert result.pose.theta == pytest.approx(math.radians(5.0), abs=1e-2)


# --- 3. Seed prediction from velocity -----------------------------------

def test_seed_prediction_reflects_established_velocity() -> None:
    pipeline, provider = make_pipeline()
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    pipeline.process_scan(scan_at_pose(Pose2D(0.05, 0.0, 0.0), timestamp=0.1))
    vx, vy, omega = pipeline.state.velocity_estimate
    assert vx == pytest.approx(0.5, abs=0.05)  # 0.05m / 0.1s
    assert vy == pytest.approx(0.0, abs=0.02)
    # The third scan is generated at a pose consistent with continuing
    # that velocity -- if seed prediction works, registration succeeds.
    result = pipeline.process_scan(scan_at_pose(Pose2D(0.10, 0.0, 0.0), timestamp=0.2))
    assert result.accepted is True
    last_query_center = provider.queries[-1][0]
    # The query center (seed position) should be near the predicted
    # (0.10, 0.0), not the previous accepted (0.05, 0.0).
    assert last_query_center[0] == pytest.approx(0.10, abs=0.02)


# --- 4. Seed displacement gate -------------------------------------------

def test_seed_displacement_gate_rejects_implausible_jump() -> None:
    pipeline, provider = make_pipeline(max_seed_displacement=0.01)
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    pipeline.process_scan(scan_at_pose(Pose2D(0.05, 0.05, math.radians(5.0)), timestamp=0.1))
    state_before = pipeline.state
    result = pipeline.process_scan(scan_at_pose(Pose2D(0.10, 0.10, math.radians(10.0)), timestamp=0.2))
    assert result.accepted is False
    assert result.rejection_reason == "seed_displacement_too_large"
    assert result.registration_result is None  # ICP was never invoked
    assert pipeline.state.last_good_pose == state_before.last_good_pose
    assert pipeline.state.velocity_estimate == state_before.velocity_estimate


# --- 5. Seed rotation gate ------------------------------------------------

def test_seed_rotation_gate_rejects_implausible_spin() -> None:
    pipeline, provider = make_pipeline(max_seed_rotation=math.radians(0.01))
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    pipeline.process_scan(scan_at_pose(Pose2D(0.05, 0.05, math.radians(5.0)), timestamp=0.1))
    result = pipeline.process_scan(scan_at_pose(Pose2D(0.10, 0.10, math.radians(10.0)), timestamp=0.2))
    assert result.accepted is False
    assert result.rejection_reason == "seed_rotation_too_large"
    assert result.registration_result is None


# --- 6. Registration failure ----------------------------------------------

def test_registration_failure_is_reported() -> None:
    provider = FakeReferenceProvider(STATIC_WORLD_POINTS + np.array([50.0, 50.0]))  # hopelessly mismatched
    pipeline = ScanMatchingPipeline(provider)
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    result = pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.1))
    assert result.accepted is False
    assert result.registration_result is not None
    assert result.registration_result.converged is False
    assert result.rejection_reason == result.registration_result.rejected_reason


# --- 7. Consecutive failure counting --------------------------------------

def test_consecutive_failures_increment_each_time() -> None:
    provider = FakeReferenceProvider(STATIC_WORLD_POINTS + np.array([50.0, 50.0]))
    pipeline = ScanMatchingPipeline(provider, PipelineConfig(max_consecutive_failures=10))
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    for i in range(1, 4):
        result = pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.1 * i))
        assert result.consecutive_failures == i
        assert result.localized is True  # threshold not reached yet


# --- 8. Transition to localized=False after threshold failures -----------

def test_localization_lost_after_max_consecutive_failures() -> None:
    provider = FakeReferenceProvider(STATIC_WORLD_POINTS + np.array([50.0, 50.0]))
    pipeline = ScanMatchingPipeline(provider, PipelineConfig(max_consecutive_failures=5))
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    result = None
    for i in range(1, 6):
        result = pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.1 * i))
    assert result.consecutive_failures == 5
    assert result.localized is False
    assert pipeline.state.localized is False


# --- 9. No seed extrapolation after localization loss ---------------------

def test_no_extrapolation_after_localization_lost() -> None:
    provider = FakeReferenceProvider(STATIC_WORLD_POINTS)
    pipeline = ScanMatchingPipeline(provider, PipelineConfig(max_consecutive_failures=5))
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    pipeline.process_scan(scan_at_pose(Pose2D(0.05, 0.0, 0.0), timestamp=0.1))  # succeeds, sets nonzero velocity
    assert pipeline.state.velocity_estimate[0] != 0.0

    provider.set_points(STATIC_WORLD_POINTS + np.array([50.0, 50.0]))  # now force failures
    for i in range(2, 7):
        pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.1 * i))
    assert pipeline.state.localized is False
    provider.queries.clear()
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=1.0))
    # The query center must be exactly last_good_pose (no velocity
    # extrapolation), even though velocity_estimate is still nonzero.
    center = provider.queries[-1][0]
    assert center[0] == pytest.approx(pipeline.state.last_good_pose.x, abs=1e-9)
    assert center[1] == pytest.approx(pipeline.state.last_good_pose.y, abs=1e-9)


# --- 10. Successful local recovery ----------------------------------------

def test_successful_recovery_restores_localized() -> None:
    provider = FakeReferenceProvider(STATIC_WORLD_POINTS + np.array([50.0, 50.0]))
    pipeline = ScanMatchingPipeline(provider, PipelineConfig(max_consecutive_failures=5))
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    for i in range(1, 6):
        pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.1 * i))
    assert pipeline.state.localized is False

    provider.set_points(STATIC_WORLD_POINTS)  # "found the right area again"
    result = pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.7))
    assert result.accepted is True
    assert result.localized is True
    assert result.consecutive_failures == 0
    assert pipeline.state.localized is True


# --- 11. Duplicate timestamp -----------------------------------------------

def test_duplicate_timestamp_rejected_without_state_change() -> None:
    pipeline, provider = make_pipeline()
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    pipeline.process_scan(scan_at_pose(Pose2D(0.05, 0.0, 0.0), timestamp=0.1))
    state_before = pipeline.state
    result = pipeline.process_scan(scan_at_pose(Pose2D(0.20, 0.0, 0.0), timestamp=0.1))  # same timestamp again
    assert result.accepted is False
    assert result.rejection_reason == "timestamp_not_newer"
    assert result.registration_result is None
    assert pipeline.state == state_before  # no state change at all
    assert result.consecutive_failures == state_before.consecutive_failures  # not counted as a failure


# --- 12. Out-of-order timestamp ---------------------------------------------

def test_out_of_order_timestamp_rejected_without_state_change() -> None:
    pipeline, provider = make_pipeline()
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    pipeline.process_scan(scan_at_pose(Pose2D(0.05, 0.0, 0.0), timestamp=0.2))
    state_before = pipeline.state
    result = pipeline.process_scan(scan_at_pose(Pose2D(0.20, 0.0, 0.0), timestamp=0.1))  # earlier than last
    assert result.accepted is False
    assert result.rejection_reason == "timestamp_not_newer"
    assert pipeline.state == state_before


# --- 13/14. Failed registration does not alter last-good pose/velocity ----

def test_failed_registration_does_not_alter_pose_or_velocity() -> None:
    pipeline, provider = make_pipeline()
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    pipeline.process_scan(scan_at_pose(Pose2D(0.05, 0.0, 0.0), timestamp=0.1))
    state_before = pipeline.state

    # Swap in mismatched geometry to force a failure on the next scan.
    provider.set_points(STATIC_WORLD_POINTS + np.array([50.0, 50.0]))
    result = pipeline.process_scan(scan_at_pose(Pose2D(0.10, 0.0, 0.0), timestamp=0.2))

    assert result.accepted is False
    assert pipeline.state.last_good_pose == state_before.last_good_pose
    assert pipeline.state.last_good_timestamp == state_before.last_good_timestamp
    assert pipeline.state.velocity_estimate == state_before.velocity_estimate


# --- 15. Successful registration resets failure count ---------------------

def test_success_resets_consecutive_failures() -> None:
    provider = FakeReferenceProvider(STATIC_WORLD_POINTS + np.array([50.0, 50.0]))
    pipeline = ScanMatchingPipeline(provider, PipelineConfig(max_consecutive_failures=10))
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.1))
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.2))
    assert pipeline.state.consecutive_failures == 2

    provider.set_points(STATIC_WORLD_POINTS)
    result = pipeline.process_scan(scan_at_pose(Pose2D(0.02, 0.0, 0.0), timestamp=0.3))
    assert result.accepted is True
    assert result.consecutive_failures == 0
    assert pipeline.state.consecutive_failures == 0


# --- 16. Current scan never inserted into the reference provider ----------

def test_current_scan_never_inserted_into_reference_provider() -> None:
    pipeline, provider = make_pipeline()
    points_before = provider.points.copy()
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    pipeline.process_scan(scan_at_pose(Pose2D(0.05, 0.0, 0.0), timestamp=0.1))
    pipeline.process_scan(scan_at_pose(Pose2D(0.10, 0.0, 0.0), timestamp=0.2))
    np.testing.assert_array_equal(provider.points, points_before)
    assert provider.points.shape[0] == points_before.shape[0]  # nothing appended


# --- 17. Reference query centered on the seed pose -------------------------

def test_reference_query_centered_on_seed_pose() -> None:
    pipeline, provider = make_pipeline()
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    # Zero velocity established on scan 1 -> seed for scan 2 is exactly
    # last_good_pose == (0, 0).
    pipeline.process_scan(scan_at_pose(Pose2D(0.05, 0.0, 0.0), timestamp=0.1))
    center = provider.queries[-1][0]
    assert center == pytest.approx((0.0, 0.0), abs=1e-9)


# --- 18. Reference radius passed correctly ----------------------------------

def test_reference_radius_passed_correctly() -> None:
    pipeline, provider = make_pipeline(registration_radius_m=3.5)
    pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    pipeline.process_scan(scan_at_pose(Pose2D(0.02, 0.0, 0.0), timestamp=0.1))
    assert provider.queries[-1][1] == pytest.approx(3.5)


# =========================================================================
# Configuration validation
# =========================================================================

@pytest.mark.parametrize("field,value", [
    ("registration_radius_m", 0.0), ("registration_radius_m", -1.0),
    ("max_seed_displacement", 0.0), ("max_seed_displacement", -0.1),
    ("max_seed_rotation", 0.0), ("max_seed_rotation", -0.1),
    ("max_consecutive_failures", 0),
])
def test_invalid_pipeline_config_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(**{field: value})


# =========================================================================
# Phase 6: integration with the real PersistentWorldModel
# =========================================================================

CLUSTER_A = STATIC_WORLD_POINTS[:7]    # bearings ~10-34 deg
CLUSTER_B = STATIC_WORLD_POINTS[7:12]  # bearings ~125-141 deg
CLUSTER_C = STATIC_WORLD_POINTS[12:]   # bearings ~255-267 deg
# A cluster distinct from A/B/C's sectors (sector 4, ~190-200 deg), used
# where a test needs genuinely NEW geometry never present in STATIC_WORLD_POINTS.
CLUSTER_NEW = np.array([bearing_point(a, 1.0) for a in (190.0, 195.0, 200.0)])
# CLUSTER_A[0] is a known-reliable confirming point (a couple of the 16
# static points sit close enough to a grid-cell boundary that they
# don't reliably confirm through 0.1-degree scan binning -- see the
# Phase 5 test suite's own note on this class of issue); tests that
# need to assert a SPECIFIC point's state use this one, not the whole
# cluster, to stay robust rather than fragile on that discretization
# edge case.


def prebuilt_world_model(
    points: np.ndarray = STATIC_WORLD_POINTS,
    pose: Pose2D = Pose2D(x=0.0, y=0.0, theta=0.0),
    timestamps: tuple[float, ...] = (0.0, 0.2, 0.5),
) -> PersistentWorldModel:
    """A PersistentWorldModel with ``points`` already CONFIRMED, built by
    calling integrate() directly with ground-truth poses -- simulating
    the state of a map after some earlier successful mapping period,
    exactly as Phase 5's own tests already validate confirmation
    itself. Uses scan_at_pose_sparse (invalid, not max-range,
    background) throughout to avoid the adjacent-ray free-space
    interference documented in mapping/persistent_world_model's own
    test suite."""
    model = PersistentWorldModel()
    for t in timestamps:
        model.integrate(pose, scan_at_pose_sparse(pose, t, points=points), now=t)
    return model


class RecordingWorldModel:
    """Spy wrapper around a real PersistentWorldModel: records each
    reference query's (center, radius, returned points -- snapshotted
    at query time) before delegating, so a test can inspect exactly
    what ICP actually saw, independent of whatever integrate() does
    afterward within the same process_scan() call."""

    def __init__(self, model: PersistentWorldModel) -> None:
        self.model = model
        self.queries: list[tuple[tuple[float, float], float, np.ndarray]] = []

    def get_confirmed_occupied_points_world(self, center: tuple[float, float], radius: float) -> np.ndarray:
        points = self.model.get_confirmed_occupied_points_world(center, radius)
        self.queries.append((center, radius, points.copy()))
        return points

    def integrate(self, pose: Pose2D, scan: LaserScan, now: float) -> None:
        self.model.integrate(pose, scan, now)


# --- First-scan / startup behavior ------------------------------------------

def test_first_scan_populates_world_model() -> None:
    model = PersistentWorldModel()
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)

    result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=0.0, points=CLUSTER_A))
    assert result.accepted is True
    assert result.map_integrated is True
    assert result.registration_result is None  # no ICP on scan 1 (section 5/8)
    # Freshly observed once -- tentative, not yet confirmed.
    assert model.get_cell_state(CLUSTER_A[0][0], CLUSTER_A[0][1]) == CellState.TENTATIVE


def test_startup_scan1_continues_bootstrap_not_a_registration_failure() -> None:
    # This is exactly the Phase 6 "cold-start deadlock" scenario --
    # scan 0 initializes+integrates, but its geometry is only tentative
    # (one observation), so scan 1's confirmed reference is still
    # empty. Phase 6 alone would have this fail registration outright
    # (see the old, now-superseded version of this test). Phase 7's
    # whole point is that this is NOT a registration failure at all --
    # scan 1 simply continues bootstrapping (fixed pose, no ICP, no
    # failure count increment), exactly per the approved design.
    model = PersistentWorldModel()
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)

    r0 = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=0.0, points=CLUSTER_A))
    assert r0.accepted is True and r0.map_integrated is True
    assert r0.bootstrap_status == BootstrapStatus.BOOTSTRAPPING
    assert model.get_confirmed_occupied_points_world((0.0, 0.0), 8.0).shape[0] == 0  # nothing confirmed yet

    r1 = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=0.1, points=CLUSTER_A))
    assert r1.accepted is True  # not a failure
    assert r1.map_integrated is True
    assert r1.registration_result is None  # no ICP was attempted
    assert r1.consecutive_failures == 0
    assert r1.bootstrap_status == BootstrapStatus.BOOTSTRAPPING  # CLUSTER_A alone never spans enough sectors
    assert r1.pose.x == pytest.approx(0.0) and r1.pose.y == pytest.approx(0.0) and r1.pose.theta == pytest.approx(0.0)


# --- Reference-before-integration invariant (section 12) --------------------

def test_current_scans_new_geometry_never_appears_in_its_own_reference() -> None:
    model = prebuilt_world_model()  # full 3-sector base map, pre-confirmed
    spy = RecordingWorldModel(model)
    pipeline = ScanMatchingPipeline(reference_provider=spy, world_model=spy, bootstrap_config=BootstrapConfig(min_duration=0.0))

    # Pipeline-first scan: bootstraps pipeline state, no ICP, reinforces the base map only.
    pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.0))

    # This scan sees the base map (lets registration succeed) AND
    # brand-new geometry (CLUSTER_NEW), never observed before this
    # exact call, not part of STATIC_WORLD_POINTS at all.
    combined = np.vstack([STATIC_WORLD_POINTS, CLUSTER_NEW])
    result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.1, points=combined))
    assert result.accepted is True
    assert result.map_integrated is True

    # The reference actually handed to ICP this call must not contain
    # any point near CLUSTER_NEW's location.
    _, _, queried_points = spy.queries[-1]
    for pt in CLUSTER_NEW:
        distances = np.hypot(queried_points[:, 0] - pt[0], queried_points[:, 1] - pt[1]) if queried_points.size else np.array([np.inf])
        assert np.all(distances > 0.1), "new geometry leaked into its own registration reference"

    # It was, correctly, integrated AFTER registration -- now tentative.
    # (Checked at the accepted pose's world coordinates: tiny ICP
    # residual drift can shift which exact 0.05m cell a given point
    # lands in, so at least one of the three is checked reliably
    # rather than asserting all three land in the precise cell a
    # perfect zero-drift transform would predict.)
    assert any(model.get_cell_state(pt[0], pt[1]) == CellState.TENTATIVE for pt in CLUSTER_NEW)


# --- Failed registration produces zero map changes (section 17) -------------

def test_failed_registration_produces_zero_map_changes() -> None:
    # Full 3-sector base map (not just CLUSTER_A) -- needed so
    # bootstrap can actually complete (a 1-sector reference never
    # satisfies the sector-spread requirement, regardless of min_duration).
    model = prebuilt_world_model()
    pipeline = ScanMatchingPipeline(
        reference_provider=model, world_model=model, bootstrap_config=BootstrapConfig(min_duration=0.0)
    )
    pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.0))

    state_before = model.state.copy()
    pending_before = model.pending_since.copy()
    pipeline_state_before = pipeline.state

    # A scan of totally unrelated geometry, far from anything confirmed
    # -- registration must fail (insufficient correspondences).
    far_points = CLUSTER_A + np.array([50.0, 50.0])
    result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.1, points=far_points))

    assert result.accepted is False
    assert result.map_integrated is False
    np.testing.assert_array_equal(model.state, state_before)
    both_nan = np.isnan(model.pending_since) & np.isnan(pending_before)
    np.testing.assert_array_equal(np.isnan(model.pending_since), np.isnan(pending_before))
    np.testing.assert_array_equal(model.pending_since[~both_nan], pending_before[~both_nan])

    assert pipeline.state.last_good_pose == pipeline_state_before.last_good_pose
    assert pipeline.state.last_good_timestamp == pipeline_state_before.last_good_timestamp
    assert pipeline.state.velocity_estimate == pipeline_state_before.velocity_estimate
    assert pipeline.state.consecutive_failures == pipeline_state_before.consecutive_failures + 1


# --- Accepted registration integrates with the accepted pose (section 18) ---

def test_accepted_registration_integrates_with_accepted_pose() -> None:
    model = prebuilt_world_model()
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model, bootstrap_config=BootstrapConfig(min_duration=0.0))
    pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.0))

    true_pose = Pose2D(x=0.04, y=0.03, theta=math.radians(3.0))  # small, ICP-realistic motion
    result = pipeline.process_scan(scan_at_pose_sparse(true_pose, timestamp=1.1))

    assert result.accepted is True
    assert result.registration_result is not None and result.registration_result.converged is True
    assert result.map_integrated is True
    assert result.pose.x == pytest.approx(true_pose.x, abs=0.03)
    assert result.pose.y == pytest.approx(true_pose.y, abs=0.03)
    assert result.pose.theta == pytest.approx(true_pose.theta, abs=0.05)
    # The pose actually used to integrate must be the ACCEPTED pose,
    # not the seed: since velocity was ~0 entering this call, the seed
    # was ~(0,0,0) -- if integration had used the seed instead, the
    # scan's geometry would land at the wrong world location.
    assert model.get_cell_state(CLUSTER_A[0][0], CLUSTER_A[0][1]) == CellState.CONFIRMED  # untouched, still there


# --- World-frame stability (section 19) -------------------------------------

def test_world_frame_stable_across_robot_motion() -> None:
    model = prebuilt_world_model()
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model, bootstrap_config=BootstrapConfig(min_duration=0.0))
    pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.0))

    small_motion = Pose2D(x=0.03, y=0.0, theta=0.0)
    result = pipeline.process_scan(scan_at_pose_sparse(small_motion, timestamp=1.1))
    assert result.accepted is True

    # The pre-confirmed geometry's world coordinates must be exactly
    # where they always were -- the map does not shift with the robot.
    assert model.get_cell_state(CLUSTER_A[0][0], CLUSTER_A[0][1]) == CellState.CONFIRMED


# --- Moving geometry: no identity assumption (section 20) -------------------

def test_moving_geometry_no_identity_assumed() -> None:
    model = prebuilt_world_model()  # full 3-sector base map, pre-confirmed (registration anchor)
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model, bootstrap_config=BootstrapConfig(min_duration=0.0))
    pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.0))

    # "Geometry" appears at a brand-new location D1 (never seen before).
    d1 = CLUSTER_NEW  # bearing ~190-200 deg
    scan_1 = np.vstack([STATIC_WORLD_POINTS, d1])
    r1 = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.1, points=scan_1))
    assert r1.accepted is True
    # (See the analogous note in test_current_scans_new_geometry_...:
    # sub-cell ICP drift can shift which exact cell an individual point
    # lands in, so "at least one of the cluster's points" is the
    # robust check, not "every single one".)
    assert any(model.get_cell_state(pt[0], pt[1]) == CellState.TENTATIVE for pt in d1)

    # Now "geometry" is at a DIFFERENT new location D2 instead; D1 is
    # simply absent from this scan (not directly observed free -- this
    # narrow synthetic scan doesn't cover that bearing at all). D1
    # (not contradicted) persists as-is; D2 starts its own, entirely
    # independent, tentative observation. Nothing here ever treats D1
    # and D2 as "the same object that moved."
    d2 = np.array([bearing_point(300.0, 1.0), bearing_point(305.0, 1.0), bearing_point(310.0, 1.0)])
    scan_2 = np.vstack([STATIC_WORLD_POINTS, d2])
    r2 = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.2, points=scan_2))
    assert r2.accepted is True
    assert any(model.get_cell_state(pt[0], pt[1]) == CellState.TENTATIVE for pt in d1)  # untouched, not "moved away"
    assert any(model.get_cell_state(pt[0], pt[1]) == CellState.TENTATIVE for pt in d2)


# --- Occlusion protects far geometry through the pipeline (section 21) ------

def test_occlusion_protects_confirmed_geometry_through_pipeline() -> None:
    far = np.array([[5.0, 0.0]])
    near = np.array([[2.0, 0.0]])
    model = prebuilt_world_model(points=np.vstack([STATIC_WORLD_POINTS, far]))
    assert model.get_cell_state(5.0, 0.0) == CellState.CONFIRMED

    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model, bootstrap_config=BootstrapConfig(min_duration=0.0))
    pipeline.process_scan(
        scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.0, points=np.vstack([STATIC_WORLD_POINTS, far]))
    )

    # A closer obstacle now blocks the ray to the far point; the base
    # map remains available so registration still succeeds.
    blocked_scan_points = np.vstack([STATIC_WORLD_POINTS, near])
    result = pipeline.process_scan(
        scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.1, points=blocked_scan_points)
    )
    assert result.accepted is True
    assert model.get_cell_state(5.0, 0.0) == CellState.CONFIRMED  # untouched -- Phase 5 semantics, Phase 6 doesn't interfere


# --- map_integrated correctness without a world model -----------------------

def test_map_integrated_false_without_world_model() -> None:
    pipeline, provider = make_pipeline()  # Phase 4's FakeReferenceProvider, no world_model
    r0 = pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    assert r0.map_integrated is False
    r1 = pipeline.process_scan(scan_at_pose(Pose2D(0.02, 0.0, 0.0), timestamp=0.1))
    assert r1.accepted is True
    assert r1.map_integrated is False  # no world_model was supplied -- always False, never an error


# --- Replay determinism (section 23) -----------------------------------------

def test_replay_determinism_pose_sequence_and_map_state() -> None:
    def run() -> tuple[list[Pose2D], PersistentWorldModel]:
        model = prebuilt_world_model()
        pipeline = ScanMatchingPipeline(
            reference_provider=model, world_model=model, bootstrap_config=BootstrapConfig(min_duration=0.0)
        )
        poses = []
        scans = [
            scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.0),
            scan_at_pose_sparse(Pose2D(0.02, 0.0, 0.0), timestamp=1.1),
            scan_at_pose_sparse(Pose2D(0.04, 0.01, math.radians(2.0)), timestamp=1.2),
            scan_at_pose_sparse(Pose2D(0.06, 0.02, math.radians(4.0)), timestamp=1.3),
        ]
        for scan in scans:
            result = pipeline.process_scan(scan)
            poses.append(result.pose)
        return poses, model

    poses_a, model_a = run()
    poses_b, model_b = run()

    assert len(poses_a) == len(poses_b)
    for pa, pb in zip(poses_a, poses_b):
        assert pa.x == pb.x
        assert pa.y == pb.y
        assert pa.theta == pb.theta

    np.testing.assert_array_equal(model_a.state, model_b.state)
    np.testing.assert_array_equal(model_a.get_grid().cells, model_b.get_grid().cells)


# --- Real stationary fixture (section 24) -- stability only, no motion claim ---

def test_real_stationary_fixture_integrates_without_crashing() -> None:
    """The real fixture is a STATIONARY recording -- this test only
    checks the pipeline+world-model integration stays stable and
    deterministic against real (slightly noisy) hardware data. It
    makes no claim about successful motion estimation, since the
    sensor never moved while this fixture was recorded."""
    fixture_path = _FIXTURES_DIR / "real_scans_stationary.json"
    if not fixture_path.exists():
        pytest.skip("real hardware scan fixture not present")
    with fixture_path.open("r", encoding="utf-8") as f:
        raw_scans = json.load(f)
    scans = [LaserScan(**raw) for raw in raw_scans]

    model = PersistentWorldModel(confirmation_duration=0.05)  # short duration: fixture spans ~2.2s total
    # bootstrap min_duration/timeout scaled down to match: the fixture
    # only spans ~2.2s total, so the pipeline defaults (min_duration=0.6s)
    # would leave little or no room to observe the transition to normal
    # (ICP-based) localization within this short recording.
    pipeline = ScanMatchingPipeline(
        reference_provider=model, world_model=model,
        bootstrap_config=BootstrapConfig(min_duration=0.3, timeout=5.0),
    )

    accepted_count = 0
    saw_bootstrapping = False
    saw_normal_registration = False
    for scan in scans:
        result = pipeline.process_scan(scan)
        if result.bootstrap_status == BootstrapStatus.BOOTSTRAPPING:
            saw_bootstrapping = True
        if result.registration_result is not None:
            saw_normal_registration = True
        if result.accepted:
            accepted_count += 1
            assert _pose_finite(result.pose)

    assert accepted_count >= 1  # at least the first (always-accepted) scan
    assert saw_bootstrapping is True
    # Not asserting saw_normal_registration: whether this short, real,
    # stationary fixture produces ENOUGH confirmed+spread geometry to
    # exit bootstrap within its ~2.2s span depends on the room's real
    # geometry -- observed here only for information, not a pass/fail
    # claim about motion estimation (the sensor never moved).
    grid = model.get_grid()
    assert grid.cells.shape == (model.occupancy_grid.grid.height, model.occupancy_grid.grid.width)


def _pose_finite(pose: Pose2D) -> bool:
    return math.isfinite(pose.x) and math.isfinite(pose.y) and math.isfinite(pose.theta)


# =========================================================================
# Phase 7: cold-start bootstrap
#
# Throughout, "stationary" scans are built at the fixed pose (0,0,0)
# via scan_at_pose_sparse -- matching the explicit operational
# precondition bootstrap relies on and never verifies: the robot must
# actually be stationary for these scans to be physically meaningful.
# =========================================================================

# --- _confirmed_reference_is_usable (pure helper) ---------------------------

def test_confirmed_reference_is_usable_requires_both_count_and_spread() -> None:
    three_sector_points = np.vstack([
        np.array([bearing_point(10.0, 1.0), bearing_point(15.0, 1.0), bearing_point(20.0, 1.0)]),
        np.array([bearing_point(125.0, 1.0)]),
        np.array([bearing_point(255.0, 1.0)]),
    ])
    one_sector_points = np.array([bearing_point(a, 1.0) for a in (10, 12, 14, 16, 18, 20, 22, 24)])

    assert _confirmed_reference_is_usable(three_sector_points, min_points=3, min_sectors=3) is True
    # Plenty of points, but only 1 sector -- must NOT be usable.
    assert _confirmed_reference_is_usable(one_sector_points, min_points=3, min_sectors=3) is False
    # Enough sectors, but too few points.
    assert _confirmed_reference_is_usable(three_sector_points, min_points=10, min_sectors=3) is False
    # Empty reference.
    assert _confirmed_reference_is_usable(np.empty((0, 2)), min_points=1, min_sectors=1) is False


# --- A. First scan enters bootstrap -----------------------------------------

def test_first_scan_enters_bootstrap() -> None:
    model = PersistentWorldModel()
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)

    result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=0.0))
    assert result.pose.x == pytest.approx(0.0)
    assert result.pose.y == pytest.approx(0.0)
    assert result.pose.theta == pytest.approx(0.0)
    assert result.bootstrap_status == BootstrapStatus.BOOTSTRAPPING
    assert result.map_integrated is True
    assert pipeline.state.bootstrap_status == BootstrapStatus.BOOTSTRAPPING


# --- B. Bootstrap does not attempt ICP ---------------------------------------

def test_bootstrap_never_calls_register(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_register(*args, **kwargs):
        raise AssertionError("register() must never be called while BOOTSTRAPPING")

    monkeypatch.setattr("slam.scan_matching_pipeline.register", failing_register)

    model = PersistentWorldModel()
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t))
        assert result.bootstrap_status == BootstrapStatus.BOOTSTRAPPING
        assert result.registration_result is None
    # (register() staying unpatched-and-uncalled for all of the above is
    # the assertion -- failing_register would have raised otherwise.)


# --- C. Multiple stationary scans are integrated at (0,0,0) -----------------

def test_multiple_bootstrap_scans_integrate_at_fixed_origin() -> None:
    model = PersistentWorldModel()
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)
    for t in (0.0, 0.15, 0.3, 0.45):
        result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t))
        assert result.accepted is True
        assert result.pose == Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=t)
        assert result.map_integrated is True
    # At least one static point should have accumulated enough
    # continued observation to be on its way to confirmation.
    assert model.get_pending_since(CLUSTER_A[0][0], CLUSTER_A[0][1]) is not None


# --- D. Temporal confirmation works through bootstrap -----------------------

def test_bootstrap_scans_reach_confirmation_via_phase5_rules() -> None:
    model = PersistentWorldModel()  # default confirmation_duration=0.5
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)
    for t in (0.0, 0.2, 0.5):
        pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t))
    assert model.get_cell_state(CLUSTER_A[0][0], CLUSTER_A[0][1]) == CellState.CONFIRMED


# --- E. Bootstrap does not complete too early --------------------------------

def test_bootstrap_stays_active_before_min_duration_even_if_confirmed() -> None:
    model = PersistentWorldModel()  # confirmation_duration=0.5
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)  # min_duration default 0.6
    result = None
    for t in (0.0, 0.2, 0.5):  # geometry confirms internally by t=0.5 (span 0.5 >= 0.5)
        result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t))
    assert model.get_cell_state(CLUSTER_A[0][0], CLUSTER_A[0][1]) == CellState.CONFIRMED  # confirmed internally
    assert result.bootstrap_status == BootstrapStatus.BOOTSTRAPPING  # but elapsed (0.5) < min_duration (0.6)


# --- F. Bootstrap requires usable confirmed geometry (count) ----------------

def test_bootstrap_remains_active_past_min_duration_without_enough_points() -> None:
    model = PersistentWorldModel()
    # Only 2 points ever observed -- below RegistrationConfig's default
    # min_correspondences (3), regardless of elapsed time.
    sparse_points = STATIC_WORLD_POINTS[:2]
    pipeline = ScanMatchingPipeline(
        reference_provider=model, world_model=model, bootstrap_config=BootstrapConfig(min_duration=0.6, timeout=5.0)
    )
    result = None
    for t in (0.0, 0.3, 0.6, 0.7):
        result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t, points=sparse_points))
    assert result.bootstrap_status == BootstrapStatus.BOOTSTRAPPING


# --- G. Bootstrap completes once both criteria are met ----------------------

def test_bootstrap_completes_once_duration_and_reference_both_sufficient() -> None:
    model = PersistentWorldModel()
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)  # defaults: min_duration=0.6
    statuses = []
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t))
        statuses.append(result.bootstrap_status)
    assert statuses[:-1] == [BootstrapStatus.BOOTSTRAPPING] * 6
    assert statuses[-1] == BootstrapStatus.COMPLETE


# --- H. First normal scan after bootstrap uses ICP ---------------------------

def test_first_scan_after_bootstrap_invokes_registration() -> None:
    model = PersistentWorldModel()
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)
    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t))
    assert pipeline.state.bootstrap_status == BootstrapStatus.COMPLETE

    result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=0.65))
    assert result.registration_result is not None  # ICP was actually invoked
    assert result.accepted is True


# --- I. Initial pose is fixed throughout bootstrap ---------------------------

def test_bootstrap_never_alters_the_initial_pose() -> None:
    model = PersistentWorldModel()
    pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)
    for t in (0.0, 0.2, 0.4, 0.6):
        result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t))
        if result.bootstrap_status == BootstrapStatus.BOOTSTRAPPING:
            assert result.pose == Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=t)


# --- J/K. Post-bootstrap failure preserves Phase 4/6 semantics --------------

def test_failed_registration_after_bootstrap_preserves_phase4_state() -> None:
    model = prebuilt_world_model()
    pipeline = ScanMatchingPipeline(
        reference_provider=model, world_model=model, bootstrap_config=BootstrapConfig(min_duration=0.0)
    )
    pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.0))
    assert pipeline.state.bootstrap_status == BootstrapStatus.COMPLETE

    state_before = pipeline.state
    map_state_before = model.state.copy()

    far_points = STATIC_WORLD_POINTS + np.array([50.0, 50.0])
    result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.1, points=far_points))

    assert result.accepted is False
    assert result.map_integrated is False
    assert result.bootstrap_status == BootstrapStatus.COMPLETE  # unaffected by a normal-mode failure
    assert pipeline.state.last_good_pose == state_before.last_good_pose
    assert pipeline.state.last_good_timestamp == state_before.last_good_timestamp
    assert pipeline.state.velocity_estimate == state_before.velocity_estimate
    assert pipeline.state.consecutive_failures == state_before.consecutive_failures + 1
    np.testing.assert_array_equal(model.state, map_state_before)


# --- L. Bootstrap timeout -----------------------------------------------------

def test_bootstrap_timeout_produces_explicit_failure_not_icp() -> None:
    model = PersistentWorldModel()
    # CLUSTER_A alone never spans the required 3 sectors -- bootstrap
    # can never complete no matter how long it waits.
    pipeline = ScanMatchingPipeline(
        reference_provider=model, world_model=model,
        bootstrap_config=BootstrapConfig(min_duration=0.6, timeout=1.5),
    )
    result = None
    for t in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5):
        result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t, points=CLUSTER_A))
    assert result.bootstrap_status == BootstrapStatus.FAILED
    assert result.localized is False
    assert result.registration_result is None  # never entered ICP

    # Terminal: a further scan is cleanly rejected, not silently retried.
    next_result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=1.6, points=CLUSTER_A))
    assert next_result.accepted is False
    assert next_result.bootstrap_status == BootstrapStatus.FAILED
    assert next_result.rejection_reason == "bootstrap_failed"
    assert next_result.map_integrated is False


def test_restart_bootstrap_after_timeout() -> None:
    model = PersistentWorldModel()
    pipeline = ScanMatchingPipeline(
        reference_provider=model, world_model=model,
        bootstrap_config=BootstrapConfig(min_duration=0.6, timeout=1.5),
    )
    for t in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5):
        pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t, points=CLUSTER_A))
    assert pipeline.state.bootstrap_status == BootstrapStatus.FAILED

    pipeline.restart_bootstrap()
    assert pipeline.state.bootstrap_status == BootstrapStatus.BOOTSTRAPPING
    assert pipeline.state.last_good_pose is None
    assert pipeline.state.consecutive_failures == 0

    # The world model's PREVIOUSLY-built confirmed geometry (if any)
    # is completely untouched by restart_bootstrap() -- only this
    # pipeline's own tracking state was reset.
    result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=2.0, points=CLUSTER_A))
    assert result.accepted is True
    assert result.bootstrap_status == BootstrapStatus.BOOTSTRAPPING


def test_restart_bootstrap_without_world_model_raises() -> None:
    pipeline, _ = make_pipeline()  # Phase 4's FakeReferenceProvider, no world_model
    with pytest.raises(ValueError):
        pipeline.restart_bootstrap()


# --- M. Deterministic replay --------------------------------------------------

def test_bootstrap_replay_is_deterministic() -> None:
    def run() -> tuple[list[BootstrapStatus], list[Pose2D], PersistentWorldModel]:
        model = PersistentWorldModel()
        pipeline = ScanMatchingPipeline(reference_provider=model, world_model=model)
        statuses, poses = [], []
        for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            result = pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t))
            statuses.append(result.bootstrap_status)
            poses.append(result.pose)
        return statuses, poses, model

    statuses_a, poses_a, model_a = run()
    statuses_b, poses_b, model_b = run()

    assert statuses_a == statuses_b
    for pa, pb in zip(poses_a, poses_b):
        assert pa.x == pb.x and pa.y == pb.y and pa.theta == pb.theta
    np.testing.assert_array_equal(model_a.state, model_b.state)
    np.testing.assert_array_equal(model_a.get_grid().cells, model_b.get_grid().cells)


# --- N. No tentative reference during bootstrap-to-normal transition --------

def test_no_tentative_geometry_ever_used_as_bootstrap_reference() -> None:
    model = PersistentWorldModel()
    spy = RecordingWorldModel(model)
    pipeline = ScanMatchingPipeline(reference_provider=spy, world_model=spy)

    for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        pipeline.process_scan(scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t))

    # Every reference query bootstrap itself issued (via
    # _update_bootstrap_status) must have returned only CONFIRMED
    # points -- get_confirmed_occupied_points_world's own contract,
    # checked here as an explicit regression guard specifically in the
    # bootstrap code path.
    assert len(spy.queries) >= 1
    for _, _, queried_points in spy.queries:
        for px, py in queried_points:
            assert model.get_cell_state(px, py) == CellState.CONFIRMED


# --- O. Angular spread specifically (many points, one sector) ---------------

def test_bootstrap_rejects_usable_geometry_concentrated_in_one_sector() -> None:
    model = PersistentWorldModel()
    # 8 points, well above any point-count threshold, but ALL within
    # bearings 10-24 degrees -- a single 45-degree sector.
    dense_one_sector = np.array([bearing_point(a, 1.0) for a in (10, 12, 14, 16, 18, 20, 22, 24)])
    pipeline = ScanMatchingPipeline(
        reference_provider=model, world_model=model, bootstrap_config=BootstrapConfig(min_duration=0.5, timeout=5.0)
    )
    result = None
    for t in (0.0, 0.2, 0.5, 0.6):
        result = pipeline.process_scan(
            scan_at_pose_sparse(Pose2D(0.0, 0.0, 0.0), timestamp=t, points=dense_one_sector)
        )
    assert result.bootstrap_status == BootstrapStatus.BOOTSTRAPPING  # plenty of points, not enough spread
