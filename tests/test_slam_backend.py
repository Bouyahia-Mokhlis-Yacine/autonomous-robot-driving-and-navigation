"""Tests for slam.backend: GroundTruthBackend, OdometryBackend, and
LidarScanMatchingBackend (the real, no-encoder localization backend)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from common.config import LidarConfig, SlamConfig, load_navigation_config
from common.messages import LaserScan, Pose2D, VelocityCommand
from slam.backend import GroundTruthBackend, LidarScanMatchingBackend, OdometryBackend

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def make_scan() -> LaserScan:
    return LaserScan(
        timestamp=0.0, angle_min=0.0, angle_increment=1.0, ranges=[1.0], range_min=0.1, range_max=5.0
    )


def test_ground_truth_backend_returns_provider_pose_and_ignores_scan() -> None:
    poses = [Pose2D(x=1.0, y=2.0, theta=0.0), Pose2D(x=3.0, y=4.0, theta=1.0)]
    call_count = {"n": 0}

    def provider() -> Pose2D:
        pose = poses[call_count["n"]]
        call_count["n"] += 1
        return pose

    backend = GroundTruthBackend(pose_provider=provider)

    first = backend.estimate_pose(make_scan())
    second = backend.estimate_pose(make_scan())

    assert first == poses[0]
    assert second == poses[1]


def test_odometry_backend_integrates_across_calls() -> None:
    backend = OdometryBackend(initial_pose=Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0))

    backend.integrate(VelocityCommand(linear=1.0, angular=0.0), dt=1.0)
    backend.integrate(VelocityCommand(linear=1.0, angular=0.0), dt=1.0)

    pose = backend.estimate_pose(make_scan())
    assert pose.x == pytest.approx(2.0)
    assert pose.y == pytest.approx(0.0)


def test_odometry_backend_estimate_pose_ignores_scan_content() -> None:
    backend = OdometryBackend(initial_pose=Pose2D(x=5.0, y=5.0, theta=0.0, timestamp=0.0))
    pose_a = backend.estimate_pose(make_scan())
    pose_b = backend.estimate_pose(
        LaserScan(timestamp=99.0, angle_min=-1.0, angle_increment=0.5, ranges=[9.0, 9.0], range_min=0.1, range_max=20.0)
    )
    assert pose_a == pose_b


def test_odometry_backend_rotation_then_translation() -> None:
    backend = OdometryBackend(initial_pose=Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0))
    backend.integrate(VelocityCommand(linear=0.0, angular=math.pi / 2.0), dt=1.0)
    backend.integrate(VelocityCommand(linear=1.0, angular=0.0), dt=1.0)

    pose = backend.estimate_pose(make_scan())
    assert pose.x == pytest.approx(0.0, abs=1e-9)
    assert pose.y == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# LidarScanMatchingBackend: the real backend for this robot (no wheel
# encoders -- see the Phase 13 plan). Exercised against a sequence of
# scans recorded from the actual connected RPLiDAR, so these tests are
# deterministic and don't require hardware to be attached to run.
# ------------------------------------------------------------------ #


def make_lidar_scan_matching_backend(**overrides) -> LidarScanMatchingBackend:
    lidar_config = LidarConfig(
        range_min=0.15, range_max=8.0, angle_min=-math.pi, angle_max=math.pi,
        angular_resolution=math.radians(1.0), scan_rate=10.0,
    )
    slam_config = SlamConfig(
        map_size_pixels=200, map_size_meters=10.0, hole_width_mm=600.0,
        sigma_xy_mm=100.0, sigma_theta_degrees=20.0, max_search_iter=1000,
    )
    lidar_config = overrides.pop("lidar_config", lidar_config)
    slam_config = overrides.pop("slam_config", slam_config)
    return LidarScanMatchingBackend(lidar_config, slam_config, **overrides)


def load_real_scan_fixture() -> list[LaserScan]:
    path = _FIXTURES_DIR / "real_scans_stationary.json"
    with path.open("r", encoding="utf-8") as f:
        raw_scans = json.load(f)
    return [LaserScan(**raw) for raw in raw_scans]


def test_constructs_from_real_navigation_config() -> None:
    # Exercises the exact config path a real run would use.
    nav_config = load_navigation_config()
    backend = LidarScanMatchingBackend(nav_config.lidar, nav_config.slam)
    assert backend is not None


def test_sparse_scan_keeps_previous_pose_unchanged() -> None:
    backend = make_lidar_scan_matching_backend(min_valid_points=50)
    sparse_scan = LaserScan(
        timestamp=0.0, angle_min=-math.pi, angle_increment=math.radians(1.0),
        ranges=[8.0] * 361, range_min=0.15, range_max=8.0,  # every ray at range_max = zero valid hits
    )
    first = backend.estimate_pose(sparse_scan)
    second = backend.estimate_pose(sparse_scan)
    assert first == second == Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0)


def test_estimate_pose_returns_valid_pose2d_for_synthetic_room_scan() -> None:
    # A synthetic "square room" scan: every ray hits a wall at a plausible range.
    backend = make_lidar_scan_matching_backend()
    ranges = [3.0 + 0.5 * math.sin(i * 0.1) for i in range(361)]  # varied, matchable geometry
    scan = LaserScan(
        timestamp=1.0, angle_min=-math.pi, angle_increment=math.radians(1.0),
        ranges=ranges, range_min=0.15, range_max=8.0,
    )
    pose = backend.estimate_pose(scan)
    assert isinstance(pose, Pose2D)
    assert math.isfinite(pose.x) and math.isfinite(pose.y) and math.isfinite(pose.theta)


@pytest.mark.skipif(
    not (_FIXTURES_DIR / "real_scans_stationary.json").exists(),
    reason="real hardware scan fixture not present",
)
def test_real_recorded_scans_stationary_pose_stays_bounded() -> None:
    """Sanity check against real, recorded RPLiDAR data: the sensor was
    physically stationary for this whole recording, so the estimated
    pose should stay roughly put -- not exact (correlative scan matching
    has real noise), but nowhere near a catastrophic divergence."""
    scans = load_real_scan_fixture()
    nav_config = load_navigation_config()
    backend = LidarScanMatchingBackend(nav_config.lidar, nav_config.slam)

    poses = [backend.estimate_pose(scan) for scan in scans]

    for pose in poses:
        assert isinstance(pose, Pose2D)
        assert math.isfinite(pose.x) and math.isfinite(pose.y) and math.isfinite(pose.theta)

    # Generous bound: a stationary real sensor should not "walk" more than
    # ~2 meters or a full turn across a ~25-scan recording under
    # unmuned RMHC search noise (sigma_xy_mm=100, sigma_theta_degrees=20).
    xs = [p.x for p in poses]
    ys = [p.y for p in poses]
    assert max(xs) - min(xs) < 2.0
    assert max(ys) - min(ys) < 2.0


@pytest.mark.skipif(
    not (_FIXTURES_DIR / "real_scans_stationary.json").exists(),
    reason="real hardware scan fixture not present",
)
def test_real_recorded_scans_do_not_crash_the_backend() -> None:
    # The recorded fixture includes scans captured right after the
    # driver's own serial-buffer-overflow recovery (see
    # hardware/rplidar_driver.py) -- real, slightly-imperfect data,
    # which is exactly what this test guards against crashing on.
    scans = load_real_scan_fixture()
    nav_config = load_navigation_config()
    backend = LidarScanMatchingBackend(nav_config.lidar, nav_config.slam)
    for scan in scans:
        backend.estimate_pose(scan)  # must not raise
