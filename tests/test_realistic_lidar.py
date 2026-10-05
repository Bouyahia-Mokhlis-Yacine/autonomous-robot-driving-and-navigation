"""Tests for ``simulation.realistic_lidar`` (the RPLiDAR A2M12 sensor model)
and ``simulation.pose_history``: interface compatibility, timing, noise,
missing returns, partial scans, motion distortion, and determinism."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from common.config import load_navigation_config
from common.messages import LaserScan, Pose2D
from simulation.environment import Environment
from simulation.lidar_statistics import scan_statistics
from simulation.obstacles import RectangleObstacle
from simulation.pose_history import PoseHistory
from simulation.realistic_lidar import RealisticLidar, RealisticLidarConfig
from simulation.simulated_lidar import SimulatedLidar

NAV_CFG = load_navigation_config()
LIDAR_CFG = NAV_CFG.lidar

# Every imperfection off: the sample model reduced to a clean, steady sensor.
CLEAN = RealisticLidarConfig(
    rotation_jitter_std_s=0.0, delivery_delay_mean_s=0.0, sample_dropout_probability=0.0,
    range_noise_std_m=0.0, outlier_probability=0.0, range_quantization_m=0.0,
    partial_scan_probability=0.0, first_scan_partial=False,
)


def room(pose: Pose2D | None = None) -> Environment:
    env = Environment(
        width=6.0, height=5.0, robot_length=0.3, robot_width=0.25,
        robot_pose=pose or Pose2D(x=2.0, y=2.5, theta=0.0),
    )
    env.add_static_obstacle(RectangleObstacle(x=4.5, y=3.5, width=0.6, height=0.6))
    return env


def stationary_records(config: RealisticLidarConfig, revolutions: int, seed: int = 0, env: Environment | None = None):
    env = env or room()
    pose = env.robot_pose
    lidar = RealisticLidar(LIDAR_CFG, config, seed=seed)
    return lidar.acquire_until(env, lambda t: pose, revolutions / config.scan_rate_hz + 1e-9)


# --------------------------------------------------------------- interface ---

def test_realistic_scan_is_the_same_laserscan_contract_as_the_ideal_simulator() -> None:
    env = room()
    ideal = SimulatedLidar(LIDAR_CFG).scan(env, env.robot_pose, timestamp=0.1)
    real = next(r.scan for r in stationary_records(RealisticLidarConfig(), 5) if r.scan is not None)
    assert isinstance(real, LaserScan)
    # The realistic model bins exactly like the real driver: 361 bins (the
    # real recording has 361 too). NOTE: the pre-existing ideal simulator
    # yields 360 (floor() on a slightly truncated pi in navigation.yaml) --
    # a pre-existing ideal-vs-real mismatch, deliberately left unchanged.
    assert real.num_rays == 361
    assert ideal.num_rays == 360
    assert real.angle_min == ideal.angle_min
    assert real.angle_increment == ideal.angle_increment
    assert (real.range_min, real.range_max) == (ideal.range_min, ideal.range_max)
    # Same "no detection" convention: empty bins are exactly range_max, never NaN/inf/0.
    ranges = np.asarray(real.ranges)
    assert np.all(np.isfinite(ranges))
    assert np.all((ranges == real.range_max) | ((ranges >= real.range_min) & (ranges < real.range_max)))


def test_ideal_scan_leaves_timing_metadata_unset_and_realistic_sets_it() -> None:
    env = room()
    ideal = SimulatedLidar(LIDAR_CFG).scan(env, env.robot_pose, timestamp=0.1)
    assert ideal.scan_time == 0.0 and ideal.time_increment == 0.0
    rec = stationary_records(CLEAN, 1)[0]
    assert rec.scan.scan_time == pytest.approx(1.0 / CLEAN.scan_rate_hz)
    assert rec.scan.time_increment == pytest.approx(rec.scan.scan_time * LIDAR_CFG.angular_resolution / (2 * math.pi))


def test_laserscan_timing_fields_are_optional_and_validated() -> None:
    scan = LaserScan(timestamp=0.0, angle_min=0.0, angle_increment=0.1, ranges=[1.0, 2.0], range_min=0.1, range_max=5.0)
    assert scan.scan_time == 0.0 and scan.time_increment == 0.0
    with pytest.raises(ValueError):
        LaserScan(timestamp=0.0, angle_min=0.0, angle_increment=0.1, ranges=[1.0], range_min=0.1, range_max=5.0,
                  scan_time=-0.1)


def test_binning_matches_the_real_driver_exactly() -> None:
    """Feed the real driver's binning the same raw measurements the model
    produced (converted back to the rplidar library's (quality, deg, mm)
    tuples) and require identical ranges -- same representation of
    missing returns, same nearest-hit rule."""
    driver_mod = pytest.importorskip("hardware.rplidar_driver")
    from common.config import RPLidarHardwareConfig

    driver = driver_mod.RPLidarDriver(
        RPLidarHardwareConfig(port="/dev/null", baudrate=256000, timeout=1.0, min_measurements_per_scan=5,
                              angles_clockwise=True),
        LIDAR_CFG,
    )
    for rec in stationary_records(RealisticLidarConfig(), 6, seed=3):
        if rec.scan is None:
            continue
        ok = np.isfinite(rec.measured_ranges)
        raw = [
            (15, -math.degrees(a) % 360.0, r * 1000.0)   # as the sensor reports it: clockwise
            for a, r in zip(rec.sample_angles[ok], rec.measured_ranges[ok])
        ]
        expected = driver._bin_scan(raw, timestamp=rec.scan.timestamp)
        np.testing.assert_allclose(rec.scan.ranges, expected.ranges, rtol=0, atol=1e-9)


def test_clean_model_matches_ideal_simulator_ranges() -> None:
    """With every imperfection off and the robot stationary, the sample
    model sees the same geometry as the ideal ray-caster (up to the
    sub-degree difference between a sample's angle and its bin centre)."""
    env = room()
    ideal = np.asarray(SimulatedLidar(LIDAR_CFG).scan(env, env.robot_pose, timestamp=0.0).ranges)
    # Bin i has the same angle in both (angle_min + i * increment); the ideal
    # scan simply lacks the final (+pi, duplicate-direction) bin.
    scan = np.asarray(stationary_records(CLEAN, 1, env=env)[0].scan.ranges)[: ideal.size]
    filled = scan < LIDAR_CFG.range_max
    assert filled.mean() > 0.95
    assert np.median(np.abs(scan[filled] - ideal[filled])) < 0.02


# ------------------------------------------------------------------ timing ---

def test_scan_rate_duration_and_jitter() -> None:
    records = stationary_records(RealisticLidarConfig(partial_scan_probability=0.0), 120, seed=1)
    scans = [r.scan for r in records if r.scan is not None]
    stats = scan_statistics(scans)
    assert stats["scan_frequency_hz"] == pytest.approx(11.0, abs=0.4)
    assert stats["scan_interval_s"]["mean"] == pytest.approx(0.090, abs=0.004)
    assert stats["scan_interval_s"]["std"] > 0.005  # jitter exists
    assert 0.03 < stats["scan_interval_s"]["min"] < stats["scan_interval_s"]["max"] < 0.2
    durations = np.array([r.duration for r in records])
    assert durations.mean() == pytest.approx(0.0905, abs=0.002)
    for r in records:
        assert r.end_time <= r.timestamp <= r.end_time + RealisticLidarConfig().delivery_delay_max_s
        assert np.all((r.sample_times >= r.start_time) & (r.sample_times <= r.end_time))
    assert all(b.timestamp > a.timestamp for a, b in zip(records, records[1:]))


def test_without_jitter_timing_is_exactly_periodic() -> None:
    records = stationary_records(CLEAN, 20)
    intervals = np.diff([r.timestamp for r in records])
    np.testing.assert_allclose(intervals, 1.0 / CLEAN.scan_rate_hz, atol=1e-9)


def test_result_does_not_depend_on_how_finely_time_is_sliced() -> None:
    env = room()
    pose = env.robot_pose
    coarse = RealisticLidar(LIDAR_CFG, RealisticLidarConfig(), seed=5).acquire_until(env, lambda t: pose, 2.0)
    fine_lidar = RealisticLidar(LIDAR_CFG, RealisticLidarConfig(), seed=5)
    fine = []
    for k in range(1, 201):
        fine += fine_lidar.acquire_until(env, lambda t: pose, k * 0.01)
    assert [r.timestamp for r in coarse] == [r.timestamp for r in fine]
    assert all(a.scan.ranges == b.scan.ranges for a, b in zip(coarse, fine) if a.scan is not None)


# ------------------------------------------------------------------- noise ---

def test_valid_measurements_are_slightly_noisy() -> None:
    records = stationary_records(RealisticLidarConfig(partial_scan_probability=0.0, first_scan_partial=False), 20)
    true = np.concatenate([r.true_ranges for r in records])
    meas = np.concatenate([r.measured_ranges for r in records])
    ok = np.isfinite(meas)
    err = meas[ok] - true[ok]
    assert np.mean(err != 0.0) > 0.5
    assert np.median(np.abs(err)) < 0.003     # millimetre-level core
    assert np.max(np.abs(err)) > 0.02          # but a heavier tail exists
    assert abs(err.mean()) < 0.001             # unbiased


def test_noise_never_corrupts_missing_readings_or_leaves_the_valid_range() -> None:
    cfg = RealisticLidarConfig(outlier_probability=0.5, outlier_std_m=5.0)  # absurdly noisy on purpose
    for rec in stationary_records(cfg, 15, seed=2):
        if rec.scan is None:
            continue
        # Samples with no obstacle, dropped, or skipped (partial) never get a value.
        assert np.all(np.isnan(rec.measured_ranges[~np.isfinite(rec.true_ranges)]))
        delivered = rec.measured_ranges[np.isfinite(rec.measured_ranges)]
        assert np.all((delivered >= LIDAR_CFG.range_min) & (delivered <= LIDAR_CFG.range_max))
        ranges = np.asarray(rec.scan.ranges)
        assert np.all((ranges == LIDAR_CFG.range_max) | (ranges >= LIDAR_CFG.range_min))
    assert any(rec.out_of_range_after_noise > 0 for rec in stationary_records(cfg, 15, seed=2))


def test_noise_can_be_switched_off() -> None:
    rec = stationary_records(CLEAN, 1)[0]
    ok = np.isfinite(rec.measured_ranges)
    np.testing.assert_array_equal(rec.measured_ranges[ok], rec.true_ranges[ok])


# ---------------------------------------------------------- missing returns --

def test_missing_returns_occur_at_the_configured_rate() -> None:
    for p in (0.0, 0.145, 0.4):
        cfg = dataclasses.replace(CLEAN, sample_dropout_probability=p)
        records = stationary_records(cfg, 30, seed=4)
        hits = sum(int(np.count_nonzero(np.isfinite(r.true_ranges))) for r in records)
        dropped = sum(r.dropped_returns for r in records)
        assert dropped / hits == pytest.approx(p, abs=0.02)


def test_no_obstacle_directions_are_not_counted_as_sensor_failures() -> None:
    """Short-range sensor in a big room: most directions see nothing. Those
    are 'no detectable obstacle', not dropouts -- dropouts apply only to
    directions that really do contain something."""
    short = dataclasses.replace(LIDAR_CFG, range_max=1.0)
    env = Environment(width=20.0, height=20.0, robot_length=0.3, robot_width=0.25, robot_pose=Pose2D(10.0, 10.0, 0.0))
    env.add_static_obstacle(RectangleObstacle(x=10.6, y=10.0, width=0.2, height=0.6))  # the only thing in range
    lidar = RealisticLidar(short, dataclasses.replace(CLEAN, sample_dropout_probability=1.0), seed=0)
    rec = lidar.acquire_until(env, lambda t: env.robot_pose, 0.1)[0]
    hits = int(np.count_nonzero(np.isfinite(rec.true_ranges)))
    assert 0 < hits < 0.2 * rec.true_ranges.size
    assert rec.dropped_returns == hits         # every real hit dropped...
    assert rec.scan is None                    # ...so nothing is left to deliver


def test_dropout_rate_in_typical_bins_matches_the_recording() -> None:
    records = stationary_records(RealisticLidarConfig(), 60, seed=7)
    stats = scan_statistics([r.scan for r in records if r.scan is not None])
    assert stats["dropout_rate_in_typical_bins"] == pytest.approx(0.146, abs=0.03)


# ------------------------------------------------------------ partial scans --

def test_partial_scans_are_a_tail_arc_with_the_rest_reported_as_no_detection() -> None:
    cfg = dataclasses.replace(CLEAN, partial_scan_probability=1.0, partial_keep_fraction_min=0.2,
                              partial_keep_fraction_max=0.2)
    rec = stationary_records(cfg, 1)[0]
    assert rec.is_partial and rec.scan is not None
    delivered_times = rec.sample_times[np.isfinite(rec.measured_ranges)]
    assert delivered_times.min() >= rec.end_time - 0.2 * rec.duration - 1e-9
    valid = np.asarray(rec.scan.ranges) < LIDAR_CFG.range_max
    assert 0.15 * 361 < valid.sum() < 0.25 * 361
    assert rec.skipped_for_partial > 0


def test_partial_scans_occur_at_roughly_the_configured_probability() -> None:
    records = stationary_records(RealisticLidarConfig(), 300, seed=8)
    assert records[0].is_partial  # startup revolution
    rate = np.mean([r.is_partial for r in records[1:]])
    assert rate == pytest.approx(0.08, abs=0.04)
    stats = scan_statistics([r.scan for r in records if r.scan is not None])
    assert stats["partial_scans"] == sum(r.is_partial for r in records if r.scan is not None)


def test_too_short_revolutions_are_swallowed_like_the_rplidar_library_does() -> None:
    cfg = dataclasses.replace(CLEAN, partial_scan_probability=1.0, partial_keep_fraction_min=0.005,
                              partial_keep_fraction_max=0.005)
    rec = stationary_records(cfg, 1)[0]
    assert rec.num_delivered_samples <= cfg.min_measurements_per_scan
    assert rec.scan is None


def test_slam_pipeline_and_world_model_accept_partial_scans() -> None:
    from mapping.persistent_world_model import PersistentWorldModel
    from slam.scan_matching_pipeline import ScanMatchingPipeline

    wm = PersistentWorldModel(resolution=0.05, width_m=8.0, height_m=8.0, origin_x=-4.0, origin_y=-4.0)
    pipeline = ScanMatchingPipeline(reference_provider=wm, world_model=wm)
    records = stationary_records(RealisticLidarConfig(partial_scan_probability=0.3), 40, seed=9)
    partial_seen = 0
    for rec in records:
        if rec.scan is None:
            continue
        result = pipeline.process_scan(rec.scan)  # must not raise
        partial_seen += int(rec.is_partial)
        assert result.pose is not None
    assert partial_seen >= 3
    assert pipeline.state.localized


# ------------------------------------------------------- motion distortion ---

def _moving_history(v: float, omega: float, start: Pose2D, duration: float = 1.0, dt: float = 0.005) -> PoseHistory:
    history = PoseHistory()
    pose = start
    t = 0.0
    history.append(t, pose)
    while t < duration:
        t += dt
        pose = Pose2D(
            x=pose.x + v * math.cos(pose.theta) * dt,
            y=pose.y + v * math.sin(pose.theta) * dt,
            theta=pose.theta + omega * dt,
        )
        history.append(t, pose)
    return history


def test_forward_motion_samples_come_from_different_poses() -> None:
    env = room()
    history = _moving_history(v=1.0, omega=0.0, start=env.robot_pose)
    rec = RealisticLidar(LIDAR_CFG, CLEAN, seed=0).acquire_until(env, history.pose_at, 0.1)[0]
    xs = rec.sample_poses[:, 0]
    # Poses advance monotonically across the revolution by ~v * period.
    assert np.all(np.diff(xs) >= 0.0)
    assert xs[-1] - xs[0] == pytest.approx(1.0 * rec.duration, rel=0.05)
    # And each sample's ideal range is the ray-cast from ITS OWN pose.
    caster = SimulatedLidar(LIDAR_CFG)
    for k in (0, len(xs) // 2, len(xs) - 1):
        x, y, th = rec.sample_poses[k]
        a = th + rec.sample_angles[k]
        expected = caster._cast_ray(env, (x, y), (math.cos(a), math.sin(a)))
        if expected < LIDAR_CFG.range_max:
            assert rec.true_ranges[k] == pytest.approx(expected)


def test_rotation_samples_come_from_different_headings() -> None:
    env = room()
    history = _moving_history(v=0.0, omega=2.0, start=env.robot_pose)
    rec = RealisticLidar(LIDAR_CFG, CLEAN, seed=0).acquire_until(env, history.pose_at, 0.1)[0]
    thetas = np.unwrap(rec.sample_poses[:, 2])
    assert thetas[-1] - thetas[0] == pytest.approx(2.0 * rec.duration, rel=0.05)


def test_distortion_disabled_uses_one_pose_per_scan() -> None:
    env = room()
    history = _moving_history(v=1.0, omega=1.0, start=env.robot_pose)
    cfg = dataclasses.replace(CLEAN, enable_motion_distortion=False)
    rec = RealisticLidar(LIDAR_CFG, cfg, seed=0).acquire_until(env, history.pose_at, 0.1)[0]
    assert np.ptp(rec.sample_poses, axis=0).max() == 0.0
    end = history.pose_at(rec.end_time)
    np.testing.assert_allclose(rec.sample_poses[0], (end.x, end.y, end.theta))


def test_motion_distortion_changes_the_measured_geometry() -> None:
    """Moving toward a wall: with distortion, samples late in the sweep see
    the wall closer than a single end-of-scan pose would imply."""
    env = room()
    history = _moving_history(v=1.0, omega=0.0, start=env.robot_pose)
    distorted = RealisticLidar(LIDAR_CFG, CLEAN, seed=0).acquire_until(env, history.pose_at, 0.1)[0]
    single = RealisticLidar(LIDAR_CFG, dataclasses.replace(CLEAN, enable_motion_distortion=False), seed=0) \
        .acquire_until(env, history.pose_at, 0.1)[0]
    diff = np.abs(np.asarray(distorted.scan.ranges) - np.asarray(single.scan.ranges))
    assert diff.max() > 0.03  # centimetres of skew at 1 m/s


# ------------------------------------------------------------ determinism ---

def test_same_seed_same_scans_different_seed_different_scans() -> None:
    a = stationary_records(RealisticLidarConfig(), 20, seed=42)
    b = stationary_records(RealisticLidarConfig(), 20, seed=42)
    c = stationary_records(RealisticLidarConfig(), 20, seed=43)
    assert [r.timestamp for r in a] == [r.timestamp for r in b]
    assert all(
        (x.scan is None and y.scan is None) or x.scan.ranges == y.scan.ranges for x, y in zip(a, b)
    )
    assert [r.timestamp for r in a] != [r.timestamp for r in c]


# ------------------------------------------------------------ pose history ---

def test_pose_history_interpolates_and_clamps() -> None:
    h = PoseHistory()
    h.append(0.0, Pose2D(0.0, 0.0, 0.0))
    h.append(1.0, Pose2D(2.0, 4.0, 1.0))
    p = h.pose_at(0.25)
    assert (p.x, p.y, p.theta) == pytest.approx((0.5, 1.0, 0.25))
    assert h.pose_at(-1.0).x == 0.0 and h.pose_at(5.0).x == 2.0
    with pytest.raises(ValueError):
        h.append(0.5, Pose2D(0.0, 0.0, 0.0))


def test_pose_history_interpolates_heading_across_the_pi_seam() -> None:
    h = PoseHistory()
    h.append(0.0, Pose2D(0.0, 0.0, math.pi - 0.1))
    h.append(1.0, Pose2D(0.0, 0.0, -math.pi + 0.1))
    assert abs(abs(h.pose_at(0.5).theta) - math.pi) < 1e-9


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        RealisticLidarConfig(sample_dropout_probability=1.5)
    with pytest.raises(ValueError):
        RealisticLidarConfig(partial_keep_fraction_min=0.5, partial_keep_fraction_max=0.2)
    with pytest.raises(ValueError):
        RealisticLidarConfig(delivery_delay_max_s=0.2)  # longer than a rotation would reorder scans
    with pytest.raises(ValueError):
        RealisticLidar(dataclasses.replace(LIDAR_CFG, range_max=0.0))
