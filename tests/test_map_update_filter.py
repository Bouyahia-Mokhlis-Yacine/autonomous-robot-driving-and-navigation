"""Map-update motion filter (``PipelineConfig.min_integration_*``): a still
robot must not keep writing its own pose noise into the map it registers
against. Real-sensor regression: 30 s of a stationary RPLiDAR recorded on the
robot (2026-09-29), which drifted 39 cm in 60 s live before the filter."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from common.config import load_navigation_config, load_robot_config
from common.messages import LaserScan, Pose2D
from real_world_model_walkthrough import build_stack
from slam.scan_matching_pipeline import BootstrapStatus, PipelineConfig, ScanMatchingPipeline
from tests.test_scan_matching_pipeline import STATIC_WORLD_POINTS, FakeReferenceProvider, scan_at_pose

# Recorded BEFORE the driver learned the sensor is clockwise, so these scans are
# left/right mirrored. Irrelevant here: a mirrored static scene is self-consistent.
FIXTURE = Path(__file__).parent / "fixtures" / "real_scans_stationary_2026-09-29.npz"


class CountingWorldModel:
    def __init__(self) -> None:
        self.poses: list[Pose2D] = []

    def integrate(self, pose: Pose2D, scan: LaserScan, now: float) -> None:
        self.poses.append(pose)


def _bootstrapped(**config) -> tuple[ScanMatchingPipeline, CountingWorldModel, float]:
    world = CountingWorldModel()
    pipeline = ScanMatchingPipeline(FakeReferenceProvider(STATIC_WORLD_POINTS), PipelineConfig(**config),
                                    world_model=world)
    t = 0.0
    while pipeline.state.bootstrap_status == BootstrapStatus.BOOTSTRAPPING:
        pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), t))
        t += 0.1
    assert pipeline.state.bootstrap_status == BootstrapStatus.COMPLETE
    return pipeline, world, t


def test_default_integrates_every_accepted_scan() -> None:
    pipeline, world, t = _bootstrapped()
    before = len(world.poses)
    for i in range(5):
        assert pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), t + 0.1 * i)).map_integrated
    assert len(world.poses) == before + 5


def test_still_robot_is_not_integrated_until_it_moves_or_turns() -> None:
    pipeline, world, t = _bootstrapped(min_integration_translation=0.05,
                                       min_integration_rotation=math.radians(2.0))
    before = len(world.poses)
    for i in range(5):
        r = pipeline.process_scan(scan_at_pose(Pose2D(0.0, 0.0, 0.0), t))
        t += 0.1
        assert r.accepted and not r.map_integrated
    assert len(world.poses) == before

    moved = pipeline.process_scan(scan_at_pose(Pose2D(0.03, 0.0, 0.0), t)); t += 0.1
    assert moved.accepted and not moved.map_integrated        # 3 cm: below the threshold
    moved = pipeline.process_scan(scan_at_pose(Pose2D(0.06, 0.0, 0.0), t)); t += 0.1
    assert moved.accepted and moved.map_integrated            # 6 cm from the last integrated pose
    assert world.poses[-1].x == pytest.approx(0.06, abs=0.01)

    turned = pipeline.process_scan(scan_at_pose(Pose2D(0.06, 0.0, math.radians(3.0)), t))
    assert turned.accepted and turned.map_integrated          # 3 deg >= 2 deg


def test_bootstrap_scans_are_always_integrated() -> None:
    _, world, t = _bootstrapped(min_integration_translation=0.05, min_integration_rotation=math.radians(2.0))
    assert len(world.poses) == round(t / 0.1)


def test_negative_thresholds_rejected() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(min_integration_translation=-0.01)


def _real_scans() -> list[LaserScan]:
    d = np.load(FIXTURE)
    return [LaserScan(timestamp=float(t), angle_min=float(d["angle_min"]),
                      angle_increment=float(d["angle_increment"]), ranges=(r / 1000.0).tolist(),
                      range_min=float(d["range_min"]), range_max=float(d["range_max"]))
            for t, r in zip(d["timestamps"], d["ranges_mm"])]


def _max_pose_error(nav_cfg) -> tuple[float, float]:
    _, pipeline, _ = build_stack(nav_cfg, load_robot_config())
    scans = _real_scans()
    xy, th, accepted = 0.0, 0.0, 0
    for scan in scans:
        r = pipeline.process_scan(scan)
        accepted += r.accepted
        xy, th = max(xy, math.hypot(r.pose.x, r.pose.y)), max(th, abs(r.pose.theta))
    assert accepted >= 0.98 * len(scans)
    return xy, math.degrees(th)


@pytest.mark.parametrize("filtered", [True, False])
def test_real_stationary_lidar_does_not_drift_with_the_filter(filtered: bool) -> None:
    nav = load_navigation_config()
    nav = dataclasses.replace(nav, mapping=dataclasses.replace(nav.mapping, width=20.0, height=20.0))
    if not filtered:
        nav = dataclasses.replace(nav, localization=dataclasses.replace(
            nav.localization, map_update_min_translation_m=0.0, map_update_min_rotation_deg=0.0))
    xy, th = _max_pose_error(nav)
    if filtered:
        assert xy < 0.04 and th < 3.0, (xy, th)
    else:
        # The recording really reproduces the bug (guards the fixture).
        assert xy > 0.10 or th > 5.0, (xy, th)


# ------------------------------------------------------------ startup mapping --

def _startup(duration_s: float) -> tuple[float, int, list]:
    """Real recording through build_stack with the given bootstrap duration:
    (time bootstrap completed, confirmed cells then, per-scan results)."""
    nav = load_navigation_config()
    nav = dataclasses.replace(nav, mapping=dataclasses.replace(nav.mapping, width=20.0, height=20.0),
                              localization=dataclasses.replace(nav.localization, bootstrap_duration_s=duration_s))
    world, pipeline, _ = build_stack(nav, load_robot_config())
    done_at, confirmed, results = None, 0, []
    for scan in _real_scans():
        r = pipeline.process_scan(scan)
        results.append((scan.timestamp, r))
        if done_at is None and r.bootstrap_status == BootstrapStatus.COMPLETE:
            done_at = scan.timestamp
            confirmed = len(world.get_confirmed_occupied_points_world((0.0, 0.0), 20.0))
    return done_at, confirmed, results


def test_startup_mapping_uses_the_configured_duration_then_the_motion_filter() -> None:
    assert load_navigation_config().localization.bootstrap_duration_s == 2.0
    done_at, _, results = _startup(2.0)
    assert 2.0 <= done_at < 2.3                         # one scan period past the minimum
    during = [r for t, r in results if t < done_at]
    assert during and all(r.accepted and r.map_integrated for r in during)
    after = [r for t, r in results if t > done_at + 1.0]
    # Stationary robot after startup: the drift fix keeps the map frozen.
    assert sum(r.map_integrated for r in after) <= 0.02 * len(after)


def test_longer_startup_populates_the_map_better_than_the_old_default() -> None:
    _, old, _ = _startup(0.6)
    _, new, _ = _startup(2.0)
    assert new >= 1.3 * old, (old, new)                # measured: 81% vs 50% of near reference cells
