"""Localization robustness on REAL RPLiDAR data (the stationary recording of
2026-09-29, tests/fixtures/real_scans_stationary_2026-09-29.npz), through the
production stack (``build_stack``) and the localization health monitor.
Findings of the pre-motor validation, 2026-09-29."""

from __future__ import annotations

import dataclasses
import math

import pytest

from common.config import load_navigation_config, load_robot_config
from common.geometry import wrap_angle
from common.messages import LaserScan, VelocityCommand
from real_world_model_walkthrough import build_stack
from slam.motion_model import CommandHistory
from tests.test_map_update_filter import _real_scans

NAV = load_navigation_config()
NAV = dataclasses.replace(NAV, mapping=dataclasses.replace(NAV.mapping, width=20.0, height=20.0))
ROBOT = load_robot_config()


def _run(scans: list[LaserScan], motion_prior=None):
    _, pipeline, health = build_stack(NAV, ROBOT, motion_prior=motion_prior)
    out = []
    for scan in scans:
        r = pipeline.process_scan(scan)
        health.on_result(r, scan.timestamp)
        out.append((scan.timestamp, r, health.update(scan.timestamp).value))
    return out


def _rotated(scan: LaserScan, heading: float) -> LaserScan:
    """The same scene seen by the sensor after the robot turned in place by
    ``heading`` (whole bins; no in-sweep distortion)."""
    n, k = len(scan.ranges), int(round(heading / scan.angle_increment))
    return dataclasses.replace(scan, ranges=[scan.ranges[(i + k) % n] for i in range(n)])


def test_blackout_goes_lost_then_recovers_in_place() -> None:
    scans = [dataclasses.replace(s, ranges=[s.range_max] * len(s.ranges)) if 15.0 <= s.timestamp < 17.0 else s
             for s in _real_scans()]
    out = _run(scans)
    during = [h for t, _, h in out if 16.5 <= t < 17.0]
    assert during and all(h == "LOST" for h in during)          # lost_after_s = 1.2 s
    t_end, r, h = out[-1]
    assert h == "LOCALIZED"
    assert math.hypot(r.pose.x, r.pose.y) < 0.05 and abs(r.pose.theta) < math.radians(3.0)


def test_invalid_ranges_are_ignored() -> None:
    bad = [float("nan"), float("inf"), 0.0, -1.0]
    scans = [dataclasses.replace(s, ranges=[bad[i % 4] if i % 6 == 0 else r for i, r in enumerate(s.ranges)])
             if s.timestamp >= 10.0 else s for s in _real_scans()]
    out = _run(scans)
    assert all(h == "LOCALIZED" for t, _, h in out if t > 3.0)
    _, r, _ = out[-1]
    assert math.hypot(r.pose.x, r.pose.y) < 0.05


def test_full_turn_and_back_at_1_rad_s_with_commanded_motion() -> None:
    """Real scans turned in place (whole bins) at 1 rad/s, 360 deg and back,
    seeded by the (here perfect) commanded motion, as with motors.

    REGRESSION GUARD, NOT AN ACCURACY TARGET: measured 2026-09-29, the
    estimate lags through the turn and ends 7.8 deg off (max 9.5 deg) while
    health stays LOCALIZED -- a known limitation of point-to-point ICP in this
    sparse room. The bounds sit
    just above the measurement so that any worsening fails this test."""
    scans = _real_scans()
    omega, still = 1.0, 3.0
    turn = 2.0 * math.pi / omega

    def heading(t: float) -> float:
        if t < still:
            return 0.0
        if t < still + turn:
            return omega * (t - still)
        if t < still + 2.0 * turn:
            return 2.0 * math.pi - omega * (t - still - turn)
        return 0.0

    history = CommandHistory(ROBOT.limits.max_linear_velocity, ROBOT.limits.max_angular_velocity)
    history.record(0.0, VelocityCommand(0.0, 0.0))
    history.record(still, VelocityCommand(0.0, omega))
    history.record(still + turn, VelocityCommand(0.0, -omega))
    history.record(still + 2.0 * turn, VelocityCommand(0.0, 0.0))
    t0 = scans[0].timestamp
    moved = [_rotated(s, heading(s.timestamp - t0)) for s in scans if s.timestamp - t0 < still + 2.0 * turn + 2.0]
    out = _run(moved, motion_prior=history)
    errors = [abs(wrap_angle(r.pose.theta - heading(t - t0))) for t, r, _ in out]
    assert all(h in ("LOCALIZED", "INITIALIZING") for _, _, h in out)
    assert max(errors) < math.radians(11.0), math.degrees(max(errors))
    assert errors[-1] < math.radians(9.0), math.degrees(errors[-1])


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN LIMITATION (2026-09-29): a robot lifted and turned 90 deg while the runtime runs is NOT "
    "detected. After a few 'implausible_motion' rejections the time-aware gate (which widens with the "
    "time since the last accepted scan) admits a wrong ICP solution (-31 deg vs +90 deg) and health "
    "stays LOCALIZED. There is no global relocalization. Operational rule until fixed: never move the "
    "robot by hand while the runtime is running; restart it instead."))
def test_robot_lifted_and_turned_90_deg_is_not_silently_mislocalized() -> None:
    scans = [_rotated(s, math.pi / 2) if s.timestamp >= 20.0 else s for s in _real_scans()]
    out = _run(scans)
    _, r, h = out[-1]
    correct = abs(wrap_angle(r.pose.theta - math.pi / 2)) < math.radians(5.0)
    assert correct or h == "LOST"
