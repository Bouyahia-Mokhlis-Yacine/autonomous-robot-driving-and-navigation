"""Time-aware motion validation, commanded-motion prediction
(``slam.motion_model``) and delay compensation (``simulation.closed_loop``)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from common.config import load_navigation_config, load_robot_config
from common.messages import Pose2D, VelocityCommand
from simulation.closed_loop import ClosedLoopSimulation
from simulation.scenario import default_scenario
from simulation.sensor_simulation_config import load_sensor_simulation_config
from slam.motion_model import CommandHistory
from slam.scan_matching_pipeline import PipelineConfig, ScanMatchingPipeline, _clamp_velocity

V = VelocityCommand
ROBOT_CFG = load_robot_config()
NAV_CFG = load_navigation_config()


# ------------------------------------------------------------ motion model --

def test_propagate_straight_line() -> None:
    h = CommandHistory(0.5, 1.5)
    h.record(0.0, V(0.4, 0.0))
    p = h.propagate(Pose2D(1.0, 2.0, math.pi / 2), 0.0, 1.5)
    assert (p.x, p.y, p.theta) == pytest.approx((1.0, 2.6, math.pi / 2), abs=1e-9)


def test_propagate_pure_rotation() -> None:
    h = CommandHistory(0.5, 1.5)
    h.record(0.0, V(0.0, 1.0))
    p = h.propagate(Pose2D(0.0, 0.0, 0.0), 0.0, 1.0)
    assert (p.x, p.y) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert p.theta == pytest.approx(1.0)


def test_propagate_combined_motion_follows_the_exact_arc() -> None:
    v, w, t = 0.4, 1.0, 1.2
    h = CommandHistory(0.5, 1.5)
    h.record(0.0, V(v, w))
    p = h.propagate(Pose2D(0.0, 0.0, 0.0), 0.0, t)
    R = v / w  # circle of radius R, centre (0, R)
    assert (p.x, p.y, p.theta) == pytest.approx((R * math.sin(w * t), R - R * math.cos(w * t), w * t), abs=1e-4)


def test_propagate_uses_each_command_only_while_it_was_in_effect() -> None:
    h = CommandHistory(0.5, 1.5)
    h.record(0.0, V(0.0, 0.0))
    h.record(1.0, V(0.5, 0.0))
    h.record(2.0, V(0.0, 0.0))
    p = h.propagate(Pose2D(0.0, 0.0, 0.0), 0.5, 3.0)
    assert p.x == pytest.approx(0.5)
    assert h.propagate(Pose2D(0.0, 0.0, 0.0), 2.0, 1.0).x == 0.0  # never backwards


def test_commands_are_clamped_to_the_robot_limits() -> None:
    h = CommandHistory(0.5, 1.5)
    h.record(0.0, V(3.0, -9.0))
    assert h.command_at(0.1) == (0.5, -1.5)


def test_predict_motion_is_in_the_start_pose_frame_and_unwrapped() -> None:
    h = CommandHistory(0.5, 1.5)
    assert h.predict_motion(0.0, 1.0) is None  # nothing commanded yet
    h.record(0.0, V(0.0, 1.5))
    dx, dy, dth = h.predict_motion(0.0, 4.0)
    assert dth == pytest.approx(6.0)  # > pi, not folded
    h2 = CommandHistory(0.5, 1.5)
    h2.record(0.0, V(0.5, 0.0))
    assert h2.predict_motion(1.0, 2.0) == pytest.approx((0.5, 0.0, 0.0))


# ------------------------------------------------------- motion validation --

def test_velocity_estimate_is_clamped_to_physical_limits() -> None:
    """The baseline failure: one noisy registration became 1.2 m/s for a
    robot spinning in place, and every later seed extrapolated it."""
    vx, vy, w = _clamp_velocity((-0.82, -0.89, 0.2), 0.5, 1.5)
    assert math.hypot(vx, vy) == pytest.approx(0.5)
    assert _clamp_velocity((0.0, 0.0, 3.0), 0.5, 1.5)[2] == 1.5


def test_plausibility_bound_scales_with_elapsed_time() -> None:
    cfg = PipelineConfig(max_linear_velocity=0.5, max_angular_velocity=1.5)
    assert cfg.time_aware
    assert cfg.max_rotation(0.05) < math.radians(10) < cfg.max_rotation(0.5)
    assert cfg.max_translation(0.5) == pytest.approx(0.5 * 1.5 * 0.5 + 0.05)
    assert not PipelineConfig().time_aware  # default: the original fixed gate
    with pytest.raises(ValueError):
        PipelineConfig(max_linear_velocity=0.5)  # both or neither


def _room_points(n: int = 360) -> np.ndarray:
    """A 6 x 4 m rectangular room outline (world frame), plus a pillar to break symmetry."""
    pts = []
    for x in np.linspace(-3, 3, 120):
        pts += [(x, -2.0), (x, 2.0)]
    for y in np.linspace(-2, 2, 80):
        pts += [(-3.0, y), (3.0, y)]
    for a in np.linspace(0, 2 * math.pi, 40):
        pts.append((1.0 + 0.2 * math.cos(a), 0.8 + 0.2 * math.sin(a)))
    return np.array(pts)


class StaticReference:
    def __init__(self, points: np.ndarray) -> None:
        self.points = points

    def get_confirmed_occupied_points_world(self, center, radius):
        return self.points


def _scan_from(pose: Pose2D, t: float, points: np.ndarray):
    """A LaserScan of the static point set seen from ``pose`` (nearest point per 1-degree bin)."""
    from common.messages import LaserScan
    local = points - np.array([pose.x, pose.y])
    c, s = math.cos(-pose.theta), math.sin(-pose.theta)
    lx, ly = c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]
    ang = np.arctan2(ly, lx)
    rng = np.hypot(lx, ly)
    ranges = [8.0] * 361
    for a, r in zip(ang, rng):
        i = int(round((a + math.pi) / math.radians(1)))
        if 0 <= i < 361 and r < ranges[i]:
            ranges[i] = float(r)
    return LaserScan(timestamp=t, angle_min=-math.pi, angle_increment=math.radians(1), ranges=ranges,
                     range_min=0.15, range_max=8.0)


def _pipeline(time_aware: bool, prior=None) -> ScanMatchingPipeline:
    cfg = PipelineConfig(max_linear_velocity=0.5, max_angular_velocity=1.5) if time_aware else PipelineConfig()
    return ScanMatchingPipeline(StaticReference(_room_points()), pipeline_config=cfg, motion_prior=prior)


def test_fast_rotation_between_slow_scans_is_accepted_when_time_aware() -> None:
    """14 degrees in 0.3 s at 0.8 rad/s is physically fine; the fixed gate
    (10 degrees, independent of dt) rejected it once the robot was turning."""
    pts = _room_points()
    for time_aware, expect in ((False, False), (True, True)):
        h = CommandHistory(0.5, 1.5)
        h.record(0.0, V(0.0, 0.8))
        p = _pipeline(time_aware, prior=h)
        p.process_scan(_scan_from(Pose2D(0.0, 0.0, 0.0), 0.0, pts))
        res = p.process_scan(_scan_from(Pose2D(0.0, 0.0, 0.24), 0.3, pts))
        assert res.accepted is expect, (time_aware, res.rejection_reason)
        if expect:
            assert res.pose.theta == pytest.approx(0.24, abs=math.radians(1))


def test_physically_impossible_registration_is_rejected(monkeypatch) -> None:
    """A registration that 'converged' to a pose 1 m / 60 degrees away only
    0.1 s later is refused in time-aware mode (the fixed gate only looked at
    the prediction, so it accepted it). Note the limit of this rule: a WRONG
    registration that implies plausible motion (e.g. ICP locking onto its
    seed after the robot was carried away) cannot be caught this way."""
    import slam.scan_matching_pipeline as spm
    from slam.scan_registration import RegistrationResult

    def far_away(reference, local, seed, cfg):
        return RegistrationResult(pose=Pose2D(1.0, 0.0, math.radians(60), seed.timestamp), converged=True,
                                  residual=0.01, correspondence_count=200, iterations=5, rejected_reason=None)

    pts = _room_points()
    for time_aware, expect_accepted in ((True, False), (False, True)):
        p = _pipeline(time_aware)
        p.process_scan(_scan_from(Pose2D(0.0, 0.0, 0.0), 0.0, pts))
        monkeypatch.setattr(spm, "register", far_away)
        res = p.process_scan(_scan_from(Pose2D(0.0, 0.0, 0.0), 0.1, pts))
        monkeypatch.undo()
        assert res.accepted is expect_accepted
        if not expect_accepted:
            assert res.rejection_reason == "implausible_motion"
            assert (res.pose.x, res.pose.theta) == (0.0, 0.0)  # pose frozen, never jumped
            assert p.state.velocity_estimate == (0.0, 0.0, 0.0)


def test_commanded_motion_seeds_the_start_of_a_turn() -> None:
    """Robot was stationary (velocity estimate 0), then commanded to turn:
    the command-seeded registration lands on the true heading."""
    pts = _room_points()
    h = CommandHistory(0.5, 1.5)
    h.record(0.0, V(0.0, 0.0))
    h.record(0.1, V(0.0, 1.5))
    p = _pipeline(True, prior=h)
    p.process_scan(_scan_from(Pose2D(0.0, 0.0, 0.0), 0.0, pts))
    p.process_scan(_scan_from(Pose2D(0.0, 0.0, 0.0), 0.1, pts))
    true_theta = 1.5 * 0.2
    res = p.process_scan(_scan_from(Pose2D(0.0, 0.0, true_theta), 0.3, pts))
    assert res.accepted
    assert res.pose.theta == pytest.approx(true_theta, abs=math.radians(1))


def test_commands_are_not_trusted_over_the_scans() -> None:
    """Commanded to drive, but the robot did not move (stalled/blocked):
    the scans still win; the command only moved the seed."""
    pts = _room_points()
    h = CommandHistory(0.5, 1.5)
    h.record(0.0, V(0.3, 0.0))
    p = _pipeline(True, prior=h)
    p.process_scan(_scan_from(Pose2D(0.0, 0.0, 0.0), 0.0, pts))
    res = p.process_scan(_scan_from(Pose2D(0.0, 0.0, 0.0), 0.2, pts))  # commanded 6 cm, moved 0
    assert res.accepted
    assert (res.pose.x, res.pose.y) == pytest.approx((0.0, 0.0), abs=0.01)


# ------------------------------------------------------- delay compensation --

@pytest.mark.parametrize("script", [
    [(6.0, V(0.3, 0.0))],                                # straight
    [(6.0, V(0.0, 1.2))],                                # rotation
    [(6.0, V(0.25, 0.8))],                               # combined arc
], ids=["straight", "rotation", "combined"])
def test_delay_compensation_reduces_the_error_of_the_pose_navigation_uses(script) -> None:
    """Same run with and without compensation; the realistic sensor + an
    overloaded processor (0.4-0.6 s) make the latency large. Compared at
    every tick against ground truth."""
    from simulation.sensor_processing import ProcessingConfig

    def mean_error(compensate: bool) -> tuple[float, float]:
        nav = dataclasses.replace(NAV_CFG, localization=dataclasses.replace(NAV_CFG.localization,
                                                                             delay_compensation=compensate))
        sensor = dataclasses.replace(load_sensor_simulation_config(mode="realistic", seed=5),
                                     processing=ProcessingConfig(delay_min_s=0.4, delay_max_s=0.6))
        sc = default_scenario(ROBOT_CFG.robot.width, ROBOT_CFG.robot.length)
        sc.environment.robot_pose = Pose2D(3.5, 3.0, 0.0)  # more room for the straight run
        sim = ClosedLoopSimulation(sc.environment, sc.environment.robot_pose, sc.slam_frame_goal, nav, ROBOT_CFG,
                                   max_simulation_time=5.0, sensor_simulation=sensor, drive_script=script)
        pos, head = [], []
        while sim._sim_time < 5.0 - 1e-9:
            sim.step()
            if sim._script_start_time is not None:
                gt = sim._ground_truth_to_slam(sim.environment.robot_pose)
                est = sim.nav_robot.get_pose()
                pos.append(math.hypot(gt.x - est.x, gt.y - est.y))
                head.append(abs(math.remainder(gt.theta - est.theta, 2 * math.pi)))
        return float(np.mean(pos)), float(np.mean(head))

    (p_off, h_off), (p_on, h_on) = mean_error(False), mean_error(True)
    assert p_on + h_on < 0.5 * (p_off + h_off), ((p_off, h_off), (p_on, h_on))
