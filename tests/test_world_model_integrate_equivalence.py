"""The optimized ``PersistentWorldModel.integrate()`` must produce exactly
the same map as the original per-cell implementation.

``ReferenceWorldModel.integrate`` below is a VERBATIM copy of the original
loop (one ``_observe_free``/``_observe_occupied`` call per traversed cell).
Tolerance: none -- internal ``state``, ``pending_since`` (NaN-aware) and the
public grid must be bit-identical after every scan, over long, adversarial
scan sequences (moving poses, out-of-bounds rays, invalid/max-range
readings, repeated hits, confirmation and clearing over time)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.config import load_navigation_config
from common.messages import LaserScan, Pose2D
from mapping.persistent_world_model import PersistentWorldModel
from mapping.ray_casting import bresenham_line
from simulation.environment import Environment
from simulation.obstacles import RectangleObstacle
from simulation.realistic_lidar import RealisticLidar


class ReferenceWorldModel(PersistentWorldModel):
    def integrate(self, pose: Pose2D, scan: LaserScan, now: float) -> None:  # original implementation
        origin_row, origin_col = self.occupancy_grid.world_to_grid(pose.x, pose.y)

        for i in range(scan.num_rays):
            r = scan.ranges[i]
            if not math.isfinite(r) or r < scan.range_min:
                continue  # invalid: no free-space or occupied observation

            is_hit = r < scan.range_max
            angle = pose.theta + scan.angle_at(i)
            end_x = pose.x + r * math.cos(angle)
            end_y = pose.y + r * math.sin(angle)
            end_row, end_col = self.occupancy_grid.world_to_grid(end_x, end_y)

            traced = bresenham_line(origin_row, origin_col, end_row, end_col)

            for row, col in traced[:-1]:
                if self.occupancy_grid.in_bounds(row, col):
                    self._observe_free(row, col)

            last_row, last_col = traced[-1]
            if not self.occupancy_grid.in_bounds(last_row, last_col):
                continue
            if is_hit:
                self._observe_occupied(last_row, last_col, now)
            else:
                self._observe_free(last_row, last_col)

        self._rebuild_public_grid()


def assert_identical(a: PersistentWorldModel, b: PersistentWorldModel) -> None:
    np.testing.assert_array_equal(a.state, b.state)
    np.testing.assert_array_equal(a.pending_since, b.pending_since)  # NaN == NaN here
    np.testing.assert_array_equal(a.get_grid().cells, b.get_grid().cells)


def pair(**kwargs) -> tuple[PersistentWorldModel, ReferenceWorldModel]:
    return PersistentWorldModel(**kwargs), ReferenceWorldModel(**kwargs)


def test_identical_on_random_adversarial_scans() -> None:
    rng = np.random.default_rng(0)
    # Small map so many rays leave it (out-of-bounds handling).
    fast, ref = pair(resolution=0.05, width_m=4.0, height_m=4.0, origin_x=-2.0, origin_y=-2.0)
    for k in range(80):
        n = 181
        ranges = rng.uniform(0.05, 4.0, n)
        ranges[rng.random(n) < 0.15] = 3.0            # range_max: no-detection rays
        ranges[rng.random(n) < 0.05] = 0.1            # below range_min: invalid
        ranges[rng.random(n) < 0.03] = math.inf       # non-finite: invalid
        # Repeat a few scans exactly so tentative cells get confirmed/cleared over time.
        if k % 7 == 3:
            ranges = prev
        prev = ranges
        scan = LaserScan(timestamp=0.1 * k, angle_min=-math.pi, angle_increment=2 * math.pi / (n - 1),
                         ranges=list(ranges), range_min=0.15, range_max=3.0)
        pose = Pose2D(x=rng.uniform(-0.5, 0.5), y=rng.uniform(-0.5, 0.5), theta=rng.uniform(-math.pi, math.pi))
        now = 0.1 * k
        fast.integrate(pose, scan, now)
        ref.integrate(pose, scan, now)
        assert_identical(fast, ref)


def test_identical_on_a_realistic_sensor_sequence_with_motion() -> None:
    lidar_cfg = load_navigation_config().lidar
    env = Environment(width=8.0, height=6.0, robot_length=0.3, robot_width=0.25, robot_pose=Pose2D(3.0, 3.0, 0.0))
    env.add_static_obstacle(RectangleObstacle(x=5.5, y=2.0, width=0.4, height=1.5))
    env.add_static_obstacle(RectangleObstacle(x=1.5, y=4.5, width=1.0, height=0.3))
    lidar = RealisticLidar(lidar_cfg, seed=11)

    def pose_at(t: float) -> Pose2D:  # an arc through the room
        return Pose2D(3.0 + 0.3 * t, 3.0 + 0.1 * math.sin(t), 0.6 * t)

    fast, ref = pair(resolution=0.05, width_m=10.0, height_m=8.0, origin_x=-3.0, origin_y=-3.0)
    count = 0
    for rec in lidar.acquire_until(env, pose_at, 6.0):
        if rec.scan is None:
            continue
        p = pose_at(rec.timestamp)
        slam_pose = Pose2D(p.x - 3.0, p.y - 3.0, p.theta)  # world model frame
        fast.integrate(slam_pose, rec.scan, rec.timestamp)
        ref.integrate(slam_pose, rec.scan, rec.timestamp)
        assert_identical(fast, ref)
        count += 1
    assert count > 50


@pytest.mark.parametrize("duration", [0.05, 0.5, 2.0])
def test_identical_for_different_confirmation_durations(duration: float) -> None:
    rng = np.random.default_rng(int(duration * 100))
    fast, ref = pair(resolution=0.1, width_m=6.0, height_m=6.0, origin_x=-3.0, origin_y=-3.0,
                     confirmation_duration=duration)
    base = rng.uniform(0.5, 2.5, 90)
    for k in range(40):
        ranges = base + rng.normal(0.0, 0.05, base.size)  # jitter makes cells flip free/hit
        scan = LaserScan(timestamp=0.1 * k, angle_min=-math.pi, angle_increment=2 * math.pi / 89,
                         ranges=list(ranges), range_min=0.15, range_max=8.0)
        fast.integrate(Pose2D(0.0, 0.0, 0.0), scan, 0.1 * k)
        ref.integrate(Pose2D(0.0, 0.0, 0.0), scan, 0.1 * k)
        assert_identical(fast, ref)


def test_scan_with_no_valid_rays_changes_nothing() -> None:
    fast, ref = pair()
    scan = LaserScan(timestamp=0.0, angle_min=0.0, angle_increment=0.1, ranges=[0.0, math.nan, 0.01],
                     range_min=0.15, range_max=8.0)
    fast.integrate(Pose2D(0.0, 0.0, 0.0), scan, 0.0)
    ref.integrate(Pose2D(0.0, 0.0, 0.0), scan, 0.0)
    assert_identical(fast, ref)
