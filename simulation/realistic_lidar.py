"""Realistic RPLiDAR (A2M12) sensor model, layered on the ideal simulator.

``simulation.simulated_lidar.SimulatedLidar`` is an idealized sensor:
every one of the 361 bins is measured, perfectly, instantaneously, from
one pose, at exactly ``scan_rate``. This module keeps that ray-caster for
the physics (``SimulatedLidar._cast_ray`` -- the same geometry, reused,
not reimplemented) and adds the imperfections of the real sensor + driver
chain on top. It still emits the SAME ``LaserScan`` message, built the
SAME way ``hardware.rplidar_driver.RPLidarDriver`` builds it, so SLAM
cannot tell the ideal simulator, this model, and real hardware apart by
type or convention.

What the real chain does (and what is modeled here)
---------------------------------------------------
1. The head spins continuously (~11 Hz). Each revolution is a stream of
   individual *samples*, each at its own angle AND its own instant, from
   wherever the robot is at that instant (motion distortion). Modeled:
   ``samples_per_second`` samples per second, evenly spaced in time over
   a rotation period that jitters slightly revolution to revolution, with
   a random angular phase per revolution (real sample angles do not land
   on the same bins every revolution). Each sample ray-casts from the
   interpolated ground-truth pose at its own acquisition time
   (``enable_motion_distortion``), or -- when disabled -- from one pose
   at the revolution's end time.
2. Physics first, imperfections second. The ideal range is computed
   first; a ray that reaches ``range_max`` without hitting anything is
   "no detectable obstacle" and simply yields no sample -- that is not a
   sensor failure. A ray that DOES hit something can still fail to
   return (``sample_dropout_probability``: low reflectivity, grazing
   incidence, ...) -- the rplidar library drops such quality-0 samples,
   so they vanish exactly like the no-obstacle case does.
3. Range noise (see ``_noisy_range``), then the protocol's 0.25 mm
   distance quantization, then the driver's own range gate: anything
   outside ``[range_min, range_max]`` is discarded (never clamped).
4. Binning: identical to ``RPLidarDriver._bin_scan`` -- nearest 1-degree
   bin, angles folded into (-pi, pi], nearest hit wins when several
   samples share a bin, empty bins report ``range_max``. So a missing
   return, a partial scan's unmeasured sector, and "nothing there" are
   ALL represented as ``range_max``, exactly as the real driver does
   (it cannot tell them apart either -- see the final report's note on
   what that means for free-space carving).
5. Partial scans: in the real recording every partial scan is the TAIL
   of a revolution (a contiguous arc ending at the revolution's end),
   consistent with the rplidar library flushing its serial buffer on
   overflow and with the very first revolution starting mid-rotation.
   Modeled: with ``partial_scan_probability`` (and always for the first
   revolution when ``first_scan_partial``), only samples from the final
   ``U(partial_keep_fraction_min, partial_keep_fraction_max)`` of the
   revolution's duration survive.
6. The rplidar library silently swallows revolutions with
   ``<= min_measurements_per_scan`` samples; so does this model (the
   revolution is counted as generated but never delivered).
7. Timestamps: the real driver stamps each scan with ``time.time()`` when
   the library yields it -- i.e. once the revolution is complete and the
   host gets around to it. Modeled: ``timestamp = revolution end +
   delivery delay``, the delay exponential with mean
   ``delivery_delay_mean_s`` (host/USB/GIL scheduling), capped at
   ``delivery_delay_max_s``. This is what produces the observed 70-136 ms
   interval spread while the head itself spins steadily.

Acquisition order vs. bin order (see ``LaserScan.scan_time``): each
revolution starts at ``scan_start_angle`` (default -6 degrees in the
driver's bin frame, where the recording's partial scans place the
revolution boundary) and sweeps in the direction of increasing
``LaserScan`` angle, wrapping from +pi to -pi -- the same time-to-bin
mapping the real driver produces from the raw angles it is given.

Angle direction (resolved 2026-09-29): the real sensor's raw angles ARE
    clockwise; ``RPLidarDriver`` now negates them (config/hardware.yaml
    ``rplidar.angles_clockwise``), confirmed with the visual left/right check.
    OPEN FOLLOW-UP: this model still sweeps in the direction of INCREASING
    LaserScan angle, but a clockwise sensor sweeps in the direction of
    DECREASING (counter-clockwise) angle in the driver's output, and
    ``scan_start_angle`` was fitted in the old, mirrored bin frame. Only the
    in-sweep motion-distortion direction and the partial-scan boundary are
    affected, not the geometry; not changed yet.

Every random draw goes through one ``numpy.random.Generator`` seeded from
``seed``, so a given seed reproduces the exact same noise, dropouts,
partial scans and timing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from common.config import LidarConfig
from common.messages import LaserScan, Pose2D
from simulation.environment import Environment
from simulation.simulated_lidar import SimulatedLidar

PoseProvider = Callable[[float], Pose2D]

TWO_PI = 2.0 * math.pi


@dataclass
class RealisticLidarConfig:
    """Tunable sensor-imperfection parameters. Defaults are EMPIRICAL
    starting points from ``tests/fixtures/real_scans_stationary.json``
    (25 stationary scans, ~2.2 s), not specifications -- see
    ``config/lidar_simulation.yaml`` for where each number came from.
    The geometric shape (range limits, FOV, bin width) is NOT here: it
    comes from the shared ``LidarConfig`` in ``navigation.yaml``, exactly
    like the ideal simulator and the real driver."""

    scan_rate_hz: float = 11.05
    rotation_jitter_std_s: float = 0.002
    samples_per_second: float = 4000.0
    scan_start_angle: float = -0.1047

    sample_dropout_probability: float = 0.145

    range_noise_std_m: float = 0.001
    range_noise_proportional: float = 0.0
    outlier_probability: float = 0.02
    outlier_std_m: float = 0.04
    range_quantization_m: float = 0.00025

    partial_scan_probability: float = 0.08
    partial_keep_fraction_min: float = 0.15
    partial_keep_fraction_max: float = 0.30
    first_scan_partial: bool = True
    min_measurements_per_scan: int = 5

    delivery_delay_mean_s: float = 0.009
    delivery_delay_max_s: float = 0.06

    enable_motion_distortion: bool = True

    def __post_init__(self) -> None:
        if self.scan_rate_hz <= 0.0:
            raise ValueError("scan_rate_hz must be > 0")
        if self.rotation_jitter_std_s < 0.0:
            raise ValueError("rotation_jitter_std_s must be >= 0")
        if self.samples_per_second <= 0.0:
            raise ValueError("samples_per_second must be > 0")
        for name in ("sample_dropout_probability", "outlier_probability", "partial_scan_probability"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("range_noise_std_m", "range_noise_proportional", "outlier_std_m", "range_quantization_m"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 < self.partial_keep_fraction_min <= self.partial_keep_fraction_max <= 1.0:
            raise ValueError("need 0 < partial_keep_fraction_min <= partial_keep_fraction_max <= 1")
        if self.min_measurements_per_scan < 0:
            raise ValueError("min_measurements_per_scan must be >= 0")
        if self.delivery_delay_mean_s < 0.0 or self.delivery_delay_max_s < 0.0:
            raise ValueError("delivery delays must be >= 0")
        # A delay cap shorter than one rotation keeps delivery order equal
        # to acquisition order, as with the real single serial stream.
        if self.delivery_delay_max_s >= 0.8 * self.nominal_period_s:
            raise ValueError("delivery_delay_max_s must be well under one rotation period")

    @property
    def nominal_period_s(self) -> float:
        """Nominal rotation period == nominal acquisition duration of a full
        scan (the head measures continuously; there is no gap between
        revolutions)."""
        return 1.0 / self.scan_rate_hz


@dataclass
class SimulatedScanRecord:
    """Everything known about one simulated revolution. ``scan`` is what
    downstream code receives (``None`` if the revolution was swallowed
    by the ``min_measurements_per_scan`` rule, like the real library
    does). The per-sample arrays are simulator-only ground truth for
    diagnostics and tests -- they never reach SLAM."""

    index: int
    start_time: float
    end_time: float
    timestamp: float  # == scan.timestamp; the moment the scan reaches the host
    is_partial: bool
    scan: LaserScan | None
    sample_times: np.ndarray       # (N,) acquisition instant of every attempted sample
    sample_angles: np.ndarray      # (N,) sensor-frame angle, folded into (-pi, pi]
    sample_poses: np.ndarray       # (N, 3) ground-truth (x, y, theta) at each sample's instant
    true_ranges: np.ndarray        # (N,) ideal physical range (inf = nothing within range_max)
    measured_ranges: np.ndarray    # (N,) delivered range, NaN if not delivered for any reason
    dropped_returns: int = 0       # samples that hit something but did not return
    out_of_range_after_noise: int = 0
    skipped_for_partial: int = 0

    @property
    def delivered(self) -> bool:
        return self.scan is not None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_delivered_samples(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.measured_ranges)))

    def delivered_hit_points_world(self) -> np.ndarray:
        """(M, 2) world points of delivered samples, placed using each
        sample's TRUE acquisition pose (where the sensor actually was)."""
        ok = np.isfinite(self.measured_ranges)
        poses = self.sample_poses[ok]
        ang = poses[:, 2] + self.sample_angles[ok]
        r = self.measured_ranges[ok]
        return np.column_stack((poses[:, 0] + r * np.cos(ang), poses[:, 1] + r * np.sin(ang)))


@dataclass
class RealisticLidar:
    """Stateful revolution-by-revolution RPLiDAR model.

    Call ``acquire_until(environment, pose_at, now)`` once the ground-truth
    pose history covers ``now``: it returns every revolution that has
    fully completed by ``now`` (in order), each stamped with its own
    delivery timestamp -- which may be slightly AFTER ``now``; callers
    decide when a scan "arrives" (see ``simulation.sensor_processing``).
    """

    lidar_config: LidarConfig
    config: RealisticLidarConfig = field(default_factory=RealisticLidarConfig)
    seed: int | None = None
    start_time: float = 0.0

    def __post_init__(self) -> None:
        self._ray_caster = SimulatedLidar(config=self.lidar_config)  # validates the shared shape config
        c = self.lidar_config
        self._num_bins = int(round((c.angle_max - c.angle_min) / c.angular_resolution)) + 1
        self._rng = np.random.default_rng(self.seed)
        self._next_start = self.start_time
        self._index = 0
        # A period drawn for a revolution that had not finished by the last
        # acquire_until() call; reused next call so the RNG stream does not
        # depend on how finely the caller slices time.
        self._pending_period: float | None = None

    @property
    def num_bins(self) -> int:
        return self._num_bins

    @property
    def next_revolution_start(self) -> float:
        return self._next_start

    # ----------------------------------------------------------- acquisition --

    def acquire_until(self, environment: Environment, pose_at: PoseProvider, now: float) -> list[SimulatedScanRecord]:
        records: list[SimulatedScanRecord] = []
        while True:
            period = self._draw_period()
            end = self._next_start + period
            if end > now + 1e-12:
                self._pending_period = period  # not complete yet; keep it for next call
                break
            records.append(self._acquire_revolution(environment, pose_at, self._next_start, period))
            self._next_start = end
        return records

    def _draw_period(self) -> float:
        if self._pending_period is not None:
            period, self._pending_period = self._pending_period, None
            return period
        c = self.config
        jitter = self._rng.normal(0.0, c.rotation_jitter_std_s) if c.rotation_jitter_std_s > 0.0 else 0.0
        return max(0.5 * c.nominal_period_s, c.nominal_period_s + jitter)

    def _acquire_revolution(
        self, environment: Environment, pose_at: PoseProvider, start: float, period: float
    ) -> SimulatedScanRecord:
        c = self.config
        lc = self.lidar_config
        rng = self._rng

        n = max(1, int(round(c.samples_per_second * period)))
        phase = rng.random()
        fractions = (np.arange(n, dtype=np.float64) + phase) / n  # in (0, 1)
        times = start + fractions * period
        raw_angles = c.scan_start_angle + fractions * TWO_PI
        angles = np.mod(raw_angles + math.pi, TWO_PI) - math.pi  # fold into [-pi, pi)
        angles[angles == -math.pi] = math.pi                       # ... then (-pi, pi], like the driver

        # Partial-scan decision (tail of the revolution survives).
        is_first = self._index == 0
        is_partial = (is_first and c.first_scan_partial) or (rng.random() < c.partial_scan_probability)
        keep_from = rng.uniform(c.partial_keep_fraction_min, c.partial_keep_fraction_max) if is_partial else 1.0
        keep_mask = fractions >= 1.0 - keep_from

        end = start + period
        end_pose = pose_at(end)
        poses = np.empty((n, 3), dtype=np.float64)
        true_ranges = np.full(n, math.inf, dtype=np.float64)
        measured = np.full(n, math.nan, dtype=np.float64)

        # Draw every random number for every sample unconditionally, so the
        # RNG stream (and hence reproducibility for a given seed) does not
        # depend on the geometry of the scene.
        dropout_draws = rng.random(n)
        outlier_draws = rng.random(n)
        normal_draws = rng.standard_normal(n)

        dropped = 0
        out_of_range = 0
        for k in range(n):
            pose = pose_at(float(times[k])) if c.enable_motion_distortion else end_pose
            poses[k] = (pose.x, pose.y, pose.theta)
            world_angle = pose.theta + angles[k]
            hit = self._ray_caster._cast_ray(environment, (pose.x, pose.y), (math.cos(world_angle), math.sin(world_angle)))
            if hit >= lc.range_max:
                continue  # no detectable obstacle: no sample, not a failure
            true_ranges[k] = hit
            if not keep_mask[k]:
                continue
            if dropout_draws[k] < c.sample_dropout_probability:
                dropped += 1
                continue
            r = self._noisy_range(hit, outlier_draws[k], normal_draws[k])
            if r < lc.range_min or r > lc.range_max:
                out_of_range += 1
                continue  # the driver's range gate: discarded, never clamped
            measured[k] = r

        skipped = int(np.count_nonzero(~keep_mask & np.isfinite(true_ranges)))
        timestamp = end + self._draw_delivery_delay()

        delivered_count = int(np.count_nonzero(np.isfinite(measured)))
        scan: LaserScan | None = None
        if delivered_count > c.min_measurements_per_scan:
            scan = self._bin(angles, measured, timestamp, period)

        record = SimulatedScanRecord(
            index=self._index, start_time=start, end_time=end, timestamp=timestamp,
            is_partial=is_partial, scan=scan, sample_times=times, sample_angles=angles,
            sample_poses=poses, true_ranges=true_ranges, measured_ranges=measured,
            dropped_returns=dropped, out_of_range_after_noise=out_of_range, skipped_for_partial=skipped,
        )
        self._index += 1
        return record

    def _noisy_range(self, true_range: float, outlier_draw: float, normal_draw: float) -> float:
        """Two-component noise model: a Gaussian core whose std grows
        linearly with range (``range_noise_std_m + range_noise_proportional
        * r``), mixed with a wider Gaussian for a small fraction of
        samples (``outlier_probability``/``outlier_std_m``). A single
        Gaussian cannot match the recording's ~4 mm typical vs. ~16 mm
        worst-decile spread (a ratio of ~4, where a Gaussian's p90/median
        of |error| is ~2.4); the mixture's heavier tail can. Then quantized
        to the protocol's distance resolution."""
        c = self.config
        if outlier_draw < c.outlier_probability:
            std = c.outlier_std_m
        else:
            std = c.range_noise_std_m + c.range_noise_proportional * true_range
        r = true_range + std * normal_draw
        if c.range_quantization_m > 0.0:
            r = round(r / c.range_quantization_m) * c.range_quantization_m
        return r

    def _draw_delivery_delay(self) -> float:
        c = self.config
        if c.delivery_delay_mean_s <= 0.0:
            return 0.0
        return min(float(self._rng.exponential(c.delivery_delay_mean_s)), c.delivery_delay_max_s)

    def _bin(self, angles: np.ndarray, measured: np.ndarray, timestamp: float, period: float) -> LaserScan:
        """Same binning as ``RPLidarDriver._bin_scan`` (nearest bin,
        nearest hit wins, empty bins = range_max)."""
        lc = self.lidar_config
        ranges = [lc.range_max] * self._num_bins
        for angle, r in zip(angles, measured):
            if not math.isfinite(r):
                continue
            bin_index = round((float(angle) - lc.angle_min) / lc.angular_resolution)
            if 0 <= bin_index < self._num_bins and r < ranges[bin_index]:
                ranges[bin_index] = float(r)
        return LaserScan(
            timestamp=timestamp,
            angle_min=lc.angle_min,
            angle_increment=lc.angular_resolution,
            ranges=ranges,
            range_min=lc.range_min,
            range_max=lc.range_max,
            scan_time=period,
            time_increment=period * lc.angular_resolution / TWO_PI,
        )
