"""Per-run sensor + processing diagnostics for the closed-loop simulation.

Collects just enough per revolution / per processed scan to report the
realism metrics (scan timing, valid/missing readings, partial scans,
noise, acquisition duration, processing latency, queue behavior) without
keeping every ``SimulatedScanRecord``'s full per-sample arrays alive for
the whole run. ``report()`` reuses ``simulation.lidar_statistics`` so
these numbers are computed exactly like the real-recording comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.messages import LaserScan
from simulation.lidar_statistics import sample_noise_statistics, scan_statistics
from simulation.realistic_lidar import SimulatedScanRecord
from simulation.sensor_processing import CompletedScan, ProcessingMetrics


@dataclass
class SensorRunLog:
    revolutions_generated: int = 0
    revolutions_swallowed: int = 0      # <= min_measurements_per_scan, never delivered (like the rplidar library)
    partial_revolutions: int = 0        # model ground truth, delivered or not
    delivered_scans: list[LaserScan] = field(default_factory=list)
    acquisition_durations: list[float] = field(default_factory=list)
    scan_age_at_use: list[float] = field(default_factory=list)  # finish - revolution START (oldest data in the scan)
    _acq_start_by_timestamp: dict[float, float] = field(default_factory=dict)
    _true: list[np.ndarray] = field(default_factory=list)
    _measured: list[np.ndarray] = field(default_factory=list)

    def record_revolution(self, record: SimulatedScanRecord) -> None:
        self.revolutions_generated += 1
        self.partial_revolutions += int(record.is_partial)
        self.acquisition_durations.append(record.duration)
        self._true.append(record.true_ranges)
        self._measured.append(record.measured_ranges)
        if record.scan is None:
            self.revolutions_swallowed += 1
        else:
            self.delivered_scans.append(record.scan)
            self._acq_start_by_timestamp[record.scan.timestamp] = record.start_time

    def record_completion(self, completed: CompletedScan) -> None:
        start = self._acq_start_by_timestamp.pop(completed.scan.timestamp, None)
        # Scans dropped by the queue policy never complete; their entries are
        # pruned lazily here so the dict stays small.
        for ts in [ts for ts in self._acq_start_by_timestamp if ts < completed.scan.timestamp]:
            del self._acq_start_by_timestamp[ts]
        if start is not None:
            self.scan_age_at_use.append(completed.finish_time - start)

    def report(self, processing: ProcessingMetrics) -> dict:
        out: dict = {
            "revolutions_generated": self.revolutions_generated,
            "revolutions_swallowed": self.revolutions_swallowed,
            "partial_revolutions_model": self.partial_revolutions,
            "scans_generated": len(self.delivered_scans),
            "scans_received": processing.scans_received,
            "scans_processed": processing.scans_processed,
            "scans_dropped": processing.scans_dropped,
            "average_processing_latency_s": processing.average_processing_latency,
            "max_processing_latency_s": processing.max_processing_latency,
            "mean_processing_time_s": float(np.mean(processing.processing_times)) if processing.processing_times else None,
            "mean_queue_size": processing.mean_queue_size,
            "max_queue_size": processing.max_queue_size,
            "mean_acquisition_duration_s": float(np.mean(self.acquisition_durations)) if self.acquisition_durations else None,
            "mean_scan_age_at_use_s": float(np.mean(self.scan_age_at_use)) if self.scan_age_at_use else None,
            "max_scan_age_at_use_s": float(np.max(self.scan_age_at_use)) if self.scan_age_at_use else None,
        }
        if self._true:
            out["range_noise"] = sample_noise_statistics(np.concatenate(self._true), np.concatenate(self._measured))
        if len(self.delivered_scans) >= 2:
            out["scans"] = scan_statistics(self.delivered_scans)
        return out


def _f(value: float | None, scale: float = 1.0, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * scale:.{digits}f}"


def format_sensor_report(report: dict) -> str:
    """Human-readable version of ``SensorRunLog.report()``."""
    lines = ["Sensor / processing diagnostics"]
    scans = report.get("scans")
    if scans is not None:
        iv = scans["scan_interval_s"]
        vp = scans["valid_per_scan"]
        mp = scans["missing_per_scan"]
        lines += [
            f"  scan frequency:        {_f(scans['scan_frequency_hz'], digits=2)} Hz",
            f"  scan interval:         mean {_f(iv['mean'], 1000)} ms, std {_f(iv['std'], 1000)} ms, "
            f"min {_f(iv['min'], 1000)} ms, max {_f(iv['max'], 1000)} ms",
            f"  valid readings/scan:   mean {_f(vp['mean'])} (min {_f(vp['min'], digits=0)}, max {_f(vp['max'], digits=0)})",
            f"  missing readings/scan: mean {_f(mp['mean'])}",
            f"  dropout (typical bins): {_f(scans.get('dropout_rate_in_typical_bins'), 100)} %",
            f"  partial scans:         {scans['partial_scans']} of {scans['num_scans']} delivered "
            f"(model: {report['partial_revolutions_model']} partial revolutions, "
            f"{report['revolutions_swallowed']} swallowed as too short)",
        ]
    noise = report.get("range_noise")
    if noise and noise.get("n"):
        lines.append(
            f"  range noise (vs truth): median |e| {_f(noise['median_abs_m'], 1000)} mm, "
            f"p90 |e| {_f(noise['p90_abs_m'], 1000)} mm, std {_f(noise['std_m'], 1000)} mm, "
            f"max |e| {_f(noise['max_abs_m'], 1000)} mm"
        )
    lines += [
        f"  acquisition duration:  {_f(report['mean_acquisition_duration_s'], 1000)} ms per revolution",
        f"  scans generated/received/processed/dropped: {report['scans_generated']}/"
        f"{report['scans_received']}/{report['scans_processed']}/{report['scans_dropped']}",
        f"  processing time:       mean {_f(report['mean_processing_time_s'], 1000)} ms",
        f"  arrival->result latency: mean {_f(report['average_processing_latency_s'], 1000)} ms, "
        f"max {_f(report['max_processing_latency_s'], 1000)} ms",
        f"  scan age when used:    mean {_f(report['mean_scan_age_at_use_s'], 1000)} ms, "
        f"max {_f(report['max_scan_age_at_use_s'], 1000)} ms (from start of acquisition)",
        f"  queue size:            mean {_f(report['mean_queue_size'], digits=2)}, max {report['max_queue_size']}",
    ]
    return "\n".join(lines)
