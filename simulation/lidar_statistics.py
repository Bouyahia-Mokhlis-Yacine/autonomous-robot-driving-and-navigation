"""Sensor-behavior statistics computed identically for real and simulated scans.

``scan_statistics()`` only looks at ``LaserScan`` messages, so the exact
same code measures the real recording (``tests/fixtures/
real_scans_stationary.json``) and any simulated run -- the comparison
tool (``compare_lidar_behavior.py``) and the closed-loop diagnostics both
use it, so "real vs. simulated" is never two different definitions of the
same metric.

``sample_noise_statistics()`` is simulator-only: it compares delivered
ranges against the simulator's own ground-truth ranges, which no real
recording can provide.

Definitions (chosen to be measurable from binned scans alone):
    partial scan      -- valid bins < ``partial_threshold`` x the median
                         valid-bin count of the run.
    typically-valid   -- a bin valid in >= 50% of the run's non-partial
                         scans (i.e. "a direction that normally sees
                         something").
    dropout rate      -- over non-partial scans, the fraction of
                         (scan, typically-valid bin) pairs that are empty.
    stationary jitter -- per-bin standard deviation of valid ranges across
                         non-partial scans (bins with >= 5 values). Only
                         meaningful when the sensor was stationary; it
                         includes both range noise and the effect of
                         samples landing at slightly different angles
                         within a 1-degree bin, exactly as the real
                         measurement does.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from common.messages import LaserScan

PARTIAL_THRESHOLD = 0.5


def _summary(values: Iterable[float]) -> dict[str, float | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def valid_mask(scan: LaserScan) -> np.ndarray:
    r = np.asarray(scan.ranges, dtype=np.float64)
    return np.isfinite(r) & (r >= scan.range_min) & (r < scan.range_max)


def scan_statistics(scans: list[LaserScan], partial_threshold: float = PARTIAL_THRESHOLD) -> dict:
    if not scans:
        raise ValueError("need at least one scan")
    num_bins = scans[0].num_rays
    masks = np.array([valid_mask(s) for s in scans])
    ranges = np.array([np.asarray(s.ranges, dtype=np.float64) for s in scans])
    valid_counts = masks.sum(axis=1)

    median_valid = float(np.median(valid_counts))
    is_partial = valid_counts < partial_threshold * median_valid
    full = ~is_partial

    timestamps = np.array([s.timestamp for s in scans], dtype=np.float64)
    intervals = np.diff(timestamps)

    stats: dict = {
        "num_scans": len(scans),
        "num_bins": num_bins,
        "scan_interval_s": _summary(intervals),
        "scan_frequency_hz": (1.0 / float(intervals.mean())) if intervals.size else None,
        "valid_per_scan": _summary(valid_counts),
        "missing_per_scan": _summary(num_bins - valid_counts),
        "partial_scans": int(is_partial.sum()),
        "partial_fraction": float(is_partial.mean()),
        "scan_time_s": _summary(s.scan_time for s in scans),
        "range_m": _summary(ranges[masks]),
        "interval_histogram_ms": _histogram(intervals * 1000.0, edges=np.arange(40.0, 181.0, 10.0)),
    }

    if full.any():
        full_masks = masks[full]
        typical = full_masks.mean(axis=0) >= 0.5
        stats["typically_valid_bins"] = int(typical.sum())
        stats["dropout_rate_in_typical_bins"] = (
            float(1.0 - full_masks[:, typical].mean()) if typical.any() else None
        )
        jitter = []
        full_ranges = ranges[full]
        for b in range(num_bins):
            vals = full_ranges[full_masks[:, b], b]
            if vals.size >= 5:
                jitter.append(float(vals.std()))
        jitter_arr = np.asarray(jitter)
        stats["stationary_bin_std_m"] = {
            "n_bins": int(jitter_arr.size),
            "median": float(np.median(jitter_arr)) if jitter_arr.size else None,
            "p90": float(np.percentile(jitter_arr, 90)) if jitter_arr.size else None,
        }
    return stats


def sample_noise_statistics(true_ranges: np.ndarray, measured_ranges: np.ndarray) -> dict:
    """Simulator-only: error of every delivered sample vs. its ideal range."""
    ok = np.isfinite(measured_ranges) & np.isfinite(true_ranges)
    err = measured_ranges[ok] - true_ranges[ok]
    if err.size == 0:
        return {"n": 0}
    abs_err = np.abs(err)
    return {
        "n": int(err.size),
        "mean_m": float(err.mean()),
        "std_m": float(err.std()),
        "median_abs_m": float(np.median(abs_err)),
        "p90_abs_m": float(np.percentile(abs_err, 90)),
        "max_abs_m": float(abs_err.max()),
    }


def _histogram(values: np.ndarray, edges: np.ndarray) -> list[tuple[float, float, int]]:
    if values.size == 0:
        return []
    clipped = np.clip(values, edges[0], edges[-1] - 1e-9)
    counts, _ = np.histogram(clipped, bins=edges)
    return [(float(lo), float(hi), int(c)) for lo, hi, c in zip(edges[:-1], edges[1:], counts)]
