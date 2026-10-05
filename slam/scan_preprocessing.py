"""LaserScan -> local Cartesian points, split by hit vs. no-detection.

Pure, dependency-free (beyond numpy) preprocessing shared by scan
registration (``slam/scan_registration.py``, a later phase) and, later,
free-space integration in the persistent world model. Kept separate
from both so each is independently testable, per the Phase 1 scope of
the scan-registration/world-model design.

Ray classification (the ``range_min``/``range_max`` semantics already
established by ``common.messages.LaserScan`` and used throughout this
project):
    - hit:            ``range_min <= r < range_max`` -- a real detection
    - no-detection:   ``r >= range_max`` -- clear line of sight out to
      range_max along this bearing; not a hit, but not discarded either
      -- kept separately since free-space carving still needs it
    - invalid:        ``r < range_min`` -- discarded entirely, not used
      for anything (matches how ``hardware/rplidar_driver.py`` already
      treats out-of-range measurements at the binning stage)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.messages import LaserScan


@dataclass
class PreprocessedScan:
    """A scan split into registration input and free-space-carving input.

    Both arrays are in the local (sensor) frame. Angle convention:
    local angle 0 = +X, increasing counterclockwise -- matches
    ``LaserScan.angle_at`` and ``mapping.occupancy_grid.update_from_scan``
    exactly (``x = r*cos(angle)``, ``y = r*sin(angle)``).
    """

    hit_points_local: np.ndarray     # (N, 2) float64 -- Cartesian (x, y) of real hits
    max_range_bearings: np.ndarray   # (K,) float64 -- radians, no-detection ray angles


def preprocess_scan(scan: LaserScan) -> PreprocessedScan:
    """Split ``scan`` into hit points and max-range bearings.

    Invalid (``r < range_min``) rays are discarded entirely -- they
    appear in neither output array.
    """
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    num_rays = ranges.shape[0]
    angles = scan.angle_min + np.arange(num_rays, dtype=np.float64) * scan.angle_increment

    is_hit = (ranges >= scan.range_min) & (ranges < scan.range_max)
    hit_ranges = ranges[is_hit]
    hit_angles = angles[is_hit]
    if hit_ranges.size:
        hit_points_local = np.column_stack(
            (hit_ranges * np.cos(hit_angles), hit_ranges * np.sin(hit_angles))
        ).astype(np.float64)
    else:
        hit_points_local = np.empty((0, 2), dtype=np.float64)

    is_max_range = ranges >= scan.range_max
    max_range_bearings = angles[is_max_range].astype(np.float64)

    return PreprocessedScan(hit_points_local=hit_points_local, max_range_bearings=max_range_bearings)
