"""PersistentWorldModel(occupied_timeout=...): forget occupied geometry that
has not been hit recently (opt-in; the default keeps the map persistent)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.messages import LaserScan, Pose2D
from mapping.persistent_world_model import CellState, PersistentWorldModel

N = 360


def scan(t: float, wall: bool = True, blob: bool = False) -> LaserScan:
    """A wall at x = 2 m ahead (+-20 deg); optionally a small object at 1 m, 90 deg left."""
    ranges = []
    for i in range(N):
        a = math.remainder(i * 2 * math.pi / N, 2 * math.pi)
        r = 8.0
        if wall and abs(a) < math.radians(20):
            r = 2.0 / math.cos(a)
        if blob and abs(a - math.pi / 2) < math.radians(3):
            r = 1.0
        ranges.append(r)
    return LaserScan(timestamp=t, angle_min=0.0, angle_increment=2 * math.pi / N, ranges=ranges,
                     range_min=0.15, range_max=8.0)


def model(timeout: float | None) -> PersistentWorldModel:
    return PersistentWorldModel(resolution=0.05, width_m=10.0, height_m=10.0, origin_x=-5.0, origin_y=-5.0,
                                max_range_policy="ignore", occupied_timeout=timeout)


def run(wm: PersistentWorldModel, t0: float, t1: float, **kw) -> None:
    t = t0
    while t < t1 - 1e-9:
        wm.integrate(Pose2D(0.0, 0.0, 0.0), scan(t, **kw), t)
        t = round(t + 0.1, 6)


def test_default_never_forgets() -> None:
    wm = model(None)
    run(wm, 0.0, 1.0, blob=True)
    assert wm.get_cell_state(0.0, 1.0) == CellState.CONFIRMED
    run(wm, 1.0, 6.0, blob=False)                         # object gone from view for 5 s
    assert wm.get_cell_state(0.0, 1.0) == CellState.CONFIRMED


def test_unseen_geometry_is_removed_after_the_timeout() -> None:
    wm = model(2.0)
    run(wm, 0.0, 1.0, blob=True)
    assert wm.get_cell_state(0.0, 1.0) == CellState.CONFIRMED
    run(wm, 1.0, 2.8, blob=False)                          # unseen for < 2 s: kept
    assert wm.get_cell_state(0.0, 1.0) == CellState.CONFIRMED
    run(wm, 2.8, 3.3, blob=False)                          # unseen for > 2 s: gone
    assert wm.get_cell_state(0.0, 1.0) == CellState.UNKNOWN
    assert wm.get_grid().cells[wm.occupancy_grid.world_to_grid(0.0, 1.0)] == -1


def test_continuously_seen_geometry_is_kept() -> None:
    wm = model(2.0)
    run(wm, 0.0, 10.0)
    assert wm.get_cell_state(2.0, 0.0) == CellState.CONFIRMED


def test_tentative_noise_is_removed_too() -> None:
    wm = model(2.0)
    wm.integrate(Pose2D(0.0, 0.0, 0.0), scan(0.0, blob=True), 0.0)   # a single spurious sighting
    assert wm.get_cell_state(0.0, 1.0) == CellState.TENTATIVE
    run(wm, 0.1, 2.5)
    assert wm.get_cell_state(0.0, 1.0) == CellState.UNKNOWN
    assert math.isnan(wm.get_pending_since(0.0, 1.0) or math.nan)


def test_free_space_is_not_forgotten() -> None:
    wm = model(2.0)
    run(wm, 0.0, 5.0)
    assert wm.get_cell_state(1.0, 0.0) == CellState.FREE


def test_timeout_validation() -> None:
    with pytest.raises(ValueError):
        model(0.0)
