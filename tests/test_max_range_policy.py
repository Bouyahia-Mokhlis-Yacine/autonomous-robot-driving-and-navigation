"""How ``PersistentWorldModel`` treats max-range ("no return") rays.

A ``range_max`` bin means only "no return in this direction": nothing
within range, a missed return (dark/glancing surface, dropout), or -- in a
partial scan -- a sector that was never measured. The driver and the
realistic sensor model cannot tell these apart. The original behavior
("clear") treats every such bin as a clear line of sight to ``range_max``
and erases everything along it, including confirmed walls. These tests pin
down that behavior and each alternative policy."""

from __future__ import annotations

import math

import pytest

from common.messages import LaserScan, Pose2D
from mapping.persistent_world_model import CellState, PersistentWorldModel

RANGE_MAX = 8.0
N = 360
INC = 2 * math.pi / N


def scan(ranges: list[float], t: float) -> LaserScan:
    return LaserScan(timestamp=t, angle_min=0.0, angle_increment=INC, ranges=ranges,
                     range_min=0.15, range_max=RANGE_MAX)


def wall_scan(t: float, missing: set[int] = frozenset(), wall_at: float = 1.0) -> LaserScan:
    """A wall at x = wall_at in front of the robot (bins within +-20 deg);
    everything else sees nothing within range. ``missing`` bins drop out."""
    ranges = []
    for i in range(N):
        a = i * INC
        a = a - 2 * math.pi if a > math.pi else a
        r = wall_at / math.cos(a) if abs(a) < math.radians(20) else RANGE_MAX
        ranges.append(RANGE_MAX if i in missing else r)
    return scan(ranges, t)


def model(policy: str | None = None, **kw) -> PersistentWorldModel:
    args = dict(resolution=0.05, width_m=20.0, height_m=20.0, origin_x=-10.0, origin_y=-10.0)
    if policy is not None:
        args["max_range_policy"] = policy
    return PersistentWorldModel(**args, **kw)


def confirmed_wall(wm: PersistentWorldModel) -> None:
    for k in range(8):  # 0.7 s of steady observation -> confirmed
        wm.integrate(Pose2D(0.0, 0.0, 0.0), wall_scan(0.1 * k), 0.1 * k)
    assert wm.get_cell_state(1.0, 0.0) == CellState.CONFIRMED


def dropout_at_wall_center(wm: PersistentWorldModel, t: float = 1.0) -> None:
    # The few bins that all land on the wall cell at (1.0, 0.0) miss their return.
    wm.integrate(Pose2D(0.0, 0.0, 0.0), wall_scan(t, missing={357, 358, 359, 0, 1, 2, 3}), t)


def test_default_is_the_original_clear_policy() -> None:
    assert model().max_range_policy == "clear"


def test_clear_policy_lets_a_few_dropouts_erase_a_confirmed_wall() -> None:
    """Documents the original behavior this change addresses."""
    wm = model("clear")
    confirmed_wall(wm)
    dropout_at_wall_center(wm)
    assert wm.get_cell_state(1.0, 0.0) == CellState.FREE
    assert wm.get_cell_state(1.5, 0.0) == CellState.FREE  # and everything behind it


@pytest.mark.parametrize("policy", ["ignore", "unknown_only"])
def test_dropouts_never_erase_a_confirmed_wall(policy: str) -> None:
    wm = model(policy)
    confirmed_wall(wm)
    for k in range(10):
        dropout_at_wall_center(wm, t=1.0 + 0.1 * k)
    assert wm.get_cell_state(1.0, 0.0) == CellState.CONFIRMED


@pytest.mark.parametrize("policy", ["ignore", "unknown_only"])
def test_dropouts_do_not_prevent_a_wall_from_being_confirmed(policy: str) -> None:
    """A tentative wall cell must not be reset by a max-range ray passing
    through it in between hits (that kept real walls from ever confirming)."""
    wm = model(policy)
    for k in range(8):
        missing = {0} if k % 2 else set()
        wm.integrate(Pose2D(0.0, 0.0, 0.0), wall_scan(0.1 * k, missing), 0.1 * k)
    assert wm.get_cell_state(1.0, 0.0) == CellState.CONFIRMED


def test_limited_policy_only_clears_up_to_its_distance() -> None:
    near = model("limited", max_range_clear_distance=0.5)
    confirmed_wall(near)
    dropout_at_wall_center(near)
    assert near.get_cell_state(1.0, 0.0) == CellState.CONFIRMED   # wall beyond 0.5 m survives
    assert near.get_cell_state(0.3, 0.0) == CellState.FREE

    far = model("limited", max_range_clear_distance=2.0)
    confirmed_wall(far)
    dropout_at_wall_center(far)
    assert far.get_cell_state(1.0, 0.0) == CellState.FREE          # within 2 m: still erodes
    assert far.get_cell_state(3.0, 0.0) == CellState.UNKNOWN       # nothing past 2 m is touched


def test_unknown_only_still_maps_open_space_as_free() -> None:
    wm = model("unknown_only")
    wm.integrate(Pose2D(0.0, 0.0, 0.0), wall_scan(0.0), 0.0)
    assert wm.get_cell_state(0.0, 5.0) == CellState.FREE      # a no-return direction, never seen before
    ignore = model("ignore")
    ignore.integrate(Pose2D(0.0, 0.0, 0.0), wall_scan(0.0), 0.0)
    assert ignore.get_cell_state(0.0, 5.0) == CellState.UNKNOWN


@pytest.mark.parametrize("policy", ["clear", "ignore", "unknown_only", "limited"])
def test_real_returns_still_clear_a_removed_obstacle(policy: str) -> None:
    """Hits are positive evidence under every policy: if the wall moves back,
    rays that now reach further clear the old wall cells."""
    wm = model(policy)
    confirmed_wall(wm)
    for k in range(3):
        t = 1.0 + 0.1 * k
        wm.integrate(Pose2D(0.0, 0.0, 0.0), wall_scan(t, wall_at=2.0), t)
    assert wm.get_cell_state(1.0, 0.0) == CellState.FREE


def test_partial_scan_does_not_erase_the_unmeasured_sector() -> None:
    """A partial scan reports its unmeasured bins as range_max, exactly like the driver."""
    for policy, expected in (("clear", CellState.FREE), ("unknown_only", CellState.CONFIRMED)):
        wm = model(policy)
        confirmed_wall(wm)
        partial = wall_scan(1.0, missing=set(range(0, 30)) | set(range(330, 360)))  # the wall's whole sector
        wm.integrate(Pose2D(0.0, 0.0, 0.0), partial, 1.0)
        assert wm.get_cell_state(1.0, 0.0) == expected, policy


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        model("forget")
    with pytest.raises(ValueError):
        model("limited", max_range_clear_distance=0.0)
