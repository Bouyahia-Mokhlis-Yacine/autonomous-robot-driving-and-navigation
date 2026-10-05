"""Phase 5: PersistentWorldModel -- the temporal/persistent interpretation
of LiDAR observations, layered on top of the existing occupancy-grid
representation.

Core invariant: **NOT OBSERVED != GONE.** A confirmed occupied cell is
never cleared merely because the current scan doesn't currently see
it (out of range, occluded, or simply not covered by this scan's
rays). It is cleared only when the current scan gives the LiDAR a
*direct, unobstructed line of sight* to that location and the
observation says the location is free.

Architecture (composition, not modification, per the approved design):
this class owns the temporal state (``_state``, ``_pending_since``)
and composes an existing, unmodified ``mapping.occupancy_grid.
OccupancyGridMap`` purely for grid storage, world<->grid coordinate
conversion, and bounds -- ``OccupancyGridMap`` itself is untouched.
Ray tracing reuses ``mapping.ray_casting.bresenham_line`` unchanged;
no second ray-tracing implementation is introduced.

No object classification, identity, tracking, or velocity of any kind
exists anywhere in this module -- every occupied cell is treated
uniformly, purely as geometry. This module is independent of and does
NOT import ``slam.scan_matching_pipeline`` -- the intended future
dependency direction is pipeline -> this module, never the reverse;
that integration is explicitly a later phase's job, not this one's.

Internal cell states -- FOUR, not the minimum three, and this is a
deliberate, documented choice: UNKNOWN, FREE, TENTATIVE, CONFIRMED.
The task's own minimum is UNKNOWN/TENTATIVE/CONFIRMED; a distinct FREE
state is added because a direct free-space observation is genuine,
positive evidence (not the absence of information UNKNOWN represents),
and the public ``OccupancyGrid`` needs an explicit FREE value to stay
compatible with existing consumers (inflation, planning, GUI) without
requiring any changes to them, exactly as this phase requires.

Public ``OccupancyGrid`` mapping:
    UNKNOWN   (internal) -> UNKNOWN  (-1)
    FREE      (internal) -> FREE     ( 0)
    TENTATIVE (internal) -> UNKNOWN  (-1)  -- see below
    CONFIRMED (internal) -> OCCUPIED (100)

TENTATIVE deliberately maps to public UNKNOWN, not FREE: showing a
cell with active (if unconfirmed) occupied evidence as "safe to drive
through" (FREE) would misrepresent what is actually known, and is
less conservative than treating it as "not yet decided" (UNKNOWN) --
which is also literally true. This matches
``get_confirmed_occupied_points_world()``'s requirement to never
expose tentative geometry as confirmed: the public grid doesn't
expose it as occupied OR free, only as not-yet-decided. A future
visualization phase can render a distinct "tentative" overlay
straight from ``get_cell_state()``/``get_pending_since()`` (below)
without needing the public grid itself to carry that distinction --
consistent with how the inflation overlay is already a separate layer
on top of the base map, not a third value baked into the base grid.

Confirmation rule (see ``_observe_occupied``): a cell's
``pending_since`` is set to the scan timestamp of its FIRST occupied
observation and never advances or resets merely because time passes
without further evidence (occlusion). On every SUBSEQUENT occupied
observation of that same still-tentative cell, confirmation is
checked as ``now - pending_since >= confirmation_duration``, using
that observation's own timestamp as ``now`` -- never an externally
advancing clock disconnected from an actual hit. This is span-based
continuity (first sighting to most recent sighting), not a wall-clock
timer and not a scan-count: an occluded gap in between neither resets
nor artificially advances anything, and confirmation can never happen
without a fresh corroborating observation actually occurring at or
after the 0.5s mark.
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np

from common.messages import LaserScan, OccupancyGrid, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from mapping.ray_casting import bresenham_line

DEFAULT_RESOLUTION = 0.05
DEFAULT_WIDTH_M = 20.0
DEFAULT_HEIGHT_M = 20.0
DEFAULT_ORIGIN_X = -10.0
DEFAULT_ORIGIN_Y = -10.0
DEFAULT_CONFIRMATION_DURATION = 0.5

# How max-range ("no return") rays are used -- see PersistentWorldModel.__init__.
MAX_RANGE_POLICIES = ("clear", "ignore", "limited", "unknown_only")


class CellState(IntEnum):
    """Internal temporal cell state -- see module docstring for the
    full mapping to the public OccupancyGrid values and the rationale
    for FREE beyond the task's minimum three states."""

    UNKNOWN = 0
    FREE = 1
    TENTATIVE = 2
    CONFIRMED = 3


_STATE_TO_PUBLIC = {
    CellState.UNKNOWN: OccupancyGrid.UNKNOWN,
    CellState.FREE: OccupancyGrid.FREE,
    CellState.TENTATIVE: OccupancyGrid.UNKNOWN,
    CellState.CONFIRMED: OccupancyGrid.OCCUPIED,
}


class PersistentWorldModel:
    """Owns ``state``, ``pending_since``, and an ``occupancy_grid``
    (composed ``OccupancyGridMap``, reused unchanged for storage and
    coordinate conversion). See module docstring for the full design.
    """

    def __init__(
        self,
        resolution: float = DEFAULT_RESOLUTION,
        width_m: float = DEFAULT_WIDTH_M,
        height_m: float = DEFAULT_HEIGHT_M,
        origin_x: float = DEFAULT_ORIGIN_X,
        origin_y: float = DEFAULT_ORIGIN_Y,
        confirmation_duration: float = DEFAULT_CONFIRMATION_DURATION,
        max_range_policy: str = "clear",
        max_range_clear_distance: float = 2.0,
        occupied_timeout: float | None = None,
    ) -> None:
        """``max_range_policy`` -- what a no-return (``r >= range_max``) ray
        is evidence of. The sensor cannot distinguish "nothing within
        range" from a missed return (dark or glancing surface, dropout) or
        an unmeasured sector of a partial scan; all arrive as range_max.
            "clear"        (default, original): clear line of sight out to
                           range_max -- every traced cell observed FREE,
                           clearing TENTATIVE and CONFIRMED cells. One missed
                           return can erase a confirmed wall.
            "ignore"       no evidence at all (treated like an invalid ray).
            "limited"      as "clear", but only up to
                           ``max_range_clear_distance`` metres.
            "unknown_only" weak evidence: may turn UNKNOWN cells FREE, never
                           overrides occupied (TENTATIVE/CONFIRMED) or any
                           evidence from real returns in the same scan.
        Real returns (hits) are positive evidence under every policy:
        everything before the hit is observed free, as before.

        ``occupied_timeout`` (seconds, optional): forget occupied geometry
        that has not been HIT for longer than this. After each scan, every
        TENTATIVE or CONFIRMED cell whose most recent occupied observation is
        older than ``now - occupied_timeout`` returns to UNKNOWN (not FREE:
        not seeing something is not evidence the space is empty). This
        deliberately overrides the NOT OBSERVED != GONE invariant above,
        trading persistence (walls behind the robot or occluded ones fade
        too, and scan matching can only use recently seen geometry) for
        removal of stale clutter and noise. ``None`` (default) keeps the
        original, fully persistent behavior."""
        if confirmation_duration <= 0.0:
            raise ValueError("confirmation_duration must be > 0")
        if max_range_policy not in MAX_RANGE_POLICIES:
            raise ValueError(f"max_range_policy must be one of {MAX_RANGE_POLICIES}")
        if max_range_clear_distance <= 0.0:
            raise ValueError("max_range_clear_distance must be > 0")
        if occupied_timeout is not None and occupied_timeout <= 0.0:
            raise ValueError("occupied_timeout must be > 0 (or None)")
        self._occupied_timeout = occupied_timeout
        self._confirmation_duration = confirmation_duration
        self.max_range_policy = max_range_policy
        self._max_range_clear_distance = max_range_clear_distance

        self.occupancy_grid = OccupancyGridMap(resolution, width_m, height_m, origin_x, origin_y)
        shape = (self.occupancy_grid.grid.height, self.occupancy_grid.grid.width)
        self.state = np.full(shape, CellState.UNKNOWN, dtype=np.int8)
        self.pending_since = np.full(shape, math.nan, dtype=np.float64)
        # Time of each cell's most recent occupied observation (used only by occupied_timeout).
        self.last_hit = np.full(shape, math.nan, dtype=np.float64)

    # ------------------------------------------------------------ integrate ---

    def integrate(self, pose: Pose2D, scan: LaserScan, now: float) -> None:
        """Integrate one scan, observed from ``pose`` (world frame), as
        of time ``now`` (caller-supplied -- this method never reads a
        clock itself, so replay stays fully deterministic).

        Per-ray processing (see module docstring for the exact
        confirmation rule and the free/occupied semantics):
            - invalid (``r < range_min`` or non-finite): no effect at
              all -- neither free nor occupied evidence.
            - hit (``range_min <= r < range_max``): every cell traced
              strictly before the endpoint is directly observed free;
              the endpoint cell is an occupied observation.
            - no-detection (``r >= range_max``): every cell traced up
              to and including the endpoint (at range r) is directly
              observed free -- there is no occupied endpoint.
        Cells never traced by any ray this scan (occluded, or beyond
        every ray's own endpoint) are left completely untouched --
        this is what makes "not observed" distinct from "observed
        free" throughout this class.

        Implementation: the rays are traced exactly as before (same
        ``bresenham_line`` cells, same endpoint handling, same bounds
        rules), producing the scan's observations as one ordered event
        list -- identical to the order the original per-cell loop applied
        them in. The state transitions are then applied per CELL with
        array operations instead of one Python call per traversed cell
        (~30k per scan, which was ~95% of all SLAM processing time).
        That is exact, not approximate, because within one scan (a single
        ``now``) a cell's final state depends only on:
            - its LAST observation: free -> FREE, pending cleared;
            - if that was occupied: whether ANY free observation came
              earlier in the scan. If so, the cell went through FREE, so
              the trailing occupied run leaves it TENTATIVE since ``now``
              (repeat occupied hits at the same ``now`` never confirm,
              because ``confirmation_duration > 0``). If not, the result
              is one ``_observe_occupied`` applied to the cell's state
              before the scan (repeats are idempotent at a fixed ``now``).
        ``_observe_free``/``_observe_occupied`` remain the single-cell
        definition of these semantics; tests check the vectorized path
        against a verbatim copy of the original loop.
        """
        grid = self.occupancy_grid.grid
        height, width = grid.height, grid.width
        origin_row, origin_col = self.occupancy_grid.world_to_grid(pose.x, pose.y)

        rows: list[int] = []
        cols: list[int] = []
        occupied: list[bool] = []  # per event: True = occupied observation, False = free
        weak_rows: list[int] = []  # "unknown_only": cells seen only along no-return rays
        weak_cols: list[int] = []
        policy = self.max_range_policy
        for i in range(scan.num_rays):
            r = scan.ranges[i]
            if not math.isfinite(r) or r < scan.range_min:
                continue  # invalid: no free-space or occupied observation

            is_hit = r < scan.range_max
            if not is_hit:
                if policy == "ignore":
                    continue
                if policy == "limited":
                    r = min(r, self._max_range_clear_distance)
            angle = pose.theta + scan.angle_at(i)
            end_x = pose.x + r * math.cos(angle)
            end_y = pose.y + r * math.sin(angle)
            end_row, end_col = self.occupancy_grid.world_to_grid(end_x, end_y)

            traced = bresenham_line(origin_row, origin_col, end_row, end_col)
            if not is_hit and policy == "unknown_only":
                for row, col in traced:
                    weak_rows.append(row)
                    weak_cols.append(col)
                continue
            n = len(traced)
            for row, col in traced:
                rows.append(row)
                cols.append(col)
            occupied.extend([False] * (n - 1))
            occupied.append(is_hit)  # endpoint: occupied if a hit, else free

        strong_cells = None
        if rows:
            strong_cells = self._apply_observations(
                np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64),
                np.asarray(occupied, dtype=bool), now, height, width,
            )
        if weak_rows:
            self._apply_weak_free(np.asarray(weak_rows, dtype=np.int64), np.asarray(weak_cols, dtype=np.int64),
                                  strong_cells, height, width)
        if self._occupied_timeout is not None:
            self._forget_unseen(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64),
                                np.asarray(occupied, dtype=bool), now, height, width)
        self._rebuild_public_grid()

    def _forget_unseen(
        self, rows: np.ndarray, cols: np.ndarray, occupied: np.ndarray, now: float, height: int, width: int
    ) -> None:
        """``occupied_timeout``: stamp this scan's hits, then return occupied
        cells not hit within the timeout to UNKNOWN."""
        if rows.size:
            inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width) & occupied
            self.last_hit.reshape(-1)[(rows * width + cols)[inside]] = now
        occupied_state = (self.state == CellState.TENTATIVE) | (self.state == CellState.CONFIRMED)
        stale = occupied_state & ~(self.last_hit >= now - self._occupied_timeout)  # NaN counts as stale
        self.state[stale] = CellState.UNKNOWN
        self.pending_since[stale] = math.nan

    def _apply_weak_free(
        self, rows: np.ndarray, cols: np.ndarray, strong_cells: np.ndarray | None, height: int, width: int
    ) -> None:
        """"unknown_only" no-return evidence: UNKNOWN -> FREE, nothing else.
        Cells that real returns observed in this same scan are left to that
        evidence (so the result does not depend on ray order)."""
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        flat = np.unique((rows * width + cols)[inside])
        if strong_cells is not None and strong_cells.size:
            flat = flat[~np.isin(flat, strong_cells, assume_unique=True)]
        state = self.state.reshape(-1)
        unknown = flat[state[flat] == CellState.UNKNOWN]
        state[unknown] = CellState.FREE

    def _apply_observations(
        self, rows: np.ndarray, cols: np.ndarray, occupied: np.ndarray, now: float, height: int, width: int
    ) -> np.ndarray:
        """Apply one scan's ordered observation events (see ``integrate``).
        Out-of-bounds events are dropped individually, exactly as the
        per-cell loop skipped them. Returns the (sorted, unique) flat
        indices of every cell observed."""
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        flat = (rows * width + cols)[inside]
        occupied = occupied[inside]
        if flat.size == 0:
            return flat

        # Last event per cell: unique over the reversed sequence finds each
        # cell's final occurrence.
        cells, first_in_reversed = np.unique(flat[::-1], return_index=True)
        last_is_occupied = occupied[::-1][first_in_reversed]
        freed_anywhere = np.zeros(height * width, dtype=bool)
        freed_anywhere[flat[~occupied]] = True
        any_free = freed_anywhere[cells]

        # Flat VIEWS, so writes land in self.state / self.pending_since.
        if not (self.state.flags.c_contiguous and self.pending_since.flags.c_contiguous):
            self.state = np.ascontiguousarray(self.state)
            self.pending_since = np.ascontiguousarray(self.pending_since)
        state = self.state.reshape(-1)
        pending = self.pending_since.reshape(-1)
        before = state[cells]

        # Last observation free -> FREE (pending cleared if it was tentative).
        to_free = cells[~last_is_occupied]
        was_tentative = to_free[state[to_free] == CellState.TENTATIVE]
        pending[was_tentative] = math.nan
        state[to_free] = CellState.FREE

        # Last observation occupied, preceded by a free one -> TENTATIVE since now.
        restarted = cells[last_is_occupied & any_free]
        state[restarted] = CellState.TENTATIVE
        pending[restarted] = now

        # Only occupied observations this scan -> one _observe_occupied step.
        only_hit = last_is_occupied & ~any_free
        hit_cells = cells[only_hit]
        hit_before = before[only_hit]
        new_tentative = hit_cells[(hit_before != CellState.CONFIRMED) & (hit_before != CellState.TENTATIVE)]
        state[new_tentative] = CellState.TENTATIVE
        pending[new_tentative] = now
        tentative = hit_cells[hit_before == CellState.TENTATIVE]
        since = pending[tentative]
        confirm = tentative[~np.isnan(since) & ((now - since) >= self._confirmation_duration)]
        state[confirm] = CellState.CONFIRMED
        pending[confirm] = math.nan
        return cells

    def _observe_free(self, row: int, col: int) -> None:
        """A cell with direct, unobstructed line of sight, observed as
        free right now. Clears CONFIRMED (the only way confirmed
        geometry is ever removed, per the core invariant) and
        discards any in-progress TENTATIVE evidence (contradicted
        before confirmation)."""
        if self.state[row, col] == CellState.TENTATIVE:
            self.pending_since[row, col] = math.nan
        self.state[row, col] = CellState.FREE

    def _observe_occupied(self, row: int, col: int, now: float) -> None:
        """A cell hit by a valid ray this scan. See module docstring
        for the exact span-based confirmation rule."""
        state = self.state[row, col]
        if state == CellState.CONFIRMED:
            return  # reinforced, nothing to do
        if state != CellState.TENTATIVE:
            self.state[row, col] = CellState.TENTATIVE
            self.pending_since[row, col] = now
            return
        since = self.pending_since[row, col]
        if not math.isnan(since) and (now - since) >= self._confirmation_duration:
            self.state[row, col] = CellState.CONFIRMED
            self.pending_since[row, col] = math.nan
        # else: still tentative, still pending -- unchanged.

    def _rebuild_public_grid(self) -> None:
        cells = self.occupancy_grid.grid.cells
        for state_value, public_value in _STATE_TO_PUBLIC.items():
            cells[self.state == state_value] = public_value

    # ------------------------------------------------------------- queries ---

    def get_grid(self) -> OccupancyGrid:
        """The existing public message type -- unchanged, and already
        kept in sync with ``state`` after every ``integrate()`` call.
        Compatible with existing planning/navigation/visualization
        without any changes to those systems."""
        return self.occupancy_grid.grid

    def get_confirmed_occupied_points_world(self, center: tuple[float, float], radius: float) -> np.ndarray:
        """CONFIRMED occupied cell centers, in world coordinates, within
        ``radius`` meters of ``center`` -- never tentative, free, or
        unknown cells. Matches the ``ReferenceProvider`` protocol
        Phase 4's pipeline already expects (an ``(N, 2)`` float array),
        though this module does not import or depend on that pipeline
        in any way.
        """
        return self._points_world_by_state(center, radius, CellState.CONFIRMED)

    def get_tentative_points_world(self, center: tuple[float, float], radius: float) -> np.ndarray:
        """TENTATIVE cell centers, in world coordinates, within
        ``radius`` meters of ``center``. Phase 8, explicitly
        pre-authorized as a minimal, read-only, visualization-oriented
        accessor: tentative geometry is deliberately NOT reconstructable
        from ``get_grid()`` alone (``TENTATIVE`` maps to public
        ``UNKNOWN``, per this module's design -- see the module
        docstring), and a per-cell ``get_cell_state()`` loop over the
        whole grid every GUI frame would be far too slow for a live
        display. Read-only: does not touch ``pending_since`` or
        ``state``, and cannot confirm/clear anything.
        """
        return self._points_world_by_state(center, radius, CellState.TENTATIVE)

    def _points_world_by_state(self, center: tuple[float, float], radius: float, wanted: CellState) -> np.ndarray:
        """Shared bounding-box-then-exact-radius query behind both
        ``get_confirmed_occupied_points_world`` and
        ``get_tentative_points_world`` -- bounded to a grid-index
        bounding box around ``center`` first (never scans the whole
        grid), then filtered to the exact circular radius.
        """
        grid = self.occupancy_grid.grid
        cx, cy = center

        row_a, col_a = self.occupancy_grid.world_to_grid(cx - radius, cy - radius)
        row_b, col_b = self.occupancy_grid.world_to_grid(cx + radius, cy + radius)
        row_lo, row_hi = max(0, min(row_a, row_b)), min(grid.height - 1, max(row_a, row_b))
        col_lo, col_hi = max(0, min(col_a, col_b)), min(grid.width - 1, max(col_a, col_b))
        if row_hi < row_lo or col_hi < col_lo:
            return np.empty((0, 2), dtype=np.float64)

        sub_state = self.state[row_lo : row_hi + 1, col_lo : col_hi + 1]
        rows, cols = np.nonzero(sub_state == wanted)
        if rows.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        rows = rows + row_lo
        cols = cols + col_lo

        xs = grid.origin_x + (cols.astype(np.float64) + 0.5) * grid.resolution
        ys = grid.origin_y + (rows.astype(np.float64) + 0.5) * grid.resolution
        within = np.hypot(xs - cx, ys - cy) <= radius
        return np.column_stack((xs[within], ys[within]))

    # ---------------------------------------------------- test/diagnostic ---

    def get_cell_state(self, x: float, y: float) -> CellState:
        """The internal temporal state of the cell containing world
        point ``(x, y)`` -- exposed for direct testing of temporal
        behavior (per the approved design), and as a diagnostic hook
        a future visualization phase can use for a tentative overlay.
        """
        row, col = self.occupancy_grid.world_to_grid(x, y)
        return CellState(self.state[row, col])

    def get_pending_since(self, x: float, y: float) -> float | None:
        """``pending_since`` for the cell containing world point
        ``(x, y)``, or ``None`` if it isn't currently tentative."""
        row, col = self.occupancy_grid.world_to_grid(x, y)
        value = self.pending_since[row, col]
        return None if math.isnan(value) else float(value)
