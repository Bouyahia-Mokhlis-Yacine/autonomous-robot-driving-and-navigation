"""Tk-free GUI state: the data the GUIs read, kept out of ``visualization.gui``
so headless code (the robot runtime on the Pi, the live walkthrough's
processing thread) can build it without importing tkinter/PIL.ImageTk.
``visualization.gui`` re-exports every name here."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from common.messages import LaserScan, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from mapping.persistent_world_model import PersistentWorldModel
from slam.scan_matching_pipeline import BootstrapStatus, PipelineResult
from visualization.overlays import scan_hit_points_world

@dataclass
class GuiDataSource:
    """Read-only data the GUI needs but doesn't own -- supplied by whatever
    is driving the simulation/hardware (kept out of gui.py deliberately)."""

    get_static_map: Callable[[], OccupancyGridMap]
    get_planning_map: Callable[[], OccupancyGridMap]
    get_pose: Callable[[], Pose2D]
    get_scan: Callable[[], LaserScan | None]


@dataclass(frozen=True)
class VisualizationState:
    """Everything one WorldModelGUI redraw needs, snapshotted once per
    processed scan. Strictly read-only data -- nothing in this
    dataclass, ``build_visualization_state()``, or ``WorldModelGUI``
    ever calls ``register()``, ``world_model.integrate()``, or any
    other pipeline/world-model mutator; see ``build_visualization_state``.

    ``robot_pose`` is always the pipeline's AUTHORITATIVE pose --
    ``PipelineResult.pose`` (last accepted pose, frozen on any
    rejection; never a seed or diagnostic extrapolation) -- per the
    approved design's explicit truth-model requirement.
    """

    grid: object  # common.messages.OccupancyGrid
    tentative_points_world: np.ndarray
    current_scan_points_world: list[tuple[float, float]]
    robot_pose: Pose2D
    trajectory: tuple[tuple[float, float], ...]
    bootstrap_status: BootstrapStatus
    localized: bool
    consecutive_failures: int
    scan_timestamp: float
    registration_accepted: bool | None
    correspondence_count: int | None
    residual: float | None
    processing_time_ms: float | None


def build_visualization_state(
    world_model: PersistentWorldModel,
    result: PipelineResult,
    scan: LaserScan,
    trajectory: tuple[tuple[float, float], ...],
    processing_time_ms: float | None = None,
) -> VisualizationState:
    """Pure function: pipeline/world-model outputs -> one
    ``VisualizationState`` snapshot. Read-only throughout -- only ever
    calls ``get_grid()``/``get_tentative_points_world()`` (both
    read-only queries) on ``world_model``, and reads (never writes)
    ``result``/``scan``. The caller (a scan-processing loop, never the
    GUI itself) is responsible for calling ``pipeline.process_scan()``
    beforehand and for maintaining ``trajectory`` (append ``result.pose``
    only when ``result.accepted`` is True -- never a rejected/seed pose).
    """
    pose = result.pose  # PipelineResult's own documented contract: always the authoritative pose

    grid = world_model.get_grid()
    center_x = grid.origin_x + grid.width * grid.resolution / 2.0
    center_y = grid.origin_y + grid.height * grid.resolution / 2.0
    full_map_radius = math.hypot(grid.width * grid.resolution, grid.height * grid.resolution)
    tentative_points = world_model.get_tentative_points_world((center_x, center_y), full_map_radius)

    # Reuses overlays.scan_hit_points_world (unchanged, already tested)
    # for the local-scan -> world-frame transform -- the exact same
    # convention (world = R(theta) @ local + [x, y]) used everywhere
    # else in this project, applied here with the AUTHORITATIVE pose.
    current_scan_points_world = scan_hit_points_world(pose, scan)

    reg = result.registration_result
    return VisualizationState(
        grid=grid,
        tentative_points_world=tentative_points,
        current_scan_points_world=current_scan_points_world,
        robot_pose=pose,
        trajectory=trajectory,
        bootstrap_status=result.bootstrap_status,
        localized=result.localized,
        consecutive_failures=result.consecutive_failures,
        scan_timestamp=scan.timestamp,
        registration_accepted=(reg.converged if reg is not None else None),
        correspondence_count=(reg.correspondence_count if reg is not None else None),
        residual=(reg.residual if reg is not None else None),
        processing_time_ms=processing_time_ms,
    )


class LatestStateBox:
    """Single-slot, lock-free latest-state handoff between a scan-
    processing thread/loop and the GUI's own independent redraw tick --
    the same pattern already established elsewhere in this project
    (``real_sensor_walkthrough.py``'s ``latest_scan_box``): a plain
    single-element list, safe under CPython's GIL for this exact
    single-writer/single-reader access pattern, with no lock needed.

    The GUI always reads only whatever is currently in the slot and
    never queues or replays older states -- if several scans are
    processed while the GUI is busy on one redraw, ``set()`` simply
    overwrites the slot each time, so the next GUI tick renders only
    the newest one. Nothing accumulates.
    """

    def __init__(self) -> None:
        self._slot: list[VisualizationState | None] = [None]

    def set(self, state: VisualizationState) -> None:
        self._slot[0] = state

    def get(self) -> VisualizationState | None:
        return self._slot[0]
