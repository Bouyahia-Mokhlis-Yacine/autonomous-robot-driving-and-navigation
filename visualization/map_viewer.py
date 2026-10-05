"""Occupancy-grid -> image rendering, independent of any GUI toolkit.

Renders at one pixel per grid cell (the GUI layer scales the resulting
``PIL.Image`` to the desired on-screen size); vectorized with numpy so
even large grids render fast enough for a live redraw loop. Producing
plain ``PIL.Image`` objects (rather than drawing into a Tk canvas
directly) keeps this fully testable by inspecting pixel values, with no
display required.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from common.messages import OccupancyGrid

DEFAULT_PALETTE: dict[int, tuple[int, int, int]] = {
    OccupancyGrid.UNKNOWN: (128, 128, 128),
    OccupancyGrid.FREE: (255, 255, 255),
    OccupancyGrid.OCCUPIED: (0, 0, 0),
}

# Phase 8: black-background, RViz-like scheme for the new world-model
# GUI (visualization/gui.py's WorldModelGUI) -- a separate named
# palette, NOT a change to DEFAULT_PALETTE, specifically so the
# existing white-background NavigationGUI (which calls
# render_grid_to_image() with no explicit palette, i.e. relies on
# DEFAULT_PALETTE) keeps rendering exactly as it always has. FREE is
# pure black so it blends into the canvas background (bg="black");
# UNKNOWN is a dark-but-distinct gray so unexplored area stays visible
# against that background; OCCUPIED is near-white for maximum contrast.
DARK_PALETTE: dict[int, tuple[int, int, int]] = {
    OccupancyGrid.UNKNOWN: (40, 40, 40),
    OccupancyGrid.FREE: (0, 0, 0),
    OccupancyGrid.OCCUPIED: (230, 230, 230),
}

DEFAULT_INFLATION_COLOR: tuple[int, int, int] = (255, 165, 0)  # orange


def render_grid_to_image(
    grid: OccupancyGrid, palette: dict[int, tuple[int, int, int]] | None = None
) -> Image.Image:
    """Render ``grid`` as an RGB image, one pixel per cell.

    Row 0 (world y = origin_y) is placed at the *bottom* of the image so
    the image's vertical orientation matches the world frame (y up);
    callers displaying it on a screen-space canvas (y down) must flip it
    or rely on ``visualization.overlays.world_to_screen`` for placement,
    exactly as they would for any other world-frame content.
    """
    colors = palette or DEFAULT_PALETTE
    height, width = grid.cells.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for value, color in colors.items():
        rgb[grid.cells == value] = color

    rgb = np.flipud(rgb)  # row 0 (origin_y) at the bottom
    return Image.fromarray(rgb, mode="RGB")


def render_inflation_overlay_image(
    raw_grid: OccupancyGrid,
    inflated_grid: OccupancyGrid,
    color: tuple[int, int, int] = DEFAULT_INFLATION_COLOR,
    alpha: int = 160,
) -> Image.Image:
    """Transparent RGBA overlay highlighting cells occupied only *after* inflation.

    This is the "planning/inflated obstacles" layer the spec asks to be
    shown distinctly from the raw map: cells that were free/unknown
    before inflation but became occupied afterward (the added safety
    margin), not the original obstacle cells themselves.
    """
    if raw_grid.cells.shape != inflated_grid.cells.shape:
        raise ValueError("raw_grid and inflated_grid must have the same shape")

    height, width = raw_grid.cells.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    inflated_only = (inflated_grid.cells == OccupancyGrid.OCCUPIED) & (
        raw_grid.cells != OccupancyGrid.OCCUPIED
    )
    rgba[inflated_only] = (*color, alpha)

    rgba = np.flipud(rgba)
    return Image.fromarray(rgba, mode="RGBA")
