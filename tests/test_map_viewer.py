"""Phase 11 tests: occupancy-grid -> image rendering (no display required)."""

from __future__ import annotations

import numpy as np
import pytest

from common.messages import OccupancyGrid
from visualization.map_viewer import (
    DEFAULT_INFLATION_COLOR,
    DEFAULT_PALETTE,
    render_grid_to_image,
    render_inflation_overlay_image,
)


def make_grid(cells: np.ndarray, resolution: float = 1.0) -> OccupancyGrid:
    height, width = cells.shape
    return OccupancyGrid(resolution=resolution, width=width, height=height, origin_x=0.0, origin_y=0.0, cells=cells)


def test_render_grid_image_size_matches_grid_dimensions() -> None:
    cells = np.full((5, 8), OccupancyGrid.UNKNOWN, dtype=np.int8)
    grid = make_grid(cells)
    image = render_grid_to_image(grid)
    assert image.size == (8, 5)  # PIL size is (width, height)
    assert image.mode == "RGB"


def test_render_grid_image_colors_match_palette() -> None:
    cells = np.array(
        [[OccupancyGrid.FREE, OccupancyGrid.OCCUPIED], [OccupancyGrid.UNKNOWN, OccupancyGrid.FREE]],
        dtype=np.int8,
    )
    grid = make_grid(cells)
    image = render_grid_to_image(grid)
    pixels = np.array(image)

    for row in range(2):
        for col in range(2):
            # Image row 0 = world row (height-1) since row 0 (origin_y) is
            # placed at the bottom of the image.
            image_row = (grid.height - 1) - row
            expected_value = cells[row, col]
            assert tuple(pixels[image_row, col]) == DEFAULT_PALETTE[expected_value]


def test_render_grid_image_custom_palette() -> None:
    cells = np.full((2, 2), OccupancyGrid.OCCUPIED, dtype=np.int8)
    grid = make_grid(cells)
    custom = {OccupancyGrid.OCCUPIED: (255, 0, 0), OccupancyGrid.FREE: (0, 255, 0), OccupancyGrid.UNKNOWN: (0, 0, 255)}
    image = render_grid_to_image(grid, palette=custom)
    pixels = np.array(image)
    assert tuple(pixels[0, 0]) == (255, 0, 0)


def test_render_grid_row0_placed_at_image_bottom() -> None:
    # Row 0 = occupied (should render at the bottom); last row = free (top).
    cells = np.full((4, 3), OccupancyGrid.UNKNOWN, dtype=np.int8)
    cells[0, :] = OccupancyGrid.OCCUPIED
    cells[-1, :] = OccupancyGrid.FREE
    grid = make_grid(cells)
    pixels = np.array(render_grid_to_image(grid))

    assert tuple(pixels[-1, 0]) == DEFAULT_PALETTE[OccupancyGrid.OCCUPIED]  # bottom of image
    assert tuple(pixels[0, 0]) == DEFAULT_PALETTE[OccupancyGrid.FREE]       # top of image


def test_inflation_overlay_highlights_only_newly_occupied_cells() -> None:
    raw = np.full((3, 3), OccupancyGrid.FREE, dtype=np.int8)
    raw[1, 1] = OccupancyGrid.OCCUPIED
    inflated = raw.copy()
    inflated[0, 1] = OccupancyGrid.OCCUPIED  # newly occupied by inflation
    inflated[1, 0] = OccupancyGrid.OCCUPIED  # newly occupied by inflation

    raw_grid = make_grid(raw)
    inflated_grid = make_grid(inflated)
    overlay = render_inflation_overlay_image(raw_grid, inflated_grid)
    pixels = np.array(overlay)

    assert overlay.mode == "RGBA"
    # The original obstacle cell (1,1) -> image row (3-1-1)=1, col 1: must stay transparent.
    assert pixels[1, 1][3] == 0
    # A newly-inflated cell (0,1) -> image row (3-1-0)=2, col 1: must be colored.
    assert tuple(pixels[2, 1][:3]) == DEFAULT_INFLATION_COLOR
    assert pixels[2, 1][3] > 0


def test_inflation_overlay_fully_transparent_when_grids_match() -> None:
    cells = np.full((3, 3), OccupancyGrid.FREE, dtype=np.int8)
    grid = make_grid(cells)
    overlay = render_inflation_overlay_image(grid, grid)
    pixels = np.array(overlay)
    assert np.all(pixels[:, :, 3] == 0)


def test_inflation_overlay_rejects_mismatched_shapes() -> None:
    small = make_grid(np.full((2, 2), OccupancyGrid.FREE, dtype=np.int8))
    big = make_grid(np.full((3, 3), OccupancyGrid.FREE, dtype=np.int8))
    with pytest.raises(ValueError):
        render_inflation_overlay_image(small, big)
