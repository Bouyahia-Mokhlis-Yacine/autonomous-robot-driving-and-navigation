"""2D navigation GUI (tkinter).

Displays the occupancy map, inflated/planning obstacles, robot pose and
heading, LiDAR points, goal, global path, local target, and navigation
state. Left-click sets a destination; right-click cancels it.

This module is a thin wiring layer: it reads state from
``NavigationManager`` and an injected ``GuiDataSource`` and calls
``set_goal``/``cancel_goal`` on click -- it does not implement any
navigation, planning, or simulation logic itself, and holds no
navigation state of its own beyond what it needs to render.

Phase 8 adds a second, independent GUI (``WorldModelGUI``, at the
bottom of this file) for the newer scan-matching-pipeline /
persistent-world-model architecture (Phases 1-7), which has no
``NavigationManager`` or goal-setting concept at all. ``NavigationGUI``
above is completely unmodified and unaffected by that addition.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import numpy as np
from PIL import ImageTk

from common.messages import Goal, LaserScan, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from navigation.navigation_manager import NavigationManager
from slam.scan_matching_pipeline import BootstrapStatus
from visualization.gui_state import (  # noqa: F401 -- re-exported; moved there so headless code avoids tkinter
    GuiDataSource,
    LatestStateBox,
    VisualizationState,
    build_visualization_state,
)
from visualization.map_viewer import DARK_PALETTE, render_grid_to_image, render_inflation_overlay_image
from visualization.overlays import (
    ViewTransform,
    heading_line_endpoints,
    scan_hit_points_world,
    screen_to_world,
    world_to_screen,
)

ROBOT_COLOR = "#1f77b4"
HEADING_COLOR = "#1f77b4"
LIDAR_COLOR = "#d62728"
PATH_COLOR = "#2ca02c"
GOAL_COLOR = "#9467bd"
LOCAL_TARGET_COLOR = "#ff7f0e"


class NavigationGUI:
    def __init__(
        self,
        nav_manager: NavigationManager,
        data_source: GuiDataSource,
        scale: float = 40.0,
        dt: float = 0.05,
        step_callback: Callable[[float], None] | None = None,
    ) -> None:
        self.nav_manager = nav_manager
        self.data_source = data_source
        self.dt = dt
        self.step_callback = step_callback

        static_grid = data_source.get_static_map().grid
        world_width = static_grid.width * static_grid.resolution
        world_height = static_grid.height * static_grid.resolution
        canvas_width = max(1, int(round(world_width * scale)))
        canvas_height = max(1, int(round(world_height * scale)))
        self.view = ViewTransform(
            world_origin_x=static_grid.origin_x,
            world_origin_y=static_grid.origin_y,
            scale=scale,
            canvas_height=canvas_height,
        )

        self.root = tk.Tk()
        self.root.title("Autonomous Navigation")

        self.canvas = tk.Canvas(
            self.root, width=canvas_width, height=canvas_height, bg="white", highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var, anchor="w", font=("TkFixedFont",)).pack(fill="x")

        # Tk garbage-collects PhotoImages with no surviving Python reference,
        # even while still displayed -- these keep them alive.
        self._map_photo: ImageTk.PhotoImage | None = None
        self._overlay_photo: ImageTk.PhotoImage | None = None

        self.redraw()
        if self.step_callback is not None:
            self._schedule_tick()

    # ------------------------------------------------------------- input ---

    def _on_left_click(self, event: tk.Event) -> None:
        x, y = screen_to_world(event.x, event.y, self.view)
        planning_map = self.data_source.get_planning_map()
        self.nav_manager.set_goal(Goal(x=x, y=y), planning_map)
        self.redraw()

    def _on_right_click(self, event: tk.Event) -> None:
        self.nav_manager.cancel_goal()
        self.redraw()

    # ------------------------------------------------------------- ticking ---

    def _schedule_tick(self) -> None:
        self.root.after(max(1, int(self.dt * 1000)), self._tick)

    def _tick(self) -> None:
        if self.step_callback is not None:
            self.step_callback(self.dt)
        self.redraw()
        self._schedule_tick()

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.root.destroy()

    # ------------------------------------------------------------- drawing ---

    def redraw(self) -> None:
        self.canvas.delete("all")

        static_grid = self.data_source.get_static_map().grid
        planning_grid = self.data_source.get_planning_map().grid
        pose = self.data_source.get_pose()
        scan = self.data_source.get_scan()

        self._draw_map_layers(static_grid, planning_grid)
        if scan is not None:
            self._draw_scan(pose, scan)
        self._draw_path()
        self._draw_local_target()
        self._draw_goal()
        self._draw_robot(pose)
        self._update_status(pose)

    def _draw_map_layers(self, static_grid, planning_grid) -> None:
        world_w = static_grid.width * static_grid.resolution
        world_h = static_grid.height * static_grid.resolution
        px_w = max(1, int(round(world_w * self.view.scale)))
        px_h = max(1, int(round(world_h * self.view.scale)))

        map_image = render_grid_to_image(static_grid).resize((px_w, px_h))
        self._map_photo = ImageTk.PhotoImage(map_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self._map_photo)

        overlay_image = render_inflation_overlay_image(static_grid, planning_grid).resize((px_w, px_h))
        self._overlay_photo = ImageTk.PhotoImage(overlay_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self._overlay_photo)

    def _draw_scan(self, pose: Pose2D, scan: LaserScan) -> None:
        for x, y in scan_hit_points_world(pose, scan):
            px, py = world_to_screen(x, y, self.view)
            self.canvas.create_oval(px - 1, py - 1, px + 1, py + 1, fill=LIDAR_COLOR, outline="")

    def _draw_path(self) -> None:
        path = self.nav_manager.get_path()
        if path is None or len(path.points) < 2:
            return
        screen_points = [world_to_screen(p.x, p.y, self.view) for p in path.points]
        flat = [coord for point in screen_points for coord in point]
        self.canvas.create_line(*flat, fill=PATH_COLOR, width=2)

    def _draw_local_target(self) -> None:
        target = self.nav_manager.get_local_target()
        if target is None:
            return
        px, py = world_to_screen(target.x, target.y, self.view)
        r = 5
        self.canvas.create_oval(px - r, py - r, px + r, py + r, outline=LOCAL_TARGET_COLOR, width=2)

    def _draw_goal(self) -> None:
        goal = self.nav_manager.get_goal()
        if goal is None:
            return
        px, py = world_to_screen(goal.x, goal.y, self.view)
        r = 7
        self.canvas.create_line(px - r, py - r, px + r, py + r, fill=GOAL_COLOR, width=2)
        self.canvas.create_line(px - r, py + r, px + r, py - r, fill=GOAL_COLOR, width=2)

    def _draw_robot(self, pose: Pose2D) -> None:
        px, py = world_to_screen(pose.x, pose.y, self.view)
        r = 6
        self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=ROBOT_COLOR, outline="")

        (sx, sy), (ex, ey) = heading_line_endpoints(pose, length=0.4)
        spx, spy = world_to_screen(sx, sy, self.view)
        epx, epy = world_to_screen(ex, ey, self.view)
        self.canvas.create_line(spx, spy, epx, epy, fill=HEADING_COLOR, width=2)

    def _update_status(self, pose: Pose2D) -> None:
        goal = self.nav_manager.get_goal()
        goal_text = f"({goal.x:.2f}, {goal.y:.2f})" if goal is not None else "none"
        self.status_var.set(
            f"state={self.nav_manager.state.value}  "
            f"pose=({pose.x:.2f}, {pose.y:.2f}, {pose.theta:.2f})  "
            f"goal={goal_text}  "
            f"[left-click: set goal, right-click: cancel]"
        )


# =========================================================================
# Phase 8: WorldModelGUI -- live visualization for the scan-matching-
# pipeline / persistent-world-model architecture (Phases 1-7). No
# NavigationManager, no goals, no path -- purely a read-only viewer.
# =========================================================================

WORLD_MODEL_ROBOT_COLOR = "#4da3ff"      # localized
WORLD_MODEL_ROBOT_LOST_COLOR = "#e74c3c"  # localized == False
TENTATIVE_COLOR = "#ffb703"               # amber, distinct from OCCUPIED's near-white
TRAJECTORY_COLOR = "#3a5a7a"              # subtle, must not compete with the map
STATUS_OK_COLOR = "#7CFC00"
STATUS_WARN_COLOR = "#ffb703"
STATUS_BAD_COLOR = "#e74c3c"


class WorldModelGUI:
    """Live, read-only 2D view of a ``PersistentWorldModel`` +
    ``ScanMatchingPipeline`` (Phases 1-7) -- an RViz-like viewer for
    development/debugging. Never calls ``register()``,
    ``world_model.integrate()``, or any other pipeline/world-model
    mutator; consumes only whatever ``VisualizationState`` a
    ``LatestStateBox`` currently holds (see module-level docs above).

    Camera: pan via left-click-drag, zoom via the mouse wheel (centered
    on the cursor), reset/fit via the "f" key or the Reset button.
    ``self._view`` is reassigned (never mutated in place) on every
    pan/zoom -- ``overlays.ViewTransform`` itself stays the existing,
    unmodified, immutable dataclass.
    """

    def __init__(
        self,
        state_box: LatestStateBox,
        scale: float = 30.0,
        dt: float = 0.1,
        world_size_hint_m: float = 20.0,
    ) -> None:
        self.state_box = state_box
        self.dt = dt

        canvas_px = max(1, int(round(world_size_hint_m * scale)))
        self._view = ViewTransform(
            world_origin_x=-world_size_hint_m / 2.0,
            world_origin_y=-world_size_hint_m / 2.0,
            scale=scale,
            canvas_height=canvas_px,
        )

        self.root = tk.Tk()
        self.root.title("World Model Viewer")

        self.canvas = tk.Canvas(self.root, width=canvas_px, height=canvas_px, bg="black", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_drag)
        self.canvas.bind("<MouseWheel>", self._on_zoom_windows)  # Windows/Mac
        self.canvas.bind("<Button-4>", self._on_zoom_linux_up)   # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_zoom_linux_down)  # Linux scroll down
        self.canvas.bind("<KeyPress-f>", lambda _event: self.reset_view())
        self.canvas.focus_set()

        status_frame = tk.Frame(self.root, bg="black")
        status_frame.pack(side="right", fill="y")
        self.status_var = tk.StringVar()
        tk.Label(
            status_frame, textvariable=self.status_var, anchor="nw", justify="left",
            font=("TkFixedFont",), fg="white", bg="black", padx=8, pady=8,
        ).pack(side="top", fill="both", expand=True)
        tk.Button(status_frame, text="Reset view (f)", command=self.reset_view).pack(side="bottom", fill="x")

        self._pan_last: tuple[int, int] | None = None
        self._map_photo: ImageTk.PhotoImage | None = None
        self._last_pose: Pose2D = Pose2D(x=0.0, y=0.0, theta=0.0)

        self._schedule_tick()

    # ------------------------------------------------------------- camera ---

    def reset_view(self) -> None:
        """Recenter on the latest known robot pose at the construction
        scale -- "fit map / reset view", per the approved design."""
        scale = self._view.scale
        half_span_m = (self.canvas.winfo_width() or self._view.canvas_height) / scale / 2.0
        self._view = ViewTransform(
            world_origin_x=self._last_pose.x - half_span_m,
            world_origin_y=self._last_pose.y - half_span_m,
            scale=scale,
            canvas_height=self._view.canvas_height,
        )

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_last = (event.x, event.y)

    def _on_pan_drag(self, event: tk.Event) -> None:
        if self._pan_last is None:
            return
        dx_px = event.x - self._pan_last[0]
        dy_px = event.y - self._pan_last[1]
        self._pan_last = (event.x, event.y)
        # Screen +y is down; world +y is up (matches world_to_screen's
        # own convention) -- dragging the mouse down must move the
        # visible world DOWN too, i.e. increase world_origin_y.
        self._view = ViewTransform(
            world_origin_x=self._view.world_origin_x - dx_px / self._view.scale,
            world_origin_y=self._view.world_origin_y + dy_px / self._view.scale,
            scale=self._view.scale,
            canvas_height=self._view.canvas_height,
        )

    def _zoom(self, factor: float, cursor_x: int, cursor_y: int) -> None:
        world_x, world_y = screen_to_world(cursor_x, cursor_y, self._view)
        new_scale = max(2.0, min(500.0, self._view.scale * factor))
        # Keep the world point under the cursor fixed on screen.
        new_origin_x = world_x - cursor_x / new_scale
        new_origin_y = world_y - (self._view.canvas_height - cursor_y) / new_scale
        self._view = ViewTransform(
            world_origin_x=new_origin_x, world_origin_y=new_origin_y,
            scale=new_scale, canvas_height=self._view.canvas_height,
        )

    def _on_zoom_windows(self, event: tk.Event) -> None:
        self._zoom(1.15 if event.delta > 0 else 1.0 / 1.15, event.x, event.y)

    def _on_zoom_linux_up(self, event: tk.Event) -> None:
        self._zoom(1.15, event.x, event.y)

    def _on_zoom_linux_down(self, event: tk.Event) -> None:
        self._zoom(1.0 / 1.15, event.x, event.y)

    # ------------------------------------------------------------- ticking ---

    def _schedule_tick(self) -> None:
        self.root.after(max(1, int(self.dt * 1000)), self._tick)

    def _tick(self) -> None:
        state = self.state_box.get()
        if state is not None:
            self._last_pose = state.robot_pose
            self.redraw(state)
        self._schedule_tick()

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.root.destroy()

    # ------------------------------------------------------------- drawing ---

    def redraw(self, state: VisualizationState) -> None:
        self.canvas.delete("all")
        self._draw_map(state.grid)
        self._draw_tentative(state.tentative_points_world)
        self._draw_trajectory(state.trajectory)
        self._draw_scan(state.current_scan_points_world)
        self._draw_robot(state.robot_pose, state.localized)
        self._update_status(state)

    def _draw_map(self, grid) -> None:
        world_w = grid.width * grid.resolution
        world_h = grid.height * grid.resolution
        px_w = max(1, int(round(world_w * self._view.scale)))
        px_h = max(1, int(round(world_h * self._view.scale)))
        image = render_grid_to_image(grid, palette=DARK_PALETTE).resize((px_w, px_h))
        self._map_photo = ImageTk.PhotoImage(image)
        origin_px, origin_py = world_to_screen(grid.origin_x, grid.origin_y, self._view)
        # world_to_screen gives the BOTTOM-left corner in screen space
        # (world +y is up, screen +y is down) -- the image's anchor is
        # its top-left, which is px_h above that point.
        self.canvas.create_image(origin_px, origin_py - px_h, anchor="nw", image=self._map_photo)

    def _draw_tentative(self, points_world: np.ndarray) -> None:
        r = 2
        for x, y in points_world:
            px, py = world_to_screen(float(x), float(y), self._view)
            self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=TENTATIVE_COLOR, outline="")

    def _draw_scan(self, points_world: list[tuple[float, float]]) -> None:
        r = 1
        for x, y in points_world:
            px, py = world_to_screen(x, y, self._view)
            self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=LIDAR_COLOR, outline="")

    def _draw_trajectory(self, trajectory: tuple[tuple[float, float], ...]) -> None:
        if len(trajectory) < 2:
            return
        flat = [coord for x, y in trajectory for coord in world_to_screen(x, y, self._view)]
        self.canvas.create_line(*flat, fill=TRAJECTORY_COLOR, width=1)

    def _draw_robot(self, pose: Pose2D, localized: bool) -> None:
        color = WORLD_MODEL_ROBOT_COLOR if localized else WORLD_MODEL_ROBOT_LOST_COLOR
        px, py = world_to_screen(pose.x, pose.y, self._view)
        r = 6
        self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=color, outline="")
        (sx, sy), (ex, ey) = heading_line_endpoints(pose, length=0.4)
        spx, spy = world_to_screen(sx, sy, self._view)
        epx, epy = world_to_screen(ex, ey, self._view)
        self.canvas.create_line(spx, spy, epx, epy, fill=color, width=2)

    def _update_status(self, state: VisualizationState) -> None:
        if state.bootstrap_status == BootstrapStatus.BOOTSTRAPPING:
            bootstrap_line = "Bootstrap: BOOTSTRAPPING -- keep robot stationary"
        elif state.bootstrap_status == BootstrapStatus.FAILED:
            bootstrap_line = "Bootstrap: FAILED"
        else:
            bootstrap_line = "Bootstrap: COMPLETE"
        localization_line = f"Localization: {'LOCALIZED' if state.localized else 'NOT LOCALIZED'}"

        lines = [
            bootstrap_line,
            localization_line,
            "",
            "Pose:",
            f"  X:     {state.robot_pose.x:8.3f}",
            f"  Y:     {state.robot_pose.y:8.3f}",
            f"  Theta: {state.robot_pose.theta:8.3f}",
            "",
            f"Failures: {state.consecutive_failures}",
            f"Scan t:   {state.scan_timestamp:.3f}",
        ]
        if state.registration_accepted is not None:
            lines += [
                "",
                "Registration:",
                f"  {'accepted' if state.registration_accepted else 'rejected'}",
                f"  Correspondences: {state.correspondence_count}",
                f"  Residual: {state.residual:.4f}" if state.residual is not None else "  Residual: n/a",
            ]
        if state.processing_time_ms is not None:
            lines.append(f"Processing: {state.processing_time_ms:.1f} ms")

        self.status_var.set("\n".join(lines))
