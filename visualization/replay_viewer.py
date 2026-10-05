"""Phase 12.5: deterministic visual replay for a recorded
``ClosedLoopSimulation`` run.

Reuses the existing rendering primitives (``visualization.map_viewer``,
``visualization.overlays``) exactly as ``visualization.gui`` does -- this
is a new tkinter *shell*, not a new visualization architecture. It is
driven by an already-computed ``list[simulation.diagnostics.ReplayFrame]``
instead of a live callback: ``PersistentWorldModel.integrate()`` is too
slow (~0.2s/scan) for responsive interactive stepping against a live
simulation (see ``simulation/diagnostics.py``'s module docstring), so the
simulation is run once, fully, beforehand, and this module only ever
scrubs an index into that already-computed list. No simulation or
navigation logic runs anywhere in this module.

Ground truth vs. estimated: each ``ReplayFrame`` carries both
``ground_truth_pose_slam_frame`` (ground truth, translated into the SLAM
frame for display only) and ``estimated_pose`` (what the navigation stack
actually used). Both are drawn, in different colors, purely for human
comparison -- this module has no connection to the navigation stack at
all, so there is no way for it to leak ground truth into navigation
decisions even in principle.
"""

from __future__ import annotations

import tkinter as tk

from PIL import ImageTk

from simulation.diagnostics import ReplayFrame
from visualization.map_viewer import render_grid_to_image, render_inflation_overlay_image
from visualization.overlays import ViewTransform, heading_line_endpoints, world_to_screen

GROUND_TRUTH_COLOR = "#2ca02c"
ESTIMATED_COLOR = "#1f77b4"
PATH_COLOR = "#2ca02c"
RAW_PATH_COLOR = "#ff9896"  # light red/pink, distinct from the final path -- Phase 14
GOAL_COLOR = "#9467bd"
LOCAL_TARGET_COLOR = "#ff7f0e"
BLOCKED_COLOR = "#e74c3c"


class ReplayViewer:
    """Play / Pause / Step / Reset / Jump-to-failure through a
    pre-recorded ``list[ReplayFrame]``."""

    def __init__(
        self,
        frames: list[ReplayFrame],
        failure_tick: int | None = None,
        scale: float = 40.0,
        dt_ms: int = 100,
    ) -> None:
        if not frames:
            raise ValueError("frames must be non-empty")
        self.frames = frames
        self.failure_tick = failure_tick
        self.dt_ms = dt_ms
        self.index = 0
        self.playing = False

        first_grid = frames[0].raw_grid
        world_width = first_grid.width * first_grid.resolution
        world_height = first_grid.height * first_grid.resolution
        canvas_width = max(1, int(round(world_width * scale)))
        canvas_height = max(1, int(round(world_height * scale)))
        self.view = ViewTransform(
            world_origin_x=first_grid.origin_x, world_origin_y=first_grid.origin_y,
            scale=scale, canvas_height=canvas_height,
        )

        self.root = tk.Tk()
        self.root.title("Phase 12.5 -- closed-loop simulation replay")

        self.canvas = tk.Canvas(
            self.root, width=canvas_width, height=canvas_height, bg="white", highlightthickness=0
        )
        self.canvas.pack(side="top")

        controls = tk.Frame(self.root)
        controls.pack(side="top", fill="x")
        tk.Button(controls, text="|<< Reset", command=self.reset).pack(side="left")
        tk.Button(controls, text="< Step", command=self.step_backward).pack(side="left")
        self.play_button = tk.Button(controls, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left")
        tk.Button(controls, text="Step >", command=self.step_forward).pack(side="left")
        if failure_tick is not None:
            tk.Button(controls, text="Jump to failure", command=self.jump_to_failure).pack(side="left")

        self.status_var = tk.StringVar()
        tk.Label(
            self.root, textvariable=self.status_var, anchor="w", justify="left", font=("TkFixedFont",)
        ).pack(fill="x")

        self._map_photo: ImageTk.PhotoImage | None = None
        self._overlay_photo: ImageTk.PhotoImage | None = None

        self.redraw()

    # ------------------------------------------------------------- controls ---

    def reset(self) -> None:
        self.playing = False
        self.play_button.config(text="Play")
        self.index = 0
        self.redraw()

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self._schedule_tick()

    def step_forward(self) -> None:
        self.playing = False
        self.play_button.config(text="Play")
        self.index = min(self.index + 1, len(self.frames) - 1)
        self.redraw()

    def step_backward(self) -> None:
        self.playing = False
        self.play_button.config(text="Play")
        self.index = max(self.index - 1, 0)
        self.redraw()

    def jump_to_failure(self) -> None:
        if self.failure_tick is None:
            return
        self.playing = False
        self.play_button.config(text="Play")
        for i, frame in enumerate(self.frames):
            if frame.tick >= self.failure_tick:
                self.index = i
                break
        self.redraw()

    def _schedule_tick(self) -> None:
        if self.playing:
            self.root.after(self.dt_ms, self._play_tick)

    def _play_tick(self) -> None:
        if not self.playing:
            return
        if self.index >= len(self.frames) - 1:
            self.playing = False
            self.play_button.config(text="Play")
            self.redraw()
            return
        self.index += 1
        self.redraw()
        self._schedule_tick()

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.root.destroy()

    # -------------------------------------------------------------- drawing ---

    def redraw(self) -> None:
        frame = self.frames[self.index]
        self.canvas.delete("all")

        world_w = frame.raw_grid.width * frame.raw_grid.resolution
        world_h = frame.raw_grid.height * frame.raw_grid.resolution
        px_w = max(1, int(round(world_w * self.view.scale)))
        px_h = max(1, int(round(world_h * self.view.scale)))

        map_image = render_grid_to_image(frame.raw_grid).resize((px_w, px_h))
        self._map_photo = ImageTk.PhotoImage(map_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self._map_photo)

        overlay_image = render_inflation_overlay_image(frame.raw_grid, frame.planning_grid).resize((px_w, px_h))
        self._overlay_photo = ImageTk.PhotoImage(overlay_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self._overlay_photo)

        self._draw_path(frame)
        self._draw_local_target(frame)
        self._draw_goal(frame)
        self._draw_robot(frame.ground_truth_pose_slam_frame, GROUND_TRUTH_COLOR)
        self._draw_robot(frame.estimated_pose, ESTIMATED_COLOR)

        is_failed = self.failure_tick is not None and frame.tick >= self.failure_tick
        if is_failed:
            self.canvas.create_rectangle(1, 1, px_w - 1, px_h - 1, outline=BLOCKED_COLOR, width=4)
            self.canvas.create_text(
                10, 10, anchor="nw", text="BLOCKED", fill=BLOCKED_COLOR, font=("TkFixedFont", 16, "bold")
            )

        self._update_status(frame, is_failed)

    def _draw_path(self, frame: ReplayFrame) -> None:
        # Phase 14: the raw A* path (dashed, orange) drawn first/underneath,
        # so the final executed path (solid, green) is easy to compare
        # against -- makes LOS/smoothing corner-cutting visible at a glance.
        if len(frame.raw_path_points) >= 2:
            screen_points = [world_to_screen(p.x, p.y, self.view) for p in frame.raw_path_points]
            flat = [coord for point in screen_points for coord in point]
            self.canvas.create_line(*flat, fill=RAW_PATH_COLOR, width=1, dash=(4, 3))

        if len(frame.path_points) < 2:
            return
        screen_points = [world_to_screen(p.x, p.y, self.view) for p in frame.path_points]
        flat = [coord for point in screen_points for coord in point]
        self.canvas.create_line(*flat, fill=PATH_COLOR, width=2)

    def _draw_local_target(self, frame: ReplayFrame) -> None:
        if frame.local_target is None:
            return
        px, py = world_to_screen(frame.local_target.x, frame.local_target.y, self.view)
        r = 5
        self.canvas.create_oval(px - r, py - r, px + r, py + r, outline=LOCAL_TARGET_COLOR, width=2)

    def _draw_goal(self, frame: ReplayFrame) -> None:
        if frame.goal is None:
            return
        px, py = world_to_screen(frame.goal.x, frame.goal.y, self.view)
        r = 7
        self.canvas.create_line(px - r, py - r, px + r, py + r, fill=GOAL_COLOR, width=2)
        self.canvas.create_line(px - r, py + r, px + r, py - r, fill=GOAL_COLOR, width=2)

    def _draw_robot(self, pose, color: str) -> None:
        px, py = world_to_screen(pose.x, pose.y, self.view)
        r = 6
        self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=color, outline="")
        (sx, sy), (ex, ey) = heading_line_endpoints(pose, length=0.4)
        spx, spy = world_to_screen(sx, sy, self.view)
        epx, epy = world_to_screen(ex, ey, self.view)
        self.canvas.create_line(spx, spy, epx, epy, fill=color, width=2)

    def _update_status(self, frame: ReplayFrame, is_failed: bool) -> None:
        goal_text = f"({frame.goal.x:.2f}, {frame.goal.y:.2f})" if frame.goal is not None else "none"
        lines = [
            f"frame {self.index + 1}/{len(self.frames)}   tick={frame.tick}   t={frame.timestamp:.2f}s"
            f"   state={frame.navigation_state}" + ("   *** BLOCKED ***" if is_failed else ""),
            f"ground truth (SLAM frame): ({frame.ground_truth_pose_slam_frame.x:.3f}, "
            f"{frame.ground_truth_pose_slam_frame.y:.3f}, {frame.ground_truth_pose_slam_frame.theta:.3f})"
            f"   [{GROUND_TRUTH_COLOR}]",
            f"estimated:                 ({frame.estimated_pose.x:.3f}, {frame.estimated_pose.y:.3f}, "
            f"{frame.estimated_pose.theta:.3f})   [{ESTIMATED_COLOR}]",
            f"position error: {frame.position_error:.3f} m   heading error: {frame.heading_error:.3f} rad"
            f"   goal: {goal_text}   path length: {len(frame.path_points)}",
        ]
        if frame.navigation_state == "RECOVERING" or frame.recovery_attempts > 0:
            lines.append(
                f"recovery: attempt {frame.recovery_attempts}   elapsed {frame.recovery_elapsed:.2f}s"
                f"   direction: reverse (heading + pi)"
            )
        self.status_var.set("\n".join(lines))
