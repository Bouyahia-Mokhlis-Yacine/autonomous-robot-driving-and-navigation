"""Side-by-side replay: ideal simulated LiDAR vs. realistic RPLiDAR model.

Same pattern as ``visualization.replay_viewer``: the simulations are run
to completion beforehand (live stepping is too slow -- see that module's
docstring), and this shell only scrubs an index into pre-recorded frames.
Reuses the existing rendering primitives (``render_grid_to_image``,
``ViewTransform``/``world_to_screen``, ``heading_line_endpoints``); no
simulation or navigation logic runs here.

Per panel:
    map          -- the world model's public grid as of that instant
    green robot  -- ground truth (SLAM frame, display only)
    blue robot   -- the pose the navigation stack is currently using
    grey dots    -- where the latest revolution's returns REALLY were
                    (each sample placed at its true acquisition pose)
    orange dots  -- the most recently PROCESSED scan, drawn the way SLAM
                    interpreted it (every point from its one estimated pose)
In the realistic panel the gap between grey and orange is the combined
effect of latency, motion distortion, missing returns and partial scans.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass

import numpy as np
from PIL import ImageTk

from common.messages import OccupancyGrid, Pose2D
from visualization.map_viewer import render_grid_to_image
from visualization.overlays import ViewTransform, heading_line_endpoints, world_to_screen

GROUND_TRUTH_COLOR = "#2ca02c"
ESTIMATED_COLOR = "#1f77b4"
RAW_HITS_COLOR = "#7f7f7f"
PROCESSED_SCAN_COLOR = "#ff7f0e"
WARNING_COLOR = "#d62728"


@dataclass
class SensorDemoFrame:
    time: float
    grid: OccupancyGrid
    ground_truth: Pose2D            # SLAM frame, display only
    estimated: Pose2D
    raw_hits: np.ndarray            # (N, 2) SLAM frame, true acquisition poses
    processed_scan_points: np.ndarray  # (M, 2) SLAM frame, as SLAM placed them
    status_lines: tuple[str, ...]
    localized: bool


class SensorComparisonViewer:
    def __init__(
        self,
        panels: list[tuple[str, list[SensorDemoFrame]]],
        scale: float = 55.0,
        dt_ms: int = 100,
        view_bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        if not panels or any(not frames for _, frames in panels):
            raise ValueError("every panel needs at least one frame")
        self.panels = panels
        self.num_frames = max(len(frames) for _, frames in panels)
        self.dt_ms = dt_ms
        self.index = 0
        self.playing = False

        if view_bounds is None:
            g = panels[0][1][0].grid
            view_bounds = (g.origin_x, g.origin_y, g.origin_x + g.width * g.resolution,
                           g.origin_y + g.height * g.resolution)
        self.x0, self.y0, self.x1, self.y1 = view_bounds
        self.px_w = int(round((self.x1 - self.x0) * scale))
        self.px_h = int(round((self.y1 - self.y0) * scale))
        self.view = ViewTransform(world_origin_x=self.x0, world_origin_y=self.y0, scale=scale, canvas_height=self.px_h)

        self.root = tk.Tk()
        self.root.title("LiDAR realism: ideal vs. realistic RPLiDAR")
        body = tk.Frame(self.root)
        body.pack(side="top")
        self.canvases: list[tk.Canvas] = []
        self.status_vars: list[tk.StringVar] = []
        for column, (title, _) in enumerate(panels):
            frame = tk.Frame(body, padx=4)
            frame.grid(row=0, column=column, sticky="n")
            tk.Label(frame, text=title, font=("TkDefaultFont", 12, "bold")).pack()
            canvas = tk.Canvas(frame, width=self.px_w, height=self.px_h, bg="white", highlightthickness=0)
            canvas.pack()
            var = tk.StringVar()
            tk.Label(frame, textvariable=var, anchor="w", justify="left", font=("TkFixedFont", 9)).pack(fill="x")
            self.canvases.append(canvas)
            self.status_vars.append(var)

        controls = tk.Frame(self.root)
        controls.pack(side="top", fill="x")
        tk.Button(controls, text="|<< Reset", command=self.reset).pack(side="left")
        tk.Button(controls, text="< Step", command=self.step_backward).pack(side="left")
        self.play_button = tk.Button(controls, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left")
        tk.Button(controls, text="Step >", command=self.step_forward).pack(side="left")
        self.slider = tk.Scale(controls, from_=0, to=self.num_frames - 1, orient="horizontal",
                               showvalue=False, length=400, command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True)
        legend = (f"green = ground truth   blue = pose navigation uses   "
                  f"grey = true returns (latest revolution)   orange = last processed scan as SLAM placed it")
        tk.Label(self.root, text=legend, font=("TkDefaultFont", 9)).pack(side="top")

        self._photos: list[ImageTk.PhotoImage | None] = [None] * len(panels)
        self.redraw()

    # ------------------------------------------------------------- controls ---

    def reset(self) -> None:
        self._stop()
        self.index = 0
        self.redraw()

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self.root.after(self.dt_ms, self._tick)

    def step_forward(self) -> None:
        self._stop()
        self.index = min(self.index + 1, self.num_frames - 1)
        self.redraw()

    def step_backward(self) -> None:
        self._stop()
        self.index = max(self.index - 1, 0)
        self.redraw()

    def _on_slider(self, value: str) -> None:
        index = int(float(value))
        if index != self.index:
            self.index = index
            self.redraw()

    def _stop(self) -> None:
        self.playing = False
        self.play_button.config(text="Play")

    def _tick(self) -> None:
        if not self.playing:
            return
        if self.index >= self.num_frames - 1:
            self._stop()
            return
        self.index += 1
        self.redraw()
        self.root.after(self.dt_ms, self._tick)

    def run(self) -> None:
        self.root.mainloop()

    # -------------------------------------------------------------- drawing ---

    def redraw(self) -> None:
        self.slider.set(self.index)
        for i, (_, frames) in enumerate(self.panels):
            frame = frames[min(self.index, len(frames) - 1)]
            self._draw_panel(i, frame)

    def _draw_panel(self, i: int, frame: SensorDemoFrame) -> None:
        canvas = self.canvases[i]
        canvas.delete("all")
        g = frame.grid
        image = render_grid_to_image(g)
        gw, gh = g.width * g.resolution, g.height * g.resolution
        full_w = int(round(gw * self.view.scale))
        full_h = int(round(gh * self.view.scale))
        image = image.resize((max(1, full_w), max(1, full_h)))
        left = int(round((g.origin_x - self.x0) * self.view.scale))
        top = int(round((self.y1 - (g.origin_y + gh)) * self.view.scale))
        self._photos[i] = ImageTk.PhotoImage(image)
        canvas.create_image(left, top, anchor="nw", image=self._photos[i])

        self._draw_points(canvas, frame.raw_hits, RAW_HITS_COLOR, 1.5)
        self._draw_points(canvas, frame.processed_scan_points, PROCESSED_SCAN_COLOR, 2.0)
        self._draw_robot(canvas, frame.ground_truth, GROUND_TRUTH_COLOR)
        self._draw_robot(canvas, frame.estimated, ESTIMATED_COLOR)
        if not frame.localized:
            canvas.create_text(8, 8, anchor="nw", text="LOCALIZATION LOST", fill=WARNING_COLOR,
                               font=("TkFixedFont", 13, "bold"))
        self.status_vars[i].set("\n".join(frame.status_lines))

    def _draw_points(self, canvas: tk.Canvas, points: np.ndarray, color: str, r: float) -> None:
        for x, y in points:
            px, py = world_to_screen(float(x), float(y), self.view)
            canvas.create_oval(px - r, py - r, px + r, py + r, fill=color, outline="")

    def _draw_robot(self, canvas: tk.Canvas, pose: Pose2D, color: str) -> None:
        px, py = world_to_screen(pose.x, pose.y, self.view)
        r = 6
        canvas.create_oval(px - r, py - r, px + r, py + r, outline=color, width=2)
        (sx, sy), (ex, ey) = heading_line_endpoints(pose, length=0.35)
        spx, spy = world_to_screen(sx, sy, self.view)
        epx, epy = world_to_screen(ex, ey, self.view)
        canvas.create_line(spx, spy, epx, epy, fill=color, width=2)
