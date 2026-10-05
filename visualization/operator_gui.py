"""Operator GUI (laptop side): the existing ``NavigationGUI``, fed from the robot.

``NavigationGUI`` already draws map + inflation, LiDAR, robot, goal, path and
local target from a ``nav_manager`` and a ``GuiDataSource``, and turns clicks
into ``nav_manager.set_goal`` / ``cancel_goal``. Here both are REMOTE:

    RemoteNavigation  answers get_path/get_goal/get_local_target/state from the
                      runtime's latest published state; set_goal/cancel_goal
                      SEND commands. It plans nothing -- the path drawn is the
                      one the robot's own NavigationManager produced.
    RemoteDataSource  the robot's map / planning map / pose / scan, as the
                      existing message types (OccupancyGridMap, Pose2D,
                      LaserScan), so the existing drawing code is unchanged.

Added on top: a large STOP button (also Space / Escape), RESUME, CANCEL GOAL,
connection + localization + safety status, the simulator's ground-truth pose
(sim backend only, hollow green circle), and a clear STALE banner whenever
the robot has not been heard from recently. Safety never depends on this
window: STOP is enforced on the robot, and losing this window triggers the
robot's operator-loss policy.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from types import SimpleNamespace

import numpy as np

from common.messages import Goal, LaserScan, OccupancyGrid, Path, PathPoint, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from network.client import OperatorClient
from visualization.gui import GuiDataSource, NavigationGUI
from visualization.overlays import world_to_screen

STALE_AFTER_S = 1.0
GROUND_TRUTH_COLOR = "#2ca02c"


def grid_map_from(client_map: dict, cells_key: str) -> OccupancyGridMap:
    res, w, h = client_map["resolution"], client_map["width"], client_map["height"]
    grid_map = OccupancyGridMap(resolution=res, width_m=w * res, height_m=h * res,
                                origin_x=client_map["origin_x"], origin_y=client_map["origin_y"])
    grid_map.grid = OccupancyGrid(resolution=res, width=w, height=h, origin_x=client_map["origin_x"],
                                  origin_y=client_map["origin_y"], cells=np.asarray(client_map[cells_key]))
    return grid_map


def pose_from(d: dict | None) -> Pose2D | None:
    return None if d is None else Pose2D(x=d["x"], y=d["y"], theta=d["theta"])


class RemoteNavigation:
    """The subset of NavigationManager that NavigationGUI uses, backed by the robot."""

    def __init__(self, client: OperatorClient) -> None:
        self.client = client
        self.last_feedback = ""

    def _nav(self) -> dict:
        state = self.client.state
        return state["nav"] if state else {"state": "WAITING", "goal": None, "path": [], "local_target": None}

    @property
    def state(self):
        return SimpleNamespace(value=self._nav()["state"])

    def get_goal(self) -> Goal | None:
        g = self._nav()["goal"]
        return None if g is None else Goal(x=g["x"], y=g["y"])

    def get_path(self) -> Path | None:
        points = self._nav()["path"]
        return Path(points=[PathPoint(x=x, y=y) for x, y in points]) if points else None

    def get_local_target(self) -> PathPoint | None:
        t = self._nav()["local_target"]
        return None if t is None else PathPoint(x=t[0], y=t[1])

    def set_goal(self, goal: Goal, grid_map=None, raw_grid_map=None) -> None:
        seq = self.client.set_goal(goal.x, goal.y)
        self.last_feedback = (f"goal ({goal.x:.2f}, {goal.y:.2f}) sent (#{seq})" if seq is not None
                              else "NOT CONNECTED: goal not sent")

    def cancel_goal(self) -> None:
        seq = self.client.cancel_goal()
        self.last_feedback = f"cancel sent (#{seq})" if seq is not None else "NOT CONNECTED: cancel not sent"


class RemoteDataSource:
    def __init__(self, client: OperatorClient) -> None:
        self.client = client
        self._version = None
        self._static: OccupancyGridMap | None = None
        self._planning: OccupancyGridMap | None = None

    def _refresh(self) -> None:
        m = self.client.map
        if m is not None and (m.get("version"), id(m)) != self._version:
            self._static, self._planning = grid_map_from(m, "cells"), grid_map_from(m, "planning_cells")
            self._version = (m.get("version"), id(m))

    def static_map(self) -> OccupancyGridMap:
        self._refresh()
        return self._static

    def planning_map(self) -> OccupancyGridMap:
        self._refresh()
        return self._planning

    def pose(self) -> Pose2D:
        s = self.client.state
        return pose_from(s["pose"]) if s and s.get("pose") else Pose2D(0.0, 0.0, 0.0)

    def scan_pose(self) -> Pose2D | None:
        """The pose the latest scan was taken from (the LiDAR's, not the
        footprint centre's -- they differ by the LiDAR mount)."""
        s = self.client.state
        return pose_from(s.get("scan_pose")) if s else None

    def scan(self) -> LaserScan | None:
        s = self.client.state
        if not s or not s.get("scan"):
            return None
        d = s["scan"]
        return LaserScan(timestamp=d["timestamp"], angle_min=d["angle_min"], angle_increment=d["angle_increment"],
                         ranges=d["ranges"], range_min=d["range_min"], range_max=d["range_max"])

    def gui_source(self) -> GuiDataSource:
        return GuiDataSource(get_static_map=self.static_map, get_planning_map=self.planning_map,
                             get_pose=self.pose, get_scan=self.scan)


class OperatorGUI(NavigationGUI):
    def __init__(self, client: OperatorClient, scale: float, dt: float = 0.1) -> None:
        self.client = client
        self.remote_nav = RemoteNavigation(client)
        self.remote_data = RemoteDataSource(client)
        super().__init__(self.remote_nav, self.remote_data.gui_source(), scale=scale, dt=dt,
                         step_callback=lambda _dt: None)
        self.root.title(f"Robot operator -- {client.host}:{client.port}")

        bar = tk.Frame(self.root)
        bar.pack(side="top", fill="x")
        tk.Button(bar, text="STOP", command=self.press_stop, bg="#d62728", fg="white", activebackground="#a50f15",
                  activeforeground="white", font=("TkDefaultFont", 20, "bold"), height=2, width=14
                  ).pack(side="left", padx=6, pady=6)
        tk.Button(bar, text="RESUME", command=self.press_resume, font=("TkDefaultFont", 12), height=2
                  ).pack(side="left", padx=6)
        tk.Button(bar, text="CANCEL GOAL", command=self.remote_nav.cancel_goal, font=("TkDefaultFont", 12), height=2
                  ).pack(side="left", padx=6)
        tk.Label(bar, text="left-click: goal   right-click: cancel   Space/Esc: STOP", fg="#555").pack(side="left", padx=10)
        for key in ("<space>", "<Escape>"):
            self.root.bind(key, lambda _e: self.press_stop())

    # ------------------------------------------------------------ controls --

    def press_stop(self) -> None:
        seq = self.client.stop()
        self.remote_nav.last_feedback = (f"STOP sent (#{seq})" if seq is not None else
                                         "NOT CONNECTED: STOP not sent -- robot applies its own "
                                         "operator-loss policy; use the physical e-stop if needed")

    def press_resume(self) -> None:
        seq = self.client.resume()
        self.remote_nav.last_feedback = f"RESUME sent (#{seq})" if seq is not None else "NOT CONNECTED: RESUME not sent"

    # ------------------------------------------------------------- drawing --

    def _draw_scan(self, pose: Pose2D, scan: LaserScan) -> None:
        super()._draw_scan(self.remote_data.scan_pose() or pose, scan)  # from the LiDAR, not the footprint centre

    def redraw(self) -> None:
        super().redraw()
        s = self.client.state
        if s and s.get("gt_pose"):
            gt = s["gt_pose"]
            px, py = world_to_screen(gt["x"], gt["y"], self.view)
            self.canvas.create_oval(px - 9, py - 9, px + 9, py + 9, outline=GROUND_TRUTH_COLOR, width=2)
        age = time.monotonic() - self.client.last_rx if self.client.last_rx else float("inf")
        if not self.client.connected or age > STALE_AFTER_S:
            self.canvas.create_rectangle(0, 0, 10_000, 34, fill="#d62728", outline="")
            self.canvas.create_text(8, 17, anchor="w", fill="white", font=("TkDefaultFont", 13, "bold"),
                                    text="NO LINK TO ROBOT -- display is stale")
        elif s and s["safety"]["stop_latched"]:
            self.canvas.create_rectangle(0, 0, 10_000, 34, fill="#ff7f0e", outline="")
            self.canvas.create_text(8, 17, anchor="w", fill="black", font=("TkDefaultFont", 13, "bold"),
                                    text="STOP LATCHED -- RESUME, then new goal")

    def _update_status(self, pose: Pose2D) -> None:
        super()._update_status(pose)
        s = self.client.state
        age = time.monotonic() - self.client.last_rx if self.client.last_rx else float("inf")
        link = f"link: {'CONNECTED' if self.client.connected else 'DISCONNECTED'} ({age:.1f}s ago)" \
            if age != float("inf") else f"link: {self.client.status}"
        lines = [link]
        if s:
            h, safety, st = s["health"], s["safety"], s["stats"]
            lines[0] += (f"   backend={s['backend']}  robot t={s['t']:.1f}s  "
                         f"localization={h['state']}  STOP={'LATCHED' if safety['stop_latched'] else 'off'}  "
                         f"operators={safety['operators']}  policy={safety['operator_loss_policy']}")
            lines.append(f"scans {st.get('scans_processed', 0)}/{st.get('scans_received', 0)} processed, "
                         f"dropped {st.get('scans_dropped', 0)}   control overruns {st.get('tick_overruns', 0)} "
                         f"(max {st.get('max_tick_ms', 0)} ms)")
            if safety["events"]:
                lines.append("robot: " + safety["events"][-1].strip())
        last_ack = self.client.acks[-1] if self.client.acks else None
        if last_ack is not None:
            lines.append(f"last reply #{last_ack['seq']}: {'OK' if last_ack['accepted'] else 'REJECTED'} "
                         f"{last_ack.get('reason', '')}")
        if self.remote_nav.last_feedback:
            lines.append(self.remote_nav.last_feedback)
        self.status_var.set(self.status_var.get() + "\n" + "\n".join(lines))


class OrientationCheckGUI(OperatorGUI):
    """``operator_gui.py --orientation-check``: a close-up view for checking the
    LiDAR's direction and mounting by hand. DISPLAY ONLY, goal clicks disabled.

    Drawn in the ROBOT frame, fixed on screen: FRONT (+x) always points UP,
    LEFT (+y) always points to the screen's LEFT. Each live scan point is placed
    straight from its bin angle and range exactly as the driver reports them
    (``LaserScan.angle_at(i)``, CCW, 0 = FRONT) -- no SLAM pose is involved, so
    a wrong heading estimate cannot hide or fake a mirrored or rotated sensor.
    The map is not drawn: a new object next to a still robot never enters it
    (map-update motion filter); only the live scan shows it."""

    RINGS_M = (0.5, 1.0, 1.5, 2.0)
    NEAR_M = 1.5          # scan points closer than this are drawn larger
    MIN_NEAREST_M = 0.15  # the nearest-object readout ignores anything closer

    def __init__(self, client: OperatorClient, robot_length: float, robot_width: float,
                 half_width_m: float = 2.5, scale: float = 150.0, dt: float = 0.1,
                 lidar_offset: tuple[float, float] = (0.0, 0.0)) -> None:
        self._ready = False
        self.lidar_offset = lidar_offset   # LiDAR mount in the robot frame (robot.yaml)
        self.half_width_m, self.px_per_m = half_width_m, scale
        self.robot_length, self.robot_width = robot_length, robot_width
        super().__init__(client, scale=1.0, dt=dt)   # base canvas: replaced below
        side = int(round(2 * half_width_m * scale))
        self.canvas.config(width=side, height=side, bg="white")
        self.root.title(f"LiDAR orientation check -- {client.host}:{client.port} (display only, goals disabled)")
        self._ready = True

    def _on_left_click(self, event) -> None:
        pass  # no goals in this mode

    def _on_right_click(self, event) -> None:
        pass

    def robot_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Robot frame (x FRONT, y LEFT) -> canvas pixels: FRONT up, LEFT left."""
        side = 2 * self.half_width_m * self.px_per_m
        return int(round(side / 2 - y * self.px_per_m)), int(round(side / 2 - x * self.px_per_m))

    def redraw(self) -> None:
        if not self._ready:
            return
        from visualization.overlays import direction_label
        c, S = self.canvas, self.robot_to_screen
        c.delete("all")
        h, s = self.half_width_m, self.px_per_m
        side = int(round(2 * h * s))
        cx, cy = S(0.0, 0.0)

        for ring in self.RINGS_M:
            rp = ring * s
            c.create_oval(cx - rp, cy - rp, cx + rp, cy + rp, outline="#b0b0b0", dash=(3, 4))
            c.create_text(cx + rp * 0.707 + 3, cy - rp * 0.707 - 3, text=f"{ring:g} m", fill="#888", anchor="sw")

        hl, hw = self.robot_length / 2, self.robot_width / 2
        c.create_rectangle(*S(hl, hw), *S(-hl, -hw), fill="#cfe3f7", outline="#1f77b4", width=2)
        c.create_line(*S(hl, hw), *S(hl, -hw), fill="#d62728", width=6)      # front edge
        axis = 0.6 * h
        c.create_line(cx, cy, *S(axis, 0.0), fill="#d62728", width=3, arrow="last", arrowshape=(16, 20, 6))
        c.create_line(cx, cy, *S(0.0, axis), fill="#2ca02c", width=3, arrow="last", arrowshape=(16, 20, 6))
        c.create_line(cx, cy, *S(-axis, 0.0), fill="#999", width=1, dash=(2, 3))
        c.create_line(cx, cy, *S(0.0, -axis), fill="#999", width=1, dash=(2, 3))
        big = ("TkDefaultFont", 16, "bold")
        lab = 0.84 * h
        c.create_text(*S(lab, 0.0), text="FRONT  (+x, 0°)", fill="#d62728", font=big)
        c.create_text(*S(-lab, 0.0), text="REAR  (-x, 180°)", fill="#555", font=big)
        c.create_text(*S(0.0, lab), text="LEFT\n(+y)\n+90°", fill="#2ca02c", font=big, justify="center")
        c.create_text(*S(0.0, -lab), text="RIGHT\n(-y)\n-90°", fill="#9467bd", font=big, justify="center")

        lx, ly = S(*self.lidar_offset)          # where the LiDAR sits on the robot
        c.create_oval(lx - 7, ly - 7, lx + 7, ly + 7, outline="black", width=2)
        c.create_text(lx + 10, ly + 12, text="LiDAR", anchor="w", font=("TkDefaultFont", 9))
        scan = self.data_source.get_scan()
        nearest = None
        if scan is not None:
            for i in range(scan.num_rays):
                r = scan.ranges[i]
                if not (scan.range_min <= r < scan.range_max):
                    continue
                a_sensor = scan.angle_at(i)
                # Robot frame = sensor frame shifted by the mount.
                x, y = r * math.cos(a_sensor) + self.lidar_offset[0], r * math.sin(a_sensor) + self.lidar_offset[1]
                d, a = math.hypot(x, y), math.atan2(y, x)
                if r >= self.MIN_NEAREST_M and (nearest is None or d < nearest[0]):
                    nearest = (d, a)
                px, py = S(x, y)
                dot = 4 if r < self.NEAR_M else 2
                c.create_oval(px - dot, py - dot, px + dot, py + dot,
                              fill="#e6550d" if r < self.NEAR_M else "#d62728", outline="")
        c.create_rectangle(0, side - 56, side, side, fill="white", outline="#ddd")
        if nearest is not None:
            r, a = nearest
            px, py = S(r * math.cos(a), r * math.sin(a))
            c.create_oval(px - 11, py - 11, px + 11, py + 11, outline="black", width=2)
            c.create_text(8, side - 40, anchor="w", font=("TkDefaultFont", 15, "bold"),
                          text=f"nearest object: {direction_label(a)}")
            c.create_text(8, side - 16, anchor="w", font=("TkDefaultFont", 11),
                          text=f"{r:.2f} m, bearing {math.degrees(a):+.0f}°   x {r * math.cos(a):+.2f}  y {r * math.sin(a):+.2f} m")
        else:
            c.create_text(8, side - 28, anchor="w", font=("TkDefaultFont", 14, "bold"),
                          text="no live scan yet" if scan is None else "no object in range")

        age = time.monotonic() - self.client.last_rx if self.client.last_rx else float("inf")
        if not self.client.connected or age > STALE_AFTER_S:
            c.create_rectangle(0, 0, side, 34, fill="#d62728", outline="")
            c.create_text(8, 17, anchor="w", fill="white", font=("TkDefaultFont", 13, "bold"),
                          text="NO LINK TO ROBOT -- display is stale")
        self._update_status(self.data_source.get_pose())


class CloseUpOperatorGUI(OperatorGUI):
    """The default operator view: the look of the orientation check, with
    navigation. Robot-centred close-up of the map with map +x UP and +y LEFT
    (at startup: robot FRONT up, LEFT left), range rings, the robot's
    footprint and FRONT/LEFT/RIGHT/REAR labels (turning with the robot), large
    live-scan dots, the planned path and the goal. Left-click sets a goal,
    right-click cancels, the mouse wheel zooms. Display only: every command
    goes to the robot exactly as in OperatorGUI."""

    RING_STEP_M = 0.5
    ZOOM_LIMITS_M = (1.0, 10.0)

    def __init__(self, client: OperatorClient, robot_length: float, robot_width: float,
                 view_radius_m: float = 3.0, side_px: int = 700, dt: float = 0.1) -> None:
        self._ready = False
        self.view_radius_m, self.side_px = view_radius_m, side_px
        self.robot_length, self.robot_width = robot_length, robot_width
        self.fview = None
        super().__init__(client, scale=1.0, dt=dt)   # base canvas: replaced below
        self.canvas.config(width=side_px, height=side_px, bg="#7f7f7f")
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1 / 1.25))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1.25))
        self.canvas.bind("<MouseWheel>", lambda e: self._zoom(1 / 1.25 if e.delta > 0 else 1.25))
        self.root.title(f"Robot operator -- {client.host}:{client.port}   "
                        "(left-click: goal, right-click: cancel, wheel: zoom)")
        self._ready = True

    def _zoom(self, factor: float) -> None:
        lo, hi = self.ZOOM_LIMITS_M
        self.view_radius_m = min(hi, max(lo, self.view_radius_m * factor))

    def _on_left_click(self, event) -> None:
        if self.fview is None:
            return
        x, y = self.fview.to_world(event.x, event.y)
        self.remote_nav.set_goal(Goal(x=x, y=y))

    def redraw(self) -> None:
        if not self._ready:
            return
        from PIL import Image, ImageTk
        from visualization.map_viewer import DEFAULT_INFLATION_COLOR, DEFAULT_PALETTE
        from visualization.overlays import (
            ForwardUpView, cells_forward_up, direction_label, footprint_corners_world, robot_to_world,
            scan_hit_points_world, world_to_robot,
        )
        c = self.canvas
        c.delete("all")
        pose = self.data_source.get_pose()
        side, radius = self.side_px, self.view_radius_m
        scale = side / (2.0 * radius)
        v = self.fview = ForwardUpView(center_x=pose.x, center_y=pose.y, scale=scale, side=side)
        W = v.to_screen
        Rb = lambda x, y: W(*robot_to_world(pose, x, y))  # noqa: E731  robot frame -> screen

        # Map + inflation (clearance) band, cropped around the robot.
        grid = self.data_source.get_static_map().grid
        planning = self.data_source.get_planning_map().grid
        res = grid.resolution
        reach = radius * 1.45  # covers the canvas corners
        c0 = max(0, int((pose.x - reach - grid.origin_x) / res))
        c1 = min(grid.width, int((pose.x + reach - grid.origin_x) / res) + 1)
        r0 = max(0, int((pose.y - reach - grid.origin_y) / res))
        r1 = min(grid.height, int((pose.y + reach - grid.origin_y) / res) + 1)
        if c1 > c0 and r1 > r0:
            raw = grid.cells[r0:r1, c0:c1]
            rgb = np.zeros(raw.shape + (3,), dtype=np.uint8)
            for value, color in DEFAULT_PALETTE.items():
                rgb[raw == value] = color
            band = (planning.cells[r0:r1, c0:c1] == OccupancyGrid.OCCUPIED) & (raw != OccupancyGrid.OCCUPIED)
            rgb[band] = (0.4 * rgb[band] + 0.6 * np.array(DEFAULT_INFLATION_COLOR)).astype(np.uint8)
            img = Image.fromarray(np.ascontiguousarray(cells_forward_up(rgb)), mode="RGB")
            w_px = max(1, int(round((r1 - r0) * res * scale)))   # image width spans y
            h_px = max(1, int(round((c1 - c0) * res * scale)))   # image height spans x
            self._map_photo = ImageTk.PhotoImage(img.resize((w_px, h_px), Image.NEAREST))
            c.create_image(*W(grid.origin_x + c1 * res, grid.origin_y + r1 * res), anchor="nw",
                           image=self._map_photo)

        # Range rings around the robot.
        cx, cy = W(pose.x, pose.y)
        ring = self.RING_STEP_M
        while ring <= radius * 1.4:
            rp = ring * scale
            c.create_oval(cx - rp, cy - rp, cx + rp, cy + rp, outline="#5a5a5a", dash=(3, 5))
            c.create_text(cx + rp * 0.707 + 3, cy - rp * 0.707 - 3, text=f"{ring:g} m", fill="#3a3a3a", anchor="sw")
            ring += self.RING_STEP_M if radius <= 3.5 else 1.0

        # Path, local target, goal.
        path = self.nav_manager.get_path()
        if path is not None and len(path.points) >= 2:
            pts = [W(p.x, p.y) for p in path.points]
            c.create_line(*[q for p in pts for q in p], fill="#2ca02c", width=4, capstyle="round", joinstyle="round")
            for px, py in pts[1:-1]:
                c.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#2ca02c", outline="")
        target = self.nav_manager.get_local_target()
        if target is not None:
            px, py = W(target.x, target.y)
            c.create_oval(px - 7, py - 7, px + 7, py + 7, outline="#ff7f0e", width=3)
        goal = self.nav_manager.get_goal()
        if goal is not None:
            px, py = W(goal.x, goal.y)
            c.create_line(px - 10, py - 10, px + 10, py + 10, fill="#9467bd", width=4)
            c.create_line(px - 10, py + 10, px + 10, py - 10, fill="#9467bd", width=4)
            c.create_text(px + 12, py - 12, text="GOAL", fill="#9467bd", anchor="sw", font=("TkDefaultFont", 11, "bold"))

        # Live scan, larger when near.
        scan = self.data_source.get_scan()
        if scan is not None:
            for x, y in scan_hit_points_world(self.remote_data.scan_pose() or pose, scan):
                xr, yr = world_to_robot(pose, x, y)
                near = math.hypot(xr, yr) < 1.5
                px, py = W(x, y)
                r = 4 if near else 3
                c.create_oval(px - r, py - r, px + r, py + r, fill="#e6550d" if near else "#d62728", outline="")

        # Robot footprint, front edge, axes and direction labels (robot frame).
        corners = [W(*p) for p in footprint_corners_world(pose, self.robot_length, self.robot_width)]
        c.create_polygon(*[q for p in corners for q in p], fill="#cfe3f7", outline="#1f77b4", width=2)
        c.create_line(*corners[0], *corners[1], fill="#d62728", width=6)
        axis = min(0.9, 0.3 * radius)
        c.create_line(cx, cy, *Rb(axis, 0.0), fill="#d62728", width=3, arrow="last", arrowshape=(14, 18, 6))
        c.create_line(cx, cy, *Rb(0.0, axis), fill="#2ca02c", width=3, arrow="last", arrowshape=(14, 18, 6))
        lab = axis + 0.25 * radius / 3.0 + 0.2
        font = ("TkDefaultFont", 13, "bold")
        c.create_text(*Rb(lab, 0.0), text="FRONT", fill="#d62728", font=font)
        c.create_text(*Rb(-lab, 0.0), text="REAR", fill="#333", font=font)
        c.create_text(*Rb(0.0, lab), text="LEFT", fill="#2ca02c", font=font)
        c.create_text(*Rb(0.0, -lab), text="RIGHT", fill="#9467bd", font=font)

        # Header: navigation state + banners.
        st = self.client.state
        nav = st["nav"]["state"] if st else "?"
        health = st["health"]["state"] if st else "?"
        c.create_rectangle(0, side - 30, side, side, fill="white", outline="")
        c.create_text(8, side - 15, anchor="w", font=("TkDefaultFont", 12, "bold"),
                      text=f"nav {nav}   loc {health}   ±{radius:.1f} m")
        age = time.monotonic() - self.client.last_rx if self.client.last_rx else float("inf")
        if not self.client.connected or age > STALE_AFTER_S:
            c.create_rectangle(0, 0, side, 34, fill="#d62728", outline="")
            c.create_text(8, 17, anchor="w", fill="white", font=("TkDefaultFont", 13, "bold"),
                          text="NO LINK TO ROBOT -- display is stale")
        elif st and st["safety"]["stop_latched"]:
            c.create_rectangle(0, 0, side, 34, fill="#ff7f0e", outline="")
            c.create_text(8, 17, anchor="w", fill="black", font=("TkDefaultFont", 13, "bold"),
                          text="STOP LATCHED -- RESUME, then new goal")
        self._update_status(pose)
