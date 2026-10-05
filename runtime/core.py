"""The authoritative robot runtime: control loop, command application, safety.

One CONTROL thread owns the NavigationManager: every operator command is queued
by the network layer and applied here, between ticks, never from a network
thread. Per tick (``control.rate_hz``):

    1. apply queued commands (STOP / RESUME / SET_GOAL / CANCEL_GOAL)
    2. enforce the operator-loss policy
    3. enforce a latched STOP (EMERGENCY_STOP state, gated robot)
    4. backend.tick(): perception + NavigationManager.update + robot command
    5. hand the newest state / map to the server (O(1), never blocks)

Safety rules implemented here (network/server.py adds admission checks --
malformed, duplicate/out-of-order seq, stale state_seq -- before this):
    - STOP latches immediately on the network thread (``stop_now``: backend
      zero-command + latch flag) and is enforced every tick until RESUME;
      it cancels the goal, so nothing resumes by itself afterwards.
    - RESUME only clears the latch; the robot stays IDLE until a NEW goal.
    - SET_GOAL is refused while STOP is latched, and goals from a connection
      that has since dropped are discarded (stale).
    - Operator loss (no operator heard from within ``operator_timeout_s``):
      policy ``stop`` cancels the current goal; ``continue`` keeps it (all
      navigation safety checks still apply). Reconnecting never re-sends or
      re-activates anything.
    - A connecting operator never causes motion: commands are the only input.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time

import numpy as np

from navigation.navigation_manager import NavigationState
from network.protocol import PROTOCOL_VERSION, round_list
from network.server import AdmittedCommand, RobotServer
from runtime.backends import Backend
from runtime.config import RuntimeConfig

logger = logging.getLogger("autonomous_navigation.runtime")

FRAME_DOC = {
    "frame": "map (SLAM world frame): origin = sensor pose at SLAM bootstrap, +x = its heading then, +y = left",
    "units": "metres, radians (theta counter-clockwise from +x, wrapped to (-pi, pi])",
    "grid": "row 0 = lowest y; cell (r, c) spans x in [origin_x + c*res, +res), y in [origin_y + r*res, +res)",
    "time": "t = seconds since runtime start (runtime monotonic clock)",
}


def _pose(p) -> dict | None:
    return None if p is None else {"x": round(p.x, 4), "y": round(p.y, 4), "theta": round(p.theta, 5)}


class RobotRuntime:
    def __init__(self, backend: Backend, config: RuntimeConfig, host: str | None = None, port: int | None = None) -> None:
        self.backend = backend
        self.config = config
        self._commands: queue.Queue[AdmittedCommand] = queue.Queue()
        self._stop_latched = threading.Event()
        # Motors attached -> never move until an operator explicitly RESUMEs.
        if config.safety.start_stopped or getattr(backend, "requires_resume_at_start", False):
            self._stop_latched.set()
        self._latch_applied = False
        self._had_operator = False
        self.events: list[str] = []          # recent safety events, newest last (also published)
        self.applied_goals = 0               # goals actually handed to NavigationManager
        self._start = time.monotonic()
        self._last_state_pub = -math.inf
        self._last_map_pub = -math.inf
        self._last_map_version = None
        self.tick_count = 0
        self.tick_overruns = 0
        self.max_tick_s = 0.0
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        n, s = config.network, config.safety
        self.server = RobotServer(
            command_sink=self._commands.put, stop_now=self.stop_now,
            host=n.host if host is None else host, port=n.port if port is None else port,
            operator_timeout_s=s.operator_timeout_s, max_command_state_age_s=s.max_command_state_age_s,
            send_timeout_s=n.send_timeout_s,
            welcome={"runtime_id": f"{backend.name}-{int(time.time())}", "backend": backend.name,
                     "protocol": PROTOCOL_VERSION, "frame": FRAME_DOC,
                     "rates": {"state_hz": n.state_rate_hz, "map_hz": n.map_rate_hz, "control_hz": config.control.rate_hz},
                     "safety": {"operator_loss_policy": s.operator_loss_policy,
                                "operator_timeout_s": s.operator_timeout_s,
                                "max_command_state_age_s": s.max_command_state_age_s}},
        )

    # ------------------------------------------------------------ lifecycle --

    def start(self) -> int:
        port = self.server.start()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="control-loop", daemon=True)
        self._thread.start()
        return port

    def shutdown(self) -> None:
        self.stop_now()
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.server.close()
        self.backend.close()

    def is_stop_latched(self) -> bool:
        return self._stop_latched.is_set()

    def stop_now(self) -> None:
        """Latch STOP and zero the motion command immediately. Callable from any
        thread (the network reader calls it on receipt of STOP)."""
        if not self._stop_latched.is_set():
            self._event("STOP latched")
        self._stop_latched.set()
        self._latch_applied = False
        self.backend.stop_now()

    # ------------------------------------------------------------ main loop --

    def _loop(self) -> None:
        period = 1.0 / self.config.control.rate_hz
        next_tick = time.monotonic()
        while self._running.is_set():
            started = time.monotonic()
            try:
                self.tick(period)
            except Exception:
                logger.exception("Control tick failed; latching STOP")
                self.stop_now()
            elapsed = time.monotonic() - started
            self.max_tick_s = max(self.max_tick_s, elapsed)
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self.tick_overruns += 1
                next_tick = time.monotonic()

    def tick(self, dt: float) -> None:
        self._apply_commands()
        self._enforce_operator_policy()
        if self._stop_latched.is_set() and not self._latch_applied:
            nav = self.backend.nav_manager
            nav.cancel_goal()
            nav.emergency_stop()
            self._latch_applied = True
        self.backend.tick(dt)
        self.tick_count += 1
        self._publish()

    # ------------------------------------------------------------- commands --

    def _apply_commands(self) -> None:
        while True:
            try:
                adm = self._commands.get_nowait()
            except queue.Empty:
                return
            cmd, nav = adm.command, self.backend.nav_manager
            if cmd.type == "stop":
                self.stop_now()  # idempotent; the reader thread already latched it
                self.server.send_ack(adm.connection_id, cmd.seq, True, "STOP latched")
            elif not self.server.is_connected(adm.connection_id):
                self._event(f"discarded {cmd.type} #{cmd.seq}: its operator connection is gone")
            elif cmd.type == "resume":
                if self._stop_latched.is_set():
                    self._stop_latched.clear()
                    self._latch_applied = False
                    nav.clear_emergency_stop()
                    self._event("STOP released by operator (robot stays idle until a new goal)")
                    self.server.send_ack(adm.connection_id, cmd.seq, True, "resumed; idle")
                else:
                    self.server.send_ack(adm.connection_id, cmd.seq, True, "was not stopped")
            elif cmd.type == "cancel_goal":
                if not self._stop_latched.is_set():
                    nav.cancel_goal()
                self.server.send_ack(adm.connection_id, cmd.seq, True, "goal cancelled")
            elif cmd.type == "set_goal":
                self._apply_goal(adm)

    def _apply_goal(self, adm: AdmittedCommand) -> None:
        cmd = adm.command
        if self._stop_latched.is_set():
            self.server.send_ack(adm.connection_id, cmd.seq, False, "STOP is latched: press RESUME first")
            return
        grid = self.backend.world_grid()
        inside = (grid.origin_x <= cmd.x < grid.origin_x + grid.width * grid.resolution
                  and grid.origin_y <= cmd.y < grid.origin_y + grid.height * grid.resolution)
        if not inside:
            self.server.send_ack(adm.connection_id, cmd.seq, False, "goal outside the map")
            return
        self.backend.set_goal(cmd.x, cmd.y)
        self.applied_goals += 1
        state = self.backend.nav_manager.state.value
        self._event(f"goal ({cmd.x:.2f}, {cmd.y:.2f}) from operator -> {state}")
        self.server.send_ack(adm.connection_id, cmd.seq, True, f"goal set; navigation {state}")

    def _enforce_operator_policy(self) -> None:
        present = self.server.operators_present()
        if present:
            self._had_operator = True
            return
        if not self._had_operator:
            return
        self._had_operator = False
        nav = self.backend.nav_manager
        if self.config.safety.operator_loss_policy == "stop":
            if nav.get_goal() is not None and nav.state is not NavigationState.EMERGENCY_STOP:
                nav.cancel_goal()
                self._event("operator lost: goal cancelled, robot stopped (policy 'stop')")
            else:
                self._event("operator lost (no active goal)")
        else:
            self._event("operator lost: continuing current goal (policy 'continue')")

    def _event(self, text: str) -> None:
        stamped = f"{time.monotonic() - self._start:7.1f}s {text}"
        logger.warning("%s", text)
        self.events = (self.events + [stamped])[-8:]

    # ------------------------------------------------------------ publishing --

    def _publish(self) -> None:
        now = time.monotonic()
        n = self.config.network
        version = self.backend.map_version()
        if now - self._last_map_pub >= 1.0 / n.map_rate_hz and version != self._last_map_version:
            self._last_map_pub, self._last_map_version = now, version
            self.server.publish_map(self.map_message(version, now))
        if now - self._last_state_pub >= 1.0 / n.state_rate_hz:
            self._last_state_pub = now
            self.server.publish_state(self.state_message(now))

    def map_message(self, version: int, now: float) -> dict:
        grid, planning = self.backend.world_grid(), self.backend.planning_grid()
        return {"version": version, "t": round(now - self._start, 3), "resolution": grid.resolution,
                "width": grid.width, "height": grid.height, "origin_x": grid.origin_x, "origin_y": grid.origin_y,
                "cells": np.array(grid.cells, dtype=np.int8, copy=True),
                "planning_cells": np.array(planning.cells, dtype=np.int8, copy=True)}

    def state_message(self, now: float) -> dict:
        nav, health = self.backend.nav_manager, self.backend.health
        path, goal, target = nav.get_path(), nav.get_goal(), nav.get_local_target()
        scan, scan_pose = self.backend.latest_scan()
        age = health.pose_age()
        return {
            "t": round(now - self._start, 3),
            "backend": self.backend.name,
            "pose": _pose(self.backend.pose()),
            "gt_pose": _pose(self.backend.ground_truth_pose()),
            "health": {"state": health.health.value, "pose_age": round(age, 3) if math.isfinite(age) else None,
                       "losses": health.loss_events},
            "nav": {"state": nav.state.value,
                    "goal": None if goal is None else {"x": round(goal.x, 4), "y": round(goal.y, 4)},
                    "path": [] if path is None else [[round(p.x, 4), round(p.y, 4)] for p in path.points],
                    "local_target": None if target is None else [round(target.x, 4), round(target.y, 4)]},
            "safety": {"stop_latched": self._stop_latched.is_set(),
                       "operators": self.server.operators_present(),
                       "operator_loss_policy": self.config.safety.operator_loss_policy,
                       "events": list(self.events)},
            "scan": None if scan is None else {
                "timestamp": round(scan.timestamp, 4), "angle_min": scan.angle_min,
                "angle_increment": scan.angle_increment, "range_min": scan.range_min,
                "range_max": scan.range_max, "ranges": round_list(scan.ranges, 3)},
            "scan_pose": _pose(scan_pose),
            "stats": {**self.backend.stats(), "ticks": self.tick_count, "tick_overruns": self.tick_overruns,
                      "max_tick_ms": round(self.max_tick_s * 1000.0, 1)},
            "map_version": self._last_map_version,
        }
