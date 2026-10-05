"""Tk viewers for ``real_world_model_walkthrough.py`` (moved out of it so the
walkthrough's processing classes, which the headless robot runtime uses on
the Pi, never import tkinter/PIL.ImageTk)."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from common.messages import Pose2D
from slam.localization_health import LocalizationHealth
from visualization.gui import LatestStateBox, NavigationGUI, VisualizationState, WorldModelGUI
from visualization.overlays import screen_to_world

if TYPE_CHECKING:
    from real_world_model_walkthrough import LiveProcessor, NavigationSession


class HealthAwareWorldModelGUI(WorldModelGUI):
    """``WorldModelGUI`` plus the localization-health state and live stream
    counters in its status panel. Nothing else changes; the health monitor is
    also aged on the GUI's own tick so a stalled sensor still shows LOST."""

    def __init__(self, state_box: LatestStateBox, processor: LiveProcessor, **kwargs) -> None:
        self._processor = processor
        self._started = time.time()
        super().__init__(state_box, **kwargs)

    def _update_status(self, state: VisualizationState) -> None:
        super()._update_status(state)
        p = self._processor
        health = p.health.update(time.time())
        age = p.health.pose_age()
        elapsed = max(1e-6, time.time() - self._started)
        extra = [
            "",
            f"Health: {health.value}",
            f"  pose age: {age:.2f} s" if math.isfinite(age) else "  pose age: n/a",
            f"  losses:   {p.health.loss_events}",
            "",
            "Scans:",
            f"  received:  {p.stats.received} ({p.stats.received / elapsed:.1f}/s)",
            f"  processed: {p.stats.processed}",
            f"  dropped:   {p.stats.dropped}",
            f"  accepted:  {p.stats.accepted}",
        ]
        if p.stats.rejections:
            extra.append("  rejected:  " + ", ".join(f"{k} {v}" for k, v in sorted(p.stats.rejections.items())))
        if health is LocalizationHealth.INITIALIZING:
            extra += ["", "Keep the sensor STILL until", "Bootstrap: COMPLETE"]
        self.status_var.set(self.status_var.get() + "\n" + "\n".join(extra))


class LiveNavigationGUI(NavigationGUI):
    """``NavigationGUI`` (light map, click-to-target, planned path) plus a
    second status line: localization health and live stream counters."""

    def __init__(self, session: NavigationSession, **kwargs) -> None:
        self._session = session
        self._started = time.time()
        super().__init__(session.nav_manager, session.data_source(), step_callback=session.step, **kwargs)

    def _on_left_click(self, event) -> None:
        x, y = screen_to_world(event.x, event.y, self.view)
        self._session.set_goal(x, y)
        self.redraw()

    def _update_status(self, pose: Pose2D) -> None:
        super()._update_status(pose)
        p = self._session.processor
        health = p.health.health
        age = p.health.pose_age()
        elapsed = max(1e-6, time.time() - self._started)
        line = (f"localization={health.value}"
                + (f" (pose age {age:.2f}s)" if math.isfinite(age) else "")
                + f"  losses={p.health.loss_events}  scans {p.stats.received} ({p.stats.received / elapsed:.1f}/s)"
                  f" processed={p.stats.processed} dropped={p.stats.dropped} accepted={p.stats.accepted}")
        if health is LocalizationHealth.INITIALIZING:
            line += "   KEEP THE SENSOR STILL (bootstrapping)"
        elif self._session.nav_manager.get_goal() is not None:
            line += "   carry the sensor along the green path"
        self.status_var.set(self.status_var.get() + "\n" + line)


def fit_scale(map_size_m: float) -> float:
    """Largest pixels-per-metre at which the square map (plus the status
    lines and window decorations) fits on this screen."""
    import tkinter as tk

    probe = tk.Tk()
    probe.withdraw()
    screen_w, screen_h = probe.winfo_screenwidth(), probe.winfo_screenheight()
    probe.destroy()
    usable = min(screen_w - 60, screen_h - 130)  # status lines + title bar + taskbar
    return max(10.0, usable / map_size_m)
