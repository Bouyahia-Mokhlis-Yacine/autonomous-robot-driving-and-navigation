"""Local target selection: which point on the global path to chase right now.

Stateful (tracks a waypoint index) so it can advance monotonically as the
robot progresses, rather than re-selecting the nearest point every tick
(which can jitter backward on a self-intersecting or noisy path).
"""

from __future__ import annotations

from common.geometry import distance
from common.messages import Path, PathPoint, Pose2D


class LocalPlanner:
    """Selects the next waypoint to chase from a global ``Path``."""

    def __init__(self, waypoint_tolerance: float) -> None:
        if waypoint_tolerance <= 0.0:
            raise ValueError("waypoint_tolerance must be > 0")
        self.waypoint_tolerance = waypoint_tolerance
        self._index = 0

    @property
    def index(self) -> int:
        """Index of the current target waypoint in the path."""
        return self._index

    def reset(self) -> None:
        """Call when a new path is assigned (e.g. after replanning)."""
        self._index = 0

    def select_target(self, pose: Pose2D, path: Path) -> PathPoint | None:
        """Return the current local target waypoint, advancing past reached ones."""
        if path.is_empty():
            return None
        self._index = min(self._index, len(path.points) - 1)

        while self._index < len(path.points) - 1:
            target = path.points[self._index]
            if distance(pose.x, pose.y, target.x, target.y) >= self.waypoint_tolerance:
                break
            self._index += 1

        return path.points[self._index]
