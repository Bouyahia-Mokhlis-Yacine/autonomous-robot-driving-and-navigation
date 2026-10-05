"""Time-indexed ground-truth pose history for sub-tick sensor sampling.

``simulation.realistic_lidar.RealisticLidar`` acquires each of a
rotation's samples at a different instant (a real RPLiDAR takes ~90 ms
per revolution), so it needs the robot's pose at arbitrary times between
simulation ticks, not just "the pose now". This is the smallest mechanism
that answers that: the closed loop appends every physics substep's pose,
and ``pose_at(t)`` linearly interpolates between the bracketing entries.

Ground truth only: like ``Environment.robot_pose`` itself, this is read
by the sensor simulator and by evaluation code, never by the navigation
stack (see ``simulation.closed_loop``'s ground-truth isolation guarantee).
"""

from __future__ import annotations

import bisect
from collections import deque

from common.geometry import wrap_angle
from common.messages import Pose2D


class PoseHistory:
    """Append-only (by time) pose buffer with linear interpolation.

    ``max_age`` bounds memory: entries older than ``newest - max_age``
    are discarded, which is safe as long as it comfortably exceeds the
    longest span any single query reaches back (one sensor revolution
    plus delivery delay -- well under a second).
    """

    def __init__(self, max_age: float = 2.0) -> None:
        if max_age <= 0.0:
            raise ValueError("max_age must be > 0")
        self._max_age = max_age
        self._times: deque[float] = deque()
        self._poses: deque[Pose2D] = deque()

    def append(self, time: float, pose: Pose2D) -> None:
        if self._times and time < self._times[-1]:
            raise ValueError("PoseHistory times must be non-decreasing")
        if self._times and time == self._times[-1]:
            self._poses[-1] = pose  # same instant: latest pose wins
        else:
            self._times.append(time)
            self._poses.append(pose)
        while self._times and self._times[0] < time - self._max_age:
            self._times.popleft()
            self._poses.popleft()

    def __len__(self) -> int:
        return len(self._times)

    @property
    def latest_time(self) -> float:
        if not self._times:
            raise ValueError("PoseHistory is empty")
        return self._times[-1]

    def pose_at(self, time: float) -> Pose2D:
        """Interpolated pose at ``time`` (clamped to the recorded span).

        Heading interpolates along the shortest arc, so a rotation across
        the +/-pi seam interpolates correctly instead of spinning the
        long way round.
        """
        if not self._times:
            raise ValueError("PoseHistory is empty")
        times = self._times
        if time <= times[0]:
            p = self._poses[0]
            return Pose2D(p.x, p.y, p.theta, timestamp=time)
        if time >= times[-1]:
            p = self._poses[-1]
            return Pose2D(p.x, p.y, p.theta, timestamp=time)

        hi = bisect.bisect_right(times, time)
        lo = hi - 1
        t0, t1 = times[lo], times[hi]
        a, b = self._poses[lo], self._poses[hi]
        w = (time - t0) / (t1 - t0)
        dtheta = wrap_angle(b.theta - a.theta)
        return Pose2D(
            x=a.x + w * (b.x - a.x),
            y=a.y + w * (b.y - a.y),
            theta=wrap_angle(a.theta + w * dtheta),
            timestamp=time,
        )
