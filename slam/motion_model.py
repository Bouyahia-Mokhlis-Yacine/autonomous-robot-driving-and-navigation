"""Commanded-motion model: what the robot was TOLD to do, over time.

The robot has no wheel encoders, so there is no measured odometry. What IS
known, exactly, is the stream of velocity commands navigation sent. This
module records that stream and integrates it, for two purposes:

    - motion PREDICTION between two scans (seeding ICP): at the start of a
      turn the scan-derived velocity estimate is still ~0, but the command
      already says "turn at 1.5 rad/s" -- that is exactly where the
      velocity-only seed failed;
    - delay COMPENSATION: propagating a SLAM pose from the time its scan was
      captured to "now" (see ``simulation.closed_loop``).

Terminology used throughout the localization code:
    commanded motion -- integrated velocity COMMANDS (this module); known
                        exactly, but only what was requested
    estimated motion -- what SLAM measured between accepted scans (the
                        pipeline's velocity estimate / pose deltas)
    actual motion    -- what the robot really did (known only to the
                        simulator's ground truth; never used here)

Commands are NOT trusted blindly. Commanded motion only ever supplies a
SEED (ICP still has to converge on real geometry) and a short-horizon
propagation that is re-anchored at every accepted scan. The plausibility
check that accepts or rejects a registration uses physical limits only
(see ``ScanMatchingPipeline``), so a robot that ignores its commands --
stalled, blocked, slipping -- is still localized from the scans; the cost
is a worse seed, not a wrong acceptance. Commands are clamped to the
robot's velocity limits, as the motor layer clamps them.
"""

from __future__ import annotations

import bisect
import math
from collections import deque

from common.geometry import wrap_angle
from common.messages import Pose2D, VelocityCommand

_MAX_STEP_S = 0.01  # integration substep; exact enough for unicycle arcs at these speeds


class CommandHistory:
    """Time-stamped, piecewise-constant velocity commands.

    ``record(t, cmd)`` means ``cmd`` is in effect from ``t`` until the next
    record. Before the first record the robot is assumed stationary.
    """

    def __init__(self, max_linear_velocity: float, max_angular_velocity: float, max_age_s: float = 5.0) -> None:
        if max_linear_velocity <= 0.0 or max_angular_velocity <= 0.0:
            raise ValueError("velocity limits must be > 0")
        self._v_max = max_linear_velocity
        self._w_max = max_angular_velocity
        self._max_age = max_age_s
        self._times: deque[float] = deque()
        self._cmds: deque[tuple[float, float]] = deque()

    def record(self, time: float, cmd: VelocityCommand) -> None:
        if self._times and time < self._times[-1]:
            raise ValueError("commands must be recorded in time order")
        v = max(-self._v_max, min(self._v_max, cmd.linear))
        w = max(-self._w_max, min(self._w_max, cmd.angular))
        if self._times and time == self._times[-1]:
            self._cmds[-1] = (v, w)
        elif self._cmds and self._cmds[-1] == (v, w):
            return  # unchanged command: no new segment needed
        else:
            self._times.append(time)
            self._cmds.append((v, w))
        # Keep the latest segment that starts before the horizon (it is still in effect there).
        while len(self._times) > 1 and self._times[1] < time - self._max_age:
            self._times.popleft()
            self._cmds.popleft()

    def command_at(self, time: float) -> tuple[float, float]:
        i = bisect.bisect_right(self._times, time) - 1
        return self._cmds[i] if i >= 0 else (0.0, 0.0)

    def _segments(self, t0: float, t1: float):
        """(start, end, v, w) pieces of constant command covering [t0, t1]."""
        t = t0
        while t < t1:
            i = bisect.bisect_right(self._times, t) - 1
            v, w = self._cmds[i] if i >= 0 else (0.0, 0.0)
            nxt = self._times[i + 1] if i + 1 < len(self._times) else math.inf
            end = min(t1, nxt)
            yield t, end, v, w
            t = end

    def propagate(self, pose: Pose2D, t0: float, t1: float) -> Pose2D:
        """``pose`` (world frame, at ``t0``) moved by the commands in effect
        over [t0, t1] (unicycle model, small substeps). ``t1 <= t0``
        returns ``pose`` unchanged -- never integrates backwards."""
        x, y, th = pose.x, pose.y, pose.theta
        if t1 > t0:
            for start, end, v, w in self._segments(t0, t1):
                span = end - start
                steps = max(1, math.ceil(span / _MAX_STEP_S))
                h = span / steps
                for _ in range(steps):
                    mid = th + 0.5 * w * h  # midpoint heading: exact for arcs to 2nd order
                    x += v * math.cos(mid) * h
                    y += v * math.sin(mid) * h
                    th += w * h
        return Pose2D(x=x, y=y, theta=wrap_angle(th), timestamp=max(t0, t1))

    def predict_motion(self, t0: float, t1: float) -> tuple[float, float, float] | None:
        """Commanded motion over [t0, t1] as ``(dx_local, dy_local, dtheta)``
        in the frame of the pose at ``t0`` (the pipeline's ``MotionPrior``
        protocol). ``None`` if nothing was ever commanded."""
        if not self._times:
            return None
        end = self.propagate(Pose2D(0.0, 0.0, 0.0), t0, t1)
        # Unwrapped rotation, so a >180-degree prediction is not folded.
        dtheta = 0.0
        if t1 > t0:
            dtheta = sum((end_ - start) * w for start, end_, _, w in self._segments(t0, t1))
        return end.x, end.y, dtheta


class MountedMotionPrior:
    """A body-motion prior (``CommandHistory``) re-expressed for a LiDAR
    mounted at ``(lidar_x, lidar_y)`` in the robot frame. Commands move the
    body about its centre; the sensor sits off-centre, so a turn in place
    also swings it sideways -- 8 cm behind the centre, a 5 deg turn moves it
    ~7 mm. ``predict_motion`` returns the SENSOR's motion in the sensor's own
    frame at t0 (the pipeline's ``MotionPrior`` protocol; the pipeline tracks
    the sensor). The mount has no yaw, so both frames share a heading."""

    def __init__(self, body_prior: CommandHistory, lidar_x: float, lidar_y: float) -> None:
        self.body = body_prior
        self.mount = (lidar_x, lidar_y)

    def predict_motion(self, t0: float, t1: float) -> tuple[float, float, float] | None:
        body = self.body.predict_motion(t0, t1)
        if body is None:
            return None
        dx, dy, dtheta = body
        mx, my = self.mount
        c, s = math.cos(dtheta), math.sin(dtheta)
        return dx + c * mx - s * my - mx, dy + s * mx + c * my - my, dtheta

    def __getattr__(self, name):          # record / propagate / ... -> the body history
        return getattr(self.body, name)


def mounted_prior(body_prior, lidar_x: float, lidar_y: float):
    """``body_prior`` unchanged for a centred LiDAR (or no prior), else wrapped."""
    if body_prior is None or (lidar_x == 0.0 and lidar_y == 0.0):
        return body_prior
    return MountedMotionPrior(body_prior, lidar_x, lidar_y)
