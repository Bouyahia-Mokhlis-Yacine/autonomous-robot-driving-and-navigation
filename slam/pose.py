"""Pose-integration utilities shared by localization backends.

Pure kinematics, no sensor fusion or optimization — used by
``backend.OdometryBackend`` for open-loop dead reckoning, and reusable
later by real wheel-encoder odometry (Phase 13).
"""

from __future__ import annotations

import math

from common.geometry import wrap_angle
from common.messages import Pose2D, VelocityCommand


def integrate_pose(pose: Pose2D, cmd: VelocityCommand, dt: float) -> Pose2D:
    """Advance ``pose`` by applying ``cmd`` for ``dt`` seconds (unicycle model).

    Uses the pose's heading at the start of the interval (a first-order
    Euler integration) — adequate for short control-loop timesteps, not a
    substitute for real odometry/SLAM correction over long trajectories.
    """
    if dt < 0.0:
        raise ValueError("dt must be >= 0")

    new_x = pose.x + cmd.linear * math.cos(pose.theta) * dt
    new_y = pose.y + cmd.linear * math.sin(pose.theta) * dt
    new_theta = wrap_angle(pose.theta + cmd.angular * dt)

    return Pose2D(x=new_x, y=new_y, theta=new_theta, timestamp=pose.timestamp + dt)
