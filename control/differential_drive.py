"""Unicycle (v, omega) <-> differential/skid-steer wheel-speed conversion.

Pure kinematics, no controller or hardware logic -- used by anything that
needs to turn a ``VelocityCommand`` into per-side wheel speeds (a real
motor driver, Phase 14) or reconstruct (v, omega) from measured wheel
speeds (real encoder odometry, Phase 13).
"""

from __future__ import annotations

from common.messages import VelocityCommand


def unicycle_to_wheel_speeds(v: float, omega: float, wheel_base: float) -> tuple[float, float]:
    """(linear, angular) -> (v_left, v_right)."""
    if wheel_base <= 0.0:
        raise ValueError("wheel_base must be > 0")
    v_left = v - omega * wheel_base / 2.0
    v_right = v + omega * wheel_base / 2.0
    return v_left, v_right


def wheel_speeds_to_unicycle(v_left: float, v_right: float, wheel_base: float) -> tuple[float, float]:
    """(v_left, v_right) -> (linear, angular). Inverse of ``unicycle_to_wheel_speeds``."""
    if wheel_base <= 0.0:
        raise ValueError("wheel_base must be > 0")
    v = (v_left + v_right) / 2.0
    omega = (v_right - v_left) / wheel_base
    return v, omega


def velocity_command_to_wheel_speeds(cmd: VelocityCommand, wheel_base: float) -> tuple[float, float]:
    return unicycle_to_wheel_speeds(cmd.linear, cmd.angular, wheel_base)
