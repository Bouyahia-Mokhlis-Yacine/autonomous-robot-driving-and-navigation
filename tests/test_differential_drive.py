"""Phase 8 tests: unicycle <-> differential-drive wheel-speed conversion."""

from __future__ import annotations

import pytest

from common.messages import VelocityCommand
from control.differential_drive import (
    unicycle_to_wheel_speeds,
    velocity_command_to_wheel_speeds,
    wheel_speeds_to_unicycle,
)


def test_straight_line_gives_equal_wheel_speeds() -> None:
    v_left, v_right = unicycle_to_wheel_speeds(v=0.5, omega=0.0, wheel_base=0.3)
    assert v_left == pytest.approx(0.5)
    assert v_right == pytest.approx(0.5)


def test_pure_rotation_gives_opposite_wheel_speeds() -> None:
    v_left, v_right = unicycle_to_wheel_speeds(v=0.0, omega=2.0, wheel_base=0.3)
    assert v_left == pytest.approx(-0.3)
    assert v_right == pytest.approx(0.3)


def test_combined_motion() -> None:
    v_left, v_right = unicycle_to_wheel_speeds(v=1.0, omega=1.0, wheel_base=0.5)
    assert v_left == pytest.approx(0.75)
    assert v_right == pytest.approx(1.25)


def test_round_trip_conversion() -> None:
    v, omega, wheel_base = 0.4, -0.8, 0.35
    v_left, v_right = unicycle_to_wheel_speeds(v, omega, wheel_base)
    v_back, omega_back = wheel_speeds_to_unicycle(v_left, v_right, wheel_base)
    assert v_back == pytest.approx(v)
    assert omega_back == pytest.approx(omega)


def test_velocity_command_helper_matches_direct_call() -> None:
    cmd = VelocityCommand(linear=0.3, angular=0.5)
    assert velocity_command_to_wheel_speeds(cmd, wheel_base=0.3) == unicycle_to_wheel_speeds(0.3, 0.5, 0.3)


def test_rejects_nonpositive_wheel_base() -> None:
    with pytest.raises(ValueError):
        unicycle_to_wheel_speeds(1.0, 0.0, wheel_base=0.0)
    with pytest.raises(ValueError):
        wheel_speeds_to_unicycle(1.0, 1.0, wheel_base=-0.1)
