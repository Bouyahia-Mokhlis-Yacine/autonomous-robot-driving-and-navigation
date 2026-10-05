"""Wheel motors: Raspberry Pi GPIO -> two L298N H-bridges -> 4 DC motors.

Skid-steer, front + back motor on each side
driven together, speed = software PWM on the L298N IN pins through
``gpiozero.Motor`` (lgpio pin factory on the Pi). Pins and tuning live in
config/hardware.yaml ``motors:``.

``GpioDiffDriveRobot`` is a ``RobotInterface``:

    set_velocity(cmd)  body velocity (m/s, rad/s) -> left/right wheel speed
                       -> PWM duty, with, in order:
                         - invalid-command rejection: non-finite -> stop
                         - clamping to robot.yaml's velocity limits
                         - v_side = v -/+ omega * track / 2
                         - duty = v_side / max_wheel_speed_mps, both sides
                           scaled down together above ``max_duty`` (keeps the
                           turn radius instead of clipping one side)
                         - inversion + per-side trim
                         - deadband: |duty| < zero_threshold -> 0; otherwise
                           raised to at least ``min_duty`` (below it the
                           motors stall and just hum)
    stop()             all four motors off, immediately
    get_pose/set_pose  the SLAM estimate (this class owns no odometry --
                       there are no encoders)

Safety in this class (the runtime adds the STOP latch and operator-loss policy
on top, via ``runtime.backends.SafetyGatedRobot``):
    - watchdog thread: no ``set_velocity`` within ``command_timeout`` -> stop
      (robot.yaml limits.command_timeout)
    - ``close()`` always stops first
    - a SEPARATE process, ``hardware.motor_guard``, forces the pins low if
      this process dies without cleaning up (see that module) -- the one
      failure an in-process watchdog cannot cover.

Without gpiozero (e.g. this laptop), or with ``dry_run=True``, motors are
``RecordingMotor`` stand-ins: every duty is kept, nothing touches hardware.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Protocol

from common.config import MotionLimits, MotorConfig
from common.messages import Pose2D, VelocityCommand

logger = logging.getLogger("autonomous_navigation.hardware.motor_driver")

SIDES = {"left": ("front_left", "back_left"), "right": ("front_right", "back_right")}


class MotorLike(Protocol):
    def forward(self, speed: float = 1.0) -> None: ...
    def backward(self, speed: float = 1.0) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class RecordingMotor:
    """Hardware-free stand-in with gpiozero.Motor's method names."""

    def __init__(self, forward_pin: int, backward_pin: int) -> None:
        self.pins = (forward_pin, backward_pin)
        self.value = 0.0          # signed duty, like gpiozero.Motor.value
        self.closed = False

    def forward(self, speed: float = 1.0) -> None:
        self.value = speed

    def backward(self, speed: float = 1.0) -> None:
        self.value = -speed

    def stop(self) -> None:
        self.value = 0.0

    def close(self) -> None:
        self.value, self.closed = 0.0, True


def gpiozero_motor_factory() -> Callable[[int, int], MotorLike] | None:
    """``gpiozero.Motor`` if gpiozero is importable, else None."""
    try:
        from gpiozero import Motor
    except (ImportError, NotImplementedError):
        return None
    return lambda forward_pin, backward_pin: Motor(forward=forward_pin, backward=backward_pin, pwm=True)


def body_to_duty(linear: float, angular: float, cfg: MotorConfig) -> tuple[float, float]:
    """(v, omega) -> signed (left, right) duty, before deadband/trim/inversion."""
    half = cfg.track_width_m / 2.0
    left = (linear - angular * half) / cfg.max_wheel_speed_mps
    right = (linear + angular * half) / cfg.max_wheel_speed_mps
    peak = max(abs(left), abs(right))
    if peak > cfg.max_duty:
        left, right = left * cfg.max_duty / peak, right * cfg.max_duty / peak
    return left, right


def shape_duty(duty: float, trim: float, invert: bool, cfg: MotorConfig, min_duty: float | None = None) -> float:
    """``min_duty``: the stall floor for this motion (default ``cfg.min_duty``;
    see ``motion_min_duty``)."""
    floor = cfg.min_duty if min_duty is None else min_duty
    duty = (-duty if invert else duty) * trim
    if abs(duty) < cfg.zero_threshold:
        return 0.0
    return math.copysign(min(cfg.max_duty, max(floor, abs(duty))), duty)


def motion_min_duty(linear: float, angular: float, cfg: MotorConfig) -> float:
    """Stall floor for a body motion: ``min_turn_duty`` for a turn in place
    (no forward/backward component), ``min_reverse_duty`` when reversing,
    ``min_duty`` otherwise."""
    if abs(linear) <= 1e-6 and abs(angular) > 1e-6:
        return cfg.min_turn_duty
    if linear < -1e-6:
        return cfg.min_reverse_duty
    return cfg.min_duty


class GpioDiffDriveRobot:
    def __init__(
        self,
        motors: MotorConfig,
        limits: MotionLimits,
        motor_factory: Callable[[int, int], MotorLike] | None = None,
        dry_run: bool = False,
        initial_pose: Pose2D | None = None,
        watchdog_period_s: float = 0.05,
        hold_pins_low: Callable[[list[int]], bool] | None = None,
    ) -> None:
        """``hold_pins_low``: what ``close()`` uses to leave the pins driven low
        (default ``hardware.pin_hold.hold_pins_low``; injectable for tests)."""
        if hold_pins_low is None:
            from hardware.pin_hold import hold_pins_low
        self._hold_pins_low = hold_pins_low
        self.cfg = motors
        self.limits = limits
        factory = None if dry_run else (motor_factory or gpiozero_motor_factory())
        self.dry_run = factory is None
        if self.dry_run:
            if not dry_run:
                logger.warning("gpiozero not available -- motors in DRY-RUN mode (nothing is driven)")
            factory = RecordingMotor
        self._motors = {name: factory(*motors.pins[name]) for name in ("front_left", "back_left",
                                                                        "front_right", "back_right")}
        self._lock = threading.Lock()
        self._pose = initial_pose or Pose2D(0.0, 0.0, 0.0)
        self.last_command = VelocityCommand(0.0, 0.0)
        self.last_duty = (0.0, 0.0)
        self._last_command_time = time.monotonic()
        self.watchdog_trips = 0
        self.rejected_commands = 0
        self._closed = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, args=(watchdog_period_s,), name="motor-watchdog",
                                          daemon=True)
        self._watchdog.start()
        self._apply(0.0, 0.0)
        logger.info("Motors ready (%s): pins %s", "dry-run" if self.dry_run else "gpiozero", motors.pins)

    # -------------------------------------------------------- RobotInterface --

    def set_velocity(self, cmd: VelocityCommand) -> None:
        if not (math.isfinite(cmd.linear) and math.isfinite(cmd.angular)):
            self.rejected_commands += 1
            logger.error("Rejected invalid velocity command %s -- stopping", cmd)
            self.stop()
            return
        lim = self.limits
        linear = max(-lim.max_linear_velocity, min(lim.max_linear_velocity, cmd.linear))
        angular = max(-lim.max_angular_velocity, min(lim.max_angular_velocity, cmd.angular))
        left, right = body_to_duty(linear, angular, self.cfg)
        with self._lock:
            if self._closed.is_set():
                return
            self._last_command_time = time.monotonic()
            self.last_command = VelocityCommand(linear, angular)
            self._apply(left, right, motion_min_duty(linear, angular, self.cfg))

    def stop(self) -> None:
        with self._lock:
            self.last_command = VelocityCommand(0.0, 0.0)
            self._apply(0.0, 0.0)

    def get_pose(self) -> Pose2D:
        return self._pose

    def set_pose(self, pose: Pose2D) -> None:
        self._pose = pose

    def close(self) -> None:
        with self._lock:
            self._apply(0.0, 0.0)
            self._closed.set()
            for motor in self._motors.values():
                try:
                    motor.close()
                except Exception:  # noqa: BLE001 -- closing must never prevent closing the others
                    logger.exception("Error closing a motor")
            if not self.dry_run:
                # gpiozero leaves closed pins as FLOATING inputs, which the L298N
                # can read as HIGH (a motor runs) -- drive them low instead.
                self._hold_pins_low([p for pair in self.cfg.pins.values() for p in pair])
        logger.info("Motors stopped and released")

    # --------------------------------------------------------------- helpers --

    def motor_values(self) -> dict[str, float]:
        """Signed duty per motor (RecordingMotor / gpiozero ``value``)."""
        return {name: float(getattr(m, "value", 0.0)) for name, m in self._motors.items()}

    def _apply(self, left: float, right: float, min_duty: float | None = None) -> None:
        """Caller holds the lock (or is the constructor)."""
        c = self.cfg
        duties = {"left": shape_duty(left, c.left_trim, c.invert_left, c, min_duty),
                  "right": shape_duty(right, c.right_trim, c.invert_right, c, min_duty)}
        for side, names in SIDES.items():
            for name in names:
                motor, duty = self._motors[name], duties[side]
                if duty > 0.0:
                    motor.forward(duty)
                elif duty < 0.0:
                    motor.backward(-duty)
                else:
                    motor.stop()
        self.last_duty = (duties["left"], duties["right"])

    def _watch(self, period: float) -> None:
        while not self._closed.wait(period):
            with self._lock:
                idle = time.monotonic() - self._last_command_time
                moving = self.last_duty != (0.0, 0.0)
                if moving and idle > self.limits.command_timeout:
                    self._apply(0.0, 0.0)
                    self.last_command = VelocityCommand(0.0, 0.0)
                    self.watchdog_trips += 1
                    logger.warning("Motor watchdog: no command for %.2f s -- stopped", idle)
