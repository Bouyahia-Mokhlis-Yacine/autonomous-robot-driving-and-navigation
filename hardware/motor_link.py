"""Pi side of the motor-microcontroller link (protocol: hardware.motor_protocol).

``MotorCommandLink`` frames and sends commands and reads the controller's
status lines. ``MicrocontrollerDiffDriveRobot`` is the ``RobotInterface`` for
a robot whose motors hang off such a controller: the same velocity -> duty
shaping as ``GpioDiffDriveRobot`` (limits, mixing, deadband, trim, inversion),
sent as per-mille duty.

The runtime's control loop calls ``set_velocity``/``stop`` every tick (10 Hz,
well inside the controller's 300 ms watchdog). If the loop stalls, crashes or
the USB link drops, frames stop and the CONTROLLER stops the motors -- nothing
on this side has to work for that. After a trip the controller refuses motion
until it is sent a zero command; ``controller_tripped()`` exposes the trip so
the runtime can latch STOP and require an operator RESUME before re-arming.

Not wired into the runtime yet: the robot has no motor microcontroller
(motors are on the Pi's GPIO). ``hardware.fake_motor_controller`` stands in
for it in tests.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Callable

from common.config import MotionLimits, MotorConfig
from common.messages import Pose2D, VelocityCommand
from hardware.motor_driver import body_to_duty, shape_duty
from hardware.motor_protocol import MAX_DUTY, TRIPPED, STOPPED, encode_motion, encode_stop, parse_status

logger = logging.getLogger("autonomous_navigation.hardware.motor_link")


class MotorCommandLink:
    """Sequenced, timestamped command frames to the controller. Thread-safe."""

    def __init__(self, write: Callable[[bytes], object], clock: Callable[[], float] = time.monotonic,
                 session: int | None = None) -> None:
        self._write = write
        self._clock = clock
        self.session = random.randrange(1, 0x10000) if session is None else session
        self._seq = 0
        self._lock = threading.Lock()
        self.last_status: dict | None = None
        self.write_errors = 0

    def send_duty(self, left: float, right: float) -> None:
        """Signed duty in [-1, 1] per side (already shaped)."""
        l, r = (int(round(max(-1.0, min(1.0, d)) * MAX_DUTY)) for d in (left, right))
        with self._lock:
            self._seq += 1
            self._send(encode_motion(self.session, self._seq, self._host_ms(), l, r))

    def send_stop(self) -> None:
        with self._lock:
            self._seq += 1
            self._send(encode_stop(self.session, self._seq, self._host_ms()))

    def handle_line(self, line: bytes) -> None:
        """Feed a line received from the controller."""
        status = parse_status(line)
        if status is not None:
            self.last_status = status

    def controller_tripped(self) -> bool:
        s = self.last_status
        return s is not None and s["session"] == self.session and s["state"] in (TRIPPED, STOPPED)

    def _host_ms(self) -> int:
        return int(self._clock() * 1000.0)

    def _send(self, frame: bytes) -> None:
        try:
            self._write(frame)
        except OSError:
            # The controller stops on its own when frames stop arriving.
            self.write_errors += 1
            logger.exception("Motor link write failed")


class MicrocontrollerDiffDriveRobot:
    """``RobotInterface`` over a ``MotorCommandLink``."""

    def __init__(self, link: MotorCommandLink, motors: MotorConfig, limits: MotionLimits,
                 initial_pose: Pose2D | None = None) -> None:
        self.link = link
        self.cfg = motors
        self.limits = limits
        self._pose = initial_pose or Pose2D(0.0, 0.0, 0.0)
        self.last_command = VelocityCommand(0.0, 0.0)
        self.last_duty = (0.0, 0.0)
        self.rejected_commands = 0

    def set_velocity(self, cmd: VelocityCommand) -> None:
        if not (math.isfinite(cmd.linear) and math.isfinite(cmd.angular)):
            self.rejected_commands += 1
            logger.error("Rejected invalid velocity command %s -- stopping", cmd)
            self.stop()
            return
        lim, c = self.limits, self.cfg
        linear = max(-lim.max_linear_velocity, min(lim.max_linear_velocity, cmd.linear))
        angular = max(-lim.max_angular_velocity, min(lim.max_angular_velocity, cmd.angular))
        left, right = body_to_duty(linear, angular, c)
        duty = (shape_duty(left, c.left_trim, c.invert_left, c), shape_duty(right, c.right_trim, c.invert_right, c))
        self.last_command = VelocityCommand(linear, angular)
        self.last_duty = duty
        self.link.send_duty(*duty)

    def stop(self) -> None:
        self.last_command = VelocityCommand(0.0, 0.0)
        self.last_duty = (0.0, 0.0)
        self.link.send_stop()

    def arm(self) -> None:
        """A zero motion command: re-arms the controller after a trip/STOP."""
        self.last_command = VelocityCommand(0.0, 0.0)
        self.last_duty = (0.0, 0.0)
        self.link.send_duty(0.0, 0.0)

    def get_pose(self) -> Pose2D:
        return self._pose

    def set_pose(self, pose: Pose2D) -> None:
        self._pose = pose
