"""Robot command abstraction shared by simulated and real hardware.

Everything upstream (control, navigation) talks to this interface only,
so ``SimulatedRobot`` (Phase 8) can later be swapped for a real motor
driver (Phase 14) without changing any caller. Full hardware safety
machinery (dedicated e-stop line, invalid-command rejection at the wire
protocol level, etc.) is Phase 14 scope; velocity clamping and a command
watchdog are implemented here already, since robot-command code is
safety-critical from the first line per project convention.
"""

from __future__ import annotations

from typing import Protocol

from common.messages import Pose2D, VelocityCommand


class RobotInterface(Protocol):
    """Contract every robot driver (simulated or real) must satisfy."""

    def set_velocity(self, cmd: VelocityCommand) -> None:
        """Command a new body-frame velocity. Implementations must clamp
        to configured limits rather than reject out-of-range commands
        outright, so a slightly-over-limit request degrades safely."""
        ...

    def stop(self) -> None:
        """Immediately command zero velocity."""
        ...

    def get_pose(self) -> Pose2D:
        """Current best pose estimate for this robot."""
        ...
