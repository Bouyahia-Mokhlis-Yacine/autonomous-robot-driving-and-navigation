"""Simulated robot: drives a ``simulation.Environment``'s robot pose from
commanded velocities, implementing the same ``RobotInterface`` a real
motor driver will (Phase 14).

Safety behavior (per project convention, not deferred to Phase 14):
- commanded velocities are clamped to configured max limits.
- a command watchdog stops the robot if no new command arrives within
  ``command_timeout`` seconds (call ``step`` once per simulation tick).
- motion is applied via ``Environment.try_move_robot``, so a command
  that would drive the robot into an obstacle simply does not move it
  (the existing Phase 1 collision check is the safety backstop).

Known limitation: collision is checked against the candidate end pose
for each ``step()`` call, not a continuous swept path. At the control
rates this is designed for (10-30 Hz per the project's timing guidance,
i.e. dt on the order of 0.03-0.1s), that is not a practical gap; a
single call with a very large ``dt`` could in principle tunnel through a
thin obstacle. Continuous collision detection would be the fix if that
ever becomes a real requirement.
"""

from __future__ import annotations

from common.geometry import clamp
from common.messages import Pose2D, VelocityCommand
from simulation.environment import Environment
from slam.pose import integrate_pose


class SimulatedRobot:
    def __init__(
        self,
        environment: Environment,
        max_linear_velocity: float,
        max_angular_velocity: float,
        command_timeout: float,
    ) -> None:
        if max_linear_velocity <= 0.0 or max_angular_velocity <= 0.0:
            raise ValueError("max velocities must be > 0")
        if command_timeout <= 0.0:
            raise ValueError("command_timeout must be > 0")

        self._environment = environment
        self._max_linear_velocity = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._command_timeout = command_timeout
        self._last_command = VelocityCommand(linear=0.0, angular=0.0)
        self._time_since_command = 0.0

    def set_velocity(self, cmd: VelocityCommand) -> None:
        self._last_command = VelocityCommand(
            linear=clamp(cmd.linear, -self._max_linear_velocity, self._max_linear_velocity),
            angular=clamp(cmd.angular, -self._max_angular_velocity, self._max_angular_velocity),
        )
        self._time_since_command = 0.0

    def stop(self) -> None:
        self._last_command = VelocityCommand(linear=0.0, angular=0.0)

    def get_pose(self) -> Pose2D:
        return self._environment.robot_pose

    def step(self, dt: float) -> None:
        """Advance the robot by ``dt`` seconds: apply the watchdog, then integrate motion."""
        if dt < 0.0:
            raise ValueError("dt must be >= 0")

        self._time_since_command += dt
        if self._time_since_command > self._command_timeout:
            self._last_command = VelocityCommand(linear=0.0, angular=0.0)

        candidate_pose = integrate_pose(self._environment.robot_pose, self._last_command, dt)
        self._environment.try_move_robot(candidate_pose)
