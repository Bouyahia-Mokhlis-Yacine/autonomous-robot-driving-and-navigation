"""A ``RobotInterface`` implementation with no motors: pose comes from a
real sensor pipeline (pushed in externally each tick), velocity commands
go nowhere.

Exists for hardware-in-the-loop testing before any motor is connected
(carry the real LiDAR by hand instead of driving it): the exact same
``NavigationManager``/``GlobalPlanner``/GUI code that will eventually
drive a real robot can run unmodified, with a human providing the
"actuation" instead. ``NavigationManager`` calling ``set_velocity``/
``stop`` still works (and still matters for its state machine and
logging) -- there is just nothing physical listening on the other end.
"""

from __future__ import annotations

from common.messages import Pose2D, VelocityCommand


class SensorOnlyRobot:
    def __init__(self, initial_pose: Pose2D) -> None:
        self._pose = initial_pose
        self.last_command = VelocityCommand(linear=0.0, angular=0.0)

    def set_pose(self, pose: Pose2D) -> None:
        """Called externally each tick with the latest real localization estimate."""
        self._pose = pose

    def set_velocity(self, cmd: VelocityCommand) -> None:
        self.last_command = cmd  # recorded only; nothing physical to command

    def stop(self) -> None:
        self.last_command = VelocityCommand(linear=0.0, angular=0.0)

    def get_pose(self) -> Pose2D:
        return self._pose
