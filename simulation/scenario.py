"""Phase 12: a deterministic scenario for the closed-loop simulation.

A ``Scenario`` bundles a ground-truth ``Environment`` (obstacles + the
robot's ground-truth starting pose, in the environment's own corner-origin
frame -- ``0 <= x <= width``, ``0 <= y <= height``) with the goal the
navigation stack should be given, expressed in the SLAM/world frame.

Those two frames are related by a fixed, known translational offset: the
scan-matching pipeline always defines its own world-frame origin
``(0, 0, 0)`` at wherever the robot's first scan was taken (see
``slam/scan_matching_pipeline.py``'s module docstring), which is exactly
``ground_truth_start`` here. ``simulation.closed_loop.ClosedLoopSimulation``
uses that fixed offset for evaluation only -- it is never fed into the
navigation stack itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.messages import Goal, Pose2D
from simulation.environment import Environment
from simulation.obstacles import RectangleObstacle


@dataclass(frozen=True)
class Scenario:
    environment: Environment
    slam_frame_goal: Goal
    ground_truth_start: Pose2D


def default_scenario(robot_width: float, robot_length: float) -> Scenario:
    """One room, one wall (with a doorway gap) between start and goal.

    The robot starts at the room's center (ground truth), which becomes
    SLAM-frame ``(0, 0, 0)``. A wall at SLAM ``x = 1.5`` blocks the direct
    straight line to the goal at SLAM ``(3.0, 0.0)`` -- not a trivial
    straight-line scenario -- except for a generously wide gap (a
    "doorway") centered at SLAM ``y = 1.6``, well clear of both the direct
    line and either wall segment's inflated corner (gap width 2.2m vs. a
    robot on the order of half a meter wide plus inflation, so the planned
    route passes through the middle of the gap, not grazing a corner).
    Deterministic: no randomness anywhere.
    """
    start = Pose2D(x=5.0, y=4.0, theta=0.0)
    environment = Environment(
        width=10.0, height=8.0,
        robot_length=robot_length, robot_width=robot_width,
        robot_pose=start,
    )
    wall_x = start.x + 1.5
    # Doorway gap at SLAM y in [0.5, 2.7] (env y in [4.5, 6.7]).
    environment.add_static_obstacle(RectangleObstacle(
        x=wall_x, y=2.25, width=0.3, height=4.5,  # SLAM y in [-4.0, 0.5]
    ))
    environment.add_static_obstacle(RectangleObstacle(
        x=wall_x, y=7.35, width=0.3, height=1.3,  # SLAM y in [2.7, 4.0]
    ))
    return Scenario(
        environment=environment,
        slam_frame_goal=Goal(x=3.0, y=0.0),
        ground_truth_start=start,
    )
