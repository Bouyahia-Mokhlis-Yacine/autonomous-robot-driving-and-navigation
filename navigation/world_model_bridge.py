"""Phase 10: the bridge between the new SLAM/world-model stack (Phases
1-8: ``slam.scan_matching_pipeline``, ``mapping.persistent_world_model``)
and the existing, unmodified navigation stack (``planning.global_planner``,
``navigation.navigation_manager``, ``robot.sensor_only_robot``).

Per the Phase 9 inspection, these two subsystems were previously
completely disconnected -- no coordinate conversion is needed (both
already use ``mapping.occupancy_grid.OccupancyGridMap``'s exact
world<->grid convention), only plumbing. This module supplies exactly
that plumbing, as two small, independent, read-only functions -- it
implements no new logic of its own (inflation, pose truth-model, and
map semantics all remain owned by their existing modules).

``navigation.replanner.DynamicObstacleLayer`` is deliberately NOT
integrated here -- whether it still has a role once
``PersistentWorldModel`` owns tentative/confirmed persistence is an
open design question the Phase 9 inspection raised and explicitly
deferred; this bridge neither uses nor removes it.

The existing ``UNKNOWN``-is-traversable policy in ``planning.astar``/
``planning.path_smoothing``/``navigation.goal_manager`` is unchanged
and untouched by this module.
"""

from __future__ import annotations

from common.config import RobotConfig
from common.geometry import offset_pose
from mapping.inflation import compute_inflation_radius, inflate_occupancy_grid
from mapping.occupancy_grid import OccupancyGridMap
from mapping.persistent_world_model import PersistentWorldModel
from robot.sensor_only_robot import SensorOnlyRobot
from slam.scan_matching_pipeline import PipelineResult


def build_planning_map(world_model: PersistentWorldModel, robot_config: RobotConfig) -> OccupancyGridMap:
    """``PersistentWorldModel`` -> an inflated ``OccupancyGridMap`` ready
    for ``GlobalPlanner.plan()`` / ``NavigationManager.set_goal()`` /
    ``NavigationManager.update()`` / ``Replanner`` -- all consumed
    completely unmodified, exactly the same "construct an
    ``OccupancyGridMap``, then assign its ``.grid``" pattern already
    used by ``real_sensor_walkthrough.py`` and proven by
    ``tests/test_global_planner.py::test_global_planner_narrow_passage_rejected_after_inflation``.

    Read-only with respect to ``world_model``: only calls
    ``world_model.get_grid()`` (itself a read-only accessor) and never
    writes back to it. ``inflate_occupancy_grid()`` already returns a
    copy, so the world model's own grid is never mutated -- verified
    explicitly by this module's tests, not just assumed.

    Inflation radius comes from the project's existing
    ``mapping.inflation.compute_inflation_radius`` helper and the
    existing ``config/robot.yaml``-sourced ``RobotConfig`` -- no
    dimensions are hard-coded here.

    Dimensions/resolution/origin (world/grid frame convention) are
    carried over exactly from the world model's own grid, so no
    coordinate conversion happens anywhere in this function -- per the
    Phase 9 finding that both stacks already share the same
    ``OccupancyGridMap`` convention.
    """
    source_grid = world_model.get_grid()

    inflation_radius = compute_inflation_radius(
        robot_config.robot.width, robot_config.robot.length, robot_config.safety.inflation_radius
    )
    inflated_grid = inflate_occupancy_grid(source_grid, inflation_radius)

    planning_map = OccupancyGridMap(
        resolution=source_grid.resolution,
        width_m=source_grid.width * source_grid.resolution,
        height_m=source_grid.height * source_grid.resolution,
        origin_x=source_grid.origin_x,
        origin_y=source_grid.origin_y,
    )
    planning_map.grid = inflated_grid
    return planning_map


def apply_accepted_pose(robot: SensorOnlyRobot, result: PipelineResult,
                        lidar_offset: tuple[float, float] = (0.0, 0.0)) -> bool:
    """Push ``result.pose`` into ``robot`` -- but only when
    ``result.accepted`` is True.

    This is the entire localization contract established across
    Phases 4/6/8, restated here as the one place navigation code
    touches it: a rejected registration must NEVER update the robot's
    pose (not the seed, not a diagnostic prediction, not
    ``result.pose`` itself, which on rejection is already frozen at
    the last accepted pose per ``PipelineResult``'s own contract --
    but this function does not even rely on that; it simply never
    calls ``set_pose`` at all when ``accepted`` is False, so the
    previous pose already held by ``robot`` remains authoritative by
    construction, regardless of what ``result.pose`` happens to
    contain).

    Returns whether the pose was actually updated, for the caller's
    own logging/diagnostics -- purely informational.

    ``lidar_offset``: the LiDAR mount in the robot frame (robot.yaml
    ``robot.lidar_x``/``lidar_y``). ``result.pose`` is the LiDAR's pose (SLAM
    tracks the sensor); the robot receives its footprint centre's pose.
    """
    if not result.accepted:
        return False
    robot.set_pose(robot_pose_from_lidar(result.pose, lidar_offset))
    return True


def robot_pose_from_lidar(lidar_pose, lidar_offset: tuple[float, float]):
    """The robot (footprint centre) pose from the LiDAR pose."""
    if lidar_offset == (0.0, 0.0):
        return lidar_pose
    return offset_pose(lidar_pose, -lidar_offset[0], -lidar_offset[1])
