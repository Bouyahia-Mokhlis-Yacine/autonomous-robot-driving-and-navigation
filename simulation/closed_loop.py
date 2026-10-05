"""Phase 12: closed-loop software-only navigation simulation.

Wires the ground-truth simulator (``simulation.environment.Environment`` +
``simulation.simulated_lidar.SimulatedLidar`` + ``robot.simulated_robot.
SimulatedRobot``) to the real, UNMODIFIED navigation stack
(``slam.scan_matching_pipeline.ScanMatchingPipeline``,
``mapping.persistent_world_model.PersistentWorldModel``,
``navigation.world_model_bridge``, ``planning.global_planner.GlobalPlanner``,
``navigation.navigation_manager.NavigationManager``,
``robot.sensor_only_robot.SensorOnlyRobot``) -- the same composition
``real_sensor_walkthrough.py`` uses for real hardware, except every sensor
reading here is synthetic and ground truth exists only for evaluation.

Architecture::

    Environment (ground truth: obstacles, robot_pose)
            |
            v
    SimulatedLidar.scan(environment, ground_truth_pose)  -- reads ground truth
            |
            v  LaserScan (synthetic, but the same message type real hardware produces)
    ScanMatchingPipeline.process_scan()  -- existing localization, unmodified
            |                              (also integrates into PersistentWorldModel)
            v  PipelineResult (estimated pose)
    apply_accepted_pose() -> SensorOnlyRobot.set_pose()  -- ONLY on result.accepted
            |
            v
    build_planning_map(world_model, robot_cfg)  -- existing bridge, unmodified
            |
            v
    NavigationManager.update()  -- reads SensorOnlyRobot.get_pose() (the ESTIMATE),
            |                       plans/replans with GlobalPlanner, returns a
            |                       VelocityCommand (also recorded, as a no-op, on
            |                       SensorOnlyRobot -- nothing physical listens there)
            v
    (this module) forwards that same VelocityCommand to SimulatedRobot
            |
            v
    SimulatedRobot.step() -- integrates GROUND-TRUTH motion, collision-checked
            |
            v
    Environment.robot_pose updated (ground truth) -> next tick's SimulatedLidar.scan()

Ground-truth isolation (see the class docstring below for the precise
guarantee): ``Environment``/``SimulatedRobot`` ground truth is read in
exactly two places in this module -- the LiDAR ray-casting call, and the
evaluation/metrics code at the end of ``step()``. It is never passed to
``ScanMatchingPipeline``, ``PersistentWorldModel``, ``GlobalPlanner``, or
``NavigationManager``.

Bootstrap (see ``slam/scan_matching_pipeline.py``): the pipeline assumes
the robot is stationary until its own ``bootstrap_status`` reaches
``COMPLETE``. This module honors that by never calling
``NavigationManager.set_goal()`` before that point -- until then,
``NavigationManager`` stays ``IDLE``, and ``update()`` always returns a
zero ``VelocityCommand``, so the ground-truth robot never moves. No
separate "hold still" state machine is needed; it falls directly out of
the existing ``NavigationManager`` behavior.

Sensor realism (``sensor_simulation``, see ``config/lidar_simulation.yaml``):
``None`` or ``mode: ideal`` runs exactly the original loop above -- one
perfect, instantaneous ``SimulatedLidar`` scan per tick, processed
synchronously in that same tick. ``mode: realistic`` replaces only the
sensing half of the loop, and only because the original one structurally
assumes an instantaneous scan processed instantly:

    per control tick (dt):
      1. ground truth advances in ``physics_dt`` substeps under the command
         chosen last tick; every substep pose goes into a ``PoseHistory``
      2. ``RealisticLidar`` emits every revolution completed by now, each
         sample ray-cast from the interpolated pose at ITS OWN instant
         (motion distortion), with noise/dropouts/partial scans, stamped
         with its delivery time
      3. ``ScanProcessingSimulator`` replays scan arrivals and SLAM
         completions in time order under the configured queue policy and
         processing-delay model; ``ScanMatchingPipeline.process_scan`` runs
         at each completion instant, so its pose AND its map update become
         visible only after the modeled latency
      4. planning/navigation run exactly as before, on whatever the most
         recent completed estimate is -- i.e. on a pose that is typically
         0.2-0.35 s old, as it will be on the real robot
The navigation stack itself (pipeline, world model, planner, manager) is
the same unmodified code in both modes.

``goal=None``: no goal is set automatically after bootstrap; goals come
only from whoever drives ``nav_manager`` (the operator runtime,
``runtime/backends.py``).

``drive_script`` (optional, both modes): a fixed list of
``(duration_s, VelocityCommand)`` segments that replaces the navigation
manager's commands, starting once bootstrap completes (no goal is set).
Used to drive the ideal and realistic sensors along the SAME ground-truth
trajectory for side-by-side comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from common.config import NavigationConfig, RobotConfig
from common.geometry import distance, offset_pose, wrap_angle
from common.messages import Goal, Pose2D, VelocityCommand
from control.path_follower import PathFollower, path_follower_config
from navigation.motion_safety import MotionSafety, motion_safety_config
from mapping.persistent_world_model import PersistentWorldModel
from navigation.navigation_manager import NavigationManager, NavigationState
from navigation.replanner import Replanner, ReplannerConfig
from navigation.world_model_bridge import apply_accepted_pose, build_planning_map, robot_pose_from_lidar
from slam.motion_model import mounted_prior
from planning.global_planner import GlobalPlanner
from robot.sensor_only_robot import SensorOnlyRobot
from robot.simulated_robot import SimulatedRobot
from simulation.environment import Environment
from simulation.pose_history import PoseHistory
from simulation.realistic_lidar import RealisticLidar, SimulatedScanRecord
from simulation.sensor_processing import CompletedScan, ScanProcessingSimulator
from simulation.sensor_run_log import SensorRunLog
from simulation.sensor_simulation_config import SensorSimulationConfig
from simulation.simulated_lidar import SimulatedLidar
from slam.localization_health import LocalizationHealthConfig, LocalizationHealthMonitor
from slam.motion_model import CommandHistory
from slam.pose import integrate_pose
from slam.scan_matching_pipeline import BootstrapConfig, BootstrapStatus, PipelineConfig, PipelineResult, ScanMatchingPipeline

PLANNING_REBUILD_INTERVAL = 0.3  # seconds; matches real_sensor_walkthrough.py


def scan_pose_time(scan) -> float:
    """The instant a SLAM pose from ``scan`` best describes. A scan is
    acquired over ``scan_time`` and stamped when delivered, so a pose
    registered from all of it corresponds to roughly the middle of the
    rotation. Sources that report no ``scan_time`` (the real driver today)
    fall back to the timestamp itself."""
    return scan.timestamp - 0.5 * scan.scan_time


@dataclass
class EvaluationMetrics:
    """Evaluation-only metrics: nothing here is ever read back into the
    navigation stack (see the module docstring's ground-truth-isolation
    guarantee). All errors are measured, never asserted in advance."""

    goal_reached: bool = False
    collision: bool = False
    simulation_time: float = 0.0
    ticks: int = 0
    final_nav_state: str = ""

    bootstrap_completed: bool = False
    bootstrap_duration: float | None = None

    final_position_error: float | None = None
    final_heading_error: float | None = None
    mean_position_error: float | None = None
    max_position_error: float = 0.0

    ground_truth_distance: float = 0.0
    planned_path_length: float | None = None

    # Localization health (slam.localization_health): how often, and for how
    # long, navigation had to stop because the pose was not trustworthy.
    localization_loss_events: int = 0
    time_localization_invalid: float = 0.0

    _error_sum: float = field(default=0.0, repr=False)
    _error_count: int = field(default=0, repr=False)


class ClosedLoopSimulation:
    """One deterministic, fixed-timestep closed-loop simulation run.

    See the module docstring for the full architecture and the
    ground-truth isolation guarantee. ``environment``/``ground_truth_start``
    are the ONLY ground-truth inputs; everything the navigation stack
    receives is derived from simulated sensor observations.
    """

    def __init__(
        self,
        environment: Environment,
        ground_truth_start: Pose2D,
        goal: Goal | None,
        nav_cfg: NavigationConfig,
        robot_cfg: RobotConfig,
        dt: float = 0.1,
        max_simulation_time: float = 90.0,
        world_model_width_m: float = 14.0,
        world_model_height_m: float = 12.0,
        sensor_simulation: SensorSimulationConfig | None = None,
        drive_script: list[tuple[float, VelocityCommand]] | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if max_simulation_time <= 0.0:
            raise ValueError("max_simulation_time must be > 0")

        self.environment = environment
        self.ground_truth_start = ground_truth_start
        self.goal = goal
        self.dt = dt
        self.max_simulation_time = max_simulation_time
        self.robot_cfg = robot_cfg
        # LiDAR mount (robot.yaml): the simulated sensor sits there, SLAM
        # tracks it, navigation receives the footprint-centre pose.
        self.lidar_offset = (robot_cfg.robot.lidar_x, robot_cfg.robot.lidar_y)

        self.lidar = SimulatedLidar(config=nav_cfg.lidar)
        self.ground_truth_robot = SimulatedRobot(
            environment=environment,
            max_linear_velocity=robot_cfg.limits.max_linear_velocity,
            max_angular_velocity=robot_cfg.limits.max_angular_velocity,
            command_timeout=robot_cfg.limits.command_timeout,
        )

        # A world model sized to comfortably cover this scenario's
        # SLAM-frame extent (not the project-wide 20x20 default, which
        # would cost more A*/inflation time than this small room needs)
        # -- same resolution as real hardware config, just a smaller area.
        self.world_model = PersistentWorldModel(
            resolution=nav_cfg.mapping.resolution,
            width_m=world_model_width_m,
            height_m=world_model_height_m,
            origin_x=-world_model_width_m / 2.0,
            origin_y=-world_model_height_m / 2.0,
            max_range_policy=nav_cfg.mapping.max_range_policy,
            max_range_clear_distance=nav_cfg.mapping.max_range_clear_distance,
        )
        lh = nav_cfg.localization
        limits = robot_cfg.limits
        # Every command navigation sends, time-stamped (commanded motion --
        # see slam.motion_model): seeds ICP and, in realistic mode,
        # propagates the pose over the processing latency.
        self.command_history = CommandHistory(limits.max_linear_velocity, limits.max_angular_velocity)
        map_filter = dict(min_integration_translation=lh.map_update_min_translation_m,
                          min_integration_rotation=math.radians(lh.map_update_min_rotation_deg))
        pipeline_config = PipelineConfig(**map_filter)
        if lh.time_aware_validation:
            pipeline_config = PipelineConfig(
                **map_filter,
                max_linear_velocity=limits.max_linear_velocity,
                max_angular_velocity=limits.max_angular_velocity,
                velocity_margin=lh.velocity_margin,
                translation_tolerance=lh.translation_tolerance_m,
                rotation_tolerance=math.radians(lh.rotation_tolerance_deg),
            )
        self.pipeline = ScanMatchingPipeline(
            reference_provider=self.world_model,
            world_model=self.world_model,
            pipeline_config=pipeline_config,
            bootstrap_config=BootstrapConfig(min_duration=lh.bootstrap_duration_s, timeout=lh.bootstrap_timeout_s),
            motion_prior=mounted_prior(self.command_history, *self.lidar_offset) if lh.use_commanded_motion else None,
        )

        # The SLAM/world frame is always born at (0, 0, 0) by the
        # pipeline's own design (see module docstring) -- this is that
        # same fixed convention, not a ground-truth value.
        self.nav_robot = SensorOnlyRobot(initial_pose=Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=0.0))
        follower = PathFollower(path_follower_config(robot_cfg))
        # Newest raw scan (robot frame) for direction-aware motion safety.
        self._latest_scan = None
        # Kept as public attributes (not just passed into NavigationManager)
        # purely so diagnostic/instrumentation code (simulation.diagnostics)
        # can wrap their public methods from the outside -- NavigationManager
        # itself is completely unaware of and unaffected by this; it just
        # holds the same objects, per its existing dependency-injection design.
        self.global_planner = GlobalPlanner(
            allow_diagonal=nav_cfg.planning.allow_diagonal,
            unknown_cost_multiplier=nav_cfg.planning.unknown_cost_multiplier,
        )
        self.replanner = Replanner(ReplannerConfig(
            replan_interval=nav_cfg.planning.replan_interval,
            deviation_threshold=robot_cfg.safety.path_deviation_threshold,
        ))
        self.delay_compensation = lh.delay_compensation
        # (pose, time the pose describes) of the newest accepted SLAM result.
        self._pose_anchor: tuple[Pose2D, float] | None = None
        self.localization_health = LocalizationHealthMonitor(LocalizationHealthConfig(
            degraded_after_s=lh.degraded_after_s, lost_after_s=lh.lost_after_s,
            recovery_confirmations=lh.recovery_confirmations,
        ))
        self.nav_manager = NavigationManager(
            global_planner=self.global_planner,
            path_follower=follower,
            robot=self.nav_robot,
            replanner=self.replanner,
            obstacle_stop_distance=robot_cfg.safety.emergency_stop_distance,
            path_clearance_margin=nav_cfg.planning.path_clearance_margin,
            localization=self.localization_health,
            motion_safety=MotionSafety(motion_safety_config(robot_cfg), scan_provider=lambda: self._latest_scan),
        )

        self.metrics = EvaluationMetrics()
        self.planning_map = build_planning_map(self.world_model, robot_cfg)
        self._last_planning_rebuild = -math.inf
        self._goal_set = False
        self._sim_time = 0.0
        self._last_ground_truth_pose = environment.robot_pose

        self.drive_script = list(drive_script) if drive_script is not None else None
        self._script_start_time: float | None = None

        self.sensor_simulation = sensor_simulation
        self.realistic = sensor_simulation is not None and sensor_simulation.is_realistic
        # Realistic-mode state (unused in ideal mode).
        self.last_scan_record: SimulatedScanRecord | None = None
        self.last_completed: CompletedScan | None = None
        self.sensor_log = SensorRunLog()
        if self.realistic:
            seed = sensor_simulation.seed
            self.realistic_lidar = RealisticLidar(
                lidar_config=nav_cfg.lidar, config=sensor_simulation.lidar, seed=seed,
            )
            # A separate, derived seed so processing-delay draws never shift
            # the sensor's own random stream (and vice versa).
            self.processor = ScanProcessingSimulator(
                sensor_simulation.processing, seed=None if seed is None else seed + 1_000_003,
            )
            self.pose_history = PoseHistory()
            self.pose_history.append(0.0, environment.robot_pose)
            self._current_cmd = VelocityCommand(linear=0.0, angular=0.0)

    def _sensor_pose(self, robot_pose: Pose2D) -> Pose2D:
        """Ground-truth LiDAR pose from the ground-truth robot (footprint) pose."""
        if self.lidar_offset == (0.0, 0.0):
            return robot_pose
        return offset_pose(robot_pose, *self.lidar_offset)

    def _ground_truth_to_slam(self, pose: Pose2D) -> Pose2D:
        """Translate a ground-truth ``Environment`` (footprint-centre) pose into
        the SLAM/world frame, whose origin is the LiDAR's pose at the start
        (the pipeline tracks the sensor). Evaluation-only -- never passed to
        anything in the navigation stack."""
        origin = self._sensor_pose(self.ground_truth_start)
        return Pose2D(
            x=pose.x - origin.x,
            y=pose.y - origin.y,
            theta=wrap_angle(pose.theta - origin.theta),
        )

    def _scripted_command(self, bootstrap_done: bool) -> VelocityCommand | None:
        """The drive-script command for the current time, or ``None`` when no
        script is configured (navigation decides). Zero before bootstrap
        completes and after the script ends."""
        if self.drive_script is None:
            return None
        if not bootstrap_done:
            return VelocityCommand(linear=0.0, angular=0.0)
        if self._script_start_time is None:
            self._script_start_time = self._sim_time
        elapsed = self._sim_time - self._script_start_time
        for duration, cmd in self.drive_script:
            if elapsed < duration:
                return cmd
            elapsed -= duration
        return VelocityCommand(linear=0.0, angular=0.0)

    def step(self) -> None:
        """Advance the simulation by one ``dt``. See the module docstring
        for the exact, ground-truth-isolated data flow."""
        if self.realistic:
            self._step_realistic()
            return
        self._sim_time += self.dt

        ground_truth_pose = self.environment.robot_pose  # read here ONLY for sensing + evaluation
        scan = self.lidar.scan(self.environment, self._sensor_pose(ground_truth_pose), timestamp=self._sim_time)
        self._latest_scan = scan

        result = self.pipeline.process_scan(scan)
        apply_accepted_pose(self.nav_robot, result, self.lidar_offset)
        self.localization_health.on_result(result, scan.timestamp)
        self.localization_health.update(self._sim_time)

        bootstrap_done = result.bootstrap_status == BootstrapStatus.COMPLETE
        if bootstrap_done and not self.metrics.bootstrap_completed:
            self.metrics.bootstrap_completed = True
            self.metrics.bootstrap_duration = self._sim_time

        if self._sim_time - self._last_planning_rebuild >= PLANNING_REBUILD_INTERVAL:
            self.planning_map = build_planning_map(self.world_model, self.robot_cfg)
            self._last_planning_rebuild = self._sim_time

        # Bootstrap precondition: never command a goal (and therefore never
        # move -- see module docstring) before the pipeline itself reports
        # BOOTSTRAPPING complete.
        if bootstrap_done and not self._goal_set and self.drive_script is None and self.goal is not None:
            self.nav_manager.set_goal(self.goal, self.planning_map, self.world_model.occupancy_grid)
            self._goal_set = True

        cmd = self.nav_manager.update(self.planning_map, self.dt, self.world_model.occupancy_grid)
        scripted = self._scripted_command(bootstrap_done)
        if scripted is not None:
            cmd = scripted

        # Collision check against the SAME candidate motion about to be
        # applied to the ground-truth robot -- mirrors SimulatedRobot.step's
        # own integration exactly, so this reflects a real attempted
        # collision (an evaluation failure), never silently absorbed.
        candidate_pose = integrate_pose(ground_truth_pose, cmd, self.dt)
        if self.environment.check_collision(candidate_pose):
            self.metrics.collision = True

        self.command_history.record(self._sim_time, cmd)
        self.ground_truth_robot.set_velocity(cmd)
        self.ground_truth_robot.step(self.dt)

        new_ground_truth_pose = self.environment.robot_pose
        self.metrics.ground_truth_distance += distance(
            self._last_ground_truth_pose.x, self._last_ground_truth_pose.y,
            new_ground_truth_pose.x, new_ground_truth_pose.y,
        )
        self._last_ground_truth_pose = new_ground_truth_pose
        self._update_evaluation(new_ground_truth_pose)

    def _update_evaluation(self, new_ground_truth_pose: Pose2D) -> None:
        """Evaluation-only bookkeeping shared by both sensor modes."""
        expected = self._ground_truth_to_slam(new_ground_truth_pose)
        estimated = self.nav_robot.get_pose()
        position_error = distance(expected.x, expected.y, estimated.x, estimated.y)
        heading_error = abs(wrap_angle(expected.theta - estimated.theta))

        self.metrics._error_sum += position_error
        self.metrics._error_count += 1
        self.metrics.max_position_error = max(self.metrics.max_position_error, position_error)
        self.metrics.final_position_error = position_error
        self.metrics.final_heading_error = heading_error
        self.metrics.mean_position_error = self.metrics._error_sum / self.metrics._error_count

        path = self.nav_manager.get_path()
        if self.metrics.planned_path_length is None and path is not None and not path.is_empty():
            self.metrics.planned_path_length = sum(
                distance(a.x, a.y, b.x, b.y) for a, b in zip(path.points, path.points[1:])
            )

        self.metrics.localization_loss_events = self.localization_health.loss_events
        if self._goal_set and not self.localization_health.is_localization_valid():
            self.metrics.time_localization_invalid += self.dt

        self.metrics.simulation_time = self._sim_time
        self.metrics.ticks += 1
        self.metrics.final_nav_state = self.nav_manager.state.value
        if self.nav_manager.state == NavigationState.GOAL_REACHED:
            self.metrics.goal_reached = True

    # ------------------------------------------------------ realistic mode --

    def _step_realistic(self) -> None:
        """One control tick with the realistic sensor + asynchronous
        processing. See the module docstring's "Sensor realism" section."""
        assert self.sensor_simulation is not None
        t0 = self._sim_time
        t1 = t0 + self.dt
        cmd = self._current_cmd

        # 1. Ground truth moves in substeps under last tick's command.
        substeps = max(1, int(round(self.dt / self.sensor_simulation.physics_dt)))
        h = self.dt / substeps
        self.ground_truth_robot.set_velocity(cmd)
        for i in range(substeps):
            gt = self.environment.robot_pose
            if self.environment.check_collision(integrate_pose(gt, cmd, h)):
                self.metrics.collision = True
            self.ground_truth_robot.step(h)
            new_gt = self.environment.robot_pose
            self.metrics.ground_truth_distance += distance(gt.x, gt.y, new_gt.x, new_gt.y)
            self.pose_history.append(t0 + (i + 1) * h, new_gt)
        self._sim_time = t1
        self._last_ground_truth_pose = self.environment.robot_pose

        # 2. Every revolution completed by now (ground truth read for sensing only).
        for record in self.realistic_lidar.acquire_until(
                self.environment, lambda t: self._sensor_pose(self.pose_history.pose_at(t)), t1):
            self.sensor_log.record_revolution(record)
            self.last_scan_record = record
            if record.scan is not None:
                self._latest_scan = record.scan
                self.processor.submit(record.scan)

        # 3. Arrivals / SLAM completions up to now, in time order.
        for completed in self.processor.advance(t1, self.pipeline.process_scan):
            self.sensor_log.record_completion(completed)
            self.last_completed = completed
            result: PipelineResult = completed.result
            if apply_accepted_pose(self.nav_robot, result, self.lidar_offset):
                self._pose_anchor = (robot_pose_from_lidar(result.pose, self.lidar_offset),
                                     scan_pose_time(completed.scan))
            self.localization_health.on_result(result, completed.scan.timestamp)
            if result.bootstrap_status == BootstrapStatus.COMPLETE and not self.metrics.bootstrap_completed:
                self.metrics.bootstrap_completed = True
                self.metrics.bootstrap_duration = completed.finish_time

        # 4. Delay compensation: the newest accepted pose describes the robot
        #    at its scan's capture time; move it forward to now with the
        #    commands sent since (commanded motion, re-anchored every scan).
        if self.delay_compensation and self._pose_anchor is not None:
            anchor_pose, anchor_time = self._pose_anchor
            self.nav_robot.set_pose(self.command_history.propagate(anchor_pose, anchor_time, self._sim_time))

        # 5. Planning / navigation, unchanged, on that estimate.
        self.localization_health.update(self._sim_time)
        bootstrap_done = self.metrics.bootstrap_completed
        if self._sim_time - self._last_planning_rebuild >= PLANNING_REBUILD_INTERVAL:
            self.planning_map = build_planning_map(self.world_model, self.robot_cfg)
            self._last_planning_rebuild = self._sim_time
        if bootstrap_done and not self._goal_set and self.drive_script is None and self.goal is not None:
            self.nav_manager.set_goal(self.goal, self.planning_map, self.world_model.occupancy_grid)
            self._goal_set = True
        next_cmd = self.nav_manager.update(self.planning_map, self.dt, self.world_model.occupancy_grid)
        scripted = self._scripted_command(bootstrap_done)
        self._current_cmd = scripted if scripted is not None else next_cmd
        self.command_history.record(self._sim_time, self._current_cmd)  # in effect from now

        self._update_evaluation(self.environment.robot_pose)

    def sensor_report(self) -> dict | None:
        """Realistic-mode sensor/processing diagnostics (``None`` in ideal mode)."""
        if not self.realistic:
            return None
        return self.sensor_log.report(self.processor.metrics)

    def run(self) -> EvaluationMetrics:
        """Run until the goal is reached, a collision is detected, or
        ``max_simulation_time`` elapses -- whichever comes first."""
        while self._sim_time < self.max_simulation_time:
            self.step()
            if self.metrics.goal_reached or self.metrics.collision:
                break
        return self.metrics
