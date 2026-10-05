"""Typed configuration loading from YAML files.

Keeps tunable numbers (robot dimensions, safety margins, velocity limits,
sensor parameters, planning options) in config/*.yaml instead of scattered
through source code, per project convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass
class RobotDimensions:
    wheel_base: float
    width: float
    length: float
    # LiDAR mount: its centre in the ROBOT frame (x forward, y left, metres;
    # 0 deg of the sensor = robot forward). SLAM estimates the LiDAR pose;
    # navigation uses the robot centre (common.geometry.offset_pose).
    lidar_x: float = 0.0
    lidar_y: float = 0.0


@dataclass
class SafetyLimits:
    inflation_radius: float
    goal_tolerance: float
    emergency_stop_distance: float
    path_deviation_threshold: float
    # Direction-aware motion safety (navigation.motion_safety); see robot.yaml.
    side_margin: float = 0.05
    rotation_margin: float = 0.03
    min_scan_points: int = 2


@dataclass
class MotionLimits:
    max_linear_velocity: float
    max_angular_velocity: float
    max_linear_acceleration: float
    max_angular_acceleration: float
    command_timeout: float


@dataclass
class ControlTuning:
    heading_gain: float
    waypoint_tolerance: float
    # Bidirectional motion (control.path_follower). turn_in_place_enter_deg
    # None = the original forward-only controller.
    turn_in_place_enter_deg: float | None = None
    turn_in_place_exit_deg: float = 10.0
    min_turn_velocity: float = 0.3
    allow_reverse: bool = False
    max_reverse_velocity: float | None = None
    max_reverse_distance: float = 1.5
    reverse_bias_deg: float = 20.0


@dataclass
class RobotConfig:
    robot: RobotDimensions
    safety: SafetyLimits
    limits: MotionLimits
    control: ControlTuning


@dataclass
class LidarConfig:
    range_min: float
    range_max: float
    angle_min: float
    angle_max: float
    angular_resolution: float
    scan_rate: float


@dataclass
class MappingConfig:
    resolution: float
    width: float
    height: float
    # What a no-return (range_max) ray is evidence of -- see
    # mapping.persistent_world_model.PersistentWorldModel.__init__.
    max_range_policy: str = "clear"
    max_range_clear_distance: float = 2.0


@dataclass
class PlanningConfig:
    allow_diagonal: bool
    replan_interval: float
    heuristic: str
    # Phase 11: risk-aware UNKNOWN-space planning. Cost multiplier
    # (>= 1.0) for entering an UNKNOWN cell vs. an equivalent FREE one
    # -- see planning.astar's module docstring for the full rationale
    # (including why the existing distance heuristic stays admissible
    # unchanged). An initial, tunable value: strong enough that an
    # equivalent-length known-free route is clearly preferred, not so
    # large that a necessary detour through unexplored space becomes
    # impractically expensive. Not claimed to be physically optimal.
    unknown_cost_multiplier: float = 3.0
    # Phase 15: additional path-smoothing clearance margin (meters), on
    # top of the planning grid's own inflation -- see
    # planning.path_smoothing's module docstring for the full rationale.
    # 0.10m was chosen because Phase 14 measured LOS-simplified paths
    # coming as close as 0.03-0.04m to occupied geometry (enough to let
    # ordinary localization/discretization variance push the robot back
    # into its own inflation buffer); 0.10m comfortably exceeds that
    # measured gap while staying much smaller than the configured
    # inflation radius, so it should not meaningfully block legitimate
    # routes. An initial, tunable value, not claimed to be optimal.
    path_clearance_margin: float = 0.10


@dataclass
class SlamConfig:
    map_size_pixels: int
    map_size_meters: float
    hole_width_mm: float
    sigma_xy_mm: float
    sigma_theta_degrees: float
    max_search_iter: int


@dataclass
class LocalizationConfig:
    """Localization-health thresholds (see ``slam.localization_health``).
    Defaults apply when navigation.yaml has no ``localization:`` section."""

    degraded_after_s: float = 0.6
    lost_after_s: float = 1.2
    recovery_confirmations: int = 2
    # Motion validation / prediction (slam.scan_matching_pipeline, slam.motion_model).
    time_aware_validation: bool = True
    velocity_margin: float = 0.5
    translation_tolerance_m: float = 0.05
    rotation_tolerance_deg: float = 3.0
    use_commanded_motion: bool = True
    delay_compensation: bool = True
    # Map-update motion filter (slam.scan_matching_pipeline.PipelineConfig.min_integration_*).
    map_update_min_translation_m: float = 0.05
    map_update_min_rotation_deg: float = 2.0
    # Startup mapping (slam.scan_matching_pipeline.BootstrapConfig): every
    # scan is integrated at the origin for at least this long, robot still.
    bootstrap_duration_s: float = 2.0
    bootstrap_timeout_s: float = 5.0


@dataclass
class NavigationConfig:
    lidar: LidarConfig
    mapping: MappingConfig
    planning: PlanningConfig
    slam: SlamConfig
    logging_level: str
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)


@dataclass
class RPLidarHardwareConfig:
    port: str
    baudrate: int
    timeout: float
    min_measurements_per_scan: int
    # The sensor reports angles CLOCKWISE (Slamtec protocol); the stack uses
    # counter-clockwise (+y = robot LEFT). True = the driver negates them.
    angles_clockwise: bool = True


@dataclass
class MotorConfig:
    """Wheel motors: Pi GPIO -> L298N (see config/hardware.yaml ``motors:``)."""

    pins: dict[str, list[int]]
    left_trim: float = 1.0
    right_trim: float = 1.0
    max_duty: float = 0.65
    min_duty: float = 0.30
    zero_threshold: float = 0.02
    max_wheel_speed_mps: float = 0.60
    track_width_m: float = 0.30
    invert_left: bool = False
    invert_right: bool = False
    # Minimum duty when the BODY reverses / turns in place (None = min_duty).
    # A skid-steer needs more torque to pivot (all four wheels scrub sideways).
    min_reverse_duty: float | None = None
    min_turn_duty: float | None = None

    def __post_init__(self) -> None:
        if self.min_reverse_duty is None:
            self.min_reverse_duty = self.min_duty
        if self.min_turn_duty is None:
            self.min_turn_duty = self.min_duty
        required = {"front_left", "back_left", "front_right", "back_right"}
        if set(self.pins) != required or any(len(v) != 2 for v in self.pins.values()):
            raise ValueError(f"motors.pins needs exactly {sorted(required)}, each [forward, backward]")
        flat = [p for pair in self.pins.values() for p in pair]
        if len(set(flat)) != len(flat):
            raise ValueError("motor pins must all be different")
        if not (0.0 < self.left_trim <= 1.0 and 0.0 < self.right_trim <= 1.0):
            raise ValueError("trims must be in (0, 1]")
        for name in ("min_duty", "min_reverse_duty", "min_turn_duty"):
            if not (0.0 <= self.zero_threshold < getattr(self, name) <= self.max_duty <= 1.0):
                raise ValueError(f"need 0 <= zero_threshold < {name} <= max_duty <= 1")
        if self.max_wheel_speed_mps <= 0.0 or self.track_width_m <= 0.0:
            raise ValueError("max_wheel_speed_mps and track_width_m must be > 0")


@dataclass
class HardwareConfig:
    rplidar: RPLidarHardwareConfig
    motors: MotorConfig | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    return data


def load_robot_config(path: Path | str | None = None) -> RobotConfig:
    resolved = Path(path) if path is not None else _CONFIG_DIR / "robot.yaml"
    data = _load_yaml(resolved)
    return RobotConfig(
        robot=RobotDimensions(**data["robot"]),
        safety=SafetyLimits(**data["safety"]),
        limits=MotionLimits(**data["limits"]),
        control=ControlTuning(**data["control"]),
    )


def load_navigation_config(path: Path | str | None = None) -> NavigationConfig:
    resolved = Path(path) if path is not None else _CONFIG_DIR / "navigation.yaml"
    data = _load_yaml(resolved)
    return NavigationConfig(
        lidar=LidarConfig(**data["lidar"]),
        mapping=MappingConfig(**data["mapping"]),
        planning=PlanningConfig(**data["planning"]),
        slam=SlamConfig(**data["slam"]),
        logging_level=data.get("logging", {}).get("level", "INFO"),
        localization=LocalizationConfig(**data.get("localization", {})),
    )


def load_hardware_config(path: Path | str | None = None) -> HardwareConfig:
    resolved = Path(path) if path is not None else _CONFIG_DIR / "hardware.yaml"
    data = _load_yaml(resolved)
    motors = MotorConfig(**data["motors"]) if "motors" in data else None
    return HardwareConfig(rplidar=RPLidarHardwareConfig(**data["rplidar"]), motors=motors)
