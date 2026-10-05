"""Phase 4: the stateful scan-matching pipeline / pose state machine.

Wraps the Phase 3 ``register()`` function (pure geometric ICP, unchanged)
with everything a real, continuously-running localization source needs
that a pure registration function deliberately does not know about:
constant-velocity seed prediction, a pre-registration sanity gate, a
failure/recovery state machine, and cross-scan pose/velocity state.

This module does NOT own or implement a persistent world model. It
receives confirmed-occupied reference geometry through the minimal
``ReferenceProvider`` protocol below, and (Phase 6) optionally reports
accepted scans to a ``WorldModelIntegrator`` via the same kind of
minimal protocol -- ``mapping.persistent_world_model.PersistentWorldModel``
implements both, but this module imports neither it nor anything else
from ``mapping``. The dependency direction stays
``ScanMatchingPipeline -> ReferenceProvider/WorldModelIntegrator``,
never the reverse. Nothing here depends on ``OccupancyGridMap``,
navigation, visualization, GUI, or robot control. The existing
``SLAMBackend``/BreezySLAM stack (``slam/backend.py``, ``slam/slam.py``)
is untouched and remains independently available -- this pipeline is a
new, separate architecture, not a replacement forced through the old
``SLAMBackend`` shape (which has no concept of seed prediction,
gating, or a failure state machine).

Phase 6 map-integration rule (see ``process_scan``): the world model,
when supplied, is updated ONLY with an ACCEPTED pose -- the first
scan's defined ``(0,0,0)`` initialization, or a subsequent scan whose
registration actually converged. It is never updated with a seed, an
extrapolated/diagnostic pose, or ``last_good_pose`` on a rejection.
Reference geometry for a scan is always queried before that same
scan's (conditional) integration, so a scan can never become part of
its own ICP reference -- true by construction, since the query and any
integration are sequential steps within one synchronous
``process_scan()`` call, never reordered.

Velocity-frame convention (critical, see ``PipelineState.velocity_estimate``
below): ``velocity_estimate = (vx_local, vy_local, omega)`` is always in
the ROBOT-LOCAL frame, at the last accepted pose's orientation --
``vx_local``/``vy_local`` are linear velocity along the robot's local
+X/+Y axes, ``omega`` is angular velocity in rad/s, positive
counterclockwise. The pose itself (``Pose2D``) always remains in the
WORLD frame, as everywhere else in this project. Seed prediction
(local velocity -> world displacement) and velocity estimation (world
pose delta -> local velocity) are exact forward/inverse rotations of
each other -- see ``_predict_seed`` and ``_estimate_velocity``.

Timestamp handling uses three distinct clocks, never mixed:
    - ``scan.timestamp`` (source timestamp): ordering, duplicate
      detection, and the ``dt`` used for every velocity/seed
      computation below.
    - Live monotonic time (``time.monotonic()``): NOT used by this
      module at all -- it belongs to a future world-model's live
      confirmation timer (Phase 5), which is explicitly out of scope
      here. This pipeline never calls a wall-clock function itself.
    - ``time.perf_counter()``: not used here either; reserved for
      future diagnostics/benchmarking, not part of this module's
      logic.

Phase 7 -- cold-start bootstrap. The architectural gap Phase 6 left
open: the very first scan seeds the world model with only TENTATIVE
geometry (Phase 5 requires ~0.5s of continued observation before
anything CONFIRMS), so the second scan's confirmed reference can be
empty -- normal registration then fails forever, since a failed
registration can never itself contribute confirmed geometry. Bootstrap
closes this gap with an explicit precondition: **the robot is assumed
stationary until bootstrap completes.** There is no wheel odometry,
IMU, or other independent motion source to verify this -- it is not
detected, only assumed, and documented as an operational precondition
here and at every relevant call site. While bootstrapping, every valid
scan is integrated at the fixed world-frame origin ``(0,0,0)`` (never
a predicted or estimated pose), letting Phase 5's own, unmodified
temporal confirmation mechanism do the actual work. Bootstrap completes
only once BOTH an elapsed-time minimum has passed AND the confirmed
reference is actually usable (enough points, spread across enough of
the same 8 angular sectors Phase 3's own spread check uses) --
never on elapsed time alone. If usable geometry never materializes
before a configured timeout, bootstrap reports failure explicitly
rather than silently entering ICP with an inadequate reference, and
stays failed until ``restart_bootstrap()`` is called. Bootstrap only
ever engages when a ``world_model`` is supplied (see ``reset()``) --
without one there is nothing for it to build, and every pre-Phase-7
caller that never supplied a world model is completely unaffected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from common.geometry import wrap_angle
from common.messages import LaserScan, Pose2D
from slam.scan_preprocessing import preprocess_scan
from slam.scan_registration import RegistrationConfig, RegistrationResult, register

MAX_CONSECUTIVE_FAILURES_DEFAULT = 5


class ReferenceProvider(Protocol):
    """Minimal interface for obtaining confirmed-occupied reference
    geometry, deliberately decoupled from any persistent-world-model
    implementation (a later phase provides the real one).

    ``center`` is a plain world (x, y) position, not a full ``Pose2D``
    -- a radius query is orientation-independent, so only position is
    meaningful here.
    """

    def get_confirmed_occupied_points_world(self, center: tuple[float, float], radius: float) -> np.ndarray:
        ...


class WorldModelIntegrator(Protocol):
    """Minimal interface (Phase 6) for reporting an accepted scan+pose
    to a persistent world model. ``PersistentWorldModel.integrate``
    already matches this signature exactly -- this Protocol exists so
    this module names and type-checks the dependency without importing
    ``mapping.persistent_world_model`` at all.
    """

    def integrate(self, pose: Pose2D, scan: LaserScan, now: float) -> None:
        ...


class MotionPrior(Protocol):
    """Optional source of predicted motion between two times, independent
    of the scans (``slam.motion_model.CommandHistory`` implements it from
    the velocity commands navigation actually sent). Returns
    ``(dx_local, dy_local, dtheta)`` in the frame of the pose at ``t0``,
    or ``None`` when it has nothing to say."""

    def predict_motion(self, t0: float, t1: float) -> tuple[float, float, float] | None:
        ...


class BootstrapStatus(Enum):
    """Distinguishes "bootstrap incomplete" from "normal localization
    active" from "bootstrap failed" -- deliberately NOT the same
    concept as ``PipelineState.localized`` (a robot that has never
    established its initial map is not "lost", it just hasn't started
    yet). Only ``FAILED`` implies the pipeline is *not* usable for
    normal operation; ``BOOTSTRAPPING`` is an expected, temporary
    startup phase, not a fault.
    """

    BOOTSTRAPPING = "bootstrapping"
    COMPLETE = "complete"
    FAILED = "failed"


# Matches slam.scan_registration.NUM_SECTORS's exact convention
# (local/world bearing = atan2(y, x), 0 = +X, positive = CCW, 8 equal
# sectors) -- duplicated here only as this tiny, self-contained
# angular-bucketing helper, not the ICP/trimming algorithm itself,
# specifically to avoid this module importing a private (underscore)
# symbol from slam.scan_registration for one geometry check.
_BOOTSTRAP_NUM_SECTORS = 8


def _confirmed_reference_is_usable(points: np.ndarray, min_points: int, min_sectors: int) -> bool:
    """True if a ``get_confirmed_occupied_points_world()`` result has
    enough points AND enough angular spread (around the bootstrap
    origin, since bootstrap always registers at world-frame
    ``(0,0,0)``) to plausibly support ICP -- not just "many points
    clustered in one wall". Mirrors Phase 3's own correspondence-count
    + sector-spread acceptance criteria so bootstrap's notion of
    "usable" matches what registration will actually require right
    afterward.
    """
    if points.shape[0] < min_points:
        return False
    bearings = np.arctan2(points[:, 1], points[:, 0])
    normalized = np.mod(bearings, 2.0 * math.pi)
    sector_width = 2.0 * math.pi / _BOOTSTRAP_NUM_SECTORS
    sectors = np.clip(np.floor(normalized / sector_width).astype(np.int64), 0, _BOOTSTRAP_NUM_SECTORS - 1)
    return int(np.unique(sectors).shape[0]) >= min_sectors


@dataclass
class BootstrapConfig:
    """Phase 7 bootstrap tuning parameters. Deliberately does NOT
    redeclare ``min_correspondences``/``min_sectors`` -- those already
    exist on ``RegistrationConfig`` (Phase 3) and are reused directly
    from there when checking bootstrap completion, so the "is this
    reference usable" threshold can never drift out of sync with what
    ICP itself will require immediately afterward.
    """

    # Slightly greater than Phase 5's default confirmation_duration
    # (0.5s), so at least one full confirmation cycle has a chance to
    # complete before bootstrap is even eligible to finish -- an
    # initial value, not derived from a proof, consistent with every
    # other small-motion/timing assumption already documented in this
    # project.
    min_duration: float = 0.6
    # Generous initial default: long enough that a real, physically
    # reasonable indoor scene should confirm well before this fires,
    # short enough to surface a genuine problem (e.g. a nearly
    # feature-less room) rather than waiting indefinitely.
    timeout: float = 5.0

    def __post_init__(self) -> None:
        if self.min_duration < 0.0:
            raise ValueError("min_duration must be >= 0")
        if self.timeout <= self.min_duration:
            raise ValueError("timeout must be > min_duration")


@dataclass
class PipelineConfig:
    """Phase 4 pipeline tuning parameters -- deliberately separate from
    ``RegistrationConfig`` (Phase 3's ICP-internal parameters).
    Distances in meters, angles in radians. Initial, reasoned defaults
    for empirical tuning, not asserted as final.
    """

    registration_radius_m: float = 8.0
    # ~0.5 * RegistrationConfig's default max_correspondence_distance
    # (0.3m), per the approved design principle: a seed displacement
    # under half the correspondence-acceptance radius keeps ICP's
    # first-iteration correspondences plausible even in the worst case.
    max_seed_displacement: float = 0.15
    # Conservative initial value: comfortably wider than the few
    # degrees of inter-scan rotation expected at this project's
    # assumed slow indoor speed, while still catching a seed that
    # implies an implausible spin (e.g. from a stale/skipped
    # timestamp). Not derived from a proof -- an initial value for
    # empirical tuning, consistent with this project's other
    # small-motion assumptions.
    max_seed_rotation: float = math.radians(10.0)
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES_DEFAULT

    # --- Time-aware motion validation (optional). ---
    # When BOTH velocity limits are set, the fixed max_seed_displacement /
    # max_seed_rotation gate above is replaced by a physical-plausibility
    # check on the MEASURED motion between the last accepted pose and the
    # new registration:
    #     translation <= max_linear_velocity  * (1 + velocity_margin) * dt + translation_tolerance
    #     rotation    <= max_angular_velocity * (1 + velocity_margin) * dt + rotation_tolerance
    # with dt the time between the two scans' timestamps. The fixed gate
    # judged the PREDICTED motion against constants, independent of dt: it
    # rejected good scans whenever the (noisy, differenced) velocity
    # estimate was implausible or dt grew, and it could not catch a wrong
    # registration at all. The tolerances absorb ICP noise and scan-timing
    # jitter. The velocity estimate is also clamped to these limits.
    # Left as None (default): the original fixed gate, unchanged.
    max_linear_velocity: float | None = None
    max_angular_velocity: float | None = None
    velocity_margin: float = 0.5
    translation_tolerance: float = 0.05
    rotation_tolerance: float = math.radians(3.0)

    # --- Map-update motion filter (optional). ---
    # An accepted scan (after bootstrap) is integrated into the world model
    # only once the pose has moved at least min_integration_translation OR
    # turned at least min_integration_rotation since the last integrated
    # pose. Integrating every scan of a stationary robot writes each scan's
    # pose noise into the reference the next scan registers against, so the
    # map and the pose random-walk together: 33 cm / 20 deg in 60 s on a
    # recorded stationary RPLiDAR run (2026-09-29), 3 cm / 2 deg over 5 min
    # with 0.05 m / 2 deg. Same idea as slam_toolbox's minimum_travel_*.
    # 0 for both (default): every accepted scan is integrated, as before.
    min_integration_translation: float = 0.0
    min_integration_rotation: float = 0.0

    def __post_init__(self) -> None:
        if self.registration_radius_m <= 0.0:
            raise ValueError("registration_radius_m must be > 0")
        if self.max_seed_displacement <= 0.0:
            raise ValueError("max_seed_displacement must be > 0")
        if self.max_seed_rotation <= 0.0:
            raise ValueError("max_seed_rotation must be > 0")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        if (self.max_linear_velocity is None) != (self.max_angular_velocity is None):
            raise ValueError("set both max_linear_velocity and max_angular_velocity, or neither")
        if self.max_linear_velocity is not None and (self.max_linear_velocity <= 0.0 or self.max_angular_velocity <= 0.0):
            raise ValueError("velocity limits must be > 0")
        if self.velocity_margin < 0.0 or self.translation_tolerance < 0.0 or self.rotation_tolerance < 0.0:
            raise ValueError("velocity_margin and tolerances must be >= 0")
        if self.min_integration_translation < 0.0 or self.min_integration_rotation < 0.0:
            raise ValueError("min_integration_translation/rotation must be >= 0")

    @property
    def time_aware(self) -> bool:
        return self.max_linear_velocity is not None

    def max_translation(self, dt: float) -> float:
        return self.max_linear_velocity * (1.0 + self.velocity_margin) * dt + self.translation_tolerance

    def max_rotation(self, dt: float) -> float:
        return self.max_angular_velocity * (1.0 + self.velocity_margin) * dt + self.rotation_tolerance


@dataclass(frozen=True)
class PipelineState:
    """Read-only snapshot of the pipeline's cross-scan state.

    ``last_good_pose``/``last_good_timestamp`` are ``None`` only before
    the first scan has been processed. ``velocity_estimate`` is always
    ``(vx_local, vy_local, omega)`` in the robot-local frame at
    ``last_good_pose``'s orientation (see module docstring).
    """

    last_good_pose: Pose2D | None
    last_good_timestamp: float | None
    velocity_estimate: tuple[float, float, float]
    consecutive_failures: int
    localized: bool
    bootstrap_status: BootstrapStatus


@dataclass
class PipelineResult:
    """Result of one ``ScanMatchingPipeline.process_scan()`` call.

    ``pose`` is always the pipeline's current best pose after this
    call -- ``last_good_pose`` (frozen, unchanged) on any rejection,
    or the newly accepted pose. ``registration_result`` is the
    underlying Phase 3 ``RegistrationResult`` when ICP was actually
    attempted this call, or ``None`` when it wasn't (first scan,
    duplicate/invalid timestamp, or the scan was rejected by the
    pre-registration gate before ICP was ever invoked).

    ``localized=False`` on this (or any subsequent) result IS this
    pipeline's report of "LOCALIZATION LOST" -- there is no separate
    status field for it; a consumer checks ``localized``.

    ``map_integrated`` (Phase 6): ``True`` only when a ``world_model``
    was supplied AND this call actually called its ``integrate()`` --
    i.e. the first scan, or an accepted registration. Always ``False``
    when no world model was supplied, and always ``False`` on any
    rejection, regardless of world model presence.

    ``bootstrap_status`` (Phase 7): see ``BootstrapStatus``. Defaults
    to ``COMPLETE`` only for dataclass-construction convenience; every
    real ``process_scan()`` return explicitly passes the pipeline's
    actual current status.
    """

    pose: Pose2D
    accepted: bool
    localized: bool
    consecutive_failures: int
    registration_result: RegistrationResult | None
    rejection_reason: str | None
    map_integrated: bool = False
    bootstrap_status: BootstrapStatus = BootstrapStatus.COMPLETE


def _pose_is_finite(pose: Pose2D) -> bool:
    return math.isfinite(pose.x) and math.isfinite(pose.y) and math.isfinite(pose.theta)


def _predict_seed_and_motion(
    last_pose: Pose2D, velocity_estimate: tuple[float, float, float], dt: float
) -> tuple[Pose2D, float, float]:
    """Constant-velocity seed prediction (Phase 4 design, section 7).

    ``velocity_estimate = (vx_local, vy_local, omega)`` is in the
    robot-local frame at ``last_pose``'s orientation. The local-frame
    displacement (``vx_local*dt``, ``vy_local*dt``, ``omega*dt``) is
    rotated into the world frame by ``last_pose.theta`` (the standard
    ``world = R(theta) @ local`` convention used throughout this
    project) and added to ``last_pose``.

    Returns ``(seed_pose, predicted_translation, predicted_rotation)``:
    ``predicted_translation = sqrt(dx_world**2 + dy_world**2)``,
    ``predicted_rotation = abs(dtheta)`` -- both computed from the RAW
    (unwrapped) local displacement, exactly as specified, even though
    ``seed_pose.theta`` itself is wrapped into the project's normal
    angle range.
    """
    vx_local, vy_local, omega = velocity_estimate
    dx_local = vx_local * dt
    dy_local = vy_local * dt
    dtheta = omega * dt

    theta = last_pose.theta
    c, s = math.cos(theta), math.sin(theta)
    dx_world = c * dx_local - s * dy_local
    dy_world = s * dx_local + c * dy_local

    seed = Pose2D(
        x=last_pose.x + dx_world,
        y=last_pose.y + dy_world,
        theta=wrap_angle(theta + dtheta),
        timestamp=last_pose.timestamp,
    )
    predicted_translation = math.hypot(dx_world, dy_world)
    predicted_rotation = abs(dtheta)
    return seed, predicted_translation, predicted_rotation


def _estimate_velocity(
    previous_pose: Pose2D, previous_timestamp: float, current_pose: Pose2D, current_timestamp: float
) -> tuple[float, float, float]:
    """Robot-local velocity from two accepted world-frame poses (Phase 4
    design, section 16). Inverse of ``_predict_seed_and_motion``'s
    local-to-world rotation: world displacement is rotated by
    ``-previous_pose.theta`` (i.e. ``R(theta).T``) into the previous
    pose's local frame before dividing by ``dt``.

    ``dt`` must be strictly positive -- callers are responsible for
    the monotonic-timestamp guarantee (this pipeline's duplicate/
    out-of-order rejection) that makes this safe; this function does
    not itself special-case ``dt <= 0``.
    """
    dt = current_timestamp - previous_timestamp
    dx_world = current_pose.x - previous_pose.x
    dy_world = current_pose.y - previous_pose.y

    theta = previous_pose.theta
    c, s = math.cos(theta), math.sin(theta)
    dx_local = c * dx_world + s * dy_world
    dy_local = -s * dx_world + c * dy_world

    vx_local = dx_local / dt
    vy_local = dy_local / dt
    omega = wrap_angle(current_pose.theta - previous_pose.theta) / dt
    return vx_local, vy_local, omega


def _apply_local_motion(pose: Pose2D, motion: tuple[float, float, float]) -> Pose2D:
    """``pose`` moved by a ``(dx_local, dy_local, dtheta)`` expressed in its own frame."""
    dx_local, dy_local, dtheta = motion
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    return Pose2D(
        x=pose.x + c * dx_local - s * dy_local,
        y=pose.y + s * dx_local + c * dy_local,
        theta=wrap_angle(pose.theta + dtheta),
        timestamp=pose.timestamp,
    )


def _clamp_velocity(
    velocity: tuple[float, float, float], max_linear: float, max_angular: float
) -> tuple[float, float, float]:
    """A differenced velocity estimate cannot exceed what the robot can do:
    one noisy registration over a short dt otherwise becomes e.g. 1.2 m/s
    for a robot spinning in place, and every later seed extrapolates it."""
    vx, vy, omega = velocity
    speed = math.hypot(vx, vy)
    if speed > max_linear:
        vx, vy = vx * max_linear / speed, vy * max_linear / speed
    return vx, vy, max(-max_angular, min(max_angular, omega))


class ScanMatchingPipeline:
    """Stateful pose-estimation pipeline: preprocessed scans in, a
    continuously-updated world-frame pose (and the state backing it)
    out. See the module docstring for the full architecture and the
    velocity-frame convention.
    """

    def __init__(
        self,
        reference_provider: ReferenceProvider,
        pipeline_config: PipelineConfig | None = None,
        registration_config: RegistrationConfig | None = None,
        world_model: WorldModelIntegrator | None = None,
        bootstrap_config: BootstrapConfig | None = None,
        motion_prior: MotionPrior | None = None,
    ) -> None:
        """``world_model`` (Phase 6, optional, keyword-friendly and
        appended last so every existing positional call site is
        unaffected): when supplied, its ``integrate()`` is called with
        the accepted pose after the first scan and after every
        subsequent accepted registration -- never on a rejection, and
        never with anything but an accepted pose (see module
        docstring). Typically the SAME object as ``reference_provider``
        in real use (``PersistentWorldModel`` implements both
        protocols), but kept as a separate parameter so tests can
        supply a bare reference provider with no mapping side effects
        at all, exactly as every pre-Phase-6 test already does.
        Omitting it (the default) makes this pipeline behave exactly
        as it did before Phase 6 -- ``map_integrated`` is simply always
        ``False``, and (Phase 7) bootstrap never engages at all: there
        is nothing for it to build without a world model, so
        ``bootstrap_status`` starts (and stays) ``COMPLETE`` and every
        pre-Phase-7 caller is completely unaffected. ``bootstrap_config``
        is ignored entirely when no ``world_model`` is supplied.

        ``motion_prior`` (optional): commanded-motion prediction. When
        supplied and it returns a prediction, the ICP seed is the last
        accepted pose moved by that prediction -- also while localization
        is lost, where the original code seeded from the frozen pose even
        though the robot kept moving. Otherwise the original
        constant-velocity seed is used.
        """
        self._reference_provider = reference_provider
        self._motion_prior = motion_prior
        self._pipeline_config = pipeline_config or PipelineConfig()
        self._registration_config = registration_config or RegistrationConfig()
        self._world_model = world_model
        self._bootstrap_config = bootstrap_config or BootstrapConfig()
        self.reset()

    def reset(self) -> None:
        """Explicit reinitialization: forget all pose/velocity/failure/
        bootstrap state and return to "no scan processed yet". Does
        not touch the reference provider or world model (which this
        pipeline never owns or mutates) -- in particular, this does
        NOT erase any confirmed geometry already in a supplied world
        model; see ``restart_bootstrap()`` for the distinction between
        resetting this pipeline's own tracking state and the
        (untouched) persistent map."""
        self._last_good_pose: Pose2D | None = None
        self._last_good_timestamp: float | None = None
        self._velocity_estimate: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._consecutive_failures: int = 0
        self._localized: bool = True
        # Bootstrap only ever engages when there is a world model to
        # build confirmed geometry into -- without one, every
        # pre-Phase-7 caller's behavior is completely unaffected.
        self._bootstrap_status: BootstrapStatus = (
            BootstrapStatus.BOOTSTRAPPING if self._world_model is not None else BootstrapStatus.COMPLETE
        )
        self._bootstrap_start_timestamp: float | None = None
        self._last_integrated_pose: Pose2D | None = None

    def restart_bootstrap(self) -> None:
        """Explicitly re-enter BOOTSTRAPPING -- e.g. after a bootstrap
        timeout, once the robot is confirmed stationary again. Resets
        ONLY this pipeline's own tracking state (pose, velocity,
        failure count, bootstrap timer) via ``reset()``. Does NOT
        touch the persistent world model or whatever confirmed
        geometry it already holds -- this method has no way to modify
        it and does not attempt to; a world model that already has
        usable confirmed geometry from before will simply let
        bootstrap complete again quickly.
        """
        if self._world_model is None:
            raise ValueError("restart_bootstrap() has no effect: no world_model was supplied")
        self.reset()

    @property
    def state(self) -> PipelineState:
        return PipelineState(
            last_good_pose=self._last_good_pose,
            last_good_timestamp=self._last_good_timestamp,
            velocity_estimate=self._velocity_estimate,
            consecutive_failures=self._consecutive_failures,
            localized=self._localized,
            bootstrap_status=self._bootstrap_status,
        )

    def process_scan(self, scan: LaserScan) -> PipelineResult:
        """Process one ``LaserScan``, advancing (or rejecting) the
        pipeline's pose state. Synchronous, one scan at a time -- no
        threading/queuing/dropping of any kind (that belongs to a
        live sensor-integration layer, not this pipeline).
        """
        if not math.isfinite(scan.timestamp):
            return self._unchanged_result(rejection_reason="invalid_timestamp", count_as_failure=False)

        if self._last_good_pose is None:
            # First successfully processed scan: no reference geometry
            # can exist yet, so no registration is attempted -- the
            # world frame is defined to start here, at (0, 0, 0). Per
            # Phase 6, this defined pose IS an accepted pose -- the
            # world model (if supplied) is populated with it, since
            # otherwise it could never receive any geometry at all.
            # Per Phase 7, this also starts the bootstrap timer.
            pose = Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=scan.timestamp)
            self._last_good_pose = pose
            self._last_good_timestamp = scan.timestamp
            self._velocity_estimate = (0.0, 0.0, 0.0)
            self._consecutive_failures = 0
            self._localized = True
            self._bootstrap_start_timestamp = scan.timestamp
            map_integrated = self._integrate_into_world_model(pose, scan, scan.timestamp)
            self._update_bootstrap_status(scan.timestamp)
            return PipelineResult(
                pose=pose, accepted=True, localized=True, consecutive_failures=0,
                registration_result=None, rejection_reason=None, map_integrated=map_integrated,
                bootstrap_status=self._bootstrap_status,
            )

        if scan.timestamp <= self._last_good_timestamp:
            # Covers both duplicate (==) and out-of-order (<) source
            # timestamps -- the same underlying condition (not
            # strictly newer), so the same deterministic rejection.
            # Not a registration failure: no state changes, and this
            # does not count toward consecutive_failures/localization.
            # Applies identically during bootstrap -- ordering/dedup is
            # orthogonal to bootstrap status.
            return self._unchanged_result(rejection_reason="timestamp_not_newer", count_as_failure=False)

        if self._bootstrap_status == BootstrapStatus.BOOTSTRAPPING:
            return self._process_bootstrap_scan(scan)

        if self._bootstrap_status == BootstrapStatus.FAILED:
            # Terminal until an explicit restart_bootstrap() call --
            # per the approved design, do not silently keep retrying
            # forever, and do not touch the map or pose while in this
            # state (the stationary-bootstrap precondition's validity
            # window has already expired once timeout fired).
            return self._unchanged_result(rejection_reason="bootstrap_failed", count_as_failure=False)

        dt = scan.timestamp - self._last_good_timestamp
        preprocessed = preprocess_scan(scan)
        cfg = self._pipeline_config

        commanded = (
            self._motion_prior.predict_motion(self._last_good_timestamp, scan.timestamp)
            if self._motion_prior is not None else None
        )
        if cfg.time_aware:
            # No pre-registration gate on the prediction: commanded motion is
            # bounded by construction (clamped commands) and a clamped
            # velocity estimate is too; the measured motion is checked below.
            if commanded is not None:
                seed = _apply_local_motion(self._last_good_pose, commanded)
            elif self._localized:
                seed, _, _ = _predict_seed_and_motion(self._last_good_pose, self._velocity_estimate, dt)
            else:
                seed = self._last_good_pose
        elif commanded is not None:
            seed = _apply_local_motion(self._last_good_pose, commanded)
            if self._localized:
                if math.hypot(commanded[0], commanded[1]) > cfg.max_seed_displacement:
                    return self._reject("seed_displacement_too_large", registration_result=None)
                if abs(commanded[2]) > cfg.max_seed_rotation:
                    return self._reject("seed_rotation_too_large", registration_result=None)
        elif self._localized:
            seed, predicted_translation, predicted_rotation = _predict_seed_and_motion(
                self._last_good_pose, self._velocity_estimate, dt
            )
            if predicted_translation > self._pipeline_config.max_seed_displacement:
                return self._reject("seed_displacement_too_large", registration_result=None)
            if predicted_rotation > self._pipeline_config.max_seed_rotation:
                return self._reject("seed_rotation_too_large", registration_result=None)
        else:
            # Localization lost (section 15): no further velocity
            # extrapolation -- the seed is exactly the last known
            # pose, and there is no predicted-motion gate to apply
            # (predicted displacement from a zero-motion seed is 0 by
            # construction).
            seed = self._last_good_pose

        # Reference geometry is queried from the CURRENT persistent
        # state, before this scan is used for anything else -- there
        # is no code path in this pipeline that could insert the
        # current scan into what get_confirmed_occupied_points_world()
        # returns (this pipeline has no method to write to the
        # provider at all).
        reference_points = self._reference_provider.get_confirmed_occupied_points_world(
            center=(seed.x, seed.y), radius=self._pipeline_config.registration_radius_m,
        )

        result = register(reference_points, preprocessed.hit_points_local, seed, self._registration_config)

        if result.converged and _pose_is_finite(result.pose) and cfg.time_aware:
            moved = math.hypot(result.pose.x - self._last_good_pose.x, result.pose.y - self._last_good_pose.y)
            turned = abs(wrap_angle(result.pose.theta - self._last_good_pose.theta))
            if moved > cfg.max_translation(dt) or turned > cfg.max_rotation(dt):
                return self._reject("implausible_motion", registration_result=result)

        if result.converged and _pose_is_finite(result.pose):
            vx_local, vy_local, omega = _estimate_velocity(
                self._last_good_pose, self._last_good_timestamp, result.pose, scan.timestamp
            )
            if cfg.time_aware:
                vx_local, vy_local, omega = _clamp_velocity(
                    (vx_local, vy_local, omega), cfg.max_linear_velocity, cfg.max_angular_velocity
                )
            self._velocity_estimate = (vx_local, vy_local, omega)
            self._last_good_pose = result.pose
            self._last_good_timestamp = scan.timestamp
            self._consecutive_failures = 0
            self._localized = True
            # Integration time: scan.timestamp, per the approved Phase 6
            # design -- deterministic for replay, and this pipeline has
            # no live monotonic-clock source to prefer instead (see
            # module docstring). Uses the ACCEPTED result.pose, never
            # the seed -- and this happens strictly after both the
            # reference query and register() above, so the current
            # scan can never appear in its own reference.
            map_integrated = (
                self._integrate_into_world_model(result.pose, scan, scan.timestamp)
                if self._moved_since_last_integration(result.pose) else False
            )
            return PipelineResult(
                pose=result.pose, accepted=True, localized=True, consecutive_failures=0,
                registration_result=result, rejection_reason=None, map_integrated=map_integrated,
                bootstrap_status=self._bootstrap_status,
            )

        reason = result.rejected_reason if result.rejected_reason is not None else "not_converged"
        return self._reject(reason, registration_result=result)

    def _process_bootstrap_scan(self, scan: LaserScan) -> PipelineResult:
        """One scan while BOOTSTRAPPING (Phase 7). No seed prediction,
        no pre-registration gate, no ICP -- every valid, correctly-
        ordered scan received here is integrated directly at the
        fixed world-frame origin ``(0,0,0)``, on the explicit
        precondition that the robot is stationary throughout (this is
        assumed, never verified -- there is no independent motion
        source to check it against). This lets Phase 5's own,
        completely unmodified temporal confirmation mechanism build up
        confirmed geometry naturally.
        """
        pose = Pose2D(x=0.0, y=0.0, theta=0.0, timestamp=scan.timestamp)
        self._last_good_pose = pose
        self._last_good_timestamp = scan.timestamp
        # Velocity is never estimated during bootstrap -- no motion is
        # assumed to have occurred, so it stays exactly (0,0,0).
        self._consecutive_failures = 0
        map_integrated = self._integrate_into_world_model(pose, scan, scan.timestamp)
        self._update_bootstrap_status(scan.timestamp)
        if self._bootstrap_status == BootstrapStatus.FAILED:
            self._localized = False
        return PipelineResult(
            pose=pose, accepted=True, localized=self._localized, consecutive_failures=0,
            registration_result=None, rejection_reason=None, map_integrated=map_integrated,
            bootstrap_status=self._bootstrap_status,
        )

    def _update_bootstrap_status(self, now: float) -> None:
        """Evaluate bootstrap completion/timeout after integrating one
        bootstrap scan. Completion requires BOTH an elapsed-time
        minimum AND a usable confirmed reference -- never elapsed time
        alone (see module docstring and ``_confirmed_reference_is_usable``).
        Reuses ``RegistrationConfig``'s own ``min_correspondences``/
        ``min_sectors`` as the usability thresholds rather than
        redeclaring separate bootstrap-specific constants, so "usable"
        here can never drift out of sync with what ICP will actually
        require on the very next scan.
        """
        if self._bootstrap_status != BootstrapStatus.BOOTSTRAPPING:
            return
        elapsed = now - self._bootstrap_start_timestamp

        if elapsed >= self._bootstrap_config.min_duration:
            reference_points = self._reference_provider.get_confirmed_occupied_points_world(
                center=(0.0, 0.0), radius=self._pipeline_config.registration_radius_m,
            )
            if _confirmed_reference_is_usable(
                reference_points, self._registration_config.min_correspondences, self._registration_config.min_sectors
            ):
                self._bootstrap_status = BootstrapStatus.COMPLETE
                return

        if elapsed >= self._bootstrap_config.timeout:
            self._bootstrap_status = BootstrapStatus.FAILED

    def _integrate_into_world_model(self, pose: Pose2D, scan: LaserScan, now: float) -> bool:
        """Call ``world_model.integrate(pose, scan, now)`` if a world
        model was supplied; a no-op returning ``False`` otherwise. Only
        ever called from an ACCEPTED path in ``process_scan`` (first
        scan or a converged registration) -- never from ``_reject`` or
        ``_unchanged_result``, so a rejected scan can never reach this
        method at all."""
        if self._world_model is None:
            return False
        self._world_model.integrate(pose, scan, now)
        self._last_integrated_pose = pose
        return True

    def _moved_since_last_integration(self, pose: Pose2D) -> bool:
        """The map-update motion filter (``PipelineConfig.min_integration_*``)."""
        last, cfg = self._last_integrated_pose, self._pipeline_config
        if last is None:
            return True
        return (math.hypot(pose.x - last.x, pose.y - last.y) >= cfg.min_integration_translation
                or abs(wrap_angle(pose.theta - last.theta)) >= cfg.min_integration_rotation)

    def _reject(self, reason: str, registration_result: RegistrationResult | None) -> PipelineResult:
        """Shared failure path: last_good_pose/timestamp/velocity are
        left untouched (never overwritten by an unaccepted result);
        consecutive_failures increments; localized flips to False once
        the configured threshold is reached (never flips back to True
        except via a subsequent ACCEPTED registration, in
        process_scan)."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._pipeline_config.max_consecutive_failures:
            self._localized = False
        return PipelineResult(
            pose=self._last_good_pose, accepted=False, localized=self._localized,
            consecutive_failures=self._consecutive_failures, registration_result=registration_result,
            rejection_reason=reason, bootstrap_status=self._bootstrap_status,
        )

    def _unchanged_result(self, rejection_reason: str, count_as_failure: bool) -> PipelineResult:
        """A rejection that is NOT a registration attempt at all
        (invalid/non-increasing timestamp) -- per design section 10,
        this must not alter pose/velocity/timestamp, and by default
        does not count toward consecutive_failures/localization
        either, since it is a distinct, non-registration condition.
        """
        if count_as_failure:
            return self._reject(rejection_reason, registration_result=None)
        pose = self._last_good_pose if self._last_good_pose is not None else Pose2D(x=0.0, y=0.0, theta=0.0)
        return PipelineResult(
            pose=pose, accepted=False, localized=self._localized,
            consecutive_failures=self._consecutive_failures, registration_result=None,
            rejection_reason=rejection_reason, bootstrap_status=self._bootstrap_status,
        )
