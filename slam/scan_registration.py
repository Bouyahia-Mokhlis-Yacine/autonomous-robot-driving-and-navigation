"""Pure 2D point-to-point ICP registration: local scan points -> world pose.

Phase 2 of the scan-registration/persistent-world-model design. This
module knows nothing about ``LaserScan``, ``OccupancyGridMap``,
``PersistentWorldModel``, navigation, robot control, visualization,
timestamps, scan history, or failure state machines -- it operates
only on plain ``(N, 2)`` numpy point arrays and a seed ``Pose2D``, and
returns a ``RegistrationResult``. Constant-velocity seeding, the
stateful acceptance policy, and map integration are a later phase's
responsibility, not this module's.

Coordinate convention (unchanged from the rest of this project, see
``common/geometry.py`` and ``mapping/occupancy_grid.py``):
``world_point = R(theta) @ local_point + [x, y]``, with
``R(theta) = [[cos theta, -sin theta], [sin theta, cos theta]]`` and
positive theta rotating counterclockwise. This is algebraically
identical to the angle-sum form used elsewhere in this project
(``x = pose.x + r*cos(pose.theta + angle)``).

Algorithm: standard point-to-point ICP. Each iteration transforms the
local scan points into world coordinates using the current pose
estimate, finds nearest neighbors in the reference cloud via a fresh
``cKDTree``, rejects correspondences farther than
``max_correspondence_distance``, then directly re-fits the ABSOLUTE
pose (not an incremental delta) via a closed-form 2D Kabsch/SVD solve
between the ORIGINAL local points (restricted to this iteration's
matched subset) and their matched reference points. Re-fitting against
the original local points each iteration, rather than composing
incremental deltas, avoids compounding numerical error across
iterations. No randomness anywhere -- identical inputs always produce
identical output.

Robustness (Phase 3) -- two purely geometric mechanisms, applied every
iteration between max-distance rejection and the Kabsch/SVD fit, so
they shape *every* pose update, not just the final one:

1. Trimmed ICP: after max-distance rejection, the surviving
   correspondences are sorted by their Euclidean residual and only the
   best ``trim_fraction`` are kept (see ``_keep_count`` for the exact,
   floor-protected rule). This lets a minority of high-residual
   correspondences -- moved geometry, newly appeared/disappeared
   objects, isolated outliers -- be excluded from the fit without any
   semantic reasoning about what they are.
2. Angular-sector spread validation: the *retained* (post-trim)
   correspondences' LOCAL-frame bearings are bucketed into 8 fixed
   equal sectors (see ``_sector_indices``); registration is rejected
   with ``"insufficient_spatial_spread"`` unless at least
   ``min_sectors`` of the 8 are represented. This guards against a
   numerically-plausible fit that is actually dominated by one small
   angular cluster rather than genuinely well-constrained geometry.

Known, explicitly accepted limitation: if moved or newly-appeared
geometry dominates most of the scan's angular coverage, these purely
geometric mechanisms cannot reliably distinguish "the robot moved"
from "the environment moved" -- there is no semantic/object-identity
reasoning anywhere in this module, by design. Trimming only removes a
*minority* of bad correspondences, and the sector check only catches a
*spatially concentrated* bad fit; a scan dominated by moved/new
geometry is expected to fail explicitly (a rejected result) rather
than silently produce a wrong pose. Loop closure / place recognition
(a future phase) is the intended way to recover from the kind of drift
this limitation can eventually cause, not this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from common.geometry import wrap_angle
from common.messages import Pose2D

# Fixed structural choice (not configurable, unlike min_sectors below):
# the local scan's field of view is always divided into this many equal
# angular sectors for the spread-validation check.
NUM_SECTORS = 8


@dataclass
class RegistrationConfig:
    """ICP tuning parameters. Distances in meters, angles in radians,
    iteration counts as integers.

    These are initial, reasoned defaults for empirical tuning once
    real data is available (per the approved design) -- not asserted
    as final or optimal.
    """

    max_iterations: int = 50
    max_correspondence_distance: float = 0.3
    convergence_translation: float = 0.001
    convergence_rotation: float = math.radians(0.05)
    min_correspondences: int = 3
    trim_fraction: float = 0.8
    min_sectors: int = 3

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.max_correspondence_distance <= 0.0:
            raise ValueError("max_correspondence_distance must be > 0")
        if self.convergence_translation < 0.0:
            raise ValueError("convergence_translation must be >= 0")
        if self.convergence_rotation < 0.0:
            raise ValueError("convergence_rotation must be >= 0")
        if self.min_correspondences < 2:
            raise ValueError(
                "min_correspondences must be >= 2 -- rotation is mathematically "
                "undefined from a single point correspondence"
            )
        if not (0.0 < self.trim_fraction <= 1.0):
            raise ValueError("trim_fraction must be in (0, 1]")
        if not (1 <= self.min_sectors <= NUM_SECTORS):
            raise ValueError(f"min_sectors must be in [1, {NUM_SECTORS}]")


@dataclass
class RegistrationResult:
    """Result of one ``register()`` call.

    ``pose`` is only meaningful when ``rejected_reason is None``: on
    every failure except ``"max_iterations"``, no reliable transform
    could be computed at all, so ``pose`` is exactly ``seed_pose``
    unchanged rather than a fabricated estimate. On ``"max_iterations"``,
    a real (if not fully converged) pose was computed throughout the
    run and is returned with ``converged=False``.

    ``residual`` and ``correspondence_count`` describe the *final* ICP
    iteration actually performed (the one that produced ``pose``), not
    an aggregate over the whole run. ``residual`` is the RMS
    point-to-point distance, in meters, between that iteration's
    matched local points (transformed by the newly fitted pose) and
    their matched reference points -- i.e. the residual of the
    returned pose's own fit. ``correspondence_count`` is the number of
    correspondences actually used by (or, on a rejection after
    trimming, considered for) the transform estimate -- i.e. the
    *retained*, post-trim count, not the count before trimming.

    ``rejected_reason`` values introduced in Phase 3, beyond the
    Phase 2 set: ``"insufficient_spatial_spread"`` -- the retained,
    post-trim correspondences covered fewer than ``min_sectors`` of
    the 8 local-frame angular sectors.
    """

    pose: Pose2D
    converged: bool
    residual: float
    correspondence_count: int
    iterations: int
    rejected_reason: str | None


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _transform_points(points_local: np.ndarray, pose: Pose2D) -> np.ndarray:
    """local (N,2) -> world (N,2), per this module's documented convention.
    Never mutates ``points_local``."""
    R = _rotation_matrix(pose.theta)
    return points_local @ R.T + np.array([pose.x, pose.y], dtype=np.float64)


def _is_degenerate(matched_local: np.ndarray, matched_world: np.ndarray) -> bool:
    """True if the matched point sets carry no usable rotational
    information -- every matched local (or every matched reference)
    point coincides with its own centroid, so the cross-covariance
    matrix is (numerically) zero and any rotation extracted from it
    would be an arbitrary SVD tie-break, not a real fit."""
    local_centered = matched_local - matched_local.mean(axis=0)
    world_centered = matched_world - matched_world.mean(axis=0)
    return bool(np.linalg.norm(local_centered) < 1e-9 or np.linalg.norm(world_centered) < 1e-9)


def _estimate_rigid_transform_2d(
    matched_local: np.ndarray, matched_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Closed-form 2D Kabsch/SVD rigid fit: ``matched_local`` -> ``matched_world``.

    Returns ``(R, t)`` such that ``matched_local @ R.T + t`` best
    approximates ``matched_world`` in the least-squares sense, or
    ``None`` for a genuine numerical failure (non-finite intermediate
    values, SVD non-convergence, or a result that isn't a valid proper
    rotation). Assumes the caller has already ruled out degenerate
    (coincident-point) input via ``_is_degenerate``.
    """
    local_mean = matched_local.mean(axis=0)
    world_mean = matched_world.mean(axis=0)
    H = (matched_local - local_mean).T @ (matched_world - world_mean)
    if not np.all(np.isfinite(H)):
        return None

    try:
        U, _singular_values, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None

    det_sign = np.linalg.det(Vt.T @ U.T)
    d = 1.0 if det_sign >= 0.0 else -1.0
    R = Vt.T @ np.diag([1.0, d]) @ U.T

    if not np.all(np.isfinite(R)) or abs(np.linalg.det(R) - 1.0) > 1e-6:
        return None

    t = world_mean - R @ local_mean
    if not np.all(np.isfinite(t)):
        return None

    return R, t


def _keep_count(correspondence_count: int, cfg: RegistrationConfig) -> int:
    """Number of correspondences retained after trimming.

    ``keep_count = max(min_correspondences, ceil(trim_fraction *
    correspondence_count))``, clamped to ``correspondence_count``.

    The ``max()`` floor means trimming can only ever reduce an
    ALREADY-sufficient correspondence set -- it can never be the
    reason a registration ends up with fewer than
    ``min_correspondences``. A set that is already too small to trim
    up to that floor is rejected earlier (``"insufficient_correspondences"``,
    before trimming is attempted at all) rather than being silently
    padded or trimmed further.
    """
    target = math.ceil(cfg.trim_fraction * correspondence_count)
    keep = max(cfg.min_correspondences, target)
    return min(keep, correspondence_count)


def _sector_indices(points_local: np.ndarray) -> np.ndarray:
    """Assign each LOCAL-frame point to one of ``NUM_SECTORS`` equal
    angular sectors.

    Deliberately uses the scan's own local-frame bearing, never the
    world frame -- this measures the current scan geometry's angular
    coverage, independent of the pose estimate being refined. Bearing
    convention matches the rest of this project: ``theta =
    atan2(y_local, x_local)``, ``theta=0`` -> +X, positive -> CCW,
    normalized into ``[0, 2*pi)`` before sector assignment. Sector 0
    covers ``[0, 2*pi/NUM_SECTORS)``, increasing counterclockwise.
    Clamped to ``[0, NUM_SECTORS - 1]`` as a defensive guard against a
    bearing landing exactly on the ``2*pi`` boundary due to floating
    point.
    """
    bearings = np.arctan2(points_local[:, 1], points_local[:, 0])
    normalized = np.mod(bearings, 2.0 * math.pi)
    sector_width = 2.0 * math.pi / NUM_SECTORS
    sectors = np.floor(normalized / sector_width).astype(np.int64)
    return np.clip(sectors, 0, NUM_SECTORS - 1)


def register(
    reference_points_world: np.ndarray,
    scan_points_local: np.ndarray,
    seed_pose: Pose2D,
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    """Point-to-point ICP: estimate the world-frame pose of
    ``scan_points_local`` against ``reference_points_world``, starting
    from ``seed_pose``.

    Deterministic: identical inputs always produce identical output --
    no random sampling, no randomized initialization, no
    time-dependent behavior. Neither input array is modified.
    """
    cfg = config or RegistrationConfig()

    reference_points_world = np.asarray(reference_points_world, dtype=np.float64)
    scan_points_local = np.asarray(scan_points_local, dtype=np.float64)

    if scan_points_local.shape[0] == 0:
        return RegistrationResult(
            pose=seed_pose, converged=False, residual=0.0,
            correspondence_count=0, iterations=0, rejected_reason="insufficient_scan_points",
        )
    if reference_points_world.shape[0] == 0:
        return RegistrationResult(
            pose=seed_pose, converged=False, residual=0.0,
            correspondence_count=0, iterations=0, rejected_reason="insufficient_reference_points",
        )

    tree = cKDTree(reference_points_world)
    pose = seed_pose
    residual = 0.0
    correspondence_count = 0

    for iteration in range(1, cfg.max_iterations + 1):
        transformed = _transform_points(scan_points_local, pose)
        distances, indices = tree.query(transformed)
        mask = distances <= cfg.max_correspondence_distance
        raw_correspondence_count = int(np.count_nonzero(mask))

        if raw_correspondence_count < cfg.min_correspondences:
            return RegistrationResult(
                pose=seed_pose, converged=False, residual=0.0,
                correspondence_count=raw_correspondence_count, iterations=iteration,
                rejected_reason="insufficient_correspondences",
            )

        matched_local_all = scan_points_local[mask]
        matched_reference_all = reference_points_world[indices[mask]]
        matched_residuals_all = distances[mask]

        # --- Trimmed ICP (Phase 3): keep only the best-fitting fraction. ---
        keep = _keep_count(raw_correspondence_count, cfg)
        order = np.argsort(matched_residuals_all, kind="stable")[:keep]
        matched_local = matched_local_all[order]
        matched_reference = matched_reference_all[order]
        correspondence_count = keep  # the retained set is what "used by this iteration" now means

        # --- Angular-sector spread validation (Phase 3), on the RETAINED set only. ---
        distinct_sectors = int(np.unique(_sector_indices(matched_local)).shape[0])
        if distinct_sectors < cfg.min_sectors:
            return RegistrationResult(
                pose=seed_pose, converged=False, residual=0.0,
                correspondence_count=correspondence_count, iterations=iteration,
                rejected_reason="insufficient_spatial_spread",
            )

        if _is_degenerate(matched_local, matched_reference):
            return RegistrationResult(
                pose=seed_pose, converged=False, residual=0.0,
                correspondence_count=correspondence_count, iterations=iteration,
                rejected_reason="degenerate_geometry",
            )

        fit = _estimate_rigid_transform_2d(matched_local, matched_reference)
        if fit is None:
            return RegistrationResult(
                pose=seed_pose, converged=False, residual=0.0,
                correspondence_count=correspondence_count, iterations=iteration,
                rejected_reason="numerical_failure",
            )
        R, t = fit
        new_theta = wrap_angle(math.atan2(R[1, 0], R[0, 0]))
        new_pose = Pose2D(x=float(t[0]), y=float(t[1]), theta=new_theta, timestamp=seed_pose.timestamp)

        fitted_world = matched_local @ R.T + t
        residual = float(np.sqrt(np.mean(np.sum((fitted_world - matched_reference) ** 2, axis=1))))

        translation_update = math.hypot(new_pose.x - pose.x, new_pose.y - pose.y)
        rotation_update = abs(wrap_angle(new_pose.theta - pose.theta))

        if translation_update <= cfg.convergence_translation and rotation_update <= cfg.convergence_rotation:
            return RegistrationResult(
                pose=new_pose, converged=True, residual=residual,
                correspondence_count=correspondence_count, iterations=iteration, rejected_reason=None,
            )

        pose = new_pose

    return RegistrationResult(
        pose=pose, converged=False, residual=residual,
        correspondence_count=correspondence_count, iterations=cfg.max_iterations,
        rejected_reason="max_iterations",
    )
