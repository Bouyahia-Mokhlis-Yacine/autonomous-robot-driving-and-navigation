"""Localization health: is the current pose estimate safe to navigate on?

``ScanMatchingPipeline`` already reports ``localized=False`` after
``max_consecutive_failures`` rejections, but nothing downstream acted on
it: ``apply_accepted_pose`` simply stops updating the robot's pose, and
navigation kept steering on that frozen, increasingly stale estimate. And
with realistic latency a pose can also go stale WITHOUT any rejection --
when scans stop arriving or processing falls far behind.

This monitor turns both signals into one explicit health state that
navigation can query (``is_localization_valid()``):

    INITIALIZING -- bootstrap not complete yet (the robot must hold still)
    LOCALIZED    -- recent accepted pose, pipeline localized
    DEGRADED     -- pipeline still localized, but the newest accepted pose
                    is older than ``degraded_after_s`` (scans being
                    rejected or processing lagging). Still usable: short
                    gaps are normal at ~4-5 Hz processing.
    LOST         -- the pipeline itself declared localization lost, or the
                    newest accepted pose is older than ``lost_after_s``.
                    NOT usable: navigation must stop.

Leaving LOST requires ``recovery_confirmations`` consecutive accepted,
localized results -- a single lucky registration right after a failure is
exactly the kind of result that can be wrong, so it is not trusted alone.

Pose age is measured from the accepted scan's ``timestamp`` (when the data
was captured and delivered), not from when processing finished, so
processing latency counts against freshness, as it should.

Time is always supplied by the caller (``now``); this module never reads a
clock, so simulation and replay stay deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from slam.scan_matching_pipeline import BootstrapStatus, PipelineResult


class LocalizationHealth(Enum):
    INITIALIZING = "INITIALIZING"
    LOCALIZED = "LOCALIZED"
    DEGRADED = "DEGRADED"
    LOST = "LOST"


@dataclass
class LocalizationHealthConfig:
    # Normal pose age at ~4-5 Hz processing with 0.15-0.25 s latency is
    # ~0.3-0.45 s; beyond 0.6 s scans are clearly being missed.
    degraded_after_s: float = 0.6
    # ~5 missed processing cycles at the realistic rate; long enough that
    # an isolated rejection or two never stops the robot, short enough that
    # the robot cannot drive far on a stale pose (0.6 m at 0.5 m/s).
    lost_after_s: float = 1.2
    recovery_confirmations: int = 2

    def __post_init__(self) -> None:
        if self.degraded_after_s <= 0.0 or self.lost_after_s <= self.degraded_after_s:
            raise ValueError("need 0 < degraded_after_s < lost_after_s")
        if self.recovery_confirmations < 1:
            raise ValueError("recovery_confirmations must be >= 1")


class LocalizationHealthMonitor:
    def __init__(self, config: LocalizationHealthConfig | None = None) -> None:
        self.config = config or LocalizationHealthConfig()
        self._health = LocalizationHealth.INITIALIZING
        self._bootstrapped = False
        self._pipeline_localized = True
        self._last_accepted_time = -math.inf
        self._consecutive_accepted = 0
        self._now = -math.inf
        self.loss_events = 0

    # ---------------------------------------------------------------- input --

    def on_result(self, result: PipelineResult, scan_timestamp: float) -> None:
        """Feed every pipeline result, accepted or not, in processing order."""
        if result.bootstrap_status == BootstrapStatus.COMPLETE:
            self._bootstrapped = True
        self._pipeline_localized = result.localized
        if result.accepted and result.localized:
            self._last_accepted_time = max(self._last_accepted_time, scan_timestamp)
            self._consecutive_accepted += 1
        else:
            self._consecutive_accepted = 0
        self._evaluate()

    def update(self, now: float) -> LocalizationHealth:
        """Advance the clock (pose ageing) and return the current health."""
        self._now = max(self._now, now)
        self._evaluate()
        return self._health

    # --------------------------------------------------------------- output --

    @property
    def health(self) -> LocalizationHealth:
        return self._health

    def is_localization_valid(self) -> bool:
        return self._health in (LocalizationHealth.LOCALIZED, LocalizationHealth.DEGRADED)

    def pose_age(self) -> float:
        return self._now - self._last_accepted_time

    # -------------------------------------------------------------- private --

    def _evaluate(self) -> None:
        if not self._bootstrapped:
            self._health = LocalizationHealth.INITIALIZING
            return
        age = self.pose_age()
        fresh_enough = age <= self.config.lost_after_s or not math.isfinite(self._now)
        if not self._pipeline_localized or not fresh_enough:
            if self._health is not LocalizationHealth.LOST:
                self.loss_events += 1
                # Recovery must be confirmed by results that arrive AFTER the loss.
                self._consecutive_accepted = 0
            self._health = LocalizationHealth.LOST
            return
        if self._health is LocalizationHealth.LOST:
            if self._consecutive_accepted < self.config.recovery_confirmations:
                return  # stay LOST until recovery is confirmed
        self._health = (
            LocalizationHealth.DEGRADED
            if math.isfinite(self._now) and age > self.config.degraded_after_s
            else LocalizationHealth.LOCALIZED
        )
