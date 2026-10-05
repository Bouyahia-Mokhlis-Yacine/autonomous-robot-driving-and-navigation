"""Asynchronous scan-arrival -> SLAM-processing model, in simulated time.

The real system is two independent clocks: the RPLiDAR delivers a scan
roughly every 90 ms regardless of what the host is doing, and SLAM +
map integration takes time per scan (~12-24 ms measured after the
vectorized integrate(); ~150-250 ms before). So the sensor keeps producing
while SLAM is busy, and something must decide what happens to scans that
arrive in the meantime. ``ScanMatchingPipeline`` itself is deliberately
synchronous ("no threading/queuing/dropping of any kind -- that belongs
to a live sensor-integration layer"); this module IS that layer, for the
simulator.

Event-driven, not threaded, and never sleeps: ``advance(now, process)``
replays, in timestamp order, every scan arrival and every processing
completion up to ``now``. Processing a scan is a span of simulated time
``[start, finish)``; the caller's ``process`` callback is invoked AT
``finish`` (the pipeline is strictly serial, so its state at ``finish``
is exactly its state at ``start`` -- nothing else can touch it in
between), which means its result, and every map update it makes, become
visible to the rest of the simulation only once the modeled processing
time has actually elapsed. That is what answers "if a scan arrives at
simulated time T, when does SLAM actually use it?" -- ``finish``, which
``CompletedScan`` records along with the whole timeline.

Queue policies (``QueuePolicy``):
    QUEUE_ALL    -- unbounded FIFO; nothing is ever dropped (latency grows
                    without bound if the sensor outpaces processing).
                    With zero processing delay this reproduces the
                    original synchronous one-scan-per-tick behavior.
    DROP_NEWEST  -- bounded FIFO; an arrival that finds the queue full is
                    discarded.
    DROP_OLDEST  -- bounded FIFO; an arrival that finds the queue full
                    evicts the oldest waiting scan.
    LATEST_ONLY  -- at most one waiting scan; each arrival replaces it.
                    This is what ``real_sensor_walkthrough.py`` already
                    does with real hardware (a background reader thread
                    overwrites a single ``latest_scan_box``, and the main
                    loop processes whatever is latest when it is free), so
                    it is the realistic-mode default.
A scan already being processed is never interrupted by any policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from common.messages import LaserScan


class QueuePolicy(Enum):
    QUEUE_ALL = "queue_all"
    DROP_NEWEST = "drop_newest"
    DROP_OLDEST = "drop_oldest"
    LATEST_ONLY = "latest_only"


class ProcessingDelayModel(Enum):
    NONE = "none"        # processing is instantaneous (ideal/original behavior)
    UNIFORM = "uniform"  # delay ~ U(min, max); min == max gives a fixed delay


@dataclass
class ProcessingConfig:
    delay_model: ProcessingDelayModel = ProcessingDelayModel.UNIFORM
    delay_min_s: float = 0.012  # measured, see config/lidar_simulation.yaml
    delay_max_s: float = 0.024
    queue_policy: QueuePolicy = QueuePolicy.LATEST_ONLY
    queue_capacity: int = 1  # waiting scans, excluding the one in progress; ignored by QUEUE_ALL/LATEST_ONLY

    def __post_init__(self) -> None:
        if isinstance(self.delay_model, str):
            self.delay_model = ProcessingDelayModel(self.delay_model)
        if isinstance(self.queue_policy, str):
            self.queue_policy = QueuePolicy(self.queue_policy)
        if self.delay_min_s < 0.0 or self.delay_max_s < self.delay_min_s:
            raise ValueError("need 0 <= delay_min_s <= delay_max_s")
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")

    @staticmethod
    def instantaneous() -> "ProcessingConfig":
        """The original behavior: every scan processed the moment it arrives."""
        return ProcessingConfig(
            delay_model=ProcessingDelayModel.NONE, delay_min_s=0.0, delay_max_s=0.0,
            queue_policy=QueuePolicy.QUEUE_ALL,
        )


@dataclass
class CompletedScan:
    """One processed scan's full timeline, plus whatever ``process`` returned."""

    scan: LaserScan
    arrival_time: float
    start_time: float
    finish_time: float
    result: Any

    @property
    def queue_wait(self) -> float:
        return self.start_time - self.arrival_time

    @property
    def processing_latency(self) -> float:
        """Arrival -> result usable (queue wait + processing)."""
        return self.finish_time - self.arrival_time


@dataclass
class ProcessingMetrics:
    scans_received: int = 0
    scans_processed: int = 0
    scans_dropped: int = 0
    max_queue_size: int = 0
    latencies: list[float] = field(default_factory=list)       # arrival -> finish
    processing_times: list[float] = field(default_factory=list)  # start -> finish
    queue_size_samples: list[int] = field(default_factory=list)  # sampled at every arrival

    @property
    def average_processing_latency(self) -> float | None:
        return float(np.mean(self.latencies)) if self.latencies else None

    @property
    def max_processing_latency(self) -> float | None:
        return float(np.max(self.latencies)) if self.latencies else None

    @property
    def mean_queue_size(self) -> float | None:
        return float(np.mean(self.queue_size_samples)) if self.queue_size_samples else None


@dataclass
class _InFlight:
    scan: LaserScan
    arrival_time: float
    start_time: float
    finish_time: float


class ScanProcessingSimulator:
    """Event-driven single-worker processor with a configurable queue.
    See the module docstring for the timing semantics."""

    def __init__(self, config: ProcessingConfig | None = None, seed: int | None = None) -> None:
        self.config = config or ProcessingConfig()
        self._rng = np.random.default_rng(seed)
        self._arrivals: deque[LaserScan] = deque()  # not yet arrived (timestamp > now)
        self._waiting: deque[tuple[LaserScan, float]] = deque()
        self._in_flight: _InFlight | None = None
        self.metrics = ProcessingMetrics()

    # ---------------------------------------------------------------- input --

    def submit(self, scan: LaserScan) -> None:
        """Hand over a scan that will ARRIVE at ``scan.timestamp`` (possibly
        in the future relative to the last ``advance``). Submissions must be
        in timestamp order, as a single serial stream is."""
        if self._arrivals and scan.timestamp < self._arrivals[-1].timestamp:
            raise ValueError("scans must be submitted in timestamp order")
        self._arrivals.append(scan)

    # ---------------------------------------------------------------- state --

    @property
    def queue_size(self) -> int:
        return len(self._waiting)

    @property
    def busy(self) -> bool:
        return self._in_flight is not None

    @property
    def in_flight_scan_timestamp(self) -> float | None:
        return self._in_flight.scan.timestamp if self._in_flight is not None else None

    @property
    def in_flight_finish_time(self) -> float | None:
        return self._in_flight.finish_time if self._in_flight is not None else None

    # ------------------------------------------------------------ event loop --

    def advance(self, now: float, process: Callable[[LaserScan], Any]) -> list[CompletedScan]:
        """Replay every arrival/completion event with time <= ``now``, in
        time order. On an exact tie the completion is handled first, so the
        freed worker immediately takes whatever was already waiting and the
        simultaneous arrival is then queued under the policy."""
        completed: list[CompletedScan] = []
        while True:
            next_arrival = self._arrivals[0].timestamp if self._arrivals else None
            next_finish = self._in_flight.finish_time if self._in_flight is not None else None
            candidates = [t for t in (next_finish, next_arrival) if t is not None and t <= now]
            if not candidates:
                break
            if next_finish is not None and next_finish <= now and (next_arrival is None or next_finish <= next_arrival):
                completed.append(self._complete(process))
                self._start_next(next_finish)
            else:
                scan = self._arrivals.popleft()
                self._arrive(scan, scan.timestamp)
                self._start_next(scan.timestamp)
        return completed

    def _arrive(self, scan: LaserScan, t: float) -> None:
        m = self.metrics
        m.scans_received += 1
        policy = self.config.queue_policy
        if policy == QueuePolicy.QUEUE_ALL:
            self._waiting.append((scan, t))
        elif policy == QueuePolicy.LATEST_ONLY:
            if self._waiting:
                self._waiting.clear()
                m.scans_dropped += 1
            self._waiting.append((scan, t))
        elif len(self._waiting) < self.config.queue_capacity:
            self._waiting.append((scan, t))
        elif policy == QueuePolicy.DROP_NEWEST:
            m.scans_dropped += 1
        else:  # DROP_OLDEST
            self._waiting.popleft()
            self._waiting.append((scan, t))
            m.scans_dropped += 1
        m.queue_size_samples.append(len(self._waiting))
        m.max_queue_size = max(m.max_queue_size, len(self._waiting))

    def _start_next(self, t: float) -> None:
        if self._in_flight is not None or not self._waiting:
            return
        scan, arrival = self._waiting.popleft()
        self._in_flight = _InFlight(scan=scan, arrival_time=arrival, start_time=t, finish_time=t + self._draw_delay())

    def _complete(self, process: Callable[[LaserScan], Any]) -> CompletedScan:
        job = self._in_flight
        assert job is not None
        self._in_flight = None
        result = process(job.scan)
        m = self.metrics
        m.scans_processed += 1
        m.latencies.append(job.finish_time - job.arrival_time)
        m.processing_times.append(job.finish_time - job.start_time)
        return CompletedScan(
            scan=job.scan, arrival_time=job.arrival_time, start_time=job.start_time,
            finish_time=job.finish_time, result=result,
        )

    def _draw_delay(self) -> float:
        c = self.config
        if c.delay_model == ProcessingDelayModel.NONE:
            return 0.0
        if c.delay_max_s == c.delay_min_s:
            return c.delay_min_s
        return float(self._rng.uniform(c.delay_min_s, c.delay_max_s))
