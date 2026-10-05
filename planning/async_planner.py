"""Global planning off the control loop.

``NavigationManager`` used to call ``GlobalPlanner.plan`` + ``process_path``
inside its ``update()`` -- i.e. inside the robot runtime's control tick. On
the Pi that search takes 0.7 s for a far goal in a room and 1.6 s or more for
an unreachable one (it explores every reachable cell before giving up), and
before the A* speed-up it took 2.6 s / 6 s: the control loop froze for up to
4.8 s in the first real-sensor test.

``compute_plan`` is exactly that same computation, bounded by a deadline.
``ProcessPlanExecutor`` runs it in a separate, persistent worker PROCESS --
not a thread: a pure-Python search in a thread would still hold the GIL and
starve the RPLiDAR reader thread. The executor is fully non-blocking for the
caller:

    submit(request)  hands the request over (the newest request supersedes
                     any older one: the worker drops queued ones and aborts
                     a running search that has been superseded)
    poll()           returns finished results, never waits

Requests carry an id; the caller decides whether a result still matters
(``NavigationManager`` ignores results for anything but its current request).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Protocol

from common.messages import Goal, Path, Pose2D
from mapping.occupancy_grid import OccupancyGridMap
from planning.astar import PlanningTimeout
from planning.global_planner import GlobalPlanner
from planning.path_smoothing import process_path

logger = logging.getLogger("autonomous_navigation.planning.async_planner")

OK, NO_PATH, TIMEOUT, ERROR, SUPERSEDED = "ok", "no_path", "timeout", "error", "superseded"


@dataclass(frozen=True)
class PlanRequest:
    request_id: int
    grid_map: OccupancyGridMap      # the planning (inflated) map the plan is for
    start: Pose2D
    goal: Goal
    planner: GlobalPlanner          # carries allow_diagonal / unknown_cost_multiplier
    clearance_m: float              # process_path clearance (NavigationManager's path_clearance_margin)
    time_budget_s: float


@dataclass(frozen=True)
class PlanResult:
    request_id: int
    status: str                     # OK / NO_PATH / TIMEOUT / ERROR / SUPERSEDED
    path: Path | None
    elapsed_s: float
    detail: str = ""


def compute_plan(request: PlanRequest, should_abort=None) -> PlanResult:
    """``GlobalPlanner.plan`` then ``process_path`` exactly as
    ``NavigationManager._plan`` does them, within ``time_budget_s``."""
    started = time.monotonic()
    deadline = started + request.time_budget_s
    try:
        raw_path = request.planner.plan(request.grid_map, request.start, request.goal,
                                        deadline=deadline, should_abort=should_abort)
    except PlanningTimeout as exc:
        superseded = should_abort is not None and should_abort()
        return PlanResult(request.request_id, SUPERSEDED if superseded else TIMEOUT, None,
                          time.monotonic() - started, str(exc))
    except Exception as exc:  # noqa: BLE001 -- reported to the caller, which stops the robot
        return PlanResult(request.request_id, ERROR, None, time.monotonic() - started, repr(exc))
    if raw_path is None:
        return PlanResult(request.request_id, NO_PATH, None, time.monotonic() - started)
    path = process_path(raw_path, request.grid_map, simplify=True, smooth=True, clearance_m=request.clearance_m)
    return PlanResult(request.request_id, OK, path, time.monotonic() - started)


class PlanExecutor(Protocol):
    def submit(self, request: PlanRequest) -> None: ...
    def poll(self) -> list[PlanResult]: ...
    def close(self) -> None: ...


def _worker_main(requests, results, newest_id) -> None:
    """Worker process: newest request only, abort once superseded."""
    while True:
        request = requests.get()
        if request is None:
            return
        while True:  # drop everything but the newest queued request
            try:
                newer = requests.get_nowait()
            except queue.Empty:
                break
            if newer is None:
                return
            results.put(PlanResult(request.request_id, SUPERSEDED, None, 0.0))
            request = newer
        rid = request.request_id
        results.put(compute_plan(request, should_abort=lambda: newest_id.value != rid))


class ProcessPlanExecutor:
    """``compute_plan`` in one persistent worker process (spawned once at
    construction, restarted if it dies)."""

    def __init__(self) -> None:
        # spawn, not fork: the runtime is multi-threaded when this starts.
        self._ctx = mp.get_context("spawn")
        self._newest_id = self._ctx.Value("q", -1, lock=False)
        self._outstanding: set[int] = set()
        self.restarts = 0
        self._start_worker()

    def _start_worker(self) -> None:
        self._requests = self._ctx.Queue()
        self._results = self._ctx.Queue()
        self._process = self._ctx.Process(target=_worker_main, args=(self._requests, self._results, self._newest_id),
                                          name="planner-worker", daemon=True)
        self._process.start()

    def submit(self, request: PlanRequest) -> None:
        self._newest_id.value = request.request_id
        self._outstanding.add(request.request_id)
        self._requests.put(request)

    def poll(self) -> list[PlanResult]:
        out: list[PlanResult] = []
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            self._outstanding.discard(result.request_id)
            out.append(result)
        if not self._process.is_alive():
            logger.error("Planner worker died (exit code %s); restarting it", self._process.exitcode)
            out += [PlanResult(rid, ERROR, None, 0.0, "planner worker died") for rid in sorted(self._outstanding)]
            self._outstanding.clear()
            self.restarts += 1
            self._start_worker()
        return out

    def close(self) -> None:
        try:
            self._requests.put(None)
            self._process.join(timeout=2.0)
        finally:
            if self._process.is_alive():
                self._process.terminate()
