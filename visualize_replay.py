"""Phase 8 manual-verification entry point: live WorldModelGUI, driven
by the real recorded stationary fixture through the actual Phase 1-7
pipeline (scan preprocessing -> ScanMatchingPipeline -> PersistentWorldModel).

No hardware required. Since ``tests/fixtures/real_scans_stationary.json``
is real but STATIONARY (only ~2.2s of recording, sensor never moved),
this only demonstrates: bootstrap, live map building, tentative-then-
confirmed geometry, and steady-state localization against a static
scene -- it replays on a loop so the GUI stays populated for as long as
the window is open. It does NOT demonstrate moving localization; no
claim of live-motion validation is made by running this script.

Scan replay runs on a dedicated background thread, paced to the
fixture's own recorded timestamps, completely decoupled from the GUI's
own independent redraw tick -- the same architectural pattern already
established by real_sensor_walkthrough.py's background reader thread,
now applied to a replay source instead of live hardware. The GUI only
ever reads the latest ``VisualizationState`` from a ``LatestStateBox``;
it never calls ``process_scan()``, ``register()``, or
``world_model.integrate()`` itself.

Run:
    conda activate AI-env
    python visualize_replay.py
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from common.logger import configure_logging, get_logger
from common.messages import LaserScan
from mapping.persistent_world_model import PersistentWorldModel
from slam.scan_matching_pipeline import BootstrapConfig, ScanMatchingPipeline
from visualization.gui import LatestStateBox, WorldModelGUI, build_visualization_state

logger = get_logger(__name__)

FIXTURE_PATH = Path(__file__).parent / "tests" / "fixtures" / "real_scans_stationary.json"


def load_fixture_scans() -> list[LaserScan]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as f:
        raw_scans = json.load(f)
    return [LaserScan(**raw) for raw in raw_scans]


def main() -> None:
    configure_logging("INFO")
    scans = load_fixture_scans()
    logger.info("Loaded %d scans from %s", len(scans), FIXTURE_PATH)

    # Shorter confirmation/bootstrap durations than the production
    # defaults, purely so this short (~2.2s) stationary fixture
    # visibly reaches CONFIRMED geometry and COMPLETE bootstrap within
    # a single loop of the replay -- not a claim about tuned defaults.
    world_model = PersistentWorldModel(confirmation_duration=0.3)
    pipeline = ScanMatchingPipeline(
        reference_provider=world_model, world_model=world_model,
        bootstrap_config=BootstrapConfig(min_duration=0.3, timeout=5.0),
    )

    state_box = LatestStateBox()
    trajectory: list[tuple[float, float]] = []
    stop_event = threading.Event()

    def replay_loop() -> None:
        while not stop_event.is_set():
            base_scan_t: float | None = None
            base_host_t: float | None = None
            for scan in scans:
                if stop_event.is_set():
                    return
                if base_scan_t is None:
                    base_scan_t = scan.timestamp
                    base_host_t = time.time()
                else:
                    target = base_host_t + (scan.timestamp - base_scan_t)
                    delay = target - time.time()
                    if delay > 0:
                        time.sleep(delay)

                start = time.perf_counter()
                result = pipeline.process_scan(scan)
                processing_ms = (time.perf_counter() - start) * 1000.0

                if result.accepted:
                    trajectory.append((result.pose.x, result.pose.y))

                state = build_visualization_state(
                    world_model, result, scan, tuple(trajectory), processing_time_ms=processing_ms,
                )
                state_box.set(state)
            time.sleep(0.5)  # brief pause between loops of the short fixture

    reader_thread = threading.Thread(target=replay_loop, daemon=True)
    reader_thread.start()

    gui = WorldModelGUI(state_box, scale=30.0, dt=0.1)
    gui.root.title("World model viewer -- real stationary fixture replay (looping)")

    def on_close() -> None:
        logger.info("Window closed; stopping replay.")
        stop_event.set()
        gui.close()

    gui.root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        gui.run()
    finally:
        stop_event.set()
        reader_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
