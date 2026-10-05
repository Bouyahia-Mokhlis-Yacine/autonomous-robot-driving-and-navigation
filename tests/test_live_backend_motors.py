"""The real LIVE backend with GPIO motors (dry-run), end to end over TCP, with
only the serial LiDAR replaced by a stream of realistic simulated scans
stamped with wall-clock time (as the real driver stamps them). Everything else
is the production path: LiveProcessor threads -> ScanMatchingPipeline ->
PersistentWorldModel -> health -> NavigationSession/NavigationManager ->
SafetyGatedRobot -> CommandRecorder -> GpioDiffDriveRobot -> RobotRuntime ->
RobotServer -> OperatorClient."""

from __future__ import annotations

import threading
import time

import pytest

from common.config import load_navigation_config, load_robot_config
from network.client import OperatorClient
from robot_runtime import build_runtime
from runtime.backends import LiveBackend
from runtime.config import load_runtime_config


class SimulatedSerialLidar:
    """Stands in for RPLidarDriver: yields realistic scans from the middle of
    the simulator's room at ~11 Hz, timestamped with time.time()."""

    def __init__(self) -> None:
        import dataclasses
        from simulation.realistic_lidar import RealisticLidar
        from simulation.scenario import default_scenario

        robot = load_robot_config().robot
        env = default_scenario(robot.width, robot.length).environment
        pose = env.robot_pose
        lidar = RealisticLidar(load_navigation_config().lidar, seed=3)
        self._scans = [r.scan for r in lidar.acquire_until(env, lambda t: pose, 60.0) if r.scan is not None]
        self._replace = dataclasses.replace

    def iter_scans(self):
        for scan in self._scans:
            time.sleep(0.09)
            yield self._replace(scan, timestamp=time.time())

    def disconnect(self) -> None:
        pass


@pytest.fixture
def live_runtime(monkeypatch):
    def fake_start(self, rplidar_cfg, nav_cfg):
        from real_world_model_walkthrough import run_threads
        self._driver = SimulatedSerialLidar()
        self._threads = run_threads(self._driver, self.processor, self._stop_event)

    monkeypatch.setattr(LiveBackend, "_start_lidar", fake_start)
    config = load_runtime_config()
    config.live.motors, config.live.motors_dry_run, config.live.motor_guard = "gpio", True, False
    config.live.map_size_m = 14.0
    runtime = build_runtime("live", config, host="127.0.0.1", port=0)
    runtime.start()
    yield runtime
    runtime.shutdown()


def test_goal_reaches_the_motors_and_stop_cuts_them(live_runtime) -> None:
    runtime = live_runtime
    motors = runtime.backend._inner_robot
    assert motors.dry_run and runtime.backend.motors_enabled
    client = OperatorClient("127.0.0.1", runtime.server.port, heartbeat_s=0.2).start()
    assert client.wait_for(lambda c: c.state is not None and c.map is not None
                           and c.state["health"]["state"] == "LOCALIZED", 15)

    # Motors attached -> STOP latched at startup; a goal is refused.
    assert client.state["safety"]["stop_latched"] and client.state["stats"]["motors"] == "gpio"
    seq = client.set_goal(3.0, 1.5)
    assert client.wait_for(lambda c: c.ack_for(seq) is not None, 3) and not client.ack_for(seq)["accepted"]
    time.sleep(0.5)
    assert motors.last_duty == (0.0, 0.0)

    # RESUME alone never moves.
    client.resume()
    assert client.wait_for(lambda c: not c.state["safety"]["stop_latched"], 3)
    time.sleep(0.5)
    assert motors.last_duty == (0.0, 0.0)

    # A goal -> NavigationManager plans -> nonzero duty on the L298N pins.
    seq = client.set_goal(1.0, 1.2)
    assert client.wait_for(lambda c: c.ack_for(seq) is not None and c.ack_for(seq)["accepted"], 3)
    assert client.wait_for(lambda c: c.state["nav"]["path"], 5)
    assert client.wait_for(lambda c: motors.last_duty != (0.0, 0.0), 5)
    assert runtime.backend.command_history.command_at(time.time()) != (0.0, 0.0)   # commands recorded

    # STOP from the GUI -> zero duty immediately, and it stays zero.
    client.stop()
    assert client.wait_for(lambda c: motors.last_duty == (0.0, 0.0), 1.0)
    for _ in range(10):
        time.sleep(0.05)
        assert motors.last_duty == (0.0, 0.0)
    client.close()


def test_operator_loss_stops_the_motors(live_runtime) -> None:
    runtime = live_runtime
    runtime.config.safety.operator_timeout_s = 0.6
    runtime.server.operator_timeout_s = 0.6
    motors = runtime.backend._inner_robot
    client = OperatorClient("127.0.0.1", runtime.server.port, heartbeat_s=0.2).start()
    assert client.wait_for(lambda c: c.state is not None and c.state["health"]["state"] == "LOCALIZED", 15)
    client.resume()
    assert client.wait_for(lambda c: not c.state["safety"]["stop_latched"], 3)
    client.set_goal(1.0, 1.2)
    assert client.wait_for(lambda c: motors.last_duty != (0.0, 0.0), 5)
    client.close()                                                        # GUI gone
    end = time.monotonic() + 3.0
    while time.monotonic() < end and motors.last_duty != (0.0, 0.0):
        time.sleep(0.05)
    assert motors.last_duty == (0.0, 0.0)
    assert runtime.backend.nav_manager.get_goal() is None


def test_shutdown_releases_the_motors(live_runtime) -> None:
    runtime = live_runtime
    motors = runtime.backend._inner_robot
    runtime.shutdown()
    assert all(m.closed for m in motors._motors.values())
    assert all(v == 0.0 for v in motors.motor_values().values())


def test_lidar_buffer_clears_are_counted_in_the_stats(live_runtime) -> None:
    import logging
    backend = live_runtime.backend
    assert backend.stats()["lidar_buffer_clears"] == 0
    driver_log = logging.getLogger("autonomous_navigation.hardware.rplidar_driver")
    driver_log.warning("Too many measurments in the input buffer: %d/%d. Clearing buffer...", 533, 500)
    driver_log.info("Starting motor")
    assert backend.stats()["lidar_buffer_clears"] == 1
