"""End to end on localhost with the REAL simulated backend (realistic RPLiDAR
model + asynchronous SLAM + the stress-tested NavigationManager), in real
time: the same architecture as laptop GUI <-> Raspberry Pi, minus the Wi-Fi.
Also checks the GUI's remote proxies (no display needed)."""

from __future__ import annotations

import json
import math
import socket
import time

import pytest

from network.client import OperatorClient
from network.protocol import PROTOCOL_VERSION
from robot_runtime import build_runtime
from runtime.config import load_runtime_config
from visualization.operator_gui import RemoteDataSource, RemoteNavigation


@pytest.fixture
def sim_runtime():
    config = load_runtime_config()
    config.safety.operator_timeout_s = 0.6
    runtime = build_runtime("sim", config, seed=42, host="127.0.0.1", port=0)
    runtime.start()
    yield runtime
    runtime.shutdown()


def gt(client: OperatorClient) -> tuple[float, float]:
    p = client.state["gt_pose"]
    return p["x"], p["y"]


def moved(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_full_operator_session_against_the_simulated_robot(sim_runtime) -> None:
    runtime = sim_runtime
    client = OperatorClient("127.0.0.1", runtime.server.port, heartbeat_s=0.2).start()
    # 1-3: connect, receive the map / pose / scan; wait for SLAM bootstrap.
    assert client.wait_for(lambda c: c.map is not None and c.state is not None
                           and c.state["health"]["state"] == "LOCALIZED" and c.state["scan"] is not None, 15)
    assert client.state["backend"] == "sim" and client.welcome["frame"]["frame"].startswith("map")

    # 4-7: click a goal -> the backend's NavigationManager plans and drives.
    start = gt(client)
    seq = client.set_goal(3.0, 0.0)
    assert client.wait_for(lambda c: c.ack_for(seq) is not None, 3) and client.ack_for(seq)["accepted"]
    assert client.wait_for(lambda c: c.state["nav"]["goal"] == {"x": 3.0, "y": 0.0} and c.state["nav"]["path"], 3)
    assert runtime.applied_goals == 1
    assert client.wait_for(lambda c: moved(gt(c), start) > 0.3, 15)

    # 8: STOP -> the simulated robot stops and stays stopped.
    seq = client.stop()
    assert client.wait_for(lambda c: c.state["safety"]["stop_latched"]
                           and c.state["nav"]["state"] == "EMERGENCY_STOP", 3)
    time.sleep(0.5)
    at_stop = gt(client)
    time.sleep(1.5)
    assert moved(gt(client), at_stop) < 0.01
    assert not runtime.backend.stats()["collision"]

    # RESUME, new goal: moves again.
    client.resume()
    assert client.wait_for(lambda c: not c.state["safety"]["stop_latched"], 3)
    seq = client.set_goal(3.0, 0.0)
    assert client.wait_for(lambda c: c.ack_for(seq) is not None and c.ack_for(seq)["accepted"], 3)
    before = gt(client)
    assert client.wait_for(lambda c: moved(gt(c), before) > 0.1, 10)

    # 9: kill the GUI while the robot is driving -> operator-loss policy stops it.
    client.close()
    time.sleep(1.5)
    nav = runtime.backend.nav_manager
    assert nav.get_goal() is None and nav.state.value == "IDLE"
    stopped_at = runtime.backend.ground_truth_pose()
    time.sleep(1.0)
    now = runtime.backend.ground_truth_pose()
    assert math.hypot(now.x - stopped_at.x, now.y - stopped_at.y) < 0.01

    # 10: reconnect -> current state is recovered, nothing is re-executed.
    goals_before = runtime.applied_goals
    again = OperatorClient("127.0.0.1", runtime.server.port, heartbeat_s=0.2).start()
    assert again.wait_for(lambda c: c.map is not None and c.state is not None, 5)
    assert again.state["nav"]["state"] == "IDLE" and again.state["nav"]["goal"] is None
    assert any("operator lost" in e for e in again.state["safety"]["events"])
    time.sleep(1.0)
    assert runtime.applied_goals == goals_before
    again.close()


def test_gui_proxies_show_exactly_what_the_robot_published(sim_runtime) -> None:
    runtime = sim_runtime
    client = OperatorClient("127.0.0.1", runtime.server.port, heartbeat_s=0.2).start()
    assert client.wait_for(lambda c: c.map is not None and c.state is not None
                           and c.state["health"]["state"] == "LOCALIZED", 15)
    nav, data = RemoteNavigation(client), RemoteDataSource(client)
    nav.set_goal(type("G", (), {"x": 3.0, "y": 0.0})())                   # what a left-click does
    assert client.wait_for(lambda c: c.state["nav"]["path"], 5)
    assert (nav.get_goal().x, nav.get_goal().y) == (3.0, 0.0)
    assert [[p.x, p.y] for p in nav.get_path().points] == client.state["nav"]["path"]
    grid = data.static_map().grid
    robot_grid = runtime.backend.world_grid()
    assert (grid.resolution, grid.width, grid.height, grid.origin_x, grid.origin_y) == \
        (robot_grid.resolution, robot_grid.width, robot_grid.height, robot_grid.origin_x, robot_grid.origin_y)
    scan = data.scan()
    assert scan is not None and scan.num_rays == len(client.state["scan"]["ranges"])
    client.stop()
    client.close()


def test_a_stalled_operator_does_not_slow_slam(sim_runtime) -> None:
    runtime = sim_runtime
    stalled = socket.create_connection(("127.0.0.1", runtime.server.port))
    stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    stalled.sendall(json.dumps({"v": PROTOCOL_VERSION, "type": "hello", "session": "stall"}).encode() + b"\n")
    before, t0 = runtime.backend.stats()["scans_processed"], time.monotonic()
    time.sleep(3.0)
    rate = (runtime.backend.stats()["scans_processed"] - before) / (time.monotonic() - t0)
    assert rate > 9.0          # the simulated RPLiDAR's ~11 Hz is still fully processed
    stalled.close()
