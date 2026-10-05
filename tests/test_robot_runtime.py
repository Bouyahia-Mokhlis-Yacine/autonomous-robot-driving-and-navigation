"""RobotRuntime + RobotServer + OperatorClient over localhost, with a small
deterministic backend (a real NavigationManager on an open grid, a robot
that integrates its commands). The control loop is ticked MANUALLY here, so
every safety assertion is about a known tick, not a timing race."""

from __future__ import annotations

import json
import socket
import time

import pytest

from common.messages import Goal, Pose2D
from navigation.navigation_manager import NavigationState
from network.client import OperatorClient
from network.protocol import PROTOCOL_VERSION, decode_message
from network.server import AdmittedCommand
from network.protocol import Command
from runtime.config import NetworkSettings, RuntimeConfig, SafetySettings
from runtime.core import RobotRuntime
from tests.test_navigation_manager import make_grid, make_manager

DT = 0.1


class AlwaysLocalized:
    class _H:
        value = "LOCALIZED"

    health = _H()
    loss_events = 0

    def pose_age(self) -> float:
        return 0.05

    def is_localization_valid(self) -> bool:
        return True


class FakeBackend:
    """Real NavigationManager + a robot that actually moves under its commands."""

    name = "fake"

    def __init__(self) -> None:
        self.nav_manager, self.robot = make_manager(pose=Pose2D(1.0, 1.0, 0.0))
        self.grid = make_grid()
        self.health = AlwaysLocalized()
        self.goals_set: list[tuple[float, float]] = []
        self.stop_now_calls = 0
        self.version = 0

    def tick(self, dt: float) -> None:
        self.nav_manager.update(self.grid, dt)
        self.robot.integrate(dt)
        self.version += 1

    def stop_now(self) -> None:
        self.stop_now_calls += 1
        self.robot.stop()

    def set_goal(self, x: float, y: float) -> None:
        self.goals_set.append((x, y))
        self.nav_manager.set_goal(Goal(x, y), self.grid)

    def pose(self):
        return self.robot.get_pose()

    def ground_truth_pose(self):
        return None

    def latest_scan(self):
        return None, None

    def world_grid(self):
        return self.grid.grid

    def planning_grid(self):
        return self.grid.grid

    def map_version(self) -> int:
        return self.version

    def stats(self) -> dict:
        return {}

    def close(self) -> None:
        pass


def make_runtime(**safety) -> tuple[RobotRuntime, FakeBackend]:
    backend = FakeBackend()
    cfg = RuntimeConfig(network=NetworkSettings(state_rate_hz=1000.0, map_rate_hz=1000.0),
                        safety=SafetySettings(**{"operator_timeout_s": 0.5, **safety}))
    runtime = RobotRuntime(backend, cfg, host="127.0.0.1", port=0)
    runtime.server.start()  # network only; ticks are driven by the test
    return runtime, backend


def connect(runtime: RobotRuntime) -> OperatorClient:
    client = OperatorClient("127.0.0.1", runtime.server.port, heartbeat_s=0.1).start()
    assert client.wait_for(lambda c: c.connected, 3.0)
    return client


def tick_until(runtime: RobotRuntime, predicate, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        runtime.tick(DT)
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def synced(runtime: RobotRuntime, client: OperatorClient) -> None:
    assert tick_until(runtime, lambda: client.state is not None and client.map is not None)


@pytest.fixture
def rt():
    runtime, backend = make_runtime()
    yield runtime, backend
    runtime.server.close()


# ---------------------------------------------------------- goals / state --

def test_goal_is_transmitted_planned_on_the_robot_and_state_comes_back(rt) -> None:
    runtime, backend = rt
    client = connect(runtime)
    synced(runtime, client)
    seq = client.set_goal(8.0, 8.0)
    assert tick_until(runtime, lambda: client.ack_for(seq) is not None)
    assert client.ack_for(seq)["accepted"]
    assert backend.goals_set == [(8.0, 8.0)]                          # exactly the coordinates sent
    assert tick_until(runtime, lambda: client.state["nav"]["state"] == "MOVING" and client.state["nav"]["path"])
    robot_path = [[round(p.x, 4), round(p.y, 4)] for p in backend.nav_manager.get_path().points]
    assert client.state["nav"]["path"] == robot_path                   # the robot's path, not a GUI one
    assert client.state["nav"]["goal"] == {"x": 8.0, "y": 8.0}
    start = backend.pose().x, backend.pose().y
    assert tick_until(runtime, lambda: abs(backend.pose().x - start[0]) + abs(backend.pose().y - start[1]) > 0.2)
    client.close()


def test_map_and_pose_coordinates_are_consistent_end_to_end(rt) -> None:
    runtime, backend = rt
    client = connect(runtime)
    synced(runtime, client)
    g = backend.grid.grid
    m = client.map
    assert (m["resolution"], m["width"], m["height"], m["origin_x"], m["origin_y"]) == \
        (g.resolution, g.width, g.height, g.origin_x, g.origin_y)
    assert (m["cells"] == g.cells).all()
    p = backend.pose()
    assert client.state["pose"] == {"x": round(p.x, 4), "y": round(p.y, 4), "theta": round(p.theta, 5)}
    client.close()


def test_goal_outside_the_map_is_rejected(rt) -> None:
    runtime, backend = rt
    client = connect(runtime)
    synced(runtime, client)
    seq = client.set_goal(50.0, 50.0)
    assert tick_until(runtime, lambda: client.ack_for(seq) is not None)
    assert not client.ack_for(seq)["accepted"] and backend.goals_set == []
    client.close()


# ------------------------------------------------------------------- STOP --

def test_stop_latches_immediately_without_waiting_for_a_tick(rt) -> None:
    runtime, backend = rt
    client = connect(runtime)
    synced(runtime, client)
    client.set_goal(8.0, 8.0)
    tick_until(runtime, lambda: backend.robot.last_command.linear > 0)
    client.stop()
    # No tick: the network thread itself latched STOP and zeroed the robot.
    end = time.monotonic() + 2.0
    while not runtime.is_stop_latched() and time.monotonic() < end:
        time.sleep(0.01)
    assert runtime.is_stop_latched() and backend.stop_now_calls >= 1
    assert backend.robot.last_command.linear == 0.0
    runtime.tick(DT)
    assert backend.nav_manager.state is NavigationState.EMERGENCY_STOP
    assert backend.nav_manager.get_goal() is None                      # nothing to resume by itself
    for _ in range(10):
        runtime.tick(DT)
        assert backend.robot.last_command.linear == 0.0
    client.close()


def test_goals_are_refused_while_stopped_and_resume_does_not_move(rt) -> None:
    runtime, backend = rt
    client = connect(runtime)
    synced(runtime, client)
    client.stop()
    tick_until(runtime, runtime.is_stop_latched)
    seq = client.set_goal(8.0, 8.0)
    assert tick_until(runtime, lambda: client.ack_for(seq) is not None)
    assert not client.ack_for(seq)["accepted"] and backend.goals_set == []
    seq = client.resume()
    assert tick_until(runtime, lambda: client.ack_for(seq) is not None)
    assert not runtime.is_stop_latched()
    for _ in range(5):
        runtime.tick(DT)
    assert backend.nav_manager.state is NavigationState.IDLE and backend.robot.last_command.linear == 0.0
    seq = client.set_goal(8.0, 8.0)                                     # a NEW goal moves it again
    assert tick_until(runtime, lambda: backend.robot.last_command.linear > 0)
    client.close()


def test_start_stopped_requires_resume_before_any_goal() -> None:
    runtime, backend = make_runtime(start_stopped=True)
    try:
        client = connect(runtime)
        synced(runtime, client)
        seq = client.set_goal(8.0, 8.0)
        assert tick_until(runtime, lambda: client.ack_for(seq) is not None)
        assert not client.ack_for(seq)["accepted"] and backend.goals_set == []
        client.close()
    finally:
        runtime.server.close()


# ------------------------------------------------- malformed / stale input --

def raw_session(runtime: RobotRuntime, session: str = "raw-session") -> tuple[socket.socket, callable]:
    sock = socket.create_connection(("127.0.0.1", runtime.server.port), timeout=2.0)
    reader = sock.makefile("rb")

    def send(obj) -> None:
        sock.sendall((json.dumps(obj) if not isinstance(obj, bytes) else obj.decode()).encode() + b"\n")

    def replies(kind: str, n: int = 1, timeout: float = 3.0) -> list[dict]:
        found, end = [], time.monotonic() + timeout
        while len(found) < n and time.monotonic() < end:
            runtime.tick(DT)
            sock.settimeout(0.05)
            try:
                msg = decode_message(reader.readline())
            except (socket.timeout, OSError, ValueError):
                continue
            if msg["type"] == kind:
                found.append(msg)
        return found

    send({"v": PROTOCOL_VERSION, "type": "hello", "session": session, "client": "test"})
    return sock, send, replies


def test_malformed_commands_are_rejected_and_never_applied(rt) -> None:
    runtime, backend = rt
    sock, send, replies = raw_session(runtime)
    send(b"this is not json")
    send({"v": PROTOCOL_VERSION, "type": "set_goal", "seq": 1, "x": "far", "y": 0, "state_seq": 1})
    send({"v": PROTOCOL_VERSION, "type": "self_destruct", "seq": 2})
    assert len(replies("error", n=3)) == 3
    assert backend.goals_set == [] and runtime.server.rejected_count >= 3
    sock.close()


def test_commands_before_hello_are_rejected(rt) -> None:
    runtime, backend = rt
    sock = socket.create_connection(("127.0.0.1", runtime.server.port), timeout=2.0)
    sock.sendall(json.dumps({"v": PROTOCOL_VERSION, "type": "stop", "seq": 1}).encode() + b"\n")
    reply = decode_message(sock.makefile("rb").readline())
    assert reply["type"] == "error" and "hello" in reply["reason"]
    assert not runtime.is_stop_latched()
    sock.close()


def test_duplicate_and_out_of_order_sequence_numbers_are_rejected(rt) -> None:
    runtime, backend = rt
    sock, send, replies = raw_session(runtime)
    runtime.tick(DT)
    state_seq = runtime.server._state_seq
    send({"v": PROTOCOL_VERSION, "type": "set_goal", "seq": 5, "x": 8.0, "y": 8.0, "state_seq": state_seq})
    first = replies("ack")
    send({"v": PROTOCOL_VERSION, "type": "set_goal", "seq": 5, "x": 2.0, "y": 2.0, "state_seq": state_seq})
    send({"v": PROTOCOL_VERSION, "type": "set_goal", "seq": 4, "x": 3.0, "y": 3.0, "state_seq": state_seq})
    later = replies("ack", n=2)
    assert first and first[0]["accepted"]
    assert [a["accepted"] for a in later] == [False, False] and all("stale seq" in a["reason"] for a in later)
    assert backend.goals_set == [(8.0, 8.0)]                              # applied exactly once
    sock.close()


def test_goal_against_stale_state_is_rejected() -> None:
    runtime, backend = make_runtime(max_command_state_age_s=0.2)
    try:
        sock, send, replies = raw_session(runtime)
        runtime.tick(DT)
        old = runtime.server._state_seq
        time.sleep(0.3)                                                   # the operator's view is now stale
        send({"v": PROTOCOL_VERSION, "type": "set_goal", "seq": 1, "x": 8.0, "y": 8.0, "state_seq": old})
        ack = replies("ack")[0]
        assert not ack["accepted"] and "stale" in ack["reason"] and backend.goals_set == []
        send({"v": PROTOCOL_VERSION, "type": "set_goal", "seq": 2, "x": 8.0, "y": 8.0, "state_seq": 999999})
        assert not replies("ack")[0]["accepted"]                          # unknown / future state
        sock.close()
    finally:
        runtime.server.close()


def test_queued_commands_from_a_dropped_connection_are_discarded(rt) -> None:
    runtime, backend = rt
    runtime._commands.put(AdmittedCommand(Command(type="set_goal", seq=1, x=8.0, y=8.0, state_seq=1),
                                          connection_id=424242, session="gone", received_at=time.monotonic()))
    runtime.tick(DT)
    assert backend.goals_set == [] and any("discarded" in e for e in runtime.events)


# -------------------------------------------------- disconnect / reconnect --

def test_operator_loss_cancels_the_goal_under_the_stop_policy(rt) -> None:
    runtime, backend = rt
    client = connect(runtime)
    synced(runtime, client)
    client.set_goal(8.0, 8.0)
    assert tick_until(runtime, lambda: backend.robot.last_command.linear > 0)
    client.close()                                                       # GUI killed
    assert tick_until(runtime, lambda: backend.nav_manager.get_goal() is None, timeout=3.0)
    for _ in range(5):
        runtime.tick(DT)
    assert backend.robot.last_command.linear == 0.0
    assert any("operator lost" in e for e in runtime.events)


def test_operator_loss_under_the_continue_policy_keeps_the_goal() -> None:
    runtime, backend = make_runtime(operator_loss_policy="continue")
    try:
        client = connect(runtime)
        synced(runtime, client)
        client.set_goal(8.0, 8.0)
        tick_until(runtime, lambda: backend.robot.last_command.linear > 0)
        client.close()
        time.sleep(0.7)
        for _ in range(3):
            runtime.tick(DT)
        assert backend.nav_manager.get_goal() == Goal(8.0, 8.0)
        assert any("continuing" in e for e in runtime.events)
    finally:
        runtime.server.close()


def test_reconnect_resyncs_state_and_never_duplicates_a_goal(rt) -> None:
    runtime, backend = rt
    first = connect(runtime)
    synced(runtime, first)
    first.set_goal(8.0, 8.0)
    tick_until(runtime, lambda: len(backend.goals_set) == 1)
    first.close()
    tick_until(runtime, lambda: backend.nav_manager.get_goal() is None)   # operator-loss policy
    second = connect(runtime)                                            # a fresh GUI
    synced(runtime, second)                                              # full map + state on connect
    for _ in range(10):
        runtime.tick(DT)
    assert backend.goals_set == [(8.0, 8.0)]                             # nothing re-sent or re-applied
    # the snapshot synced() saw may predate the cancel; state is published asynchronously, so wait for it
    assert tick_until(runtime, lambda: second.state["nav"]["state"] == "IDLE", timeout=3.0)
    assert second.welcome["backend"] == "fake"
    second.close()


def test_client_never_queues_commands_while_disconnected(rt) -> None:
    runtime, backend = rt
    client = OperatorClient("127.0.0.1", 1, heartbeat_s=0.1).start()     # nothing listens on port 1
    time.sleep(0.2)
    assert client.set_goal(1.0, 1.0) is None and client.stop() is None
    client.close()


# ------------------------------------------------- network cannot block --

def test_a_stalled_operator_cannot_block_the_control_loop() -> None:
    """An operator that connects and then never reads: its socket buffers fill,
    its writer times out and is dropped -- the control loop keeps its rate."""
    backend = FakeBackend()
    cfg = RuntimeConfig(network=NetworkSettings(state_rate_hz=100.0, map_rate_hz=50.0, send_timeout_s=0.3),
                        safety=SafetySettings())
    runtime = RobotRuntime(backend, cfg, host="127.0.0.1", port=0)
    runtime.config.control.rate_hz = 50.0
    runtime.start()
    try:
        stalled = socket.create_connection(("127.0.0.1", runtime.server.port))
        stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        stalled.sendall(json.dumps({"v": PROTOCOL_VERSION, "type": "hello", "session": "stall"}).encode() + b"\n")
        start_ticks, start = runtime.tick_count, time.monotonic()
        time.sleep(2.0)
        rate = (runtime.tick_count - start_ticks) / (time.monotonic() - start)
        assert rate > 40.0                                                # ~50 Hz despite the stalled operator
        assert runtime.max_tick_s < 0.1
        stalled.close()
    finally:
        runtime.shutdown()


def test_publishing_is_constant_time_for_the_caller() -> None:
    runtime, backend = make_runtime()
    try:
        runtime.tick(DT)
        state = runtime.state_message(time.monotonic())
        start = time.perf_counter()
        for _ in range(1000):
            runtime.server.publish_state(state)
        assert time.perf_counter() - start < 0.2                           # no encoding / sending on the caller
    finally:
        runtime.server.close()
