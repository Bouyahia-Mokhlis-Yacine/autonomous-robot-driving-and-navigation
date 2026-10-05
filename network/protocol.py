"""Operator <-> robot-runtime wire protocol (version 1).

Transport: TCP, one UTF-8 JSON object per line (``\\n``-terminated). Stdlib
only. Every message has ``"v": 1`` and a ``"type"``.

Laptop (operator) -> robot runtime -- COMMANDS
    hello        {session, client}               must be the first message
    heartbeat    {seq}                           keeps the operator "present"
    set_goal     {seq, x, y, state_seq}          map-frame goal
    cancel_goal  {seq}
    stop         {seq}                           latched emergency stop
    resume       {seq, state_seq}                clears a latched stop (does NOT move)

Robot runtime -> laptop -- STATE
    welcome      {runtime_id, backend, protocol, frame, rates, safety}
    state        {seq, t, pose, health, nav, safety, scan, scan_pose, gt_pose?, stats, map_version}
    map          {version, t, resolution, width, height, origin_x, origin_y, cells, planning_cells}
    ack          {seq, accepted, reason}
    error        {reason}

``seq`` on commands is strictly increasing per operator SESSION (a session id
survives reconnects), so a replayed/duplicated/out-of-order command is
rejected. ``state_seq`` is the newest ``state.seq`` the operator had seen when
it issued the command: the runtime rejects goals/resumes that refer to state
it published more than ``max_command_state_age_s`` ago (clock-independent --
both ends only compare the RUNTIME's own publish times).

Coordinate conventions (the runtime is authoritative; the GUI only draws):
    map frame  : the SLAM/world frame. Origin (0, 0) = where the sensor was when
                 SLAM bootstrapped; +x = the sensor's heading at that moment,
                 +y = 90 degrees to its left. Metres.
    pose       : (x, y, theta) of the robot in the map frame; theta in radians,
                 counter-clockwise from +x, wrapped to (-pi, pi].
    goal, path : map-frame metres.
    grid       : cell (row, col) covers x in [origin_x + col*res, +res),
                 y in [origin_y + row*res, +res); row 0 is the LOWEST y
                 (nav_msgs/OccupancyGrid convention). Values: -1 unknown,
                 0 free, 100 occupied. Row-major int8, zlib, base64.
    scan       : a LaserScan (angles relative to the robot heading, CCW,
                 metres; range == range_max means "no return"), to be drawn
                 from ``scan_pose``.
    time       : ``t`` = seconds since the runtime started (runtime's own
                 monotonic clock). ``scan.timestamp`` is in the backend's
                 sensor clock (wall time for the real driver, simulated time
                 for the simulator) and is informational only.
"""

from __future__ import annotations

import base64
import json
import math
import zlib
from dataclasses import dataclass

import numpy as np

PROTOCOL_VERSION = 1
MAX_COMMAND_BYTES = 4096            # a command line longer than this is malformed
MAX_SESSION_LEN = 64

COMMAND_TYPES = ("hello", "heartbeat", "set_goal", "cancel_goal", "stop", "resume")


class ProtocolError(ValueError):
    """A message that must be rejected (malformed, wrong version, bad values)."""


@dataclass(frozen=True)
class Command:
    type: str
    seq: int | None = None          # None only for hello
    x: float | None = None
    y: float | None = None
    state_seq: int | None = None
    session: str | None = None
    client: str | None = None


def encode(message: dict) -> bytes:
    """One message -> one newline-terminated line. NaN/inf are refused."""
    return (json.dumps({"v": PROTOCOL_VERSION, **message}, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _int(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError(f"{name} must be a finite number")
    return float(value)


def _reject_constant(name: str):
    raise ProtocolError(f"{name} is not allowed")  # NaN / Infinity / -Infinity


def decode_command(line: bytes) -> Command:
    """Parse and strictly validate one inbound command line."""
    if len(line) > MAX_COMMAND_BYTES:
        raise ProtocolError("command too long")
    try:
        msg = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ProtocolError(f"not valid JSON: {err}") from None
    if not isinstance(msg, dict):
        raise ProtocolError("message must be a JSON object")
    if msg.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {msg.get('v')!r}")
    kind = msg.get("type")
    if kind not in COMMAND_TYPES:
        raise ProtocolError(f"unknown command type {kind!r}")
    allowed = {"v", "type"} | {
        "hello": {"session", "client"},
        "heartbeat": {"seq"},
        "set_goal": {"seq", "x", "y", "state_seq"},
        "cancel_goal": {"seq"},
        "stop": {"seq"},
        "resume": {"seq", "state_seq"},
    }[kind]
    extra = set(msg) - allowed
    if extra:
        raise ProtocolError(f"unexpected fields for {kind}: {sorted(extra)}")

    if kind == "hello":
        session = msg.get("session")
        if not isinstance(session, str) or not 0 < len(session) <= MAX_SESSION_LEN:
            raise ProtocolError("hello.session must be a non-empty string")
        client = msg.get("client", "")
        if not isinstance(client, str) or len(client) > 64:
            raise ProtocolError("hello.client must be a short string")
        return Command(type=kind, session=session, client=client)

    seq = _int(msg.get("seq"), "seq", minimum=1)
    if kind == "set_goal":
        return Command(type=kind, seq=seq, x=_finite(msg.get("x"), "x"), y=_finite(msg.get("y"), "y"),
                       state_seq=_int(msg.get("state_seq"), "state_seq"))
    if kind == "resume":
        return Command(type=kind, seq=seq, state_seq=_int(msg.get("state_seq"), "state_seq"))
    return Command(type=kind, seq=seq)


def decode_message(line: bytes) -> dict:
    """Parse a runtime -> operator line (used by the client)."""
    msg = json.loads(line.decode("utf-8"))
    if not isinstance(msg, dict) or msg.get("v") != PROTOCOL_VERSION or not isinstance(msg.get("type"), str):
        raise ProtocolError("bad message from runtime")
    return msg


# ------------------------------------------------------------------ grids --

def encode_cells(cells: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(cells, dtype=np.int8).tobytes(), 6)).decode("ascii")


def decode_cells(data: str, height: int, width: int) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(data))
    if len(raw) != height * width:
        raise ProtocolError("grid size does not match its metadata")
    return np.frombuffer(raw, dtype=np.int8).reshape(height, width).copy()


def round_list(values, digits: int = 4) -> list:
    return [round(float(v), digits) for v in values]
