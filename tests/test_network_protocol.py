"""network/protocol.py: framing, strict command validation, grid encoding."""

from __future__ import annotations

import json

import numpy as np
import pytest

from network.protocol import (
    MAX_COMMAND_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_cells,
    decode_command,
    decode_message,
    encode,
    encode_cells,
)


def line(**msg) -> bytes:
    return json.dumps({"v": PROTOCOL_VERSION, **msg}).encode()


def test_commands_round_trip() -> None:
    assert decode_command(encode({"type": "hello", "session": "abc", "client": "gui"})).session == "abc"
    goal = decode_command(encode({"type": "set_goal", "seq": 3, "x": 1.25, "y": -2.5, "state_seq": 17}))
    assert (goal.type, goal.seq, goal.x, goal.y, goal.state_seq) == ("set_goal", 3, 1.25, -2.5, 17)
    assert decode_command(encode({"type": "stop", "seq": 9})).type == "stop"
    assert decode_command(encode({"type": "resume", "seq": 10, "state_seq": 4})).state_seq == 4
    assert decode_command(encode({"type": "cancel_goal", "seq": 11})).seq == 11
    assert decode_command(encode({"type": "heartbeat", "seq": 12})).seq == 12


def test_integer_coordinates_are_accepted_as_floats() -> None:
    goal = decode_command(line(type="set_goal", seq=1, x=2, y=3, state_seq=1))
    assert isinstance(goal.x, float) and goal.x == 2.0


@pytest.mark.parametrize("raw", [
    b"not json",
    b"\xff\xfe",                                                        # not UTF-8
    b"[1, 2, 3]",                                                       # not an object
    json.dumps({"type": "stop", "seq": 1}).encode(),                    # no version
    json.dumps({"v": 99, "type": "stop", "seq": 1}).encode(),           # wrong version
    line(type="drive_forward", seq=1),                                  # unknown command
    line(type="stop"),                                                  # missing seq
    line(type="stop", seq=0),                                           # seq must be >= 1
    line(type="stop", seq=-5),
    line(type="stop", seq=True),                                        # bool is not an int
    line(type="stop", seq=1.5),
    line(type="stop", seq="7"),
    line(type="set_goal", seq=1, x=1.0, state_seq=1),                   # missing y
    line(type="set_goal", seq=1, x="1", y=2.0, state_seq=1),            # string coordinate
    line(type="set_goal", seq=1, x=1.0, y=2.0),                         # missing state_seq
    line(type="set_goal", seq=1, x=1.0, y=2.0, state_seq=1, speed=9),   # unexpected field
    b'{"v": 1, "type": "set_goal", "seq": 1, "x": NaN, "y": 0, "state_seq": 1}',
    b'{"v": 1, "type": "set_goal", "seq": 1, "x": Infinity, "y": 0, "state_seq": 1}',
    line(type="hello", session=""),
    line(type="hello", session="x" * 65),
    line(type="hello"),
    b"{" + b" " * MAX_COMMAND_BYTES + b"}",                             # oversized
])
def test_malformed_commands_are_rejected(raw: bytes) -> None:
    with pytest.raises(ProtocolError):
        decode_command(raw)


def test_encoder_refuses_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        encode({"type": "state", "x": float("nan")})


def test_every_message_is_one_line() -> None:
    raw = encode({"type": "state", "text": "multi\nline"})
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert decode_message(raw.rstrip(b"\n"))["text"] == "multi\nline"


def test_grid_cells_round_trip_exactly() -> None:
    rng = np.random.default_rng(0)
    cells = rng.choice(np.array([-1, 0, 100], dtype=np.int8), size=(240, 280))
    out = decode_cells(encode_cells(cells), 240, 280)
    assert out.dtype == np.int8 and np.array_equal(out, cells)
    with pytest.raises(ProtocolError):
        decode_cells(encode_cells(cells), 100, 100)
