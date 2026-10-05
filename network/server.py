"""Robot-side TCP server: operator connections, command admission, state fan-out.

Threads (none of them ever blocks the robot's control loop or SLAM):
    accept thread            one, accepts operator connections
    per connection, reader   parses + validates commands; admits them to the
                             runtime's command sink (a queue the CONTROL LOOP
                             drains -- commands are never executed here),
                             except STOP, which also latches immediately via
                             ``stop_now`` so it does not wait for a tick
    encoder thread           one; JSON-encodes / compresses the newest published
                             state and map (the control loop only hands over a
                             dict -- encoding and compression cost is off its thread)
    per connection, writer   sends acks, then the newest map, then the newest
                             state; latest-only slots, so a slow/stalled
                             operator only ever misses intermediate updates.
                             A send that blocks longer than ``send_timeout_s``
                             drops that connection.

Command admission (anything failing is answered with a rejecting ack/error and
never reaches the runtime):
    - must parse and validate (``network.protocol.decode_command``);
    - ``hello`` first; ``seq`` strictly increasing per SESSION, across
      reconnects (replays / duplicates / reordering rejected);
    - ``set_goal`` / ``resume`` must reference a ``state_seq`` this server
      published at most ``max_command_state_age_s`` ago (its own clock).
"""

from __future__ import annotations

import collections
import itertools
import logging
import select
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable

from network.protocol import Command, ProtocolError, decode_command, encode, encode_cells

logger = logging.getLogger("autonomous_navigation.network.server")

MAX_PROTOCOL_ERRORS = 20      # malformed lines tolerated per connection before it is dropped
STATE_HISTORY = 512           # published state seqs remembered for staleness checks


@dataclass(frozen=True)
class AdmittedCommand:
    command: Command
    connection_id: int
    session: str
    received_at: float        # server monotonic time


class _Connection:
    def __init__(self, conn_id: int, sock: socket.socket, address) -> None:
        self.id = conn_id
        self.sock = sock
        self.address = address
        self.session: str | None = None
        self.client = ""
        self.last_rx = time.monotonic()
        self.alive = True
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.outbox: collections.deque[bytes] = collections.deque()   # acks / welcome / errors, in order
        self.state_line: bytes | None = None
        self.map_line: bytes | None = None


class RobotServer:
    def __init__(
        self,
        command_sink: Callable[[AdmittedCommand], None],
        stop_now: Callable[[], None],
        host: str = "0.0.0.0",
        port: int = 8765,
        operator_timeout_s: float = 1.5,
        max_command_state_age_s: float = 2.0,
        send_timeout_s: float = 1.0,
        welcome: dict | None = None,
    ) -> None:
        self._sink = command_sink
        self._stop_now = stop_now
        self._host, self._port = host, port
        self.operator_timeout_s = operator_timeout_s
        self.max_command_state_age_s = max_command_state_age_s
        self.send_timeout_s = send_timeout_s
        self.welcome = dict(welcome or {})

        self._lock = threading.Lock()
        self._connections: dict[int, _Connection] = {}
        self._ids = itertools.count(1)
        self._session_last_seq: dict[str, int] = {}
        self._state_seq = 0
        self._state_published: collections.OrderedDict[int, float] = collections.OrderedDict()
        self._latest_state_line: bytes | None = None
        self._latest_map_line: bytes | None = None
        self._listener: socket.socket | None = None
        self._closed = threading.Event()
        self._pending_state: tuple[int, dict] | None = None
        self._pending_map: dict | None = None
        self._encode_wake = threading.Event()
        self.rejected_count = 0

    # ------------------------------------------------------------ lifecycle --

    def start(self) -> int:
        """Bind and start accepting. Returns the actual port (``port=0`` -> ephemeral)."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._host, self._port))
        listener.listen(4)
        listener.settimeout(0.5)
        self._listener = listener
        self._port = listener.getsockname()[1]
        threading.Thread(target=self._accept_loop, name="net-accept", daemon=True).start()
        threading.Thread(target=self._encoder_loop, name="net-encode", daemon=True).start()
        logger.info("Operator server listening on %s:%d", self._host, self._port)
        return self._port

    @property
    def port(self) -> int:
        return self._port

    def close(self) -> None:
        self._closed.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        with self._lock:
            conns = list(self._connections.values())
        for conn in conns:
            self._drop(conn, "server closing")

    # ------------------------------------------------------------ publishing --

    def publish_state(self, state: dict) -> int:
        """Hand over the newest state (plain JSON-able dict). O(1) for the
        caller: encoding and sending happen on the server's own threads, and
        an unsent older state is simply replaced (latest-only)."""
        now = time.monotonic()
        with self._lock:
            self._state_seq += 1
            seq = self._state_seq
            self._state_published[seq] = now
            while len(self._state_published) > STATE_HISTORY:
                self._state_published.popitem(last=False)
            self._pending_state = (seq, state)
        self._encode_wake.set()
        return seq

    def publish_map(self, grid_message: dict) -> None:
        """Hand over the newest map: metadata plus ``cells`` / ``planning_cells``
        as int8 numpy arrays (already copied by the caller). Compressed on the
        encoder thread; latest-only."""
        with self._lock:
            self._pending_map = grid_message
        self._encode_wake.set()

    def _encoder_loop(self) -> None:
        while not self._closed.is_set():
            if not self._encode_wake.wait(timeout=0.5):
                continue
            self._encode_wake.clear()
            with self._lock:
                pending_state, self._pending_state = self._pending_state, None
                pending_map, self._pending_map = self._pending_map, None
            try:
                if pending_map is not None:
                    msg = {k: v for k, v in pending_map.items() if k not in ("cells", "planning_cells")}
                    msg["cells"] = encode_cells(pending_map["cells"])
                    msg["planning_cells"] = encode_cells(pending_map["planning_cells"])
                    self._fan_out(map_line=encode({"type": "map", **msg}))
                if pending_state is not None:
                    seq, state = pending_state
                    self._fan_out(state_line=encode({"type": "state", "seq": seq, **state}))
            except (TypeError, ValueError):
                logger.exception("Could not encode a published message; skipped")

    def _fan_out(self, state_line: bytes | None = None, map_line: bytes | None = None) -> None:
        with self._lock:
            if state_line is not None:
                self._latest_state_line = state_line
            if map_line is not None:
                self._latest_map_line = map_line
            conns = list(self._connections.values())
        for conn in conns:
            if conn.session is not None:
                if state_line is not None:
                    conn.state_line = state_line
                if map_line is not None:
                    conn.map_line = map_line
                conn.wake.set()

    def send_ack(self, connection_id: int, seq: int | None, accepted: bool, reason: str = "") -> None:
        with self._lock:
            conn = self._connections.get(connection_id)
        if conn is not None:
            self._queue(conn, encode({"type": "ack", "seq": seq, "accepted": accepted, "reason": reason}))

    # --------------------------------------------------------------- queries --

    def operators_present(self, now: float | None = None) -> int:
        """Connections that completed hello and sent anything within the timeout."""
        now = time.monotonic() if now is None else now
        with self._lock:
            return sum(1 for c in self._connections.values()
                       if c.alive and c.session is not None and now - c.last_rx <= self.operator_timeout_s)

    def is_connected(self, connection_id: int) -> bool:
        with self._lock:
            conn = self._connections.get(connection_id)
        return conn is not None and conn.alive

    def connection_count(self) -> int:
        with self._lock:
            return len(self._connections)

    # --------------------------------------------------------------- threads --

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                sock, address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.send_timeout_s)
            conn = _Connection(next(self._ids), sock, address)
            with self._lock:
                self._connections[conn.id] = conn
            logger.info("Operator connection #%d from %s", conn.id, address)
            threading.Thread(target=self._reader, args=(conn,), name=f"net-rx-{conn.id}", daemon=True).start()
            threading.Thread(target=self._writer, args=(conn,), name=f"net-tx-{conn.id}", daemon=True).start()

    def _reader(self, conn: _Connection) -> None:
        buffer = b""
        errors = 0
        try:
            while conn.alive and not self._closed.is_set():
                # select, not a socket timeout: the socket's own timeout is the
                # writer's send timeout and must not be changed from here.
                readable, _, _ = select.select([conn.sock], [], [], 0.5)
                if not readable:
                    continue
                chunk = conn.sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        errors += self._handle_line(conn, line)
                if len(buffer) > 8192 or errors > MAX_PROTOCOL_ERRORS:
                    self._queue(conn, encode({"type": "error", "reason": "too many malformed messages"}))
                    break
        except OSError:
            pass
        finally:
            self._drop(conn, "connection closed")

    def _handle_line(self, conn: _Connection, line: bytes) -> int:
        """Returns 1 if the line was rejected as malformed, else 0."""
        now = time.monotonic()
        try:
            cmd = decode_command(line)
        except ProtocolError as err:
            self.rejected_count += 1
            self._queue(conn, encode({"type": "error", "reason": f"malformed: {err}"}))
            return 1
        conn.last_rx = now

        if cmd.type == "hello":
            if conn.session is not None:
                self._queue(conn, encode({"type": "error", "reason": "duplicate hello"}))
                return 1
            conn.session, conn.client = cmd.session, cmd.client or ""
            with self._lock:
                last = self._session_last_seq.get(cmd.session, 0)
                state_line, map_line = self._latest_state_line, self._latest_map_line
            self._queue(conn, encode({"type": "welcome", **self.welcome, "session_last_seq": last}))
            conn.map_line, conn.state_line = map_line, state_line   # full resync on (re)connect
            conn.wake.set()
            logger.info("Operator #%d hello: session=%s client=%s", conn.id, cmd.session, cmd.client)
            return 0

        if conn.session is None:
            self._queue(conn, encode({"type": "error", "reason": "hello required first"}))
            return 1

        with self._lock:
            last = self._session_last_seq.get(conn.session, 0)
            if cmd.seq <= last:
                duplicate = True
            else:
                duplicate = False
                self._session_last_seq[conn.session] = cmd.seq
        if duplicate:
            self.rejected_count += 1
            self._queue(conn, encode({"type": "ack", "seq": cmd.seq, "accepted": False,
                                      "reason": f"stale seq (last accepted {last})"}))
            return 0

        if cmd.type == "heartbeat":
            return 0

        if cmd.type in ("set_goal", "resume"):
            with self._lock:
                published = self._state_published.get(cmd.state_seq)
            if published is None or now - published > self.max_command_state_age_s:
                self.rejected_count += 1
                self._queue(conn, encode({"type": "ack", "seq": cmd.seq, "accepted": False,
                                          "reason": "stale: command refers to state older than "
                                                    f"{self.max_command_state_age_s:.1f} s"}))
                return 0

        if cmd.type == "stop":
            self._stop_now()  # latch immediately, before the control loop's next tick

        self._sink(AdmittedCommand(command=cmd, connection_id=conn.id, session=conn.session, received_at=now))
        return 0

    def _writer(self, conn: _Connection) -> None:
        try:
            while conn.alive and not self._closed.is_set():
                if not conn.wake.wait(timeout=0.5):
                    continue
                conn.wake.clear()
                while True:
                    with conn.lock:
                        line = conn.outbox.popleft() if conn.outbox else None
                    if line is None:
                        break
                    conn.sock.sendall(line)
                map_line, conn.map_line = conn.map_line, None
                if map_line is not None:
                    conn.sock.sendall(map_line)
                state_line, conn.state_line = conn.state_line, None
                if state_line is not None:
                    conn.sock.sendall(state_line)
        except OSError as err:  # includes socket.timeout: a stalled operator is dropped, never waited on
            logger.warning("Operator #%d send failed (%s); dropping connection", conn.id, err.__class__.__name__)
        finally:
            self._drop(conn, "send failed")

    # --------------------------------------------------------------- helpers --

    def _queue(self, conn: _Connection, line: bytes) -> None:
        with conn.lock:
            conn.outbox.append(line)
        conn.wake.set()

    def _drop(self, conn: _Connection, why: str) -> None:
        with self._lock:
            was_alive = conn.alive
            conn.alive = False
            self._connections.pop(conn.id, None)
        conn.wake.set()
        if was_alive:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.sock.close()
            logger.info("Operator #%d disconnected (%s)", conn.id, why)
