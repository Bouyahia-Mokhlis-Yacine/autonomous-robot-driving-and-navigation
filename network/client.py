"""Operator-side client: connection management, latest robot state, commands.

    - Connects to the robot runtime and reconnects automatically (backoff).
    - One SESSION id for the client's lifetime; command ``seq`` keeps counting
      across reconnects, so the runtime can reject any replay.
    - Commands are NEVER queued while disconnected: ``send`` either writes the
      command now or returns None. Nothing is re-sent after a reconnect, so a
      reconnect can never produce a duplicate or stale goal.
    - Heartbeats every ``heartbeat_s`` keep the operator "present" on the robot.
    - Keeps only the newest ``state`` and ``map`` (decoded) plus recent acks.
"""

from __future__ import annotations

import collections
import logging
import socket
import threading
import time
import uuid

from network.protocol import ProtocolError, decode_cells, decode_message, encode

logger = logging.getLogger("autonomous_navigation.network.client")


class OperatorClient:
    def __init__(self, host: str, port: int, client_name: str = "operator-gui", heartbeat_s: float = 0.4,
                 connect_timeout_s: float = 2.0) -> None:
        self.host, self.port = host, port
        self.client_name = client_name
        self.heartbeat_s = heartbeat_s
        self.connect_timeout_s = connect_timeout_s
        self.session = uuid.uuid4().hex
        self._seq = 0
        self._send_lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._closed = threading.Event()
        self._connected = threading.Event()

        self.welcome: dict | None = None
        self.state: dict | None = None
        self.map: dict | None = None                 # metadata + decoded "cells" / "planning_cells" arrays
        self.acks: collections.deque[dict] = collections.deque(maxlen=50)
        self.errors: collections.deque[str] = collections.deque(maxlen=20)
        self.last_rx = 0.0
        self.connects = 0
        self.status = "connecting"

    # ------------------------------------------------------------ lifecycle --

    def start(self) -> "OperatorClient":
        threading.Thread(target=self._run, name="operator-client", daemon=True).start()
        threading.Thread(target=self._heartbeat, name="operator-heartbeat", daemon=True).start()
        return self

    def close(self) -> None:
        self._closed.set()
        self._disconnect()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def wait_for(self, predicate, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate(self):
                return True
            time.sleep(0.02)
        return predicate(self)

    # ------------------------------------------------------------- commands --

    def send(self, kind: str, **fields) -> int | None:
        """Send one command now. Returns its seq, or None if not connected
        (the command is dropped, never queued). ``set_goal``/``resume`` carry
        the newest ``state_seq`` seen, as the runtime requires."""
        if kind in ("set_goal", "resume"):
            if self.state is None:
                return None
            fields.setdefault("state_seq", self.state["seq"])
        with self._send_lock:
            if self._sock is None or not self._connected.is_set():
                return None
            self._seq += 1
            seq = self._seq
            try:
                self._sock.sendall(encode({"type": kind, "seq": seq, **fields}))
            except OSError:
                self._disconnect()
                return None
        return seq

    def set_goal(self, x: float, y: float) -> int | None:
        return self.send("set_goal", x=float(x), y=float(y))

    def cancel_goal(self) -> int | None:
        return self.send("cancel_goal")

    def stop(self) -> int | None:
        return self.send("stop")

    def resume(self) -> int | None:
        return self.send("resume")

    def ack_for(self, seq: int | None) -> dict | None:
        return next((a for a in reversed(self.acks) if a.get("seq") == seq), None)

    # --------------------------------------------------------------- threads --

    def _run(self) -> None:
        backoff = 0.2
        while not self._closed.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
            except OSError as err:
                self.status = f"cannot reach robot at {self.host}:{self.port} ({err.__class__.__name__})"
                self._closed.wait(backoff)
                backoff = min(2.0, backoff * 2)
                continue
            backoff = 0.2
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(2.0)  # bounds how long a GUI button press can block in sendall
            with self._send_lock:
                self._sock = sock
                try:
                    sock.sendall(encode({"type": "hello", "session": self.session, "client": self.client_name}))
                except OSError:
                    self._sock = None
                    continue
            self.connects += 1
            self._connected.set()
            self.status = "connected"
            self._receive(sock)
            self._disconnect()
            if not self._closed.is_set():
                self.status = "disconnected -- reconnecting"
                self._closed.wait(backoff)

    def _receive(self, sock: socket.socket) -> None:
        buffer = b""
        while not self._closed.is_set():
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue  # quiet link; staleness is judged from last_rx, not here
            except OSError:
                return
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    self._handle(decode_message(line))
                except (ProtocolError, ValueError, KeyError) as err:
                    self.errors.append(f"bad message from robot: {err}")

    def _handle(self, msg: dict) -> None:
        self.last_rx = time.monotonic()
        kind = msg["type"]
        if kind == "state":
            self.state = msg
        elif kind == "map":
            h, w = msg["height"], msg["width"]
            decoded = {k: v for k, v in msg.items() if k not in ("cells", "planning_cells")}
            decoded["cells"] = decode_cells(msg["cells"], h, w)
            decoded["planning_cells"] = decode_cells(msg["planning_cells"], h, w)
            self.map = decoded
        elif kind == "ack":
            self.acks.append(msg)
        elif kind == "welcome":
            self.welcome = msg
        elif kind == "error":
            self.errors.append(msg.get("reason", "error"))

    def _heartbeat(self) -> None:
        while not self._closed.wait(self.heartbeat_s):
            if self._connected.is_set():
                self.send("heartbeat")

    def _disconnect(self) -> None:
        with self._send_lock:
            sock, self._sock = self._sock, None
            self._connected.clear()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

