"""Pi -> motor microcontroller command protocol, and the microcontroller's
watchdog logic (reference implementation).

Architecture:

    Python navigation/control (Pi)  --motor frames, USB serial-->
    microcontroller (watchdog)      --PWM-->  motor driver  -->  motors

The microcontroller stops the motors by itself when it has not received a
valid, fresh command for ``timeout_s`` (0.3 s). It needs nothing from the Pi
to do that, so a frozen, crashed or disconnected Pi / Python process cannot
keep the motors running. ``MotorControllerLogic`` below is the exact logic the
firmware must implement (it is written to port line-for-line to C: integer
milliseconds, no allocation); ``hardware.fake_motor_controller`` runs it as a
separate process for tests until the real controller exists.

Wire format -- ASCII lines, ``\\n``-terminated, at most 64 bytes:

    Pi -> MCU   M,<session>,<seq>,<host_ms>,<left>,<right>*<crc>   motion
                S,<session>,<seq>,<host_ms>*<crc>                   STOP
    MCU -> Pi   T,<state>,<session>,<last_seq>,<trips>*<crc>        status (~10 Hz)

    session   4 hex digits, random per Pi-side process start
    seq       decimal uint32, +1 per frame (wraps at 2**32)
    host_ms   decimal uint32, the Pi's monotonic clock in ms (wraps)
    left/right  signed per-mille duty, -1000..1000 (already shaped on the Pi:
              limits, deadband, trim, inversion)
    crc       2 hex digits: CRC-8 (poly 0x07, init 0x00) of every byte before '*'

A frame REFRESHES the watchdog only if it is ACCEPTED: well-formed, CRC OK,
values in range, same session, seq newer than the last accepted one, and not
delivered late (see ``max_age_s``). Rejected frames never refresh it.

States: DISARMED (power-up) / ARMED / TRIPPED (watchdog) / STOPPED (STOP frame).
Motors can run only in ARMED. Leaving any other state needs an accepted
motion frame with left = right = 0 ("arm"): after a trip or STOP the Pi has to
command zero before anything moves again -- a Python process that unfreezes
can never resume driving on its own. A STOP frame with a valid CRC is obeyed
immediately in every state (stopping is always safe, so it is not subject to
the seq/age checks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

DISARMED, ARMED, TRIPPED, STOPPED = "DISARMED", "ARMED", "TRIPPED", "STOPPED"
MAX_DUTY = 1000
MAX_FRAME_LEN = 64
U32 = 1 << 32


def crc8(data: bytes) -> int:
    """CRC-8, polynomial 0x07, init 0x00, no reflection (CRC-8/SMBUS)."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def _frame(body: str) -> bytes:
    raw = body.encode("ascii")
    return raw + b"*" + f"{crc8(raw):02X}".encode("ascii") + b"\n"


def encode_motion(session: int, seq: int, host_ms: int, left: int, right: int) -> bytes:
    return _frame(f"M,{session:04X},{seq % U32},{host_ms % U32},{int(left)},{int(right)}")


def encode_stop(session: int, seq: int, host_ms: int) -> bytes:
    return _frame(f"S,{session:04X},{seq % U32},{host_ms % U32}")


def encode_status(state: str, session: int | None, last_seq: int | None, trips: int) -> bytes:
    return _frame(f"T,{state},{'----' if session is None else f'{session:04X}'},"
                  f"{'-' if last_seq is None else last_seq},{trips}")


@dataclass(frozen=True)
class Frame:
    kind: str          # "M" / "S" / "T"
    fields: tuple[str, ...]


def parse_frame(line: bytes) -> Frame | None:
    """A CRC-checked frame, or None (malformed / bad CRC)."""
    line = line.rstrip(b"\r\n")
    if not line or len(line) > MAX_FRAME_LEN or line.count(b"*") != 1:
        return None
    body, crc_text = line.split(b"*")
    try:
        if len(crc_text) != 2 or int(crc_text, 16) != crc8(body):
            return None
        parts = body.decode("ascii").split(",")
    except (ValueError, UnicodeDecodeError):
        return None
    return Frame(parts[0], tuple(parts[1:]))


def _u32(text: str) -> int | None:
    if not text.isdigit():
        return None
    value = int(text)
    return value if value < U32 else None


def _newer(seq: int, last: int) -> bool:
    """Serial-number arithmetic: ``seq`` is after ``last`` (mod 2**32)."""
    return 0 < (seq - last) % U32 < U32 // 2


@dataclass
class ControllerCounters:
    accepted: int = 0
    malformed: int = 0          # unparsable, bad CRC, wrong fields, out of range
    wrong_session: int = 0
    stale_seq: int = 0          # duplicate or older seq
    late: int = 0               # delivered more than max_age_s late
    not_armed: int = 0          # nonzero command outside ARMED
    stops: int = 0
    trips: int = 0


@dataclass
class MotorControllerLogic:
    """The microcontroller's command handling + watchdog. Pure logic: the
    caller passes the controller's own clock (``now_ms``, integer ms) to
    every call and applies ``duty`` (left, right per-mille) to the motor
    driver after each call."""

    timeout_ms: int = 300
    max_age_ms: int = 150        # later than the fastest delivery seen by more than this -> rejected
    drift_ppm: int = 1000        # allowed clock drift Pi vs MCU (the delivery reference relaxes this fast)
    state: str = DISARMED
    duty: tuple[int, int] = (0, 0)
    session: int | None = None
    last_seq: int | None = None
    last_valid_ms: int | None = None
    trip_ms: int | None = None                  # when the watchdog last fired
    counters: ControllerCounters = field(default_factory=ControllerCounters)
    _ref_offset: int | None = None              # max(host_ms - now_ms) seen = fastest delivery
    _ref_ms: int | None = None

    # --------------------------------------------------------------- input --

    def feed_line(self, line: bytes, now_ms: int) -> bool:
        """Handle one received line. True = accepted (watchdog refreshed)."""
        self.tick(now_ms)
        frame = parse_frame(line)
        if frame is None or frame.kind not in ("M", "S"):
            self.counters.malformed += 1
            return False
        if frame.kind == "S":
            if len(frame.fields) != 3:
                self.counters.malformed += 1
                return False
            self._off(STOPPED)
            self.counters.stops += 1
            return False    # obeyed, but a STOP never refreshes the watchdog
        if len(frame.fields) != 5:
            self.counters.malformed += 1
            return False
        try:
            session = int(frame.fields[0], 16)
            left, right = int(frame.fields[3]), int(frame.fields[4])
        except ValueError:
            self.counters.malformed += 1
            return False
        seq, host_ms = _u32(frame.fields[1]), _u32(frame.fields[2])
        if seq is None or host_ms is None or not (abs(left) <= MAX_DUTY and abs(right) <= MAX_DUTY):
            self.counters.malformed += 1
            return False

        new_session = session != self.session
        if new_session:
            if self.state == ARMED:
                self.counters.wrong_session += 1   # a second sender while armed: never mix streams
                return False
            if (left, right) != (0, 0):
                self.counters.not_armed += 1
                return False
            # A new Pi-side process: its first frame must be an arming (zero) frame.
            self.session, self.last_seq, self._ref_offset = session, None, None
        elif self.last_seq is not None and not _newer(seq, self.last_seq):
            self.counters.stale_seq += 1
            return False

        offset = self._signed(host_ms - now_ms)
        if self._ref_offset is not None:
            relaxed = self._ref_offset - (now_ms - self._ref_ms) * self.drift_ppm // 1_000_000
            if relaxed - offset > self.max_age_ms:
                self.counters.late += 1
                return False
            self._ref_offset = max(relaxed, offset)
        else:
            self._ref_offset = offset
        self._ref_ms = now_ms

        if self.state != ARMED and (left, right) != (0, 0):
            self.counters.not_armed += 1
            self.last_seq = seq                   # fresh and in order, just not allowed to move
            return False
        self.last_seq = seq
        self.last_valid_ms = now_ms
        self.counters.accepted += 1
        self.state = ARMED
        self.duty = (left, right)
        return True

    def tick(self, now_ms: int) -> None:
        """Watchdog: call often (every loop pass of the firmware)."""
        if self.state == ARMED and (self.last_valid_ms is None or now_ms - self.last_valid_ms > self.timeout_ms):
            self.trip_ms = now_ms
            self.counters.trips += 1
            self._off(TRIPPED)

    def status_line(self) -> bytes:
        return encode_status(self.state, self.session, self.last_seq, self.counters.trips)

    # ------------------------------------------------------------- helpers --

    def _off(self, state: str) -> None:
        self.duty = (0, 0)
        self.state = state

    @staticmethod
    def _signed(value: int) -> int:
        value %= U32
        return value - U32 if value >= U32 // 2 else value


def parse_status(line: bytes) -> dict | None:
    frame = parse_frame(line)
    if frame is None or frame.kind != "T" or len(frame.fields) != 4:
        return None
    state, session, last_seq, trips = frame.fields
    return {"state": state, "session": None if session == "----" else int(session, 16),
            "last_seq": None if last_seq == "-" else int(last_seq), "trips": int(trips)}
