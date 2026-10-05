"""Leave the motor pins DRIVEN LOW after a process lets go of them.

gpiozero resets every pin it closes to a floating INPUT (in ``close()`` and
again at interpreter exit). The L298N inputs have no pull-downs, so a floating
IN pin can read HIGH and run a motor with the software stopped: seen on the
robot 2026-09-29, GPIO 26/20/21 read high after a normal shutdown (GPIO 26 =
front-right forward, GPIO 25 low -> that motor driven).

``hold_pins_low`` sets the pins to output-low with the Raspberry Pi's
``pinctrl`` tool, which writes the GPIO registers directly: the setting stays
after this process exits, until something reconfigures the pins (the next
runtime claims them as usual). Works without sudo for users in the gpio group.
Hardware pull-downs (10 kOhm, each IN pin to GND) remain the real backstop:
this does not cover the Pi booting or losing power.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Callable, Sequence

logger = logging.getLogger("autonomous_navigation.hardware.pin_hold")


def hold_pins_low(pins: Sequence[int], run: Callable = subprocess.run, which: Callable = shutil.which) -> bool:
    """Drive ``pins`` as outputs, LOW, persistently. True on success."""
    if not pins:
        return True
    tool = which("pinctrl")
    if tool is None:
        logger.error("pinctrl not found: motor pins %s are left FLOATING after release", list(pins))
        return False
    cmd = [tool, "set", ",".join(str(p) for p in pins), "op", "dl"]
    try:
        result = run(cmd, capture_output=True, text=True, timeout=5.0)
    except Exception:  # noqa: BLE001 -- report, never raise from a shutdown path
        logger.exception("pinctrl failed: motor pins %s may be left FLOATING", list(pins))
        return False
    if result.returncode != 0:
        logger.error("pinctrl exited %d (%s): motor pins %s may be left FLOATING",
                     result.returncode, (result.stderr or "").strip(), list(pins))
        return False
    logger.info("Motor pins %s held LOW (output) after release", list(pins))
    return True
