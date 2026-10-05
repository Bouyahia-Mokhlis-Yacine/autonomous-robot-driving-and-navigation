"""Centralized logging setup.

Every subsystem should call ``get_logger(__name__)`` rather than configuring
its own handlers, so navigation-state transitions, replanning events, and
safety-critical decisions all land in one consistent, filterable stream.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root ``autonomous_navigation`` logger once.

    Safe to call multiple times; only the first call takes effect.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("autonomous_navigation")
    root.setLevel(level.upper())

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``autonomous_navigation``.

    Configures default logging on first use so modules can call this
    without every entry point remembering to call ``configure_logging``.
    """
    if not _CONFIGURED:
        configure_logging()

    if name == "autonomous_navigation" or name.startswith("autonomous_navigation."):
        return logging.getLogger(name)
    return logging.getLogger(f"autonomous_navigation.{name}")
