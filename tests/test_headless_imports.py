"""The robot runtime runs headless on the Pi, which has no PIL.ImageTk: the live
backend's import chain must never pull in tkinter (regression: LiveBackend
imported real_world_model_walkthrough, which imported visualization.gui, and
LiveBackend.__init__ itself imported visualization.gui)."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_BLOCK_TK = """
import builtins
_real_import = builtins.__import__
def _guard(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tkinter" or name.startswith("tkinter.") or (name == "PIL" and fromlist and "ImageTk" in fromlist):
        raise ImportError("headless: " + name + " is not available")
    return _real_import(name, globals, locals, fromlist, level)
builtins.__import__ = _guard
"""

_BLOCK_TK_AND_IMPORT = _BLOCK_TK + """
import robot_runtime
import runtime.backends
from real_world_model_walkthrough import LiveProcessor, NavigationSession, run_threads
from visualization.gui_state import LatestStateBox, build_visualization_state
"""


def test_live_backend_import_chain_needs_no_tkinter():
    result = subprocess.run([sys.executable, "-c", _BLOCK_TK_AND_IMPORT], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_live_runtime_builds_and_runs_without_tkinter():
    """Constructs and runs the full live runtime (fake LiDAR) with tkinter
    blocked -- catches GUI imports done lazily inside constructors too."""
    code = _BLOCK_TK + (
        "import sys, pytest\n"
        "sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', 'tests/test_live_backend_motors.py']))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
