"""Run by `vgbuilder blender`:  blender -P bootstrap.py -- --session <folder>

Loads the add-on straight from this folder (nothing is installed in Blender's preferences) and connects it to the
session folder once the interface is up."""
import sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
session = argv[argv.index("--session") + 1] if "--session" in argv else ""

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vgbuilder_live  # noqa: E402

try:
    vgbuilder_live.register()
except ValueError:      # already registered (enabled in preferences too)
    pass


def _connect():
    if session:
        bpy.context.window_manager.vgbuilder.session_dir = session
        bpy.ops.vgbuilder.connect()
    return None


bpy.app.timers.register(_connect, first_interval=1.0)
