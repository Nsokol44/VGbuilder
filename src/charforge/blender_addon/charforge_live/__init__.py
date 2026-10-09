"""charforge live: a window into a running `charforge watch` session.

This add-on does not import charforge. It reads and writes the small JSON files of the live protocol (see
charforge/live.py): it shows the latest build, offers the job's controls as sliders, writes your changes to
live_overrides.json, and sends the selected objects into the charforge inbox.

Whatever it imports into the scene is build output: the next build replaces it. Use "Keep this version" to detach
the current objects from the link before editing them by hand.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
bl_info = {
    "name": "charforge live",
    "author": "charforge",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar > charforge",
    "description": "Live view and sliders for charforge jobs; send the selection to charforge",
    "category": "Import-Export",
}

import json
import os
import re
import subprocess
import time
from pathlib import Path

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty,
                       StringProperty)
from bpy.types import Operator, Panel, PropertyGroup

PROTOCOL = 1
POLL_S = 0.5
WRITE_DEBOUNCE_S = 0.35
HEARTBEAT_STALE_S = 5.0
DATA_KINDS = ("meshes", "materials", "images", "armatures", "actions")

_state = {
    "mtime": None, "status": None, "loaded_build": 0, "overrides": {}, "seq": 0, "dirty_at": None,
    "syncing": False, "created": {}, "watcher": None, "waiting_for_job": None, "timer": False, "sig": None,
    "adopted": False,
}
_enum_cache: dict[str, list] = {}


# ------------------------------------------------------------------ files (same rules as charforge.live)
def _write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _session(settings):
    p = Path(bpy.path.abspath(settings.session_dir)) if settings.session_dir else None
    return p if p and p.is_dir() else None


def _flush_overrides(settings):
    sd = _session(settings)
    if sd is None:
        return
    _state["seq"] = max(_state["seq"], int((_read_json(sd / "live_overrides.json") or {}).get("seq", 0))) + 1
    _write_json_atomic(sd / "live_overrides.json", {"seq": _state["seq"], "overrides": dict(_state["overrides"])})
    _state["dirty_at"] = None
    settings.pending = False


# ------------------------------------------------------------------ controls
def _enum_items(self, _ctx):
    key = self.choices_json
    items = _enum_cache.get(key)
    if items is None:
        try:
            items = [(c, c, "") for c in json.loads(key)]
        except ValueError:
            items = []
        _enum_cache[key] = items or [("-", "-", "")]
    return items


def _value_of(c):
    if c.kind == "bool":
        return bool(c.b)
    if c.kind == "choice":
        return c.e
    return int(round(c.f)) if c.kind == "int" else float(c.f)


def _changed(self, _ctx):
    if _state["syncing"]:
        return
    _state["overrides"][self.path] = _value_of(self)
    _state["dirty_at"] = time.monotonic()
    bpy.context.window_manager.charforge.pending = True


def _get_t(self):
    span = self.vmax - self.vmin
    return 0.0 if span <= 0 else min(1.0, max(0.0, (self.f - self.vmin) / span))


def _set_t(self, t):
    v = self.vmin + t * (self.vmax - self.vmin)
    if self.kind == "int":
        v = round(v)
    elif self.step > 0:
        v = round(v / self.step) * self.step
    self.f = min(self.vmax, max(self.vmin, v))


class CFControl(PropertyGroup):
    path: StringProperty()
    label: StringProperty()
    group: StringProperty()
    kind: StringProperty()                       # float | int | choice | bool
    vmin: FloatProperty()
    vmax: FloatProperty(default=1.0)
    step: FloatProperty()
    choices_json: StringProperty(default="[]")
    f: FloatProperty(update=_changed)
    b: BoolProperty(update=_changed)
    e: EnumProperty(items=_enum_items, update=_changed)
    t: FloatProperty(name="", min=0.0, max=1.0, subtype="FACTOR", get=_get_t, set=_set_t)


def _signature(controls):
    return tuple((c.get("path"), c.get("type"), c.get("min"), c.get("max"), tuple(c.get("choices", ()))) for c in controls)


def _sync_controls(settings, controls):
    """Show the server's controls. A control you have moved keeps your value; the others follow the server."""
    _state["syncing"] = True
    try:
        sig = _signature(controls)
        if sig != _state["sig"] or len(settings.controls) != len(controls):
            settings.controls.clear()
            for c in controls:
                it = settings.controls.add()
                it.path, it.label, it.group, it.kind = c["path"], c.get("label", c["path"]), c.get("group", ""), c["type"]
                if c["type"] in ("float", "int"):
                    it.vmin, it.vmax, it.step = float(c["min"]), float(c["max"]), float(c.get("step", 0))
                if c["type"] == "choice":
                    it.choices_json = json.dumps(c["choices"])
            _state["sig"] = sig
        for it, c in zip(settings.controls, controls):
            v = _state["overrides"].get(it.path, c.get("value"))
            if it.kind == "bool":
                it.b = bool(v)
            elif it.kind == "choice":
                if v in [x[0] for x in _enum_items(it, None)]:
                    it.e = v
            else:
                it.f = float(v)
    finally:
        _state["syncing"] = False


# ------------------------------------------------------------------ importing builds
def _snapshot():
    return {k: set(getattr(bpy.data, k).keys()) for k in DATA_KINDS}


def _get_collection(name, parent):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if col.name not in parent.children:
        try:
            parent.children.link(col)
        except RuntimeError:
            pass
    return col


def _layer_collection(root, col):
    if root.collection == col:
        return root
    for ch in root.children:
        r = _layer_collection(ch, col)
        if r:
            return r
    return None


def _remove_created(name):
    created = _state["created"].pop(name, None)
    if not created:
        return
    for on in created["objects"]:
        o = bpy.data.objects.get(on)
        if o is not None:
            bpy.data.objects.remove(o, do_unlink=True)
    for kind, names in created["data"].items():
        coll = getattr(bpy.data, kind)
        for dn in names:
            d = coll.get(dn)
            if d is not None and d.users == 0:
                coll.remove(d)


def _load_image(name, path):
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.load(str(path))
        img.name = name
    else:
        img.filepath = str(path)
        img.reload()
    return img


def reload_build(settings, status):
    """Replace the imported objects of this job with the files of the build in `status`. Returns a message."""
    sd = _session(settings)
    name = status.get("name", "charforge")
    if bpy.context.mode != "OBJECT":
        return "paused: leave edit mode to load the new build"
    files = status.get("files", [])
    lods = sorted((f for f in files if f.get("role") == "lod"), key=lambda f: f.get("lod", 0))
    extras = [f for f in files if f.get("role") == "collision"] if settings.import_extras else []
    if not settings.import_extras:
        lods = lods[:1]
    glbs = [(f"LOD{f['lod']}", f) for f in lods] + [("collision", f) for f in extras]
    for f in files:
        if f.get("role") in ("contact_sheet", "effect") and str(f["file"]).endswith(".png"):
            try:
                _load_image(f"charforge:{name}:{Path(f['file']).name.split('_', 1)[-1]}", sd / f["file"])
            except RuntimeError:
                pass
    if not glbs:
        return "build has no glb files (an effect job: see the images in Blender's image browser)"

    scene = bpy.context.scene
    prev_active = bpy.context.view_layer.active_layer_collection
    prev_object = bpy.context.view_layer.objects.active
    prev_object_name = prev_object.name if prev_object is not None else None
    prev_selected = [o.name for o in bpy.context.selected_objects]
    top = _get_collection(f"charforge:{name}", scene.collection)
    root = bpy.data.objects.get(f"charforge:{name}")
    if root is None:
        root = bpy.data.objects.new(f"charforge:{name}", None)
        root.empty_display_type = "ARROWS"
    if root.name not in top.objects:
        top.objects.link(root)
    _remove_created(name)
    created = {"objects": [], "data": {k: [] for k in DATA_KINDS}}
    try:
        for label, f in glbs:
            col = _get_collection(f"charforge:{name}:{label}", top)
            lc = _layer_collection(bpy.context.view_layer.layer_collection, col)
            if lc is not None:
                bpy.context.view_layer.active_layer_collection = lc
            before_o, before_d = set(bpy.data.objects.keys()), _snapshot()
            bpy.ops.import_scene.gltf(filepath=str(sd / f["file"]))
            for on in set(bpy.data.objects.keys()) - before_o:
                o = bpy.data.objects[on]
                created["objects"].append(on)
                o.select_set(False)                    # the importer selects what it makes; keep the user's selection
                if o.parent is None:
                    o.parent = root
                for c in list(o.users_collection):
                    if c != col:
                        c.objects.unlink(o)
                if o.name not in col.objects:
                    col.objects.link(o)
            after = _snapshot()
            for k in DATA_KINDS:
                created["data"][k] += sorted(after[k] - before_d[k])
            col.hide_viewport = label != "LOD0"
            col.hide_render = label != "LOD0"
    finally:
        _state["created"][name] = created
        bpy.context.view_layer.active_layer_collection = prev_active
        bpy.context.view_layer.objects.active = bpy.data.objects.get(prev_object_name) if prev_object_name else None
        for on in prev_selected:                       # the importer clears the selection; give it back
            o = bpy.data.objects.get(on)
            if o is not None:
                o.select_set(True)
    return f"loaded build {status.get('build_id')} ({len(glbs)} file(s))"


def keep_current(name):
    """Detach the objects of this job from the link: they stay in the scene as ordinary objects; the next build
    imports fresh ones."""
    _state["created"].pop(name, None)
    root = bpy.data.objects.get(f"charforge:{name}")
    stamp = time.strftime("%H%M%S")
    if root is not None:
        root.name = f"{root.name} (kept {stamp})"
    for col in list(bpy.data.collections):
        if col.name == f"charforge:{name}" or col.name.startswith(f"charforge:{name}:"):
            col.name = f"{col.name} (kept {stamp})"


# ------------------------------------------------------------------ polling
def poll(settings=None):
    """One poll of the session folder. Returns a short string saying what happened (used by tests)."""
    wm = bpy.context.window_manager
    settings = settings or wm.charforge
    did = []
    if _state["waiting_for_job"]:
        found = _find_session(_state["waiting_for_job"])
        if found:
            settings.session_dir = str(found)
            _state["waiting_for_job"] = None
            did.append("found-session")
    sd = _session(settings)
    if sd is None:
        settings.live_state = "no session folder"
        return "no-session"
    if _state["dirty_at"] is not None and time.monotonic() - _state["dirty_at"] >= WRITE_DEBOUNCE_S:
        _flush_overrides(settings)
        did.append("wrote-overrides")
    hb = sd / "live.heartbeat"
    try:
        settings.watcher_alive = time.time() - hb.stat().st_mtime < HEARTBEAT_STALE_S
    except OSError:
        settings.watcher_alive = False
    lj = sd / "live.json"
    try:
        mt = lj.stat().st_mtime_ns
    except OSError:
        settings.live_state = "waiting for live.json"
        return "no-status"
    if mt != _state["mtime"]:
        st = _read_json(lj)
        if st is None:
            return "unreadable"            # caught mid-write on a filesystem without atomic replace: next poll
        _state["mtime"], _state["status"] = mt, st
        if st.get("protocol") != PROTOCOL:
            settings.live_state, settings.message = "error", f"live.json protocol {st.get('protocol')} (this add-on speaks {PROTOCOL})"
            return "protocol-mismatch"
        settings.job_name = st.get("name", "")
        settings.live_state = st.get("state", "")
        settings.message = st.get("error") or st.get("message") or ""
        res = st.get("result", {}) or {}
        settings.build_status = res.get("status") or ""
        settings.failures_json = json.dumps([f.get("code", "?") if isinstance(f, dict) else str(f) for f in res.get("failures", [])])
        settings.build_id = int(st.get("build_id", 0))
        settings.fast = bool(st.get("fast"))
        if not _state["adopted"]:
            _state["overrides"] = dict(st.get("overrides") or {})    # overrides written elsewhere (CLI, an earlier session)
            _state["adopted"] = True
        if st.get("controls"):
            _sync_controls(settings, st["controls"])
        did.append("status")
    st = _state["status"]
    if (st and st.get("state") == "idle" and settings.auto_reload and int(st.get("build_id", 0)) > _state["loaded_build"]
            and st.get("files")):
        try:
            msg = reload_build(settings, st)
        except Exception as e:  # noqa: BLE001 - show it, keep polling
            settings.message = f"import failed: {type(e).__name__}: {e}"
            _state["loaded_build"] = int(st.get("build_id", 0))
            return "import-failed"
        if not msg.startswith("paused"):
            _state["loaded_build"] = int(st.get("build_id", 0))
            settings.loaded_build = _state["loaded_build"]
        settings.message = msg
        did.append("reloaded")
    return ",".join(did) or "idle"


def _tick():
    try:
        poll()
    except Exception as e:  # noqa: BLE001 - never kill the timer
        try:
            bpy.context.window_manager.charforge.message = f"poll error: {type(e).__name__}: {e}"
        except Exception:  # noqa: BLE001
            pass
    if not bpy.context.window_manager.charforge.connected:
        _state["timer"] = False
        return None
    return POLL_S


def _find_session(job_path):
    """A session folder under the output dir whose live.json belongs to this job file."""
    out = _state.get("waiting_out")
    if not out:
        return None
    for lj in Path(out, "_live").glob("*/live.json"):
        st = _read_json(lj)
        if st and Path(st.get("job", "")).resolve() == Path(job_path).resolve():
            return lj.parent
    return None


def _job_name(path):
    try:
        m = re.search(r"^name:\s*([A-Za-z0-9_-]+)", Path(path).read_text(), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


# ------------------------------------------------------------------ settings, operators, panel
class CFSettings(PropertyGroup):
    session_dir: StringProperty(name="Session", subtype="DIR_PATH", description="out/_live/<job name> of a running `charforge watch`")
    job_file: StringProperty(name="Job", subtype="FILE_PATH", description="A charforge job .yaml")
    out_dir: StringProperty(name="Output", subtype="DIR_PATH", default="//out")
    charforge_cmd: StringProperty(name="charforge", default="charforge", description="The charforge program (full path if it is not on Blender's PATH)")
    connected: BoolProperty()
    auto_reload: BoolProperty(name="Auto reload", default=True)
    import_extras: BoolProperty(name="All LODs + collision", default=False, description="Also import the other LODs and the collision proxy (hidden)")
    controls: CollectionProperty(type=CFControl)
    live_state: StringProperty()
    message: StringProperty()
    build_status: StringProperty()
    failures_json: StringProperty(default="[]")
    build_id: IntProperty()
    loaded_build: IntProperty()
    job_name: StringProperty()
    pending: BoolProperty()
    fast: BoolProperty()
    watcher_alive: BoolProperty()
    send_name: StringProperty(name="Name", default="from_blender")


class CHARFORGE_OT_connect(Operator):
    bl_idname = "charforge.connect"
    bl_label = "Connect"

    def execute(self, context):
        s = context.window_manager.charforge
        if _session(s) is None:
            self.report({"ERROR"}, "Session folder does not exist. Start `charforge watch` first, or use Start watcher.")
            return {"CANCELLED"}
        s.connected = True
        _state.update(mtime=None, loaded_build=0, sig=None, adopted=False)
        poll(s)
        if not _state["timer"]:
            bpy.app.timers.register(_tick, first_interval=POLL_S)
            _state["timer"] = True
        return {"FINISHED"}


class CHARFORGE_OT_disconnect(Operator):
    bl_idname = "charforge.disconnect"
    bl_label = "Disconnect"

    def execute(self, context):
        context.window_manager.charforge.connected = False
        return {"FINISHED"}


class CHARFORGE_OT_start_watcher(Operator):
    bl_idname = "charforge.start_watcher"
    bl_label = "Start watcher"
    bl_description = "Run `charforge watch` for the job file, then connect to it"

    def execute(self, context):
        s = context.window_manager.charforge
        job = Path(bpy.path.abspath(s.job_file))
        out = Path(bpy.path.abspath(s.out_dir))
        if not job.is_file():
            self.report({"ERROR"}, f"job file not found: {job}")
            return {"CANCELLED"}
        out.mkdir(parents=True, exist_ok=True)
        log = open(out / "charforge_watch.log", "ab")
        try:
            _state["watcher"] = subprocess.Popen([s.charforge_cmd, "watch", str(job), "-o", str(out)], stdout=log, stderr=log)
        except OSError as e:
            self.report({"ERROR"}, f"could not start {s.charforge_cmd!r}: {e}")
            return {"CANCELLED"}
        name = _job_name(job)
        _state["waiting_out"] = str(out)
        guess = out / "_live" / name if name else None
        if guess is not None:
            s.session_dir = str(guess)
            guess.mkdir(parents=True, exist_ok=True)
        else:
            _state["waiting_for_job"] = str(job)
        s.connected = True
        _state.update(mtime=None, loaded_build=0, sig=None, adopted=False)
        if not _state["timer"]:
            bpy.app.timers.register(_tick, first_interval=POLL_S)
            _state["timer"] = True
        return {"FINISHED"}


class CHARFORGE_OT_stop_watcher(Operator):
    bl_idname = "charforge.stop_watcher"
    bl_label = "Stop watcher"

    def execute(self, context):
        w = _state.get("watcher")
        if w is not None and w.poll() is None:
            w.terminate()
        _state["watcher"] = None
        return {"FINISHED"}


class CHARFORGE_OT_reload(Operator):
    bl_idname = "charforge.reload"
    bl_label = "Reload now"

    def execute(self, context):
        s = context.window_manager.charforge
        st = _state["status"]
        if not st or not st.get("files"):
            self.report({"WARNING"}, "no build to load yet")
            return {"CANCELLED"}
        self.report({"INFO"}, reload_build(s, st))
        _state["loaded_build"] = int(st.get("build_id", 0))
        s.loaded_build = _state["loaded_build"]
        return {"FINISHED"}


class CHARFORGE_OT_keep(Operator):
    bl_idname = "charforge.keep"
    bl_label = "Keep this version"
    bl_description = "Detach the current objects from the live link so you can edit them by hand; the next build imports fresh ones"

    def execute(self, context):
        keep_current(context.window_manager.charforge.job_name)
        return {"FINISHED"}


class CHARFORGE_OT_reset_control(Operator):
    bl_idname = "charforge.reset_control"
    bl_label = "Reset"
    path: StringProperty()

    def execute(self, context):
        _state["overrides"].pop(self.path, None)
        _flush_overrides(context.window_manager.charforge)
        return {"FINISHED"}


class CHARFORGE_OT_reset_all(Operator):
    bl_idname = "charforge.reset_all"
    bl_label = "Reset all"

    def execute(self, context):
        _state["overrides"].clear()
        _flush_overrides(context.window_manager.charforge)
        return {"FINISHED"}


def overrides_as_yaml(overrides):
    """Nested mapping as JSON, which is also valid YAML (for pasting into a job file)."""
    tree = {}
    for path, v in sorted(overrides.items()):
        cur = tree
        parts = path.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return json.dumps(tree, indent=2)


class CHARFORGE_OT_copy_overrides(Operator):
    bl_idname = "charforge.copy_overrides"
    bl_label = "Copy overrides"
    bl_description = "Copy the current overrides to the clipboard (or run `charforge overrides apply JOB --write`)"

    def execute(self, context):
        context.window_manager.clipboard = overrides_as_yaml(_state["overrides"])
        return {"FINISHED"}


def _selection_height(objs):
    zs = []
    for o in objs:
        if o.type == "MESH":
            zs += [(o.matrix_world @ v.co).z for v in o.data.vertices]
    return (max(zs) - min(zs)) if zs else 0.0


class CHARFORGE_OT_send_selection(Operator):
    bl_idname = "charforge.send_selection"
    bl_label = "Send selection to charforge"
    bl_description = "Export the selected objects as GLB into the session inbox, with a job file that imports them"

    def execute(self, context):
        s = context.window_manager.charforge
        sd = _session(s)
        if sd is None:
            self.report({"ERROR"}, "connect to a session first")
            return {"CANCELLED"}
        sel = list(context.selected_objects)
        if not sel:
            self.report({"ERROR"}, "nothing selected")
            return {"CANCELLED"}
        name = re.sub(r"[^A-Za-z0-9_-]", "_", s.send_name) or "from_blender"
        height = _selection_height(sel)
        if height <= 0.05:
            self.report({"ERROR"}, "the selected meshes have no height; select the mesh objects")
            return {"CANCELLED"}
        inbox = sd / "inbox"
        inbox.mkdir(exist_ok=True)
        tmp = inbox / f".{name}.tmp.glb"
        bpy.ops.export_scene.gltf(filepath=str(tmp), export_format="GLB", use_selection=True)
        os.replace(tmp, inbox / f"{name}.glb")
        job = (f"name: {name}\nprovider: import_glb\nprovider_args:\n  glb: {name}.glb\nstyle: stylized\n"
               f"height_m: {round(height, 3)}\ntags: [from-blender]\n")
        tmpy = inbox / f".{name}.tmp"
        tmpy.write_text(job)
        os.replace(tmpy, inbox / f"{name}.yaml")           # the job file lands last: the queue only looks at *.yaml
        self.report({"INFO"}, f"sent {name}; run: charforge queue work --inbox {inbox} --watch")
        return {"FINISHED"}


class CHARFORGE_PT_panel(Panel):
    bl_label = "charforge"
    bl_idname = "CHARFORGE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "charforge"

    def draw(self, context):
        s = context.window_manager.charforge
        lay = self.layout
        box = lay.box()
        if not s.connected:
            box.prop(s, "job_file")
            box.prop(s, "out_dir")
            box.prop(s, "charforge_cmd")
            box.operator("charforge.start_watcher", icon="PLAY")
            box.separator()
            box.prop(s, "session_dir")
            box.operator("charforge.connect", icon="LINKED")
            return
        row = box.row()
        row.label(text=f"{s.job_name or '...'}: {s.live_state}", icon="TIME" if s.live_state == "building" else "CHECKMARK")
        row.operator("charforge.disconnect", text="", icon="X")
        if not s.watcher_alive:
            box.label(text="watcher is not running (no heartbeat)", icon="ERROR")
        if s.build_status:
            box.label(text=f"build {s.build_id}: {s.build_status}" + ("  (not yet loaded)" if s.build_id != s.loaded_build else ""))
        for code in json.loads(s.failures_json or "[]"):
            box.label(text=code, icon="ERROR")
        if s.fast:
            box.label(text="fast preview builds: not the real output", icon="INFO")
        if s.message:
            box.label(text=s.message[:120])
        if s.pending:
            box.label(text="sending changes...", icon="TIME")
        row = box.row(align=True)
        row.prop(s, "auto_reload")
        row.operator("charforge.reload", text="Reload now")
        box.prop(s, "import_extras")
        box.operator("charforge.keep", icon="PINNED")
        if _state.get("watcher") is not None:
            box.operator("charforge.stop_watcher", icon="PAUSE")

        group = None
        col = None
        for c in s.controls:
            if c.group != group:
                group = c.group
                col = lay.box().column(align=True)
                col.label(text=group)
            row = col.row(align=True)
            marker = "*" if c.path in _state["overrides"] else " "
            if c.kind in ("float", "int"):
                val = f"{int(round(c.f))}" if c.kind == "int" else f"{c.f:.3g}"
                row.prop(c, "t", text=f"{marker}{c.label}: {val}", slider=True)
            elif c.kind == "bool":
                row.prop(c, "b", text=f"{marker}{c.label}")
            else:
                row.prop(c, "e", text=f"{marker}{c.label}")
            if c.path in _state["overrides"]:
                op = row.operator("charforge.reset_control", text="", icon="LOOP_BACK")
                op.path = c.path
        row = lay.row(align=True)
        row.operator("charforge.reset_all")
        row.operator("charforge.copy_overrides", icon="COPYDOWN")

        box = lay.box()
        box.label(text="Send to charforge")
        box.prop(s, "send_name")
        box.operator("charforge.send_selection", icon="EXPORT")


CLASSES = (CFControl, CFSettings, CHARFORGE_OT_connect, CHARFORGE_OT_disconnect, CHARFORGE_OT_start_watcher,
           CHARFORGE_OT_stop_watcher, CHARFORGE_OT_reload, CHARFORGE_OT_keep, CHARFORGE_OT_reset_control,
           CHARFORGE_OT_reset_all, CHARFORGE_OT_copy_overrides, CHARFORGE_OT_send_selection, CHARFORGE_PT_panel)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.WindowManager.charforge = PointerProperty(type=CFSettings)


def unregister():
    w = _state.get("watcher")
    if w is not None and w.poll() is None:
        w.terminate()
    del bpy.types.WindowManager.charforge
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
