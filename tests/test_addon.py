"""The Blender add-on, exercised through the bpy module (no GUI): registration, polling, import/replace, sliders,
override writing, send-selection, the watcher launcher. What cannot run here: the sidebar drawing, real timers,
mouse interaction. Those are only imported, never drawn."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

bpy = pytest.importorskip("bpy")

from charforge import live  # noqa: E402
from charforge.live import LiveSession  # noqa: E402
from charforge.orchestrator import Backend, build_one  # noqa: E402
from charforge.providers import inspect_glb  # noqa: E402
from charforge.spec import load_job  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ADDON_PARENT = ROOT / "src" / "charforge" / "blender_addon"
sys.path.insert(0, str(ADDON_PARENT))
import charforge_live as cf  # noqa: E402

pytestmark = pytest.mark.bpy


@pytest.fixture(scope="module", autouse=True)
def registered():
    cf.register()
    yield
    cf.unregister()


@pytest.fixture(autouse=True)
def clean():
    def wipe():
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        for kind in cf.DATA_KINDS + ("collections",):
            coll = getattr(bpy.data, kind)
            for d in list(coll):
                coll.remove(d)
    wipe()
    cf._state.update(mtime=None, status=None, loaded_build=0, overrides={}, seq=0, dirty_at=None, syncing=False,
                     created={}, waiting_for_job=None, sig=None, adopted=False)
    s = bpy.context.window_manager.charforge
    s.controls.clear()
    s.connected, s.session_dir, s.message, s.build_status, s.build_id, s.loaded_build = False, "", "", "", 0, 0
    s.import_extras, s.auto_reload = False, True
    yield
    wipe()


def job_copy(tmp_path, name):
    import shutil
    d = tmp_path / "jobs"
    d.mkdir(exist_ok=True)
    shutil.copy(ROOT / "jobs" / name, d / name)
    return d / name


@pytest.fixture(scope="module")
def hall_session(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("addon")
    job = job_copy(tmp, "struct_hall.yaml")
    sess = LiveSession(job, tmp / "out", Backend(), log=lambda s: None)
    sess.build_now()
    return sess


def settings():
    return bpy.context.window_manager.charforge


def connect(sess):
    settings().session_dir = str(sess.dir)
    assert bpy.ops.charforge.connect() == {"FINISHED"}


def tri_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def lod0_objects():
    col = bpy.data.collections.get("charforge:struct_hall:LOD0")
    return [o for o in col.objects if o.type == "MESH"] if col else []


def height(objs):
    zs = [(o.matrix_world @ v.co).z for o in objs for v in o.data.vertices]
    return max(zs) - min(zs)


def test_registered_and_settings_defaults():
    s = settings()
    assert s.auto_reload and not s.connected and s.charforge_cmd == "charforge"
    assert "charforge" in dir(bpy.ops) and hasattr(bpy.ops.charforge, "connect")


def test_connect_refuses_a_missing_session(tmp_path):
    settings().session_dir = str(tmp_path / "nope")
    with pytest.raises(RuntimeError, match="does not exist"):        # an ERROR report raises when called from Python
        bpy.ops.charforge.connect()
    assert not settings().connected


def test_connect_shows_status_builds_sliders_and_imports_lod0(hall_session):
    connect(hall_session)
    s, st = settings(), json.loads(hall_session.status_path.read_text())
    assert s.connected and s.job_name == "struct_hall" and s.build_status == "passed" and s.build_id == st["build_id"] == s.loaded_build
    assert len(s.controls) == len(st["controls"]) > 10
    assert {c.path for c in s.controls} == {c["path"] for c in st["controls"]}
    objs = lod0_objects()
    assert objs, "LOD0 was not imported"
    manifest = json.loads((hall_session.dir / "manifest.json").read_text())
    want = manifest["iterations"][-1]["metrics"]["lod_tris"][0]
    assert sum(tri_count(o) for o in objs) == want
    root = bpy.data.objects["charforge:struct_hall"]
    assert all(o.parent == root or o.parent in objs for o in objs)
    assert bpy.data.collections.get("charforge:struct_hall:LOD1") is None          # LOD0 only by default
    floors = {c.path: c for c in s.controls}["structure.floors"]
    assert floors.kind == "int" and int(round(floors.f)) == 2 and (floors.vmin, floors.vmax) == (1, 12)
    style = {c.path: c for c in s.controls}["structure.style"]
    assert style.kind == "choice" and style.e == "hall"


def test_all_lods_and_collision_are_imported_hidden_when_asked(hall_session):
    settings().import_extras = True
    connect(hall_session)
    for lab in ("LOD0", "LOD1", "LOD2", "collision"):
        col = bpy.data.collections.get(f"charforge:struct_hall:{lab}")
        assert col is not None and len(col.objects) > 0, lab
        assert col.hide_viewport == (lab != "LOD0"), lab


def test_slider_writes_overrides_debounced_then_new_build_replaces_the_objects(hall_session, tmp_path):
    connect(hall_session)
    s = settings()
    before_h = height(lod0_objects())
    before_names = {o.name for o in lod0_objects()}
    n_meshes = len(bpy.data.meshes)
    floors = {c.path: c for c in s.controls}["structure.floors"]
    floors.f = 3
    assert cf._state["overrides"] == {"structure.floors": 3} and s.pending
    assert live.read_overrides(hall_session.overrides_path)[1] == {}          # not yet: debounce
    cf._state["dirty_at"] -= 1.0
    assert "wrote-overrides" in cf.poll()
    seq, ov = live.read_overrides(hall_session.overrides_path)
    assert ov == {"structure.floors": 3} and seq >= 1 and not s.pending
    hall_session.build_now()                                                  # the watcher's job
    r = cf.poll()
    assert "status" in r and "reloaded" in r
    assert s.build_id == 2 == s.loaded_build
    after = lod0_objects()
    assert height(after) > before_h + 2.5                                     # one more storey
    assert len(bpy.data.meshes) == n_meshes                                   # old meshes were removed, not leaked
    assert not (before_names & {o.name for o in after}) or len(bpy.data.objects) < 2 * len(before_names) + 5
    assert len({c.path for c in s.controls if c.path in cf._state["overrides"]}) == 1
    assert int(round({c.path: c for c in s.controls}["structure.floors"].f)) == 3     # your value survived the refresh
    # restore for the module-scoped session
    cf._state["overrides"].clear()
    cf._flush_overrides(s)
    hall_session.build_now()
    cf.poll()
    assert height(lod0_objects()) == pytest.approx(before_h, abs=0.01)


def test_non_overridden_sliders_follow_the_server_and_placement_is_kept(hall_session):
    connect(hall_session)
    root = bpy.data.objects["charforge:struct_hall"]
    root.location = (10, 5, 0)
    root.rotation_euler[2] = 0.5
    live.write_overrides(hall_session.overrides_path, {"structure.roof": "hip"})
    hall_session.build_now()
    cf.poll()
    roof = {c.path: c for c in settings().controls}["structure.roof"]
    assert roof.e == "hip"
    root = bpy.data.objects["charforge:struct_hall"]
    assert tuple(root.location) == (10, 5, 0) and root.rotation_euler[2] == pytest.approx(0.5)   # your placement stays
    o = lod0_objects()[0]
    assert o.matrix_world.translation.x != o.location.x or o.parent is not None
    live.write_overrides(hall_session.overrides_path, {})
    hall_session.build_now()
    cf.poll()


def test_factor_slider_maps_to_the_range_and_rounds_ints(hall_session):
    connect(hall_session)
    by = {c.path: c for c in settings().controls}
    by["structure.floors"].t = 0.5
    assert by["structure.floors"].f in (6, 7) and cf._state["overrides"]["structure.floors"] in (6, 7)
    assert isinstance(cf._state["overrides"]["structure.floors"], int)
    by["structure.damage"].t = 1.0
    assert by["structure.damage"].f == pytest.approx(1.0)
    by["structure.damage"].t = 0.0
    assert by["structure.damage"].f == pytest.approx(0.0)
    assert by["structure.damage"].t == 0.0
    by["structure.floors"].f = 99                                          # typed past the end: the factor clamps
    assert by["structure.floors"].t == 1.0


def test_choice_bool_reset_and_reset_all(hall_session):
    connect(hall_session)
    by = {c.path: c for c in settings().controls}
    by["structure.roof"].e = "flat"
    by["structure.windows"].b = False
    assert cf._state["overrides"] == {"structure.roof": "flat", "structure.windows": False}
    assert bpy.ops.charforge.reset_control(path="structure.roof") == {"FINISHED"}
    assert live.read_overrides(hall_session.overrides_path)[1] == {"structure.windows": False}
    assert bpy.ops.charforge.reset_all() == {"FINISHED"}
    assert live.read_overrides(hall_session.overrides_path)[1] == {}
    assert cf._state["overrides"] == {}


def test_overrides_written_elsewhere_are_adopted_once(hall_session):
    live.write_overrides(hall_session.overrides_path, {"structure.roof": "hip"})
    hall_session.build_now()
    connect(hall_session)
    assert cf._state["overrides"] == {"structure.roof": "hip"}
    bpy.ops.charforge.reset_all()
    cf.poll()
    assert cf._state["overrides"] == {}                                    # the stale status does not bring them back
    hall_session.build_now()
    cf.poll()
    assert cf._state["overrides"] == {}


def test_copy_overrides_is_nested_valid_yaml(hall_session):
    import yaml
    connect(hall_session)
    cf._state["overrides"].update({"structure.floors": 3, "structure.footprint.0": 14.0, "wear.rust": 0.5})
    txt = cf.overrides_as_yaml(cf._state["overrides"])
    d = yaml.safe_load(txt)
    assert d["structure"]["floors"] == 3 and d["wear"] == {"rust": 0.5}
    assert d["structure"]["footprint"] == {"0": 14.0}      # index paths are for `charforge overrides apply`, which understands them
    bpy.ops.charforge.copy_overrides()


def test_keep_this_version_detaches_and_next_build_imports_fresh(hall_session):
    connect(hall_session)
    first = {o.name for o in lod0_objects()}
    assert bpy.ops.charforge.keep() == {"FINISHED"}
    assert cf._state["created"] == {}
    assert bpy.data.objects.get("charforge:struct_hall") is None
    kept = [o for o in bpy.data.objects if o.type == "MESH" and o.name in first]
    assert kept and any("(kept" in c.name for c in bpy.data.collections)
    assert bpy.ops.charforge.reload() == {"FINISHED"}
    assert bpy.data.objects.get("charforge:struct_hall") is not None
    fresh = lod0_objects()
    assert fresh and not (set(fresh) & set(kept))
    assert all(o.name in bpy.data.objects for o in kept)                      # the kept copy was not touched


def test_auto_reload_off_and_edit_mode_pause(hall_session):
    s = settings()
    s.auto_reload = False
    connect(hall_session)
    bid = json.loads(hall_session.status_path.read_text())["build_id"]
    assert not lod0_objects() and s.build_id == bid and s.loaded_build == 0
    s.auto_reload = True
    cube = bpy.data.objects.new("c", bpy.data.meshes.new("c"))
    bpy.context.scene.collection.objects.link(cube)
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        assert cf.poll().endswith("reloaded") and "paused" in s.message and s.loaded_build == 0 and not lod0_objects()
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    assert "reloaded" in cf.poll() and lod0_objects() and s.loaded_build == bid


def test_heartbeat_and_protocol_mismatch(hall_session):
    connect(hall_session)
    s = settings()
    hb = hall_session.heartbeat_path
    if hb.exists():
        hb.unlink()
    cf.poll()
    assert not s.watcher_alive
    hb.write_text(str(time.time()))
    cf.poll()
    assert s.watcher_alive
    import os
    os.utime(hb, (time.time() - 60, time.time() - 60))
    cf.poll()
    assert not s.watcher_alive
    hb.unlink()
    p = hall_session.status_path
    good = json.loads(p.read_text())
    p.write_text(json.dumps({**good, "protocol": 99}))
    assert cf.poll() == "protocol-mismatch" and "protocol 99" in s.message
    p.write_text("{ torn")
    assert cf.poll() == "unreadable"
    p.write_text(json.dumps(good))
    cf._state["mtime"] = None
    assert "status" in cf.poll()


def test_a_failed_import_is_reported_and_does_not_loop(hall_session, tmp_path):
    import shutil
    broken = tmp_path / "s"
    shutil.copytree(hall_session.dir, broken)
    st = json.loads((broken / "live.json").read_text())
    lod0 = next(f for f in st["files"] if f.get("lod") == 0)
    (broken / lod0["file"]).write_bytes(b"not a glb")
    settings().session_dir = str(broken)
    bpy.ops.charforge.connect()
    assert "import failed" in settings().message
    assert cf.poll() == "idle"                                                # not retried every tick


def test_effect_jobs_load_images(tmp_path):
    job = job_copy(tmp_path, "fx_fireball.yaml")
    sess = LiveSession(job, tmp_path / "out", Backend(), log=lambda s: None)
    sess.build_now()
    connect(sess)
    names = [i.name for i in bpy.data.images]
    assert any("flipbook" in n for n in names) and any("contact_sheet" in n for n in names)
    assert any("Effect" == c.group for c in settings().controls)
    live.write_overrides(sess.overrides_path, {"effect.radius": 0.3})
    sess.build_now()
    cf.poll()
    assert len([n for n in bpy.data.images.keys() if "flipbook" in n]) == 1       # reloaded, not duplicated


def make_box_character():
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    o = bpy.context.active_object
    o.scale = (0.5, 0.3, 1.8)
    bpy.ops.object.transform_apply(scale=True)
    o.location.z = 0.9
    bpy.context.view_layer.update()
    return o


def test_send_selection_writes_glb_then_job_into_the_inbox(hall_session):
    connect(hall_session)
    o = make_box_character()
    s = settings()
    s.send_name = "my blocky guy!"
    assert bpy.ops.charforge.send_selection() == {"FINISHED"}
    inbox = hall_session.dir / "inbox"
    assert sorted(p.name for p in inbox.iterdir()) == ["my_blocky_guy_.glb", "my_blocky_guy_.yaml"]      # no temp files left
    info = inspect_glb(inbox / "my_blocky_guy_.glb")
    assert info["triangles"] == 12
    spec = load_job(inbox / "my_blocky_guy_.yaml")
    assert spec.provider == "import_glb" and spec.name == "my_blocky_guy_" and spec.height_m == pytest.approx(1.8, abs=0.01)
    m = build_one(spec, hall_session.dir.parent / "sent", Backend())
    assert m["status"] in ("passed", "needs_review"), m.get("error")        # it goes through the same pipeline as any import
    assert m["provider"]["provider"] == "import_glb"


def test_send_selection_refuses_nothing_selected_or_flat(hall_session):
    connect(hall_session)
    before = sorted((hall_session.dir / "inbox").glob("*"))
    with pytest.raises(RuntimeError, match="nothing selected"):
        bpy.ops.charforge.send_selection()
    bpy.ops.mesh.primitive_plane_add()
    with pytest.raises(RuntimeError, match="no height"):
        bpy.ops.charforge.send_selection()
    assert sorted((hall_session.dir / "inbox").glob("*")) == before


def test_start_watcher_launches_charforge_and_connects(tmp_path):
    job = job_copy(tmp_path, "fx_shockwave.yaml")
    s = settings()
    s.job_file, s.out_dir = str(job), str(tmp_path / "out")
    s.charforge_cmd = str(Path(sys.executable).parent / "charforge")
    assert bpy.ops.charforge.start_watcher() == {"FINISHED"}
    try:
        end = time.time() + 90
        while time.time() < end:
            cf.poll()
            if s.build_id >= 1 and s.watcher_alive:
                break
            time.sleep(0.3)
        assert s.connected and s.job_name == "fx_shockwave" and s.build_status == "passed" and s.watcher_alive
        assert (tmp_path / "out" / "charforge_watch.log").exists()
    finally:
        bpy.ops.charforge.stop_watcher()
    w_end = time.time() + 20
    lj = tmp_path / "out" / "_live" / "fx_shockwave" / "live.json"
    while time.time() < w_end and json.loads(lj.read_text())["state"] != "stopped":
        time.sleep(0.2)
    assert not (tmp_path / "out" / "_live" / "fx_shockwave" / "live.heartbeat").exists()
    assert json.loads((tmp_path / "out" / "_live" / "fx_shockwave" / "live.json").read_text())["state"] == "stopped"


def test_start_watcher_reports_a_missing_program(tmp_path):
    job = job_copy(tmp_path, "fx_shockwave.yaml")
    s = settings()
    s.job_file, s.out_dir, s.charforge_cmd = str(job), str(tmp_path / "out"), "/nonexistent/charforge"
    with pytest.raises(RuntimeError, match="could not start"):
        bpy.ops.charforge.start_watcher()
    assert not s.connected


def test_bootstrap_script_registers_and_exits_cleanly(tmp_path):
    r = subprocess.run([sys.executable, str(ADDON_PARENT / "bootstrap.py"), "--", "--session", str(tmp_path)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]


def test_reload_keeps_the_users_selection_and_active_object(hall_session):
    cube = bpy.data.objects.new("mine", bpy.data.meshes.new("mine"))
    bpy.context.scene.collection.objects.link(cube)
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    connect(hall_session)
    assert lod0_objects()
    assert [o.name for o in bpy.context.selected_objects] == ["mine"] and bpy.context.view_layer.objects.active == cube


def test_zip_installs_and_enables_through_blenders_own_addon_manager(tmp_path):
    """charforge addon zip -> Install from Disk -> enable, in a throwaway Blender user folder (bpy 5.2 legacy add-on path)."""
    from charforge.cli import main
    z = tmp_path / "charforge_live.zip"
    assert main(["addon", "zip", "-O", str(z)]) == 0
    script = tmp_path / "inst.py"
    script.write_text(
        "import bpy\n"
        f"assert bpy.ops.preferences.addon_install(filepath={str(z)!r}, overwrite=True) == {{'FINISHED'}}\n"
        "assert bpy.ops.preferences.addon_enable(module='charforge_live') == {'FINISHED'}\n"
        "assert hasattr(bpy.context.window_manager, 'charforge') and hasattr(bpy.ops.charforge, 'connect')\n"
        "print('ADDON-OK')\n")
    import os
    env = {**os.environ, "HOME": str(tmp_path / "home"), "XDG_CONFIG_HOME": str(tmp_path / "home" / ".config"),
           "BLENDER_USER_CONFIG": str(tmp_path / "cfg"), "BLENDER_USER_SCRIPTS": str(tmp_path / "scripts")}
    (tmp_path / "scripts").mkdir()
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=180)
    assert "ADDON-OK" in r.stdout, r.stdout[-800:] + r.stderr[-800:]
    assert (tmp_path / "scripts" / "addons" / "charforge_live" / "__init__.py").is_file()
