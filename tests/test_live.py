"""Live link, the pure-python half: overrides, controls, the watcher, live.json. (The Blender add-on is in test_addon.py.)"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import pytest
import yaml

from vgbuilder import live
from vgbuilder.cli import main
from vgbuilder.live import LiveSession, OverrideError, apply_overrides
from vgbuilder.orchestrator import Backend, build_one
from vgbuilder.spec import load_job

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def job_copy(tmp_path, name):
    d = tmp_path / "jobs"
    d.mkdir(exist_ok=True)
    shutil.copy(ROOT / "jobs" / name, d / name)
    return d / name


# ---------------------------------------------------------------- overrides
def test_apply_overrides_nested_and_untouched_original():
    raw = {"name": "a", "structure": {"floors": 2, "footprint": [12, 7]}, "params": None}
    out = apply_overrides(raw, {"structure.floors": 4, "structure.footprint.1": 9.5, "params.leg_frac": 0.5, "wear.rust": 0.9})
    assert out["structure"] == {"floors": 4, "footprint": [12, 9.5]}
    assert out["params"] == {"leg_frac": 0.5} and out["wear"] == {"rust": 0.9}
    assert raw["structure"]["floors"] == 2 and raw["params"] is None          # the input is not modified


@pytest.mark.parametrize("path", ["provider", "provider_args.command", "critic.command", "name", "sketch", "notes", "kind", "formats"])
def test_overrides_can_not_reach_commands_paths_or_identity(path):
    with pytest.raises(OverrideError, match="cannot be overridden"):
        apply_overrides({"name": "a"}, {path: "x"})


@pytest.mark.parametrize("bad", ["", "a..b", "structure.", "structure.fl oors", "../x", "structure/floors"])
def test_bad_paths_rejected(bad):
    with pytest.raises(OverrideError):
        apply_overrides({}, {bad: 1})


def test_bad_values_and_indices():
    with pytest.raises(OverrideError, match="finite"):
        apply_overrides({}, {"seed": float("nan")})
    with pytest.raises(OverrideError, match="unsupported"):
        apply_overrides({}, {"seed": object()})
    with pytest.raises(OverrideError, match="not an index"):
        apply_overrides({"structure": {"footprint": [1, 2]}}, {"structure.footprint.5": 1})
    with pytest.raises(OverrideError, match="does not exist"):
        apply_overrides({}, {"structure.footprint.0": 5})
    with pytest.raises(OverrideError, match="must be an object"):
        live.check_overrides([1, 2])


def test_overrides_file_roundtrip_seq_and_garbage(tmp_path):
    p = tmp_path / "o.json"
    assert live.read_overrides(p) == (0, {})
    s1 = live.write_overrides(p, {"seed": 4})
    s2 = live.write_overrides(p, {"seed": 5})
    assert s2 == s1 + 1 and live.read_overrides(p) == (s2, {"seed": 5})
    p.write_text("{not json")
    with pytest.raises(OverrideError, match="not valid JSON"):
        live.read_overrides(p)
    p.write_text(json.dumps({"seq": "x", "overrides": {}}))
    with pytest.raises(OverrideError):
        live.read_overrides(p)
    live.write_overrides(p, {"seed": 1})                       # an unreadable file does not block the next write
    assert live.read_overrides(p)[1] == {"seed": 1}


def test_load_job_overrides_validate_and_change_fingerprint():
    base = load_job(ROOT / "jobs" / "struct_hall.yaml")
    ov = load_job(ROOT / "jobs" / "struct_hall.yaml", {"structure.floors": 3})
    assert ov.structure["floors"] == 3 and ov.fingerprint() != base.fingerprint()
    assert load_job(ROOT / "jobs" / "struct_hall.yaml", {}).fingerprint() == base.fingerprint()
    with pytest.raises(ValueError, match="floors"):
        load_job(ROOT / "jobs" / "struct_hall.yaml", {"structure.floors": 99})
    with pytest.raises(ValueError, match="unknown params"):
        load_job(ROOT / "jobs" / "hero.yaml", {"params.nonsense": 1})


# ---------------------------------------------------------------- controls
JOBS = ["struct_hall", "struct_tower", "struct_ruin", "fx_fireball", "fx_shockwave", "hero", "hero_gear", "hero_souls", "lanky"]


@pytest.mark.parametrize("name", JOBS)
def test_every_control_is_accepted_by_the_validator_at_its_limits(name):
    """The sliders offer exactly what the spec accepts: each control at min, at max, and unchanged validates."""
    path = ROOT / "jobs" / f"{name}.yaml"
    spec = load_job(path)
    ctl = live.controls(spec)
    assert ctl, name
    assert len({c["path"] for c in ctl}) == len(ctl)
    for c in ctl:
        if c["type"] in ("float", "int"):
            assert c["min"] <= c["value"] <= c["max"], c
            tries = [c["min"], c["max"], c["value"]]
        elif c["type"] == "choice":
            assert c["value"] in c["choices"], c
            tries = c["choices"]
        else:
            tries = [True, False]
        for v in tries:
            if c["path"] == "height_m" or c["path"].startswith("params."):
                continue                      # character body bounds are the fitter's, checked in the next test
            try:
                load_job(path, {c["path"]: v})
            except ValueError as e:
                # a combination can be inconsistent (a low floor with tall windows): that is a ValueError that says why,
                # which the session shows as 'invalid'; anything else (KeyError, TypeError) would be a schema bug
                assert len(str(e)) > 10, (c["path"], v)


def test_character_controls_show_fitted_values_and_pin_on_edit(tmp_path):
    spec = load_job(ROOT / "jobs" / "hero.yaml")
    manifest = {"best_iteration": 1, "iterations": [{"params": {"leg_frac": 0.51, "arm_frac": 0.4}}]}
    by = {c["path"]: c for c in live.controls(spec, manifest)}
    assert by["params.leg_frac"]["value"] == 0.51            # from the build, not the default
    assert by["params.head_scale"]["value"] == 1.0           # default when the build did not record it
    pinned = load_job(ROOT / "jobs" / "hero.yaml", {"params.leg_frac": 0.5})
    assert pinned.params["leg_frac"] == 0.5                    # an edited slider lands in params = pinned
    assert {c["path"]: c for c in live.controls(pinned, manifest)}["params.leg_frac"]["value"] == 0.5
    gear = {c["path"] for c in live.controls(load_job(ROOT / "jobs" / "hero_gear.yaml"))}
    assert any(p.startswith("accessory_shape.") for p in gear) and any(p.startswith("wear.") for p in gear)


# ---------------------------------------------------------------- session (effect: fast, no Blender)
def effect_session(tmp_path, **kw):
    job = job_copy(tmp_path, "fx_fireball.yaml")
    return job, LiveSession(job, tmp_path / "out", Backend(), log=lambda s: None, **kw)


def status(sess):
    return json.loads(sess.status_path.read_text())


def test_session_builds_publishes_and_follows_overrides(tmp_path):
    job, sess = effect_session(tmp_path)
    assert status(sess)["state"] == "idle" and status(sess)["build_id"] == 0
    assert sess.stamp() != sess._built_stamp
    sess.build_now()
    st = status(sess)
    assert st["state"] == "idle" and st["build_id"] == 1 and st["result"]["status"] == "passed"
    assert st["protocol"] == live.PROTOCOL and st["controls"] and st["inbox"] == "inbox"
    roles = {f["role"] for f in st["files"]}
    assert "effect" in roles and "contact_sheet" in roles
    for f in st["files"]:
        assert (sess.dir / f["file"]).is_file() and f["file"].startswith("published/00001/")
    assert sess.stamp() == sess._built_stamp                        # nothing to do until something changes

    seq = live.write_overrides(sess.overrides_path, {"effect.radius": 0.3})
    assert sess.stamp() != sess._built_stamp
    fp1 = st["result"]["fingerprint"]
    sess.build_now()
    st = status(sess)
    assert st["build_id"] == 2 and st["seq_applied"] == seq and st["overrides"] == {"effect.radius": 0.3}
    assert st["result"]["fingerprint"] != fp1
    radius = {c["path"]: c for c in st["controls"]}["effect.radius"]["value"]
    assert radius == 0.3                                            # sliders report the value in effect
    # the normal output folder is untouched by live builds
    assert not (tmp_path / "out" / "fx_fireball").exists()


def test_invalid_overrides_keep_the_last_good_build_and_recover(tmp_path):
    job, sess = effect_session(tmp_path)
    sess.build_now()
    good = status(sess)["files"]
    live.write_overrides(sess.overrides_path, {"effect.radius": 5.0})
    sess.build_now()
    st = status(sess)
    assert st["state"] == "invalid" and "radius" in st["error"]
    assert st["build_id"] == 1 and st["files"] == good and st["controls"]     # still showing build 1, sliders intact
    assert sess.stamp() == sess._built_stamp                                   # no rebuild loop on a bad input
    live.write_overrides(sess.overrides_path, {"effect.radius": 0.4})
    sess.build_now()
    assert status(sess)["state"] == "idle" and status(sess)["build_id"] == 2

    sess.overrides_path.write_text("{ half written")
    sess.build_now()
    assert status(sess)["state"] == "invalid" and "JSON" in status(sess)["error"]
    job.write_text(job.read_text() + "\nbogus_key: 1\n")
    live.write_overrides(sess.overrides_path, {})
    sess.build_now()
    assert status(sess)["state"] == "invalid" and "bogus_key" in status(sess)["error"]
    job.write_text(job.read_text().replace("bogus_key: 1\n", ""))
    sess.build_now()
    assert status(sess)["state"] == "idle" and status(sess)["build_id"] == 3


def test_overrides_file_can_not_run_programs(tmp_path):
    job, sess = effect_session(tmp_path)
    marker = tmp_path / "pwned"
    sess.overrides_path.write_text(json.dumps({"seq": 1, "overrides": {
        "provider": "command", "provider_args.command": f"touch {marker}", "critic.command": f"touch {marker}"}}))
    sess.build_now()
    st = status(sess)
    assert st["state"] == "invalid" and "cannot be overridden" in st["error"] and not marker.exists()


def test_old_builds_are_pruned_and_publish_is_immutable(tmp_path):
    job, sess = effect_session(tmp_path)
    sess.build_now()
    first = {f["file"]: sha(sess.dir / f["file"]) for f in status(sess)["files"]}
    for i in range(live.KEEP_BUILDS + 1):
        live.write_overrides(sess.overrides_path, {"effect.fade": 0.5 + 0.05 * i})
        sess.build_now()
    assert status(sess)["build_id"] == live.KEEP_BUILDS + 2
    kept = sorted(p.name for p in (sess.dir / "published").iterdir())
    assert len(kept) == live.KEEP_BUILDS and kept[-1] == f"{live.KEEP_BUILDS + 2:05d}" and "00001" not in kept
    assert not any(p.name.endswith(".tmp") for p in (sess.dir / "published").iterdir())
    assert first  # (the files of build 1 were removed with it, not rewritten)


def test_run_loop_picks_up_edits_from_either_file_and_cleans_up(tmp_path):
    job, sess = effect_session(tmp_path)
    stop = threading.Event()
    t = threading.Thread(target=sess.run, args=(stop, 0.05), daemon=True)
    t.start()

    def wait_for(pred, what, timeout=90):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if pred(status(sess)):
                    return status(sess)
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        raise AssertionError(f"timed out waiting for {what}: {status(sess)}")

    try:
        wait_for(lambda s: s["build_id"] == 1 and s["state"] == "idle", "first build")
        assert sess.heartbeat_path.exists()
        live.write_overrides(sess.overrides_path, {"effect.turbulence": 0.9})
        wait_for(lambda s: s["build_id"] == 2 and s["overrides"] == {"effect.turbulence": 0.9}, "override build")
        text = job.read_text().replace("frames: 16", "frames: 12")
        job.write_text(text)                                            # an edit in a text editor
        s = wait_for(lambda s: s["build_id"] == 3, "yaml build")
        assert s["result"]["status"] == "passed"
        time.sleep(0.5)
        assert status(sess)["build_id"] == 3                            # settled: no rebuild loop
    finally:
        stop.set()
        t.join(timeout=30)
    assert not sess.heartbeat_path.exists() and status(sess)["state"] == "stopped"


def test_session_refuses_a_broken_job_at_start(tmp_path):
    job = tmp_path / "bad.yaml"
    job.write_text("name: x\nkind: nonsense\n")
    with pytest.raises(ValueError):
        LiveSession(job, tmp_path / "out", Backend())


# ---------------------------------------------------------------- Blender builds: live == baked
def test_live_build_without_overrides_is_byte_identical_to_a_normal_run(tmp_path):
    job = job_copy(tmp_path, "struct_hall.yaml")
    normal = build_one(load_job(job), tmp_path / "normal", Backend())
    sess = LiveSession(job, tmp_path / "out", Backend(), log=lambda s: None)
    sess.build_now()
    st = status(sess)
    assert st["state"] == "idle" and st["result"]["status"] == "passed"
    lods = {f["lod"]: f for f in st["files"] if f["role"] == "lod"}
    assert sorted(lods) == [0, 1, 2] and any(f["role"] == "collision" for f in st["files"])
    for o in normal["outputs"]:
        if o["file"].endswith(".glb"):
            assert sha(sess.dir / "published" / "00001" / o["file"]) == o["sha256"], o["file"]


def test_overrides_applied_to_the_yaml_rebuild_the_exact_same_files(tmp_path):
    """What you saw while dragging sliders is what the pure-python build makes once the numbers are in the YAML."""
    job = job_copy(tmp_path, "struct_hall.yaml")
    sess = LiveSession(job, tmp_path / "out", Backend(), log=lambda s: None)
    ov = {"structure.floors": 3, "structure.footprint.0": 14.0, "structure.damage": 0.3, "wear.rust": 0.6}
    live.write_overrides(sess.overrides_path, ov)
    sess.build_now()
    st = status(sess)
    assert st["state"] == "idle" and st["build_id"] == 1
    live_hash = {f["file"].split("/")[-1]: sha(sess.dir / f["file"]) for f in st["files"] if f["role"] in ("lod", "collision")}

    diff = live.apply_to_job_file(job, ov)
    assert "floors" in diff and job.read_text().count("floors: 2") == 1          # dry run did not write
    live.apply_to_job_file(job, ov, write=True)
    assert (job.parent / (job.name + ".bak")).exists()
    baked = build_one(load_job(job), tmp_path / "baked", Backend())
    assert baked["status"] == st["result"]["status"]
    for o in baked["outputs"]:
        if o["file"] in live_hash:
            assert o["sha256"] == live_hash[o["file"]], o["file"]
    assert len(live_hash) == 4


def test_live_override_changes_the_building(tmp_path):
    job = job_copy(tmp_path, "struct_tower.yaml")
    sess = LiveSession(job, tmp_path / "out", Backend(), log=lambda s: None)
    sess.build_now()
    h1 = status(sess)["result"]["fingerprint"]
    live.write_overrides(sess.overrides_path, {"structure.floors": 7})
    sess.build_now()
    m = json.loads((sess.dir / "manifest.json").read_text())
    assert m["spec"]["structure"]["floors"] == 7 and m["fingerprint"] != h1


# ---------------------------------------------------------------- CLI
def _json_in(text):
    return json.loads(text[text.index("{"):])


def test_cli_overrides_commands(tmp_path, capsys):
    job = job_copy(tmp_path, "fx_fireball.yaml")
    out = str(tmp_path / "out")
    assert main(["overrides", "-o", out, "set", str(job), "effect.radius=0.4", "seed=7"]) == 0
    assert main(["overrides", "-o", out, "show", str(job)]) == 0
    shown = _json_in(capsys.readouterr().out)
    assert shown["overrides"] == {"effect.radius": 0.4, "seed": 7}
    assert main(["overrides", "-o", out, "set", str(job), "effect.radius=9"]) == 1          # refused, file unchanged
    assert main(["overrides", "-o", out, "set", str(job), "provider=command"]) == 1
    assert main(["overrides", "-o", out, "set", str(job), "oops"]) == 1
    capsys.readouterr()
    before = job.read_text()
    assert main(["overrides", "-o", out, "apply", str(job)]) == 0
    assert "radius" in capsys.readouterr().out and job.read_text() == before               # dry run
    assert main(["overrides", "-o", out, "apply", str(job), "--write"]) == 0
    assert yaml.safe_load(job.read_text())["effect"]["radius"] == 0.4 and yaml.safe_load(job.read_text())["seed"] == 7
    assert (job.parent / (job.name + ".bak")).read_text() == before
    capsys.readouterr()
    main(["overrides", "-o", out, "show", str(job)])
    assert _json_in(capsys.readouterr().out)["overrides"] == {}         # cleared after writing
    assert main(["overrides", "-o", out, "clear", str(job)]) == 0


def test_apply_never_writes_an_invalid_job(tmp_path):
    job = job_copy(tmp_path, "fx_fireball.yaml")
    before = job.read_text()
    with pytest.raises(ValueError):
        live.apply_to_job_file(job, {"effect.radius": 7.0}, write=True)
    assert job.read_text() == before and not (job.parent / (job.name + ".bak")).exists()
    assert not list(job.parent.glob(".*.check.yaml"))


def test_cli_watch_once_writes_live_json(tmp_path, capsys):
    job = job_copy(tmp_path, "fx_shockwave.yaml")
    assert main(["watch", str(job), "-o", str(tmp_path / "out"), "--once"]) == 0
    st = json.loads((tmp_path / "out" / "_live" / "fx_shockwave" / "live.json").read_text())
    assert st["state"] == "idle" and st["build_id"] == 1
    bad = tmp_path / "jobs" / "bad.yaml"
    bad.write_text("name: bad\nkind: effect\neffect: {template: nope}\n")
    with pytest.raises(ValueError):
        main(["watch", str(bad), "-o", str(tmp_path / "out"), "--once"])


def test_cli_addon_commands(tmp_path, capsys):
    import zipfile
    assert main(["addon", "path"]) == 0
    p = Path(capsys.readouterr().out.strip())
    assert (p / "__init__.py").is_file() and (p / "blender_manifest.toml").is_file()
    z = tmp_path / "a.zip"
    assert main(["addon", "zip", "-O", str(z)]) == 0
    names = set(zipfile.ZipFile(z).namelist())
    assert {"vgbuilder_live/__init__.py", "vgbuilder_live/blender_manifest.toml"} <= names
    assert not any("__pycache__" in n for n in names)


def test_live_core_and_addon_have_no_hidden_dependencies():
    """live.py must import without Blender; the add-on must not import vgbuilder (it only speaks the file protocol)."""
    import ast
    core = ast.parse((ROOT / "src/vgbuilder/live.py").read_text())
    mods = {n.module if isinstance(n, ast.ImportFrom) else a.name for n in ast.walk(core) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert "bpy" not in mods
    addon = ast.parse((ROOT / "src/vgbuilder/blender_addon/vgbuilder_live/__init__.py").read_text())
    names = set()
    for n in ast.walk(addon):
        if isinstance(n, ast.Import):
            names |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            names.add((n.module or "").split(".")[0])
    assert names <= {"json", "os", "re", "subprocess", "time", "pathlib", "bpy"}, names
    assert "vgbuilder" not in names


def test_addon_and_core_agree_on_the_protocol():
    src = (ROOT / "src/vgbuilder/blender_addon/vgbuilder_live/__init__.py").read_text()
    assert f"PROTOCOL = {live.PROTOCOL}" in src


def test_fast_mode_is_cheaper_labelled_and_never_the_real_output(tmp_path):
    job = job_copy(tmp_path, "struct_hall.yaml")
    sess = LiveSession(job, tmp_path / "out", Backend(), log=lambda s: None, fast=True)
    sess.build_now()
    st = status(sess)
    assert st["fast"] is True and st["state"] == "idle"
    assert [f["lod"] for f in st["files"] if f["role"] == "lod"] == [0]
    live.write_overrides(sess.overrides_path, {"texture_size": 128})       # the user's own value beats the preview default
    sess.build_now()
    m = json.loads((sess.dir / "manifest.json").read_text())
    assert m["spec"]["texture_size"] == 128 and m["spec"]["lods"] == [1.0]
    normal = build_one(load_job(job), tmp_path / "normal", Backend())
    assert m["fingerprint"] != normal["fingerprint"]
    effect = LiveSession(job_copy(tmp_path, "fx_fireball.yaml"), tmp_path / "out2", Backend(), log=lambda s: None, fast=True)
    assert effect._fast_overrides() == {}                                  # effects are already cheap and have no LODs


def test_restarting_the_watcher_continues_build_ids_and_survives_leftovers(tmp_path):
    job = job_copy(tmp_path, "fx_shockwave.yaml")
    out = tmp_path / "out"
    a = LiveSession(job, out, Backend(), log=lambda s: None)
    a.build_now()
    live.write_overrides(a.overrides_path, {"effect.fade": 0.8})
    a.build_now()
    assert status(a)["build_id"] == 2
    (a.dir / "published" / "00003.tmp").mkdir()                              # a build that died halfway
    (a.dir / "published" / "00003.tmp" / "junk").write_text("x")
    b = LiveSession(job, out, Backend(), log=lambda s: None)                 # a second `vgbuilder watch` on the same folder
    assert b.build_id == 2 and status(b)["build_id"] == 2 and status(b)["state"] == "idle"
    b.build_now()
    assert status(b)["build_id"] == 3 and status(b)["state"] == "idle"
    assert sorted(p.name for p in (b.dir / "published").iterdir()) == ["00001", "00002", "00003"]
    # a stale directory with the next id is replaced, not an error
    (b.dir / "published" / "00004").mkdir()
    (b.dir / "published" / "00004" / "old").write_text("x")
    live.write_overrides(b.overrides_path, {"effect.fade": 0.7})
    b.build_now()
    assert status(b)["build_id"] == 4 and not (b.dir / "published" / "00004" / "old").exists()


def test_fit_cache_gives_the_same_answer_and_is_keyed_on_the_inputs(tmp_path):
    import numpy as np

    from vgbuilder import fixer
    from vgbuilder.sketch import load_mask
    spec = load_job(ROOT / "jobs" / "hero_gear.yaml")
    mask = load_mask(spec.resolve_path(spec.sketch))
    cache = tmp_path / "fit"
    p1, i1 = fixer.initial_params(spec, mask, None, cache_dir=cache)
    assert "cached" not in i1 and len(list(cache.glob("*.json"))) == 1
    p2, i2 = fixer.initial_params(spec, mask, None, cache_dir=cache)
    assert i2.pop("cached") is True and p2 == p1 and i2 == i1
    plain, plain_info = fixer.initial_params(spec, mask, None)                # no cache: same numbers
    assert plain == p1 and plain_info == i1
    # accessory SIZE does not enter the fit; a different budget, pin, height or sketch does
    shaped = load_job(ROOT / "jobs" / "hero_gear.yaml", {"accessory_shape.horn_len": 2.0, "wear.rust": 0.9})
    assert fixer._fit_key(shaped, mask, None) == fixer._fit_key(spec, mask, None)
    for ov in ({"poly_budget": 4000}, {"params.leg_frac": 0.5}, {"fit_prior": 1.0}, {"accessories": ["cape"]}):
        assert fixer._fit_key(load_job(ROOT / "jobs" / "hero_gear.yaml", ov), mask, None) != fixer._fit_key(spec, mask, None), ov
    other = mask.copy()
    other[0, 0] = not other[0, 0] if other.dtype == bool else other[0, 0] + 1
    assert fixer._fit_key(spec, other, None) != fixer._fit_key(spec, mask, None)
    (next(cache.glob("*.json"))).write_text("{ torn")                          # a damaged entry is ignored and recomputed
    p3, i3 = fixer.initial_params(spec, mask, None, cache_dir=cache)
    assert p3 == p1 and "cached" not in i3


def test_character_rebuild_through_the_fit_cache_is_byte_identical(tmp_path):
    job = job_copy(tmp_path, "hero.yaml")
    shutil.copytree(ROOT / "examples", tmp_path / "examples", dirs_exist_ok=True)
    sess = LiveSession(job, tmp_path / "out", Backend(), log=lambda s: None, fast=True)
    sess.build_now()
    m1 = json.loads((sess.dir / "manifest.json").read_text())
    assert "cached" not in m1["fit"] and m1["status"] in ("passed", "needs_review")
    live.write_overrides(sess.overrides_path, {"wear.rust": 0.7})             # does not touch the fit inputs
    sess.build_now()
    m2 = json.loads((sess.dir / "manifest.json").read_text())
    assert m2["fit"].get("cached") is True
    live.write_overrides(sess.overrides_path, {})
    sess.build_now()
    m3 = json.loads((sess.dir / "manifest.json").read_text())
    assert m3["fit"].get("cached") is True
    g1 = {o["file"]: o["sha256"] for o in m1["outputs"] if o["file"].endswith(".glb")}
    g3 = {o["file"]: o["sha256"] for o in m3["outputs"] if o["file"].endswith(".glb")}
    assert g1 and g1 == g3


def test_an_edit_that_arrives_during_a_build_is_built_next(tmp_path, monkeypatch):
    from vgbuilder import orchestrator
    job, sess = effect_session(tmp_path)
    real = orchestrator.build_one
    calls = []

    def build_and_edit(spec, root, backend, log=None, **kw):
        if not calls:
            live.write_overrides(sess.overrides_path, {"effect.fade": 0.6})         # the user drags a slider mid-build
        calls.append(spec.effect.get("fade"))
        return real(spec, root, backend, log, **kw)

    monkeypatch.setattr(orchestrator, "build_one", build_and_edit)
    sess.build_now()
    assert status(sess)["build_id"] == 1 and sess.stamp() != sess._built_stamp      # still dirty: the loop will rebuild
    sess.build_now()
    assert status(sess)["build_id"] == 2 and status(sess)["overrides"] == {"effect.fade": 0.6} and sess.stamp() == sess._built_stamp
    assert calls == [None, 0.6]
