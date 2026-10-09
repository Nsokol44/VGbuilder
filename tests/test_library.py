"""Library, queue, inbox, farm, critic hook and review UI. Effect jobs are used as the workload (no Blender)."""
from __future__ import annotations

import json
import time
import sqlite3
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
import yaml

from charforge import critic as critic_mod
from charforge.farm import load_farm, make_variants
from charforge.library import Library, summarize, work
from charforge.orchestrator import Backend
from charforge.review_ui import serve
from charforge.spec import load_job

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Backend(kind="bpy-module")


def fx_job(path: Path, name="fx_a", template="fireball", **extra) -> Path:
    d = {"name": name, "kind": "effect", "effect": {"template": template, "size": 48, "frames": 8}, **extra}
    path.write_text(yaml.safe_dump(d))
    return path


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path / "db" / "library.db")


def test_queue_claims_by_priority_then_age(lib, tmp_path):
    for n, pr in (("a", 0), ("b", 5), ("c", 5), ("d", 1)):
        assert "queued" in lib.enqueue(fx_job(tmp_path / f"{n}.yaml", n, template="smoke" if n == "d" else "fireball",
                                              **({"seed": ord(n)})), pr)
    order = [lib.claim()["name"] for _ in range(4)]
    assert order == ["b", "c", "d", "a"]
    assert lib.claim() is None


def test_enqueue_dedupes_and_validates(lib, tmp_path):
    p = fx_job(tmp_path / "a.yaml")
    assert "queued" in lib.enqueue(p)
    assert "already in the queue" in lib.enqueue(p)["skipped"]
    assert "queued" in lib.enqueue(p, force=True)
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nkind: effect\neffect: {template: nope}\n")
    with pytest.raises(ValueError):
        lib.enqueue(bad)


def test_work_builds_ingests_and_skips_unchanged(lib, tmp_path):
    out = tmp_path / "out"
    lib.enqueue(fx_job(tmp_path / "a.yaml"))
    assert work(lib, out, BACKEND, once=True, log=lambda s: None) == 1
    a = lib.get("fx_a")
    assert a["version"] == 1 and a["status"] == "passed" and a["kind"] == "effect"
    assert a["thumb"] == "fx_a_contact_sheet.png" and (lib.resolve(a["final_dir"]) / a["thumb"]).exists()
    assert {f["file"] for f in a["files"]} >= {"fx_a_flipbook.png", "fx_a_particles.json"}
    assert lib.queue_rows("done")
    # same spec again: nothing to build
    assert lib.enqueue(tmp_path / "a.yaml")["skipped"].startswith("already built")
    # changed spec: next version, old one kept
    fx_job(tmp_path / "a.yaml", seed=3)
    assert "queued" in lib.enqueue(tmp_path / "a.yaml")
    work(lib, out, BACKEND, once=True, log=lambda s: None)
    assert lib.get("fx_a")["version"] == 2 and lib.get("fx_a", 1)["version"] == 1
    assert len(lib.list(latest_only=False)) == 2 and len(lib.list()) == 1


def test_ingest_is_idempotent(lib, tmp_path):
    out = tmp_path / "out"
    lib.enqueue(fx_job(tmp_path / "a.yaml"))
    work(lib, out, BACKEND, once=True, log=lambda s: None)
    m = json.loads((out / "fx_a" / "manifest.json").read_text())
    r = lib.ingest(m, out)
    assert r["new"] is False and len(lib.list(latest_only=False)) == 1


def test_recover_after_simulated_crash(lib, tmp_path):
    lib.enqueue(fx_job(tmp_path / "a.yaml"))
    job = lib.claim()
    assert job["state"] == "running" and lib.claim() is None
    # a worker died here. Fresh work() on a new Library object (new process) must pick it up
    lib2 = Library(lib.path)
    assert lib2.recover(stale_after_s=3600) == 0            # not stale yet: leave it alone
    assert work(lib2, tmp_path / "out", BACKEND, once=True, log=lambda s: None) == 1   # once-mode recovers everything
    assert lib2.get("fx_a")["status"] == "passed"
    row = lib2.queue_rows("done")[0]
    assert row["attempts"] == 2


def test_infrastructure_errors_retry_then_fail_but_needs_review_is_final(lib, tmp_path, monkeypatch):
    from charforge import orchestrator
    calls = {"n": 0}

    def boom(spec, out_root, backend, log=None):
        calls["n"] += 1
        raise RuntimeError("blender exploded")
    monkeypatch.setattr(orchestrator, "build_one", boom)
    lib.enqueue(fx_job(tmp_path / "a.yaml"))
    work(lib, tmp_path / "out", BACKEND, once=True, log=lambda s: None)
    assert calls["n"] == 3
    row = lib.queue_rows()[0]
    assert row["state"] == "failed" and "blender exploded" in row["error"]

    monkeypatch.undo()
    lib.enqueue(fx_job(tmp_path / "b.yaml", "fx_b", max_iterations=1,
                               effect={"template": "fireball", "size": 48, "frames": 8, "radius": 0.95, "fade": 0.0}))
    work(lib, tmp_path / "out", BACKEND, once=True, log=lambda s: None)
    b = lib.get("fx_b")
    assert b and b["status"] in ("passed", "needs_review")
    assert [r["state"] for r in lib.queue_rows() if r["name"] == "fx_b"] == ["done"]
    assert [r["attempts"] for r in lib.queue_rows() if r["name"] == "fx_b"] == [1]


def test_two_workers_never_claim_the_same_job(lib, tmp_path):
    for i in range(12):
        fx_job(tmp_path / f"j{i}.yaml", f"j{i}", seed=i)
        lib.enqueue(tmp_path / f"j{i}.yaml")
    got, lock = [], threading.Lock()

    def grab():
        while (j := Library(lib.path).claim()) is not None:
            with lock:
                got.append(j["id"])
    ts = [threading.Thread(target=grab) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sorted(got) == list(range(1, 13))


def test_inbox_detects_new_and_changed_files(lib, tmp_path):
    inbox = tmp_path / "inbox"; inbox.mkdir()
    fx_job(inbox / "a.yaml")
    (inbox / "broken.yaml").write_text("name: [\n")
    r = lib.scan_inbox(inbox)
    assert sum("queued" in x for x in r) == 1 and sum("error" in x for x in r) == 1
    assert lib.scan_inbox(inbox) == []                      # nothing changed
    fx_job(inbox / "a.yaml", seed=9)
    r = lib.scan_inbox(inbox)
    assert len(r) == 1 and "queued" in r[0]
    assert len(lib.queue_rows("queued")) == 2


def test_work_drains_inbox_and_workers_gt_one(lib, tmp_path):
    inbox = tmp_path / "inbox"; inbox.mkdir()
    for i, t in enumerate(["fireball", "smoke", "shockwave"]):
        fx_job(inbox / f"{t}.yaml", f"fx_{t}", template=t)
    n = work(lib, tmp_path / "out", BACKEND, workers=2, once=True, inbox=inbox, log=lambda s: None)
    assert n == 3 and len(lib.list()) == 3


def test_review_and_listing_filters(lib, tmp_path):
    lib.enqueue(fx_job(tmp_path / "a.yaml", tags=["boss"]))
    work(lib, tmp_path / "out", BACKEND, once=True, log=lambda s: None)
    assert lib.list(review="pending") and not lib.list(review="approved")
    lib.review("fx_a", "approved", "looks good")
    assert lib.list(review="approved")[0]["review_note"] == "looks good"
    assert lib.list(tag="boss") and not lib.list(tag="npc")
    with pytest.raises(ValueError):
        lib.review("fx_a", "maybe")
    with pytest.raises(KeyError):
        lib.review("nope", "approved")
    assert [e["kind"] for e in lib.get("fx_a")["events"]] == ["ingest", "review:approved"]


def test_library_is_relocatable(lib, tmp_path):
    lib.enqueue(fx_job(tmp_path / "a.yaml"))
    work(lib, tmp_path / "db" / "out", BACKEND, once=True, log=lambda s: None)
    a = lib.get("fx_a")
    assert not Path(a["final_dir"]).is_absolute() and not Path(a["manifest_path"]).is_absolute()
    import shutil
    shutil.move(tmp_path / "db", tmp_path / "moved")
    moved = Library(tmp_path / "moved" / "library.db")
    assert (moved.resolve(moved.get("fx_a")["final_dir"]) / "fx_a_flipbook.png").exists()


def test_library_silhouettes_feed_concepts(lib, tmp_path):
    """Characters register their base-body silhouette; concepts use those PNGs for the uniqueness check."""
    from PIL import Image
    final = tmp_path / "db" / "out" / "knight" / "final"; final.mkdir(parents=True)
    Image.new("L", (64, 64), 245).save(final / "knight_silhouette_front.png")
    m = {"name": "knight", "fingerprint": "f1", "status": "passed", "spec": {"kind": "character", "tags": []},
         "outputs": [{"file": "knight_silhouette_front.png", "sha256": "x", "bytes": 1}], "iterations": []}
    lib.ingest(m, tmp_path / "db" / "out")
    lib.review("knight", "approved")
    assert [p.name for p in lib.silhouette_paths(approved_only=True)] == ["knight_silhouette_front.png"]
    lib.review("knight", "pending")
    assert lib.silhouette_paths(approved_only=True) == [] and len(lib.silhouette_paths()) == 1


def test_summarize_picks_character_numbers():
    m = {"kind": "character", "best_iteration": 1, "iterations": [{"metrics": {
        "silhouette_iou": 0.91234, "height_m": 1.8, "lod_tris": [2000, 1000], "max_influences": 4,
        "animations": {"walk": {}}, "accessories": {"cape": {}}}}]}
    s = summarize(m)
    assert s["silhouette_iou"] == 0.9123 and s["animations"] == ["walk"] and s["accessories"] == ["cape"]


# ---- farm
def write_farm(tmp_path, **kw):
    base = fx_job(tmp_path / "base.yaml", "fx_base", tags=["fx"])
    cfg = {"base": "base.yaml", "count": 5, "seed": 7,
           "vary": {"effect": {"radius": [0.5, 0.9]}, "seed": True}, **kw}
    p = tmp_path / "farm.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_farm_is_deterministic_and_valid(tmp_path):
    f = write_farm(tmp_path)
    a = make_variants(f, tmp_path / "o1")
    b = make_variants(f, tmp_path / "o2")
    assert [p.read_text() for p in a] == [p.read_text() for p in b]
    specs = [load_job(p) for p in a]
    assert [s.name for s in specs] == [f"fx_base_v0{i}" for i in range(1, 6)]
    assert all(0.5 <= s.effect["radius"] <= 0.9 for s in specs) and len({s.effect["radius"] for s in specs}) > 1
    assert len({s.fingerprint() for s in specs}) == 5
    assert all("variant_of:fx_base" in s.tags and "fx" in s.tags for s in specs)
    other = make_variants(write_farm(tmp_path, seed=8), tmp_path / "o3")
    assert [p.read_text() for p in other] != [p.read_text() for p in a]


def test_farm_character_variants_use_absolute_paths_and_validate(tmp_path):
    out = tmp_path / "o"
    f = tmp_path / "farm.yaml"
    f.write_text(yaml.safe_dump({"base": str(ROOT / "jobs" / "hero_souls.yaml"), "count": 4, "seed": 1,
                                 "vary": {"hue_shift_deg": [-40, 40], "wear": {"rot": [0.0, 0.5], "rust": [0.1, 0.6]}}}))
    paths = make_variants(f, out)
    specs = [load_job(p) for p in paths]           # validates with sketches resolved from another folder
    assert all(Path(s.sketch).is_absolute() and Path(s.sketch).exists() for s in specs)
    assert len({s.style_overrides["hue_shift_deg"] for s in specs}) == 4
    assert all(0.0 <= s.wear["rot"] <= 0.5 for s in specs)
    # a hue shift reaches the resolved look
    pals = {tuple(s.load_style().palette["base"]) for s in specs}
    assert len(pals) == 4


def test_farm_rejects_bad_files(tmp_path):
    base = fx_job(tmp_path / "base.yaml")
    for cfg, msg in (({"base": "base.yaml", "vary": {"nonsense": 1}}, "unknown vary"),
                     ({"base": "base.yaml", "oops": 1}, "unknown keys"),
                     ({"base": "base.yaml", "count": 0}, "count"),
                     ({"base": "base.yaml", "vary": {"effect": {"radius": [0.9, 0.5]}}}, "lo, hi"),
                     ({"vary": {}}, "base")):
        p = tmp_path / "f.yaml"; p.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError, match=msg):
            make_variants(p, tmp_path / "o")
    # out-of-bounds values are caught by the normal job validator while expanding
    p = tmp_path / "f.yaml"
    p.write_text(yaml.safe_dump({"base": "base.yaml", "vary": {"wear": {"not_a_key": [0, 1]}}}))
    with pytest.raises(ValueError):
        make_variants(p, tmp_path / "o")


# ---- critic
STUB = "import sys,json\nprint('log noise')\nprint(json.dumps({'score': float(sys.argv[1]), 'notes': 'n', 'issues': ['x']}))\n"


def built(lib, tmp_path, **extra):
    lib.enqueue(fx_job(tmp_path / "a.yaml", **extra))
    work(lib, tmp_path / "out", BACKEND, once=True, log=lambda s: None)


def test_critic_records_score_and_flags_below_min(lib, tmp_path):
    stub = tmp_path / "stub.py"; stub.write_text(STUB)
    built(lib, tmp_path)
    r = critic_mod.run_critic(lib, "fx_a", command=f"{sys.executable} {stub} 0.4 {{contact_sheet}}", min_score=0.6)
    assert r["score"] == 0.4 and r["below_min"] is True and r["issues"] == ["x"]
    assert lib.get("fx_a")["critic"]["score"] == 0.4
    assert lib.get("fx_a")["status"] == "passed"           # the critic annotates; it never changes QA status


def test_critic_placeholders_reach_the_command(lib, tmp_path):
    stub = tmp_path / "echo.py"
    stub.write_text("import sys,json,os\nsheet,ctx=sys.argv[1:3]\n"
                    "c=json.load(open(ctx))\nok=os.path.exists(sheet) and c['name']=='fx_a' and c['kind']=='effect'\n"
                    "print(json.dumps({'score': 1.0 if ok else 0.0}))\n")
    built(lib, tmp_path)
    assert critic_mod.run_critic(lib, "fx_a", command=f"{sys.executable} {stub} {{contact_sheet}} {{context}}")["score"] == 1.0


@pytest.mark.parametrize("script", [
    "import sys; sys.exit(3)",
    "print('not json')",
    "print('{\"score\": 7}')",
    "print('{\"score\": true}')",
    "import time; time.sleep(30)",
])
def test_critic_failures_are_errors_never_passes(lib, tmp_path, script):
    stub = tmp_path / "bad.py"; stub.write_text(script)
    built(lib, tmp_path)
    r = critic_mod.run_critic(lib, "fx_a", command=f"{sys.executable} {stub}", timeout_s=2)
    assert r["score"] is None and "error" in r


def test_critic_runs_from_the_job_spec_in_the_worker(lib, tmp_path):
    stub = tmp_path / "stub.py"; stub.write_text(STUB)
    built(lib, tmp_path, critic={"command": f"{sys.executable} {stub} 0.9", "min_score": 0.5})
    c = lib.get("fx_a")["critic"]
    assert c["score"] == 0.9 and c["below_min"] is False


def test_critic_parse_output_variants():
    assert critic_mod.parse_output('{"score": 0.5}')["score"] == 0.5
    assert critic_mod.parse_output('warn\n{\n "score": 1,\n "notes": "ok"\n}')["notes"] == "ok"
    with pytest.raises(ValueError):
        critic_mod.parse_output("")


# ---- review UI
@pytest.fixture
def ui(lib, tmp_path):
    built(lib, tmp_path, tags=["<script>alert(1)</script>"])
    srv = serve(lib, port=0, log=lambda s: None)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield lib, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_ui_pages_and_files(ui):
    lib, base = ui
    st, ct, body = get(base + "/")
    assert st == 200 and b"fx_a" in body and b"<script>alert" not in body            # escaped
    assert b"&lt;script&gt;" in body
    st, _, body = get(base + "/asset/fx_a/1")
    assert st == 200 and b"Approve" in body
    st, ct, body = get(base + "/file/fx_a/1/fx_a_contact_sheet.png")
    assert st == 200 and ct == "image/png" and body[:4] == b"\x89PNG"
    assert json.loads(get(base + "/api/assets?review=pending")[2])[0]["name"] == "fx_a"
    assert get(base + "/queue")[0] == 200


def test_ui_review_post_and_safety(ui):
    lib, base = ui
    data = urllib.parse.urlencode({"decision": "approved", "note": "ship it"}).encode()
    urllib.request.urlopen(urllib.request.Request(base + "/review/fx_a/1", data=data), timeout=10).read()   # follows 303
    a = lib.get("fx_a")
    assert a["review"] == "approved" and a["review_note"] == "ship it"
    # cross-origin POST refused
    req = urllib.request.Request(base + "/review/fx_a/1", data=urllib.parse.urlencode({"decision": "rejected"}).encode(),
                                 headers={"Origin": "http://evil.example"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=10)
    assert ei.value.code == 403 and lib.get("fx_a")["review"] == "approved"
    # bad decision -> 400; unknown / unregistered / traversal paths -> 404
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(urllib.request.Request(base + "/review/fx_a/1", data=b"decision=maybe"), timeout=10)
    assert ei.value.code == 400
    for path in ("/file/fx_a/1/..%2F..%2Fmanifest.json", "/file/fx_a/1/manifest.json", "/file/nope/1/x", "/asset/nope"):
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(base + path, timeout=10)
        assert ei.value.code == 404, path


# ---------------------------------------------------------------- resource-aware queue
def res_job(tmp_path, name, provider="trellis2", **kw):
    d = {"name": name, "sketch": str(ROOT / "examples/sketches/hero_front.png"), "provider": provider,
         "provider_args": {"command": "true"} if provider != "procedural" else {}, **kw}
    p = tmp_path / f"{name}.yaml"; p.write_text(yaml.safe_dump(d)); return p


def test_claim_respects_resource_limits_without_blocking_other_work(lib, tmp_path):
    lib.enqueue(res_job(tmp_path, "g1"), 5); lib.enqueue(res_job(tmp_path, "g2"), 5)
    lib.enqueue(res_job(tmp_path, "cpu1", "procedural"), 0)
    assert [r["resource"] for r in lib.queue_rows()] == ["gpu", "gpu", None]
    first = lib.claim({"gpu": 1})
    assert first["name"] == "g1"
    second = lib.claim({"gpu": 1})
    assert second["name"] == "cpu1"            # the GPU slot is busy; a lower-priority CPU job goes ahead
    assert lib.claim({"gpu": 1}) is None       # g2 waits
    lib._set(first["id"], "done")
    assert lib.claim({"gpu": 1})["name"] == "g2"
    # no limits given -> no blocking
    lib.enqueue(res_job(tmp_path, "g3")); lib.enqueue(res_job(tmp_path, "g4"))
    assert {lib.claim()["name"], lib.claim()["name"]} == {"g3", "g4"}


def test_work_never_runs_more_gpu_jobs_than_the_limit(lib, tmp_path, monkeypatch):
    from charforge import orchestrator
    lock = threading.Lock()
    state = {"gpu": 0, "max_gpu": 0, "overlap_cpu": False, "cpu_running": 0}

    def fake_build(spec, out_root, backend, log=None):
        gpu = spec.resource() == "gpu"
        with lock:
            if gpu:
                state["gpu"] += 1; state["max_gpu"] = max(state["max_gpu"], state["gpu"])
                if state["cpu_running"]:
                    state["overlap_cpu"] = True
            else:
                state["cpu_running"] += 1
                if state["gpu"]:
                    state["overlap_cpu"] = True
        time.sleep(0.4)
        with lock:
            state["gpu" if gpu else "cpu_running"] -= 1
        return {"name": spec.name, "fingerprint": spec.fingerprint(), "status": "passed",
                "spec": {"kind": "character", "tags": []}, "outputs": [], "iterations": []}
    monkeypatch.setattr(orchestrator, "build_one", fake_build)
    for i in range(3):
        lib.enqueue(res_job(tmp_path, f"g{i}"))
    lib.enqueue(res_job(tmp_path, "cpu", "procedural"))
    assert work(lib, tmp_path / "out", BACKEND, workers=4, once=True, log=lambda s: None) == 4
    assert state["max_gpu"] == 1 and state["overlap_cpu"] is True
    assert len(lib.list()) == 4
    # raising the limit lets them overlap
    state.update(max_gpu=0)
    for i in range(3):
        lib.enqueue(res_job(tmp_path, f"h{i}"))
    work(lib, tmp_path / "out", BACKEND, workers=4, once=True, log=lambda s: None, limits={"gpu": 3})
    assert state["max_gpu"] == 3


def test_library_made_before_resource_tags_is_migrated(tmp_path):
    db = tmp_path / "old.db"
    c = sqlite3.connect(db)
    c.executescript("""CREATE TABLE queue (id INTEGER PRIMARY KEY AUTOINCREMENT, job_path TEXT NOT NULL, name TEXT NOT NULL,
      fingerprint TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'queued',
      attempts INTEGER NOT NULL DEFAULT 0, enqueued_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, asset_id INTEGER, error TEXT);
      INSERT INTO queue (job_path, name, fingerprint, enqueued_at) VALUES ('/x.yaml', 'old', 'f', '2026-01-01T00:00:00');""")
    c.commit(); c.close()
    lib = Library(db)
    assert lib.queue_rows()[0]["name"] == "old" and lib.queue_rows()[0]["resource"] is None
    lib.enqueue(res_job(tmp_path, "g1"))
    assert [r["resource"] for r in lib.queue_rows()] == [None, "gpu"] or [r["resource"] for r in lib.queue_rows()] == ["gpu", None]


def test_cli_limit_flag_is_validated(tmp_path, capsys):
    from charforge.cli import main
    assert main(["queue", "--db", str(tmp_path / "l.db"), "work", "-o", str(tmp_path), "--limit", "gpu=zero"]) == 1
    assert "bad --limit" in capsys.readouterr().out
