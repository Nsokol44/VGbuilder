from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import Backend, collect_jobs, run_jobs
from .spec import load_job


def _library(a) -> int:
    from .library import Library
    lib = Library(a.db)
    if a.lcmd == "list":
        for x in lib.list(status=a.status, tag=a.tag, review=a.review, latest_only=not a.all_versions):
            crit = x["critic"]["score"] if x["critic"] and x["critic"].get("score") is not None else "-"
            print(f"{x['name']:28s} v{x['version']:<3d} {x['kind']:9s} {x['status']:12s} {x['review']:9s} critic={crit}")
        return 0
    if a.lcmd == "show":
        x = lib.get(a.name, a.version)
        if x is None:
            print("not found"); return 1
        print(json.dumps({k: x[k] for k in ("name", "version", "kind", "status", "review", "review_note", "tags", "summary", "failures", "critic", "fingerprint")}, indent=2))
        return 0
    if a.lcmd == "review":
        x = lib.review(a.name, a.decision, a.note, a.version)
        print(f"{x['name']} v{x['version']}: {x['review']}")
        return 0
    if a.lcmd == "ingest":
        m = json.loads(a.manifest.read_text())
        r = lib.ingest(m, a.manifest.parent.parent)
        print(f"{r['name']} v{r['version']} ({'new' if r['new'] else 'already registered'})")
        return 0
    if a.lcmd == "critic":
        from .critic import run_critic
        r = run_critic(lib, a.name, command=a.command, version=a.version)
        print(json.dumps(r, indent=2))
        return 0 if "error" not in r else 2
    if a.lcmd == "silhouettes":
        import shutil
        a.dest.mkdir(parents=True, exist_ok=True)
        ps = lib.silhouette_paths(approved_only=a.approved_only)
        for p in ps:
            shutil.copy2(p, a.dest / p.name)
        print(f"copied {len(ps)} silhouette(s) to {a.dest}")
        return 0
    if a.lcmd == "serve":
        from .review_ui import serve
        srv = serve(lib, a.host, a.port)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    return 1


def _queue(a) -> int:
    from .library import Library, work
    lib = Library(a.db)
    if a.qcmd == "add":
        rc = 0
        for p in a.jobs:
            try:
                r = lib.enqueue(p, a.priority, a.force)
            except Exception as e:  # noqa: BLE001
                r = {"error": f"{type(e).__name__}: {e}"}; rc = 1
            print(f"{p}: {r}")
        return rc
    if a.qcmd == "list":
        for r in lib.queue_rows():
            print(f"#{r['id']:<4d} {r['name']:28s} prio={r['priority']:<3d} {r['state']:8s} tries={r['attempts']} {r['resource'] or '-':7s} {r['error'] or ''}")
        return 0
    if a.qcmd == "recover":
        print(f"re-queued {lib.recover(None if a.all else 3600.0)} job(s)")
        return 0
    backend = Backend(kind="blender" if a.blender else "bpy-module", exe=a.blender, timeout_s=a.timeout)
    limits = {}
    for item in a.limit:
        k, _, v = item.partition("=")
        if not k or not v.isdigit() or int(v) < 1:
            print(f"bad --limit {item!r}; expected RESOURCE=N with N >= 1"); return 1
        limits[k] = int(v)
    n = work(lib, a.out, backend, workers=a.workers, once=not a.watch, inbox=a.inbox, poll_s=a.poll, limits=limits)
    print(json.dumps({"processed": n}))
    return 0


def _live_dir(a) -> Path:
    job = load_job(a.job)
    return a.out / "_live" / job.name


def _parse_assignment(item: str):
    import yaml
    k, sep, v = item.partition("=")
    if not sep or not k:
        raise ValueError(f"expected PATH=VALUE, got {item!r}")
    return k.strip(), yaml.safe_load(v)


def _overrides(a) -> int:
    from . import live
    d = _live_dir(a)
    path = d / "live_overrides.json"
    try:
        seq, ov = live.read_overrides(path)
        if a.ocmd == "show":
            print(json.dumps({"seq": seq, "overrides": ov}, indent=2))
            return 0
        if a.ocmd == "set":
            d.mkdir(parents=True, exist_ok=True)
            ov = {**ov, **dict(_parse_assignment(x) for x in a.items)}
            load_job(a.job, ov)                        # refuse values that do not validate
            print(f"seq {live.write_overrides(path, ov)}")
            return 0
        if a.ocmd == "clear":
            d.mkdir(parents=True, exist_ok=True)
            print(f"seq {live.write_overrides(path, {})}")
            return 0
        if a.ocmd == "apply":
            diff = live.apply_to_job_file(a.job, ov, write=a.write)
            print(diff or "no change")
            if a.write and diff:
                live.write_overrides(path, {})
                print(f"written; old file kept as {a.job.name}.bak; overrides cleared. Comments in the YAML are not preserved.")
            elif diff:
                print("(dry run; add --write to replace the job file)")
            return 0
    except ValueError as e:
        print(f"error: {e}")
        return 1
    return 1


def _watch(a) -> int:
    import threading

    from .live import LiveSession
    backend = Backend(kind="blender" if a.blender else "bpy-module", exe=a.blender, timeout_s=a.timeout)
    sess = LiveSession(a.job, a.out, backend, fast=a.fast)
    print(f"watching {sess.job}\nsession folder: {sess.dir}\n(Ctrl-C to stop)", flush=True)
    if a.once:
        r = sess.build_now()
        print(json.dumps({"state": r["state"], "live_json": str(sess.status_path)}))
        return 0 if r["state"] == "built" else 2
    import signal
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())          # `kill` / the add-on's Stop button end it cleanly too
    try:
        sess.run(stop, poll_s=a.poll)
    except KeyboardInterrupt:
        stop.set()
    return 0


def _addon(a) -> int:
    from . import live  # noqa: F401
    src = Path(__file__).parent / "blender_addon" / "vgbuilder_live"
    if a.acmd == "path":
        print(src)
        return 0
    import zipfile
    dest = a.output or Path("vgbuilder_live.zip")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, Path("vgbuilder_live") / f.relative_to(src))
    print(f"wrote {dest}  (Blender: Edit > Preferences > Add-ons > Install from Disk)")
    return 0


def _blender(a) -> int:
    """Start the watcher and open Blender with the live add-on already connected to it."""
    import subprocess
    import threading

    from .live import LiveSession
    backend = Backend(kind="blender" if a.build_with_blender else "bpy-module", exe=a.build_with_blender, timeout_s=a.timeout)
    sess = LiveSession(a.job, a.out, backend, fast=a.fast)
    boot = Path(__file__).parent / "blender_addon" / "bootstrap.py"
    stop = threading.Event()
    t = threading.Thread(target=sess.run, args=(stop,), daemon=True)
    t.start()
    cmd = [a.exe, "-P", str(boot), "--", "--session", str(sess.dir)]
    print("starting:", " ".join(cmd), flush=True)
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print(f"could not start {a.exe!r}; pass --exe /path/to/blender")
        return 1
    finally:
        stop.set()
        t.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vgbuilder", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check job files without building")
    v.add_argument("jobs", type=Path, nargs="+")

    r = sub.add_parser("run", help="build one job file or a folder of them")
    r.add_argument("jobs", type=Path, help="job .yaml or a folder of them")
    r.add_argument("-o", "--out", type=Path, default=Path("out"))
    r.add_argument("-w", "--workers", type=int, default=1, help="parallel jobs (each is its own process)")
    r.add_argument("--blender", type=str, default=None, help="path to a Blender binary instead of the bpy module")
    r.add_argument("--timeout", type=int, default=1800)

    c = sub.add_parser("concepts", help="generate silhouette variants, score them, emit jobs for the diverse winners")
    c.add_argument("concept", type=Path, help="concept .yaml")
    c.add_argument("-o", "--out", type=Path, default=Path("out"))

    f = sub.add_parser("farm", help="expand a farm file into N validated variant jobs")
    f.add_argument("farm", type=Path)
    f.add_argument("-o", "--out", type=Path, default=Path("out"))
    f.add_argument("--enqueue", action="store_true", help="also add the variants to the queue")
    f.add_argument("--db", type=Path, default=None)

    lb = sub.add_parser("library", help="browse and review finished assets")
    lb.add_argument("--db", type=Path, default=Path("out/library.db"))
    ls = lb.add_subparsers(dest="lcmd", required=True)
    l1 = ls.add_parser("list")
    l1.add_argument("--status"); l1.add_argument("--tag"); l1.add_argument("--review"); l1.add_argument("--all-versions", action="store_true")
    l2 = ls.add_parser("show"); l2.add_argument("name"); l2.add_argument("--version", type=int)
    l3 = ls.add_parser("review"); l3.add_argument("name"); l3.add_argument("decision", choices=["approved", "rejected", "pending"])
    l3.add_argument("--note", default=""); l3.add_argument("--version", type=int)
    l4 = ls.add_parser("ingest", help="register an existing out/<name>/manifest.json"); l4.add_argument("manifest", type=Path)
    l5 = ls.add_parser("critic", help="run a critic command on an asset"); l5.add_argument("name")
    l5.add_argument("--command"); l5.add_argument("--version", type=int)
    l7 = ls.add_parser("silhouettes", help="copy library front silhouettes to a folder (use as a concept file's exclude_dir)")
    l7.add_argument("dest", type=Path); l7.add_argument("--approved-only", action="store_true")
    l6 = ls.add_parser("serve", help="review web UI (local)"); l6.add_argument("--host", default="127.0.0.1"); l6.add_argument("--port", type=int, default=8765)

    qu = sub.add_parser("queue", help="persistent priority queue")
    qu.add_argument("--db", type=Path, default=Path("out/library.db"))
    qs = qu.add_subparsers(dest="qcmd", required=True)
    q1 = qs.add_parser("add"); q1.add_argument("jobs", type=Path, nargs="+"); q1.add_argument("--priority", type=int); q1.add_argument("--force", action="store_true")
    qs.add_parser("list")
    q3 = qs.add_parser("recover", help="re-queue jobs left 'running' by a dead worker"); q3.add_argument("--all", action="store_true")
    q4 = qs.add_parser("work", help="process the queue")
    q4.add_argument("-o", "--out", type=Path, default=Path("out")); q4.add_argument("-w", "--workers", type=int, default=1)
    q4.add_argument("--inbox", type=Path); q4.add_argument("--watch", action="store_true", help="keep polling instead of draining once")
    q4.add_argument("--limit", action="append", default=[], metavar="RESOURCE=N",
                    help="max concurrent jobs per resource tag, e.g. gpu=1 (default gpu=1, remote=2); repeatable")
    q4.add_argument("--poll", type=float, default=2.0); q4.add_argument("--blender"); q4.add_argument("--timeout", type=int, default=1800)

    w = sub.add_parser("watch", help="live link: rebuild a job whenever its YAML or its live overrides change")
    w.add_argument("job", type=Path)
    w.add_argument("-o", "--out", type=Path, default=Path("out"))
    w.add_argument("--blender", type=str, default=None, help="build with this Blender binary instead of the bpy module")
    w.add_argument("--timeout", type=int, default=1800)
    w.add_argument("--poll", type=float, default=0.25)
    w.add_argument("--once", action="store_true", help="build once, write live.json, exit")
    w.add_argument("--fast", action="store_true", help="cheaper preview builds (LOD0 only, 256 px textures, one QA round, no animations); not the real output")

    ov = sub.add_parser("overrides", help="live overrides of a job (what the Blender sliders write)")
    ov.add_argument("-o", "--out", type=Path, default=Path("out"))
    os_ = ov.add_subparsers(dest="ocmd", required=True)
    for nm in ("show", "clear", "apply"):
        x = os_.add_parser(nm); x.add_argument("job", type=Path)
        if nm == "apply":
            x.add_argument("--write", action="store_true", help="replace the job file (a .bak is kept) and clear the overrides")
    x = os_.add_parser("set"); x.add_argument("job", type=Path); x.add_argument("items", nargs="+", metavar="PATH=VALUE")

    ad = sub.add_parser("addon", help="the Blender add-on")
    ads = ad.add_subparsers(dest="acmd", required=True)
    ads.add_parser("path", help="print the add-on folder")
    z = ads.add_parser("zip", help="make vgbuilder_live.zip for Install from Disk"); z.add_argument("-O", "--output", type=Path)

    bl = sub.add_parser("blender", help="start the watcher and open Blender with the add-on connected to it")
    bl.add_argument("job", type=Path)
    bl.add_argument("-o", "--out", type=Path, default=Path("out"))
    bl.add_argument("--exe", default="blender", help="the Blender program to open (the GUI)")
    bl.add_argument("--build-with-blender", default=None, metavar="EXE", help="build with a Blender binary instead of the bpy module")
    bl.add_argument("--timeout", type=int, default=1800)
    bl.add_argument("--fast", action="store_true", help="cheaper preview builds, see `watch --fast`")

    a = ap.parse_args(argv)
    if a.cmd == "watch":
        return _watch(a)
    if a.cmd == "overrides":
        return _overrides(a)
    if a.cmd == "addon":
        return _addon(a)
    if a.cmd == "blender":
        return _blender(a)
    if a.cmd == "farm":
        from .farm import make_variants
        paths = make_variants(a.farm, a.out / "farm" / a.farm.stem)
        res = {"jobs": [str(p) for p in paths]}
        if a.enqueue:
            from .library import Library
            lib = Library(a.db or a.out / "library.db")
            res["enqueued"] = [lib.enqueue(p) for p in paths]
        print(json.dumps(res, indent=2))
        return 0
    if a.cmd == "library":
        return _library(a)
    if a.cmd == "queue":
        return _queue(a)
    if a.cmd == "concepts":
        from .concepts import run_concepts
        m = run_concepts(a.concept, a.out)
        print(json.dumps({"selected": m["selected"], "jobs": m["jobs"], "sheet": m["sheet"]}, indent=2))
        return 0 if m["selected"] else 2
    if a.cmd == "validate":
        for p in a.jobs:
            s = load_job(p)
            print(f"ok  {p}  ({s.name}: {s.provider}, {s.style}, budget {s.resolved_budget()} tris)")
        return 0

    backend = Backend(kind="blender" if a.blender else "bpy-module", exe=a.blender, timeout_s=a.timeout)
    results = run_jobs(collect_jobs(a.jobs), a.out, backend, workers=a.workers)
    for m in results:
        fails = [f["code"] for f in m.get("final_failures", [])]
        print(f"{m['status']:13s} {m['name']:20s} iterations={len(m['iterations'])} "
              f"{'failures=' + ','.join(fails) if fails else ''} {m.get('error', '')}")
    print(json.dumps({"out": str(a.out), "jobs": len(results)}))
    return 0 if all(m["status"] == "passed" for m in results) else 2


if __name__ == "__main__":
    sys.exit(main())
