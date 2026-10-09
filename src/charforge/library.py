"""Asset library, persistent job queue and watched inbox. Standard library only (sqlite3).

* Library: every finished build becomes a versioned asset row (spec, fingerprint, status, QA summary,
  tags, paths to the files and the manifest). Same name + new fingerprint = next version; same
  fingerprint = nothing to do. Review decisions (approve / reject + note) and critic scores live here.
* Queue: jobs wait in SQLite with a priority, survive restarts (jobs that were `running` when a worker died
  are put back), are de-duplicated by fingerprint, and are retried when the *infrastructure* failed
  (a build that ends `needs_review` is a result, not a failure).
* Inbox: drop job .yaml files into a folder; `scan_inbox` enqueues new or edited ones.

Paths are stored relative to the database's folder when possible so a library can be moved.
"""
from __future__ import annotations

import calendar
import contextlib
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .spec import JobSpec, load_job

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL, version INTEGER NOT NULL, kind TEXT NOT NULL,
  status TEXT NOT NULL, review TEXT NOT NULL DEFAULT 'pending', review_note TEXT DEFAULT '',
  fingerprint TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]', style_template TEXT,
  parent TEXT, spec TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '{}', failures TEXT NOT NULL DEFAULT '[]',
  manifest_path TEXT, final_dir TEXT, thumb TEXT, silhouette TEXT, critic TEXT,
  charforge_version TEXT, created_at TEXT NOT NULL, UNIQUE(name, version)
);
CREATE TABLE IF NOT EXISTS asset_files (
  asset_id INTEGER NOT NULL REFERENCES assets(id), file TEXT NOT NULL, sha256 TEXT, bytes INTEGER,
  PRIMARY KEY (asset_id, file)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id INTEGER, ts TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT
);
CREATE TABLE IF NOT EXISTS queue (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_path TEXT NOT NULL, name TEXT NOT NULL, fingerprint TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0, resource TEXT, state TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
  enqueued_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, asset_id INTEGER, error TEXT
);
CREATE INDEX IF NOT EXISTS queue_pick ON queue(state, priority DESC, id);
CREATE TABLE IF NOT EXISTS inbox_seen (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, queue_id INTEGER);
"""
MAX_ATTEMPTS = 3
# Jobs that carry a resource tag (spec.resource(): gpu / remote / ...) run at most this many at a time per worker pool.
DEFAULT_LIMITS = {"gpu": 1, "remote": 2}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _kind(manifest: dict[str, Any]) -> str:
    return manifest.get("kind") or manifest.get("spec", {}).get("kind") or "character"


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    """The few numbers worth listing next to an asset (full detail stays in the manifest)."""
    s: dict[str, Any] = {"iterations": len(manifest.get("iterations", [])), "seconds": manifest.get("seconds")}
    bi = manifest.get("best_iteration")
    if not bi:
        return s
    m = manifest["iterations"][bi - 1].get("metrics", {})
    if _kind(manifest) == "effect":
        s.update({k: m[k] for k in ("frames", "peak_fraction", "max_coverage", "border_alpha_max") if k in m})
        return s
    if _kind(manifest) == "structure":
        st = m.get("structure", {})
        s.update({"style": st.get("style"), "lod_tris": m.get("lod_tris"), "height_m": round(m["height_m"], 2),
                  "footprint_m": [round(x, 2) for x in m["bbox"]["size"][:2]], "windows": st.get("windows"),
                  "tris_per_m2": round(m["lod_tris"][0] / m["expected"]["envelope_area"], 2),
                  "collision_tris": m.get("collision", {}).get("tris")})
        tex = (m.get("texture") or {}).get("analysis") or {}
        if "soil_fraction" in tex:
            s["soil_fraction"] = round(tex["soil_fraction"], 3)
        return s
    for k in ("silhouette_iou", "silhouette_side_iou", "height_m", "max_influences"):
        if k in m:
            s[k] = round(m[k], 4) if isinstance(m[k], float) else m[k]
    if "lod_tris" in m:
        s["lod_tris"] = m["lod_tris"]
    tex = (m.get("texture") or {}).get("analysis") or {}
    for k in ("soil_fraction", "rot_fraction", "clean_fraction", "mean_saturation"):
        if k in tex:
            s[k] = round(tex[k], 3)
    if m.get("animations"):
        s["animations"] = sorted(m["animations"])
    if m.get("accessories"):
        s["accessories"] = sorted(k for k in m["accessories"] if not k.startswith("_"))
    return s


class Library:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)
            if "resource" not in {r["name"] for r in c.execute("PRAGMA table_info(queue)")}:   # library made before resource tags
                c.execute("ALTER TABLE queue ADD COLUMN resource TEXT")

    @contextlib.contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)   # autocommit; we open transactions explicitly
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
        finally:
            c.close()

    # ---- paths
    def _store(self, p: str | Path | None) -> str | None:
        if p is None:
            return None
        p = Path(p).resolve()
        try:
            return os.path.relpath(p, self.path.parent)
        except ValueError:
            return str(p)

    def resolve(self, p: str | None) -> Path | None:
        if not p:
            return None
        q = Path(p)
        return q if q.is_absolute() else (self.path.parent / q).resolve()

    # ---- assets
    def ingest(self, manifest: dict[str, Any], out_root: Path, spec: JobSpec | None = None) -> dict[str, Any]:
        """Register a finished build. Idempotent per (name, fingerprint)."""
        name, fp = manifest["name"], manifest["fingerprint"]
        with self.conn() as c:
            row = c.execute("SELECT * FROM assets WHERE name=? AND fingerprint=? AND (status!='error' OR ?='error')",
                            (name, fp, manifest["status"])).fetchone()
            if row:
                return {**dict(row), "new": False}
            c.execute("BEGIN IMMEDIATE")
            ver = (c.execute("SELECT MAX(version) FROM assets WHERE name=?", (name,)).fetchone()[0] or 0) + 1
            spec_d = manifest.get("spec", {})
            tags = list(spec_d.get("tags", []))
            parent = next((t.split(":", 1)[1] for t in tags if t.startswith("variant_of:")), None)
            final = Path(out_root) / name / "final"
            files = manifest.get("outputs", [])
            by = {f["file"]: f for f in files}
            thumb = next((f for f in by if f.endswith("_contact_sheet.png")), None)
            sil = next((f for f in by if f.endswith("_silhouette_front.png")), None)
            cur = c.execute(
                "INSERT INTO assets (name, version, kind, status, fingerprint, tags, style_template, parent, spec, summary,"
                " failures, manifest_path, final_dir, thumb, silhouette, charforge_version, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (name, ver, _kind(manifest), manifest["status"], fp, json.dumps(tags),
                 spec_d.get("style_template"), parent, json.dumps(spec_d, default=str), json.dumps(summarize(manifest)),
                 json.dumps(manifest.get("final_failures", [])), self._store(Path(out_root) / name / "manifest.json"),
                 self._store(final), thumb, sil, manifest.get("charforge_version"), _now()))
            aid = cur.lastrowid
            for f in files:
                c.execute("INSERT INTO asset_files VALUES (?,?,?,?)", (aid, f["file"], f.get("sha256"), f.get("bytes")))
            c.execute("INSERT INTO events (asset_id, ts, kind, detail) VALUES (?,?,?,?)",
                      (aid, _now(), "ingest", f"v{ver} {manifest['status']}"))
            c.execute("COMMIT")
        return {**self.get(name, ver), "new": True}

    def get(self, name: str, version: int | None = None) -> dict[str, Any] | None:
        with self.conn() as c:
            q = "SELECT * FROM assets WHERE name=?" + (" AND version=?" if version else " ORDER BY version DESC LIMIT 1")
            row = c.execute(q, (name, version) if version else (name,)).fetchone()
            if not row:
                return None
            d = dict(row)
            for k in ("tags", "spec", "summary", "failures"):
                d[k] = json.loads(d[k])
            d["critic"] = json.loads(d["critic"]) if d["critic"] else None
            d["files"] = [dict(r) for r in c.execute("SELECT file, sha256, bytes FROM asset_files WHERE asset_id=?", (d["id"],))]
            d["events"] = [dict(r) for r in c.execute("SELECT ts, kind, detail FROM events WHERE asset_id=? ORDER BY id", (d["id"],))]
            return d

    def list(self, status: str | None = None, tag: str | None = None, review: str | None = None,
             name: str | None = None, latest_only: bool = True, kind: str | None = None) -> list[dict[str, Any]]:
        where, args = [], []
        for col, val in (("status", status), ("review", review), ("name", name), ("kind", kind)):
            if val:
                where.append(f"{col}=?"); args.append(val)
        if latest_only:
            where.append("version = (SELECT MAX(version) FROM assets b WHERE b.name = assets.name)")
        sql = "SELECT * FROM assets" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY name, version"
        with self.conn() as c:
            rows = [dict(r) for r in c.execute(sql, args)]
        out = []
        for d in rows:
            d["tags"] = json.loads(d["tags"]); d["summary"] = json.loads(d["summary"]); d["failures"] = json.loads(d["failures"])
            d["critic"] = json.loads(d["critic"]) if d["critic"] else None
            if tag and tag not in d["tags"]:
                continue
            out.append(d)
        return out

    def review(self, name: str, decision: str, note: str = "", version: int | None = None) -> dict[str, Any]:
        if decision not in ("approved", "rejected", "pending"):
            raise ValueError("decision must be approved, rejected or pending")
        a = self.get(name, version)
        if a is None:
            raise KeyError(f"no asset {name!r}")
        with self.conn() as c:
            c.execute("UPDATE assets SET review=?, review_note=? WHERE id=?", (decision, note, a["id"]))
            c.execute("INSERT INTO events (asset_id, ts, kind, detail) VALUES (?,?,?,?)",
                      (a["id"], _now(), f"review:{decision}", note))
        return self.get(name, a["version"])

    def set_critic(self, name: str, result: dict[str, Any], version: int | None = None) -> None:
        a = self.get(name, version)
        with self.conn() as c:
            c.execute("UPDATE assets SET critic=? WHERE id=?", (json.dumps(result), a["id"]))
            c.execute("INSERT INTO events (asset_id, ts, kind, detail) VALUES (?,?,?,?)",
                      (a["id"], _now(), "critic", f"score={result.get('score')}"))

    def silhouette_paths(self, approved_only: bool = False) -> list[Path]:
        """Front silhouettes of library assets: feed to concepts' `exclude_dir`-style uniqueness checks."""
        out = []
        for a in self.list(review="approved" if approved_only else None):
            if a["silhouette"] and a["final_dir"]:
                p = self.resolve(a["final_dir"]) / a["silhouette"]
                if p.exists():
                    out.append(p)
        return out

    # ---- queue
    def enqueue(self, job_path: str | Path, priority: int | None = None, force: bool = False) -> dict[str, Any]:
        """Queue a job file. Returns {"queued": id} or {"skipped": reason}. Validates the job first."""
        job_path = Path(job_path).resolve()
        spec = load_job(job_path)
        fp = spec.fingerprint()
        prio = spec.priority if priority is None else int(priority)
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            if not force:
                if c.execute("SELECT 1 FROM assets WHERE name=? AND fingerprint=?", (spec.name, fp)).fetchone():
                    c.execute("COMMIT")
                    return {"skipped": "already built (same fingerprint)", "name": spec.name}
                dup = c.execute("SELECT id FROM queue WHERE fingerprint=? AND state IN ('queued','running')", (fp,)).fetchone()
                if dup:
                    c.execute("COMMIT")
                    return {"skipped": f"already in the queue (#{dup['id']})", "name": spec.name}
            cur = c.execute("INSERT INTO queue (job_path, name, fingerprint, priority, resource, enqueued_at) VALUES (?,?,?,?,?,?)",
                            (str(job_path), spec.name, fp, prio, spec.resource(), _now()))
            c.execute("COMMIT")
        return {"queued": cur.lastrowid, "name": spec.name, "priority": prio}

    def claim(self, limits: dict[str, int] | None = None) -> dict[str, Any] | None:
        """Atomically take the highest-priority queued job (oldest first among equals).

        A job whose resource tag is already at its limit is skipped, not blocked on: a lower-priority CPU job
        may run while the single GPU slot is busy, and the GPU job keeps its place in line."""
        limits = limits or {}
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            busy: dict[str, int] = {}
            for r in c.execute("SELECT resource, COUNT(*) n FROM queue WHERE state='running' AND resource IS NOT NULL GROUP BY resource"):
                busy[r["resource"]] = r["n"]
            for row in c.execute("SELECT * FROM queue WHERE state='queued' ORDER BY priority DESC, id").fetchall():
                res = row["resource"]
                if res is not None and res in limits and busy.get(res, 0) >= limits[res]:
                    continue
                c.execute("UPDATE queue SET state='running', started_at=?, attempts=attempts+1 WHERE id=?", (_now(), row["id"]))
                c.execute("COMMIT")
                return {**dict(row), "state": "running", "attempts": row["attempts"] + 1}
            c.execute("COMMIT")
            return None

    def finish(self, qid: int, manifest: dict[str, Any] | None, out_root: Path, error: str | None = None) -> str:
        """Record the outcome. Infrastructure errors are retried up to MAX_ATTEMPTS; QA outcomes are final."""
        with self.conn() as c:
            row = c.execute("SELECT * FROM queue WHERE id=?", (qid,)).fetchone()
        if manifest is not None and manifest.get("status") in ("passed", "needs_review"):
            asset = self.ingest(manifest, out_root)
            self._set(qid, "done", asset_id=asset["id"])
            return "done"
        err = error or (manifest or {}).get("error") or "unknown error"
        retryable = (manifest or {}).get("error_retryable", True)    # structured provider errors say; unknown errors retry
        state = "queued" if (row["attempts"] < MAX_ATTEMPTS and retryable) else "failed"
        if state == "failed" and manifest is not None and manifest.get("status") == "error":
            self.ingest(manifest, out_root)          # only the final failure is shown in the library
        self._set(qid, state, error=err)
        return state

    def _set(self, qid: int, state: str, asset_id: int | None = None, error: str | None = None) -> None:
        with self.conn() as c:
            c.execute("UPDATE queue SET state=?, finished_at=?, asset_id=COALESCE(?, asset_id), error=? WHERE id=?",
                      (state, _now() if state in ("done", "failed") else None, asset_id, error, qid))

    def recover(self, stale_after_s: float | None = 3600.0) -> int:
        """Put `running` jobs back in the queue (a worker died). stale_after_s=None means all of them."""
        with self.conn() as c:
            rows = c.execute("SELECT id, started_at FROM queue WHERE state='running'").fetchall()
            n = 0
            for r in rows:
                age = time.time() - calendar.timegm(time.strptime(r["started_at"], "%Y-%m-%dT%H:%M:%S"))
                if stale_after_s is None or age > stale_after_s:
                    c.execute("UPDATE queue SET state='queued', error='recovered after interrupted run' WHERE id=?", (r["id"],))
                    n += 1
        return n

    def queue_rows(self, state: str | None = None) -> list[dict[str, Any]]:
        with self.conn() as c:
            q = "SELECT * FROM queue" + (" WHERE state=?" if state else "") + " ORDER BY priority DESC, id"
            return [dict(r) for r in c.execute(q, (state,) if state else ())]

    # ---- inbox
    def scan_inbox(self, folder: str | Path) -> list[dict[str, Any]]:
        """Enqueue job files that are new or have changed since the last scan. Invalid files are reported, not queued."""
        results = []
        for p in sorted([*Path(folder).glob("*.yaml"), *Path(folder).glob("*.yml")]):
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            with self.conn() as c:
                seen = c.execute("SELECT sha256 FROM inbox_seen WHERE path=?", (str(p.resolve()),)).fetchone()
            if seen and seen["sha256"] == sha:
                continue
            try:
                r = self.enqueue(p)
            except Exception as e:  # noqa: BLE001 - one bad file must not stop the scan
                r = {"error": f"{type(e).__name__}: {e}"}
            r["file"] = str(p)
            with self.conn() as c:
                c.execute("INSERT OR REPLACE INTO inbox_seen VALUES (?,?,?)", (str(p.resolve()), sha, r.get("queued")))
            results.append(r)
        return results


def work(lib: Library, out_root: Path, backend, workers: int = 1, once: bool = True, inbox: Path | None = None,
         poll_s: float = 2.0, log: Callable[[str], None] = print, stop: Callable[[], bool] = lambda: False,
         limits: dict[str, int] | None = None) -> int:
    """Process the queue; returns how many job attempts were run.

    once=True drains the queue and returns. It assumes it is the only worker: every job still marked `running`
    is put back first (use `queue recover` / a long-running `--watch` worker if several processes share a library).
    Otherwise it keeps polling (and scanning the inbox) until `stop()`. `limits` caps concurrent jobs per resource
    tag (default: one gpu job at a time)."""
    from concurrent.futures import FIRST_COMPLETED, wait

    from .orchestrator import build_one

    limits = {**DEFAULT_LIMITS, **(limits or {})}

    def run_job(job: dict[str, Any]) -> None:
        log(f"[queue] #{job['id']} {job['name']} (priority {job['priority']}, attempt {job['attempts']}"
            + (f", {job['resource']}" if job.get("resource") else "") + ")")
        try:
            spec = load_job(job["job_path"])
            manifest = build_one(spec, out_root, backend, log)
            state = lib.finish(job["id"], manifest, out_root)
            if state == "done" and spec.critic:
                from .critic import run_critic
                r = run_critic(lib, spec.name, min_score=spec.critic.get("min_score"))
                log(f"[critic] {spec.name}: " + (r.get("error") or f"score {r['score']:.2f}"))
        except Exception as e:  # noqa: BLE001
            state = lib.finish(job["id"], None, out_root, error=f"{type(e).__name__}: {e}")
        log(f"[queue] #{job['id']} {job['name']} -> {state}")

    done = 0
    active: set = set()
    lib.recover(stale_after_s=None if once else 3600.0)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        while not stop():
            if inbox is not None:
                for r in lib.scan_inbox(inbox):
                    log(f"[inbox] {r}")
            while len(active) < max(1, workers):
                job = lib.claim(limits)
                if job is None:
                    break
                active.add(ex.submit(run_job, job))
                done += 1
            if not active:
                if once:
                    break
                time.sleep(poll_s)
                continue
            finished, active = wait(active, timeout=None if once else poll_s, return_when=FIRST_COMPLETED)
            for f in finished:
                f.result()
    return done
