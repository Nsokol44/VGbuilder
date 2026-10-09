"""Live link: keep a job built while it is being edited, from a text editor or from inside a running Blender.

charforge never talks to Blender directly. The two sides share a folder and two small JSON files:

    <out>/_live/<name>/live_overrides.json   written by the editor side (the Blender add-on, a script, by hand)
                                             {"seq": 7, "overrides": {"structure.floors": 4, "params.leg_frac": 0.5}}
    <out>/_live/<name>/live.json             written by `charforge watch`: state, build id, failures, the files of the
                                             latest build, and the controls (sliders) this job offers
    <out>/_live/<name>/published/<build>/    an immutable copy of each finished build (so a reader never sees a
                                             half-written file); the last few are kept
    <out>/_live/<name>/live.heartbeat        touched about once a second while the watcher is alive

The build itself is the ordinary one: the same `build_one` that `charforge run` uses, on the job file with the
overrides laid over it before validation. With no overrides the glb files are byte-identical to `charforge run`
(live builds go to `_live/` so they never overwrite the normal output). Overrides live beside the job, not in it:
the YAML stays the source of truth until you apply them (`charforge overrides apply`).

Overrides are limited to the knobs of the design (see ALLOWED_ROOTS). They can never set a provider, a provider
command, a critic command or a file path, because anything that can write the overrides file would otherwise be able to
run programs.

Only the standard library is needed to read and write these files, which is what the Blender add-on does.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

import yaml

PROTOCOL = 1
KEEP_BUILDS = 3
ALLOWED_ROOTS = {
    "params", "structure", "effect", "accessory_shape", "wear", "style_overrides", "style_template", "height_m",
    "seed", "poly_budget", "texture_size", "texture", "accessories", "animations", "lods", "qa", "fit_prior",
    "target_engine", "style", "max_iterations",
}
_SEG = re.compile(r"^[A-Za-z0-9_-]+$")


class OverrideError(ValueError):
    pass


# ---------------------------------------------------------------- overrides
def _check_value(path: str, v: Any) -> None:
    if v is None or isinstance(v, (bool, int, str)):
        return
    if isinstance(v, float):
        if not math.isfinite(v):
            raise OverrideError(f"{path}: value must be finite")
        return
    if isinstance(v, list):
        for x in v:
            _check_value(path, x)
        return
    if isinstance(v, dict):
        for k, x in v.items():
            if not isinstance(k, str):
                raise OverrideError(f"{path}: keys must be strings")
            _check_value(path, x)
        return
    raise OverrideError(f"{path}: unsupported value type {type(v).__name__}")


def check_overrides(overrides: Any) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise OverrideError("overrides must be an object of {dotted.path: value}")
    for path, v in overrides.items():
        if not isinstance(path, str) or not path:
            raise OverrideError("override paths must be non-empty strings")
        segs = path.split(".")
        if not all(_SEG.match(s) for s in segs):
            raise OverrideError(f"bad override path {path!r}")
        if segs[0] not in ALLOWED_ROOTS:
            raise OverrideError(f"{path!r}: {segs[0]!r} cannot be overridden live (allowed: {sorted(ALLOWED_ROOTS)})")
        _check_value(path, v)
    return overrides


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """A copy of the raw job mapping with {"a.b.0.c": value} laid over it. Lists are edited by index (no append)."""
    check_overrides(overrides)
    out = copy.deepcopy(raw)
    for path in sorted(overrides):
        segs = path.split(".")
        cur: Any = out
        for i, s in enumerate(segs[:-1]):
            nxt_is_index = segs[i + 1].isdigit()
            if isinstance(cur, dict):
                child = cur.get(s)
                if child is None:
                    if nxt_is_index:
                        raise OverrideError(f"{path}: {'.'.join(segs[:i + 1])} is not set in the job, so index {segs[i + 1]} does not exist")
                    child = cur[s] = {}
                cur = child
            elif isinstance(cur, list):
                cur = cur[_index(path, cur, s)]
            else:
                raise OverrideError(f"{path}: cannot descend into {'.'.join(segs[:i])}")
        last = segs[-1]
        if isinstance(cur, dict):
            cur[last] = overrides[path]
        elif isinstance(cur, list):
            cur[_index(path, cur, last)] = overrides[path]
        else:
            raise OverrideError(f"{path}: parent is not a mapping or list")
    return out


def _index(path: str, lst: list, s: str) -> int:
    if not s.isdigit() or int(s) >= len(lst):
        raise OverrideError(f"{path}: {s!r} is not an index into a list of {len(lst)}")
    return int(s)


def read_overrides(path: Path) -> tuple[int, dict[str, Any]]:
    """(seq, overrides) from a live_overrides.json; (0, {}) when the file does not exist. Raises OverrideError."""
    try:
        txt = Path(path).read_text()
    except FileNotFoundError:
        return 0, {}
    try:
        d = json.loads(txt)
    except json.JSONDecodeError as e:
        raise OverrideError(f"{Path(path).name} is not valid JSON: {e}") from None
    if not isinstance(d, dict) or not isinstance(d.get("seq", 0), int) or isinstance(d.get("seq", 0), bool):
        raise OverrideError(f"{Path(path).name}: expected {{\"seq\": int, \"overrides\": {{...}}}}")
    return int(d.get("seq", 0)), check_overrides(d.get("overrides", {}))


def write_json_atomic(path: Path, data: Any) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str, sort_keys=False))
    os.replace(tmp, path)


def write_overrides(path: Path, overrides: dict[str, Any], seq: int | None = None) -> int:
    """What the editor side does (the add-on carries its own copy of this): atomic, with a rising seq."""
    check_overrides(overrides)
    path = Path(path)
    if seq is None:
        try:
            seq = read_overrides(path)[0] + 1
        except OverrideError:
            seq = int(time.time())
    write_json_atomic(path, {"seq": seq, "overrides": overrides})
    return seq


# ---------------------------------------------------------------- controls (the sliders)
def _step(lo: float, hi: float) -> float:
    span = hi - lo
    for s in (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0):
        if span / s <= 400:
            return s
    return 1.0


def _num(path: str, label: str, group: str, lo: float, hi: float, value: Any, default: Any = None, integer: bool = False) -> dict[str, Any]:
    return {"path": path, "label": label, "group": group, "type": "int" if integer else "float",
            "min": lo, "max": hi, "step": 1 if integer else _step(lo, hi), "value": value,
            "default": value if default is None else default}


def _choice(path: str, label: str, group: str, choices: list[str], value: Any) -> dict[str, Any]:
    return {"path": path, "label": label, "group": group, "type": "choice", "choices": sorted(choices), "value": value, "default": value}


def _bool(path: str, label: str, group: str, value: bool) -> dict[str, Any]:
    return {"path": path, "label": label, "group": group, "type": "bool", "value": bool(value), "default": bool(value)}


def controls(spec, manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The sliders / toggles / dropdowns this job offers, with the value in effect now.

    For a character, a slider that has never been touched shows the value of the last build (the fitted one when the
    body came from a sketch). Moving it writes `params.<name>`, which PINS that parameter: the sketch fit and the
    auto-fixer will no longer change it."""
    out: list[dict[str, Any]] = []
    if spec.kind == "structure":
        from . import structuregen as sg
        d = sg.resolve(spec.structure)
        out.append(_choice("structure.style", "Style", "Building", sorted(sg.STYLES), d["style"]))
        out.append(_choice("structure.roof", "Roof", "Building", sorted(sg.ROOFS), d["roof"]))
        out.append(_choice("structure.parapet", "Parapet", "Building", sorted(sg.PARAPETS), d["parapet"]))
        for i, lab in enumerate(("Width", "Depth")):
            out.append(_num(f"structure.footprint.{i}", f"Footprint {lab} (m)", "Building", 3.0, 60.0, float(d["footprint"][i])))
        labels = {"floors": "Floors", "floor_height": "Floor height (m)", "bay_width": "Bay width (m)", "roof_pitch_deg": "Roof pitch (deg)",
                  "overhang": "Roof overhang (m)", "wall_t": "Wall thickness (m)", "window_w": "Window width (m)", "window_h": "Window height (m)",
                  "sill_h": "Sill height (m)", "door_w": "Door width (m)", "door_h": "Door height (m)", "plinth_h": "Plinth height (m)", "damage": "Damage"}
        for k, (lo, hi) in sg.RANGES.items():
            grp = "Damage" if k == "damage" else ("Openings" if k.startswith(("window", "sill", "door")) else "Building")
            out.append(_num(f"structure.{k}", labels[k], grp, lo, hi, d[k], integer=(k == "floors")))
        for k, lab in (("windows", "Windows"), ("buttresses", "Buttresses"), ("string_courses", "String courses")):
            out.append(_bool(f"structure.{k}", lab, "Trim", d[k]))
        out.append(_num("seed", "Seed", "Damage", 0, 9999, spec.seed, integer=True))
    elif spec.kind == "effect":
        from .effects import LIMITS, TEMPLATES
        tpl = spec.effect.get("template", "fireball")
        out.append(_choice("effect.template", "Template", "Effect", sorted(TEMPLATES), tpl))
        for k, (lo, hi) in LIMITS.items():
            out.append(_num(f"effect.{k}", k.replace("_", " ").capitalize(), "Effect", lo, hi,
                            float(spec.effect.get(k, TEMPLATES[tpl].get(k, lo)))))
        out.append(_num("seed", "Seed", "Effect", 0, 9999, spec.seed, integer=True))
    else:
        from .accessories import SHAPE_DEFAULTS, SHAPE_LIMITS
        from .fit import BOUNDS
        from .spec import DEFAULT_BODY_PARAMS
        fitted = {}
        if manifest and manifest.get("iterations"):
            bi = manifest.get("best_iteration") or len(manifest["iterations"])
            fitted = manifest["iterations"][min(bi, len(manifest["iterations"])) - 1].get("params", {})
        for k, (lo, hi) in BOUNDS.items():
            v = spec.params.get(k, fitted.get(k, DEFAULT_BODY_PARAMS[k]))
            out.append(_num(f"params.{k}", k.replace("_", " ").capitalize(), "Body", lo, hi, float(v), float(DEFAULT_BODY_PARAMS[k])))
        out.append(_num("height_m", "Height (m)", "Body", 0.5, 3.0, float(spec.height_m)))
        for acc, keys in (("pauldrons", ["pauldron_scale"]), ("horns", ["horn_len", "horn_spread"]), ("cape", ["cape_len", "cape_flare"])):
            if acc in spec.accessories:
                for k in keys:
                    lo, hi = SHAPE_LIMITS[k]
                    out.append(_num(f"accessory_shape.{k}", k.replace("_", " ").capitalize(), "Accessories", lo, hi,
                                    float(spec.accessory_shape.get(k, SHAPE_DEFAULTS[k]))))
        out.append(_num("seed", "Seed", "Body", 0, 9999, spec.seed, integer=True))
    if spec.kind != "effect" and spec.style_template:
        st = spec.load_style()
        for k in ("intensity", "edge_wear", "lower_dirt", "rust", "rot", "blood"):
            out.append(_num(f"wear.{k}", k.replace("_", " ").capitalize(), "Look", 0.0, 1.0, float(st.wear.get(k, 0.0))))
        out.append(_num("wear.noise_scale", "Noise scale", "Look", 1.0, 20.0, float(st.wear.get("noise_scale", 6.0))))
    return out


# ---------------------------------------------------------------- the session
def _kind_of(name: str) -> dict[str, Any]:
    m = re.search(r"_LOD(\d+)\.glb$", name)
    if m:
        return {"role": "lod", "lod": int(m.group(1))}
    if name.endswith("_collision.glb"):
        return {"role": "collision"}
    if name.endswith(".blend"):
        return {"role": "blend"}
    if name.endswith("contact_sheet.png"):
        return {"role": "contact_sheet"}
    if name.endswith(".glb"):
        return {"role": "glb"}
    if name.endswith((".png", ".json")) and ("flipbook" in name or "sheet" in name or "particles" in name):
        return {"role": "effect"}
    return {"role": "other"}


# `charforge watch --fast`: a cheaper build for dragging sliders (LOD0 only, small textures, one QA round, no
# animations). It is a different build than the real one: its files are NOT what `charforge run` would make, and
# live.json says "fast": true. Drop --fast (or apply the overrides and run normally) for the real output.
FAST = {"lods": [1.0], "texture_size": 256, "max_iterations": 1}
FAST_CHARACTER = {"animations": []}


class LiveSession:
    """Watches one job file and its overrides; rebuilds when either changes; publishes live.json."""

    def __init__(self, job: str | Path, out_root: str | Path, backend, log: Callable[[str], None] = print, fast: bool = False):
        from .spec import load_job
        self.job = Path(job).resolve()
        self.backend, self.log, self.fast = backend, log, fast
        first = load_job(self.job)                      # raises with the real message if the job is broken at start
        self.name, self.kind = first.name, first.kind
        self.root = Path(out_root) / "_live"
        self.dir = self.root / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.overrides_path = self.dir / "live_overrides.json"
        self.status_path = self.dir / "live.json"
        self.heartbeat_path = self.dir / "live.heartbeat"
        (self.dir / "inbox").mkdir(exist_ok=True)
        self.build_id = self._previous_build_id()       # ids keep rising across watcher restarts (the add-on compares them)
        self._built_stamp: str | None = None
        self._watch_files: list[Path] = []
        self._controls: list[dict[str, Any]] = []
        self._files: list[dict[str, Any]] = []
        self._last_manifest_summary: dict[str, Any] = {}
        self._applied: dict[str, Any] = {}
        self._seq_applied = 0
        self._publish(state="idle", message="watching")

    def _previous_build_id(self) -> int:
        best = 0
        try:
            best = int(json.loads(self.status_path.read_text()).get("build_id", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        pub = self.dir / "published"
        if pub.is_dir():
            best = max([best] + [int(p.name) for p in pub.iterdir() if p.name.isdigit()])
        return best

    # -- change detection
    def stamp(self) -> str:
        h = hashlib.sha256()
        for p in (self.job, self.overrides_path):
            try:
                h.update(p.read_bytes())
            except FileNotFoundError:
                h.update(b"-")
        for p in self._watch_files:
            try:
                st = p.stat()
                h.update(f"{p}:{st.st_mtime_ns}:{st.st_size}".encode())
            except FileNotFoundError:
                h.update(f"{p}:-".encode())
        return h.hexdigest()

    def _fast_overrides(self) -> dict[str, Any]:
        if self.kind == "effect":
            return {}
        return {**FAST, **(FAST_CHARACTER if self.kind == "character" else {})}

    def _refresh_watch_files(self, spec) -> None:
        files = [p for p in (spec.resolve_path(spec.sketch), spec.resolve_path(spec.sketch_side)) if p is not None]
        for d in spec.style_dirs():
            if d.is_dir():
                files += sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml"))
        self._watch_files = files

    # -- one build cycle
    def build_now(self) -> dict[str, Any]:
        """Read job + overrides, validate, build, publish. Never raises: problems become live.json state 'invalid'/'error'."""
        from .orchestrator import build_one
        from .spec import load_job
        stamp = self.stamp()
        try:
            seq, ov = read_overrides(self.overrides_path)
            spec = load_job(self.job, {**self._fast_overrides(), **ov} if self.fast else ov)
        except Exception as e:  # noqa: BLE001 - whatever is wrong with the YAML or the overrides is shown, not raised
            self._built_stamp = stamp
            self.log(f"[live] invalid: {e}")
            self._publish(state="invalid", error=f"{type(e).__name__}: {e}")
            return {"state": "invalid"}
        if spec.name != self.name:
            self._built_stamp = stamp
            self._publish(state="invalid", error=f"the job's name changed to {spec.name!r}; restart `charforge watch`")
            return {"state": "invalid"}
        self._refresh_watch_files(spec)
        self._built_stamp = self.stamp()
        self._seq_applied, self._applied = seq, ov
        self._publish(state="building", message=f"building (overrides seq {seq})")
        self.log(f"[live] building {spec.name} with {len(ov)} override(s)")
        t0 = time.time()
        try:
            manifest = build_one(spec, self.root, self.backend, self.log, fit_cache=self.root / "_fit_cache")
        except Exception as e:  # noqa: BLE001
            self._publish(state="error", error=f"{type(e).__name__}: {e}")
            return {"state": "error"}
        ctl = []
        try:
            ctl = controls(spec, manifest)
        except Exception as e:  # noqa: BLE001
            self.log(f"[live] controls unavailable: {e}")
        if ctl:
            self._controls = ctl
        summary = {"status": manifest.get("status"), "fingerprint": manifest.get("fingerprint"),
                   "seconds": round(time.time() - t0, 2), "iterations": len(manifest.get("iterations", [])),
                   "failures": manifest.get("final_failures", []), "error": manifest.get("error"),
                   "error_code": manifest.get("error_code")}
        if manifest.get("status") in ("passed", "needs_review") and (self.dir / "final").is_dir():
            self.build_id += 1
            try:
                self._files = self._publish_files(self.dir / "final", self.build_id)
            except OSError as e:
                self.build_id -= 1
                self._publish(state="error", error=f"could not publish the build: {e}")
                return {"state": "error"}
            self._last_manifest_summary = summary
            self._publish(state="idle", message=f"build {self.build_id} {summary['status']}")
        else:
            self._last_manifest_summary = {**self._last_manifest_summary, **{"last_attempt": summary}}
            self._publish(state="error", error=summary.get("error") or "the build produced no files", attempt=summary)
        return {"state": "built", "manifest": manifest}

    def _publish_files(self, final: Path, build_id: int) -> list[dict[str, Any]]:
        dest = self.dir / "published" / f"{build_id:05d}"
        tmp = dest.with_name(dest.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(final, tmp)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
        keep = sorted(p for p in (self.dir / "published").iterdir() if p.is_dir() and not p.name.endswith(".tmp"))
        for old in keep[:-KEEP_BUILDS]:
            shutil.rmtree(old, ignore_errors=True)
        files = []
        for p in sorted(dest.iterdir()):
            if p.is_file():
                files.append({"file": f"published/{dest.name}/{p.name}", "bytes": p.stat().st_size, **_kind_of(p.name)})
        return files

    def _publish(self, state: str, message: str = "", error: str | None = None, **extra: Any) -> None:
        write_json_atomic(self.status_path, {
            "protocol": PROTOCOL, "name": self.name, "kind": self.kind, "job": str(self.job), "pid": os.getpid(),
            "state": state, "message": message, "error": error, "build_id": self.build_id,
            "seq_applied": self._seq_applied, "overrides": self._applied, "files": self._files,
            "result": self._last_manifest_summary, "controls": self._controls, "inbox": "inbox", "fast": self.fast,
            "updated": time.time(), **extra,
        })

    # -- loop
    def run(self, stop: threading.Event | None = None, poll_s: float = 0.25) -> None:
        """Build once at start, then whenever the job file, the overrides or a watched file settle on a new content.
        A change must be seen twice in a row before it triggers a build (editors write files in several steps)."""
        stop = stop or threading.Event()
        prev: str | None = None
        last_beat = 0.0
        self.build_now()
        try:
            while not stop.is_set():
                now = time.monotonic()
                if now - last_beat >= 1.0:
                    self.heartbeat_path.write_text(str(time.time()))
                    last_beat = now
                s = self.stamp()
                if s != self._built_stamp and s == prev:
                    self.build_now()
                prev = s
                stop.wait(poll_s)
        finally:
            try:
                self.heartbeat_path.unlink()
            except FileNotFoundError:
                pass
            self._publish(state="stopped", message="watcher stopped")


# ---------------------------------------------------------------- applying overrides to the YAML
def apply_to_job_file(job: str | Path, overrides: dict[str, Any], write: bool = False) -> str:
    """Return a unified diff of the job after folding the overrides into it. With write=True the file is replaced
    (the old one is kept as <file>.bak). YAML is re-dumped, so comments in the job file are NOT preserved."""
    import difflib

    job = Path(job)
    old = job.read_text()
    raw = yaml.safe_load(old)
    new_raw = apply_overrides(raw, overrides)
    new = yaml.safe_dump(new_raw, sort_keys=False, default_flow_style=None, allow_unicode=True)
    from .spec import load_job
    tmp = job.with_name(f".{job.name}.check.yaml")
    tmp.write_text(new)
    try:
        load_job(tmp)                       # never write a file that does not validate
    finally:
        tmp.unlink()
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"{job.name} (current)", f"{job.name} (with overrides)"))
    if write and diff:
        shutil.copy2(job, job.with_name(job.name + ".bak"))
        job.write_text(new)
    return diff
