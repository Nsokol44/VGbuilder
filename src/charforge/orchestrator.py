"""Job runner: provider -> build in Blender -> QA -> fix params -> rebuild, with provenance."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import __version__, engines, fixer, qa, sketch as sk
from .providers import ProviderError, provide_glb
from .spec import JobSpec, load_job

SRC_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Backend:
    """How to run Blender. `bpy-module` = the `bpy` wheel in this Python. `blender` = a binary.

    The binary mode needs numpy + pillow importable by Blender's own Python, and is the less
    exercised path. The bpy-module path is what the tests use.
    """
    kind: str = "bpy-module"
    exe: str | None = None
    timeout_s: int = 1800

    def command(self, job_json: Path, result_json: Path) -> list[str]:
        if self.kind == "bpy-module":
            return [sys.executable, "-m", "charforge.bl.entry", str(job_json), str(result_json)]
        entry = SRC_DIR / "charforge" / "bl" / "entry.py"
        return [self.exe or "blender", "-b", "--factory-startup", "-P", str(entry), "--", str(job_json), str(result_json)]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_iteration(backend: Backend, spec: JobSpec, params: dict, glb: Path | None, it_dir: Path,
                  provenance_blob: dict | None = None) -> dict:
    it_dir.mkdir(parents=True, exist_ok=True)
    job_json, result_json = it_dir / "job.json", it_dir / "result.json"
    job_json.write_text(json.dumps({
        "spec": spec.to_dict(), "params": params, "out_dir": str(it_dir),
        "imported_glb": str(glb) if glb else None, "provenance": provenance_blob,
    }, indent=2))
    env = {**os.environ, "PYTHONPATH": f"{SRC_DIR}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    t0 = time.time()
    proc = subprocess.run(backend.command(job_json, result_json), capture_output=True, text=True,
                          timeout=backend.timeout_s, env=env)
    (it_dir / "blender.log").write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
    if not result_json.exists():
        return {"ok": False, "error": f"no result written (exit {proc.returncode})", "log": str(it_dir / "blender.log")}
    res = json.loads(result_json.read_text())
    res["wall_seconds"] = round(time.time() - t0, 2)
    return res


def build_one(spec: JobSpec, out_root: Path, backend: Backend,
              log: Callable[[str], None] = lambda s: None, fit_cache: Path | None = None) -> dict[str, Any]:
    if spec.kind == "effect":              # no Blender involved: see charforge.effects
        from .effects import build_effect
        return build_effect(spec, out_root, log)
    t0 = time.time()
    out_dir = out_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "charforge_version": __version__, "name": spec.name, "fingerprint": spec.fingerprint(),
        "spec": {k: v for k, v in spec.to_dict().items() if k != "base_dir"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "iterations": [], "status": "error",
    }
    try:
        glb, provenance = provide_glb(spec, out_root / "_provider_cache")
    except ProviderError as e:
        manifest["error"] = str(e)
        manifest["error_code"], manifest["error_retryable"] = e.code, e.retryable
        _write(out_dir, manifest)
        return manifest
    manifest["provider"] = provenance
    style = spec.load_style()
    if style is not None:
        sd = style.to_dict()
        manifest["style_resolved"] = {**sd, "sha256": hashlib.sha256(json.dumps(sd, sort_keys=True, default=str).encode()).hexdigest()[:16]}
    manifest["engine_preset"] = {**engines.preset(spec.target_engine).describe(), "naming_used": spec.resolved_naming()}

    target_mask = None
    sketch_path = spec.resolve_path(spec.sketch)
    if sketch_path is not None:
        manifest["sketch_sha256"] = _sha(sketch_path)
        target_mask = sk.load_mask(sketch_path)
    side_mask = None
    side_path = spec.resolve_path(spec.sketch_side)
    if side_path is not None:
        manifest["sketch_side_sha256"] = _sha(side_path)
        side_mask = sk.load_side_mask(side_path, spec.side_facing)
    if spec.provider == "procedural":
        params, fit_info = fixer.initial_params(spec, target_mask, side_mask, cache_dir=fit_cache)
        manifest["fit"] = fit_info
        log(f"[{spec.name}] fit: {fit_info.get('fit')}")
    else:
        params, fit_info = {}, {}

    # Goes INTO every exported glb (scene extras), so the file carries its own recipe. No timestamps or
    # absolute paths: the same job must still rebuild byte-identical files.
    embed = {
        "charforge": __version__, "name": spec.name, "fingerprint": spec.fingerprint(),
        "sketch_sha256": manifest.get("sketch_sha256"), "sketch_side_sha256": manifest.get("sketch_side_sha256"),
        "provider": {k: v for k, v in provenance.items() if k not in ("glb", "cached", "seconds")},
        "style": ({"name": style.name, "sha256": manifest["style_resolved"]["sha256"]} if style is not None else None),
        "engine": spec.target_engine, "seed": spec.seed, "numpy": np.__version__,
        "spec": {k: v for k, v in manifest["spec"].items() if k not in ("notes", "tags", "critic", "priority")},
    }

    best: tuple[float, int] | None = None
    for i in range(1, spec.max_iterations + 1):
        log(f"[{spec.name}] iteration {i}")
        it_dir = out_dir / f"iter_{i}"
        res = run_iteration(backend, spec, params, glb, it_dir, embed)
        entry: dict[str, Any] = {"iteration": i, "params": dict(params)}
        if not res.get("ok"):
            entry["error"] = res.get("error")
            entry["log"] = res.get("log")
            manifest["iterations"].append(entry)
            manifest["error"] = res.get("error")
            break
        fails = qa.evaluate(res["metrics"], spec)
        s = qa.score(res["metrics"], fails)
        entry.update(metrics=res["metrics"], failures=fails, score=s, files=res["files"],
                     blender_version=res.get("blender_version"), seconds=res.get("seconds"))
        if best is None or s > best[0]:
            best = (s, i)
        if not fails:
            manifest["iterations"].append(entry)
            manifest["status"] = "passed"
            break
        new_params, changes = fixer.next_params(spec, params, fails)
        entry["next_changes"] = changes
        manifest["iterations"].append(entry)
        if not changes:
            manifest["status"] = "needs_review"
            break
        params = new_params
    else:
        manifest["status"] = "needs_review"

    if manifest["status"] != "error" and best is not None:
        manifest["best_iteration"] = best[1]
        final = out_dir / "final"
        if final.exists():
            shutil.rmtree(final)
        final.mkdir()
        outs = []
        for f in manifest["iterations"][best[1] - 1]["files"]:
            fp = Path(f)
            shutil.copy2(fp, final / fp.name)
            outs.append({"file": fp.name, "sha256": _sha(final / fp.name), "bytes": (final / fp.name).stat().st_size})
        manifest["outputs"] = outs
        last = manifest["iterations"][best[1] - 1]
        manifest["final_failures"] = last["failures"]
    manifest["seconds"] = round(time.time() - t0, 2)
    _write(out_dir, manifest)
    return manifest


def _write(out_dir: Path, manifest: dict) -> None:
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def collect_jobs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted([*path.glob("*.yaml"), *path.glob("*.yml")])
    return [path]


def run_jobs(paths: list[Path], out_root: Path, backend: Backend, workers: int = 1,
             log: Callable[[str], None] = print) -> list[dict]:
    specs = [load_job(p) for p in paths]  # validate everything up front
    out_root.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        return [build_one(s, out_root, backend, log) for s in specs]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda s: build_one(s, out_root, backend, log), specs))
