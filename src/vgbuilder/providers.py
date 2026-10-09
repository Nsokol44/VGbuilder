"""Mesh providers: where the base mesh comes from.

* procedural  - built inside Blender from numbers (no GPU, deterministic, humanoid only)
* import_glb  - use a GLB you already have (from any generator or artist)
* command     - run any shell command that turns a sketch into a GLB (a wrapper you own)
* trellis2    - `command` plus the metadata a heavy local model needs: model name/version, face-count and
                texture-size hints, and the `gpu` resource tag so the queue runs one at a time
* hunyuan     - HTTP: POST the sketch to a Hunyuan3D-style API server, get a GLB back
* http        - same client with no defaults: you say which fields and which response shape

None of the third-party model code is shipped or called by vgbuilder. Every provider ends the same way: a GLB
on disk that has been *inspected* (valid container, at least one triangle) plus a provenance dict.

Failures are structured: `ProviderError.code` is one of CODES and `.retryable` says whether trying again can help
(timeouts, an unreachable server, HTTP 429/5xx) or not (bad config, missing sketch, a file that is not a GLB).
The queue uses that flag; a job is never retried for a reason that cannot change.

Command contract: the string may contain {python} {image} {side_image} {out} {seed} {name} {faces}
{texture_size}; it runs with the job file's folder as cwd, must exit 0 and write a glb to {out}.
Results are cached by (provider, command or endpoint+payload, model, hints, sketch bytes, seed), so QA
retries never regenerate the mesh.

HTTP contract: POST {url}{endpoint} with JSON {image_field: <base64 png>, seed_field: seed, ...payload}; the
response is either the GLB bytes or JSON holding base64 GLB under `response_key` (dotted path). Header values
may reference environment variables as ${NAME} so secrets never sit in the job file or the manifest.

The default field names for `hunyuan` follow the `/generate` endpoint described for Hunyuan3D-2.x's API server.
That was not checked against a live server here: if your server names things differently, override
`image_field`, `seed_field`, `endpoint`, `payload` and `response_key` in provider_args; no code change needed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .spec import JobSpec

CODES = {"config", "missing_input", "timeout", "exit_nonzero", "no_output", "invalid_glb", "empty_glb",
         "unreachable", "http_error", "bad_response", "provider_error"}
GENERATED = {"command", "trellis2", "hunyuan", "http"}
GPU_DEFAULT = {"trellis2": "gpu", "hunyuan": "gpu", "http": "remote"}

COMMON_KEYS = {"timeout_s", "resource", "model", "model_version"}
ARG_KEYS: dict[str, set[str]] = {
    "import_glb": {"glb"},
    "command": COMMON_KEYS | {"command"},
    "trellis2": COMMON_KEYS | {"command", "faces", "texture_size"},
    "hunyuan": COMMON_KEYS | {"url", "endpoint", "payload", "headers", "image_field", "side_image_field", "seed_field",
                              "response", "response_key", "max_bytes"},
    "http": COMMON_KEYS | {"url", "endpoint", "payload", "headers", "image_field", "side_image_field", "seed_field",
                           "response", "response_key", "max_bytes"},
}
HUNYUAN_DEFAULTS = {"endpoint": "/generate", "image_field": "image", "seed_field": "seed", "response": "bytes"}
HTTP_DEFAULTS = {"endpoint": "/generate", "image_field": "image", "seed_field": "seed", "response": "auto"}
RESOURCE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ProviderError(RuntimeError):
    def __init__(self, message: str, code: str = "provider_error", retryable: bool = False):
        super().__init__(message)
        assert code in CODES, code
        self.code, self.retryable = code, retryable


# ---------------------------------------------------------------- config
def validate_args(provider: str, args: dict[str, Any], has_sketch: bool) -> None:
    """Raise ValueError for anything that can be decided without running the provider."""
    if provider == "procedural":
        return
    bad = set(args) - ARG_KEYS[provider]
    if bad:
        raise ValueError(f"provider {provider}: unknown provider_args {sorted(bad)}; allowed: {sorted(ARG_KEYS[provider])}")
    if provider == "import_glb":
        if "glb" not in args:
            raise ValueError("provider import_glb needs provider_args.glb")
        return
    if provider in ("command", "trellis2") and "command" not in args:
        raise ValueError(f"provider {provider} needs provider_args.command")
    if provider in ("hunyuan", "http"):
        url = args.get("url")
        if not url or urllib.parse.urlparse(str(url)).scheme not in ("http", "https"):
            raise ValueError(f"provider {provider} needs provider_args.url starting with http:// or https://")
        if args.get("response", "auto") not in ("auto", "bytes", "json_base64"):
            raise ValueError("response must be auto, bytes or json_base64")
        if not isinstance(args.get("payload", {}), dict) or not isinstance(args.get("headers", {}), dict):
            raise ValueError("payload and headers must be mappings")
    if not has_sketch:
        raise ValueError(f"provider {provider} needs a sketch")
    if "resource" in args and not RESOURCE_RE.match(str(args["resource"])):
        raise ValueError("resource must look like gpu / remote / my_box")
    for k in ("timeout_s", "faces", "texture_size", "max_bytes"):
        if k in args and not (isinstance(args[k], (int, float)) and args[k] > 0):
            raise ValueError(f"provider_args.{k} must be a positive number")


def resource_of(provider: str, args: dict[str, Any]) -> str | None:
    """Queue resource tag: jobs sharing a tag are limited together (see library.work(limits=...))."""
    return args.get("resource") or GPU_DEFAULT.get(provider)


# ---------------------------------------------------------------- GLB inspection (pure python)
def inspect_glb(path: Path) -> dict[str, Any]:
    """Container-level check of a GLB without Blender. Raises ProviderError(invalid_glb / empty_glb)."""
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ProviderError("output is not a binary glTF (missing 'glTF' magic)", "invalid_glb")
    version, length = struct.unpack("<II", data[4:12])
    if version != 2 or length > len(data):
        raise ProviderError(f"bad glb header (version {version}, length {length} vs {len(data)} bytes)", "invalid_glb")
    clen, ctype = struct.unpack("<II", data[12:20])
    if ctype != 0x4E4F534A or 20 + clen > len(data):
        raise ProviderError("glb has no JSON chunk", "invalid_glb")
    try:
        doc = json.loads(data[20:20 + clen].decode("utf-8"))
    except ValueError as e:
        raise ProviderError(f"glb JSON chunk does not parse: {e}", "invalid_glb") from None
    acc = doc.get("accessors", [])
    tris = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            idx = prim.get("indices")
            pos = prim.get("attributes", {}).get("POSITION")
            n = acc[idx]["count"] if idx is not None and idx < len(acc) else acc[pos]["count"] if pos is not None and pos < len(acc) else 0
            tris += n // 3
    if tris == 0:
        raise ProviderError("glb contains no triangles", "empty_glb")
    return {"meshes": len(doc.get("meshes", [])), "triangles": tris, "materials": len(doc.get("materials", [])),
            "images": len(doc.get("images", [])), "skins": len(doc.get("skins", [])),
            "animations": len(doc.get("animations", [])), "bytes": len(data)}


# ---------------------------------------------------------------- helpers
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def _cache_key(spec: JobSpec, args: dict[str, Any]) -> str:
    # timeout / resource / headers change how a run is *executed*, not what it returns, so they stay out of the key
    h = hashlib.sha256()
    h.update(spec.provider.encode())
    h.update(_canon({k: v for k, v in args.items() if k not in ("timeout_s", "resource", "headers", "url", "max_bytes")}).encode())
    h.update(str(spec.seed).encode())
    for rel in (spec.sketch, spec.sketch_side):
        sk = spec.resolve_path(rel)
        if sk is not None:
            h.update(sk.read_bytes())
    return h.hexdigest()[:16]


def _expand_env(headers: dict[str, Any]) -> dict[str, str]:
    out = {}
    for k, v in headers.items():
        def sub(m: re.Match) -> str:
            if m.group(1) not in os.environ:
                raise ProviderError(f"header {k!r} refers to ${{{m.group(1)}}} which is not set", "config")
            return os.environ[m.group(1)]
        out[str(k)] = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", sub, str(v))
    return out


def _dig(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            raise KeyError(part)
    return obj


# ---------------------------------------------------------------- command / trellis2
def _run_command(spec: JobSpec, args: dict[str, Any], sketch: Path, out: Path) -> None:
    try:
        cmd = args["command"].format(
            python=shlex.quote(sys.executable), image=shlex.quote(str(sketch)), out=shlex.quote(str(out)),
            side_image=shlex.quote(str(spec.resolve_path(spec.sketch_side) or "")),
            seed=spec.seed, name=spec.name, faces=int(args.get("faces", 0)), texture_size=int(args.get("texture_size", 0)))
    except (KeyError, IndexError, ValueError) as e:
        raise ProviderError(f"command template has an unknown or malformed placeholder: {e!r}", "config") from None
    timeout = args.get("timeout_s", 1800)
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=spec.base_dir, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ProviderError(f"command timed out after {timeout}s", "timeout", retryable=True) from None
    if proc.returncode != 0:
        raise ProviderError(f"command failed (exit {proc.returncode}): {(proc.stderr or proc.stdout)[-600:]}", "exit_nonzero")
    if not out.exists():
        raise ProviderError(f"command failed (no output file written): {(proc.stderr or proc.stdout)[-600:]}", "no_output")


# ---------------------------------------------------------------- http / hunyuan
def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_http_request(spec: JobSpec, args: dict[str, Any], sketch: Path) -> tuple[str, dict[str, Any]]:
    """URL + JSON payload (headers excluded). Pure, so contract tests can assert on it."""
    d = {**(HUNYUAN_DEFAULTS if spec.provider == "hunyuan" else HTTP_DEFAULTS), **args}
    payload: dict[str, Any] = {d["image_field"]: _b64(sketch), d["seed_field"]: spec.seed}
    side = spec.resolve_path(spec.sketch_side)
    if side is not None and d.get("side_image_field"):
        payload[d["side_image_field"]] = _b64(side)
    payload.update(d.get("payload", {}))
    return d["url"].rstrip("/") + "/" + d["endpoint"].lstrip("/"), payload


def _http_generate(spec: JobSpec, args: dict[str, Any], sketch: Path, out: Path) -> None:
    d = {**(HUNYUAN_DEFAULTS if spec.provider == "hunyuan" else HTTP_DEFAULTS), **args}
    url, payload = build_http_request(spec, args, sketch)
    headers = {"Content-Type": "application/json", **_expand_env(d.get("headers", {}))}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    timeout = d.get("timeout_s", 1800)
    limit = int(d.get("max_bytes", 512 * 1024 * 1024))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(limit + 1)
            ctype = r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        snippet = e.read(300).decode("utf-8", "replace")
        raise ProviderError(f"server answered HTTP {e.code}: {snippet}", "http_error",
                            retryable=e.code in (408, 425, 429) or e.code >= 500) from None
    except (socket.timeout, TimeoutError):
        raise ProviderError(f"no answer from {url} within {timeout}s", "timeout", retryable=True) from None
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        raise ProviderError(f"cannot reach {url}: {getattr(e, 'reason', e)}", "unreachable", retryable=True) from None
    if len(body) > limit:
        raise ProviderError(f"response larger than max_bytes ({limit})", "bad_response")
    mode = d["response"]
    if mode == "auto":
        mode = "bytes" if body[:4] == b"glTF" else "json_base64"
    if mode == "json_base64":
        key = d.get("response_key")
        try:
            doc = json.loads(body.decode("utf-8"))
        except ValueError:
            raise ProviderError(f"expected JSON but got {ctype or 'unknown content type'}: {body[:80]!r}", "bad_response") from None
        if not key:
            raise ProviderError(f"JSON response needs provider_args.response_key; keys are {sorted(doc) if isinstance(doc, dict) else type(doc).__name__}", "config")
        try:
            body = base64.b64decode(_dig(doc, key), validate=True)
        except KeyError:
            raise ProviderError(f"response has no {key!r}; keys are {sorted(doc) if isinstance(doc, dict) else '?'}", "bad_response") from None
        except (ValueError, TypeError):
            raise ProviderError(f"{key!r} is not valid base64", "bad_response") from None
    tmp = out.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(out)


# ---------------------------------------------------------------- entry point
def provide_glb(spec: JobSpec, cache_dir: Path) -> tuple[Path | None, dict]:
    """Return (glb_path or None for procedural, provenance dict). Raises ProviderError."""
    if spec.provider == "procedural":
        return None, {"provider": "procedural"}
    args = dict(spec.provider_args)

    if spec.provider == "import_glb":
        p = spec.resolve_path(args["glb"])
        if p is None or not p.exists():
            raise ProviderError(f"glb not found: {p}", "missing_input")
        info = inspect_glb(p)
        return p, {"provider": "import_glb", "glb": str(p), "sha256": _sha(p), "glb_info": info}

    sketch = spec.resolve_path(spec.sketch)
    if sketch is None or not sketch.exists():
        raise ProviderError(f"sketch not found: {sketch}", "missing_input")
    cache_dir = cache_dir.resolve()  # the command runs in another cwd, so paths must be absolute
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{spec.name}_{_cache_key(spec, args)}.glb"
    cached = out.exists()
    seconds = 0.0
    if not cached:
        t0 = time.time()
        if spec.provider in ("command", "trellis2"):
            _run_command(spec, args, sketch, out)
        else:
            _http_generate(spec, args, sketch, out)
        seconds = round(time.time() - t0, 2)
    try:
        info = inspect_glb(out)
    except ProviderError:
        out.unlink(missing_ok=True)            # never cache a file that failed inspection
        raise
    prov: dict[str, Any] = {"provider": spec.provider, "cached": cached, "sha256": _sha(out), "glb_info": info,
                            "seed": spec.seed, "seconds": seconds}
    for k in ("model", "model_version", "faces", "texture_size"):
        if k in args:
            prov[k] = args[k]
    if "command" in args:
        prov["command"] = args["command"]
    if spec.provider in ("hunyuan", "http"):
        # derived from the spec only (not from the run), so a cached and a fresh build embed identical provenance
        url, payload = build_http_request(spec, args, sketch)
        prov.update(url=url.split("?")[0], payload_keys=sorted(payload), header_names=sorted(args.get("headers", {})))
    return out, prov
