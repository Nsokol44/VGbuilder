"""Provider contracts: GLB inspection, config validation, HTTP and command providers against local mocks.
Nothing here needs a GPU or a real model; the mocks stand in for TRELLIS / Hunyuan servers."""
from __future__ import annotations

import base64
import json
import os
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from vgbuilder import providers
from glbs import make_glb
from vgbuilder.farm import make_variants
from vgbuilder.library import Library, work
from vgbuilder.orchestrator import Backend, build_one
from vgbuilder.providers import ProviderError, build_http_request, inspect_glb, provide_glb, resource_of, validate_args
from vgbuilder.spec import JobSpec, load_job

ROOT = Path(__file__).resolve().parents[1]
SKETCH = ROOT / "examples" / "sketches" / "hero_front.png"


def spec_for(provider: str, args: dict, tmp: Path, **kw) -> JobSpec:
    s = JobSpec(name="gen", sketch=str(SKETCH), provider=provider, provider_args=args, base_dir=str(tmp), **kw)
    s.validate()
    return s


# ---------------------------------------------------------------- GLB inspection
def test_inspect_glb_counts_triangles(tmp_path):
    p = tmp_path / "a.glb"; p.write_bytes(make_glb(7))
    info = inspect_glb(p)
    assert info["triangles"] == 7 and info["meshes"] == 1 and info["skins"] == 0


@pytest.mark.parametrize("blob,code", [
    (b"not a glb at all, just text" * 3, "invalid_glb"),
    (make_glb()[:30], "invalid_glb"),                                   # truncated
    (make_glb()[:4] + struct.pack("<II", 1, 99) + make_glb()[12:], "invalid_glb"),   # version 1
    (make_glb(doc_patch={"meshes": []}), "empty_glb"),
])
def test_inspect_glb_rejects(tmp_path, blob, code):
    p = tmp_path / "bad.glb"; p.write_bytes(blob)
    with pytest.raises(ProviderError) as e:
        inspect_glb(p)
    assert e.value.code == code and not e.value.retryable


def test_inspect_glb_json_chunk_garbage(tmp_path):
    g = bytearray(make_glb())
    g[20:30] = b"{{{{{{{{{{"
    p = tmp_path / "g.glb"; p.write_bytes(bytes(g))
    with pytest.raises(ProviderError, match="does not parse"):
        inspect_glb(p)


# ---------------------------------------------------------------- config
def test_validation_rules(tmp_path):
    ok_http = {"url": "http://127.0.0.1:9"}
    spec_for("hunyuan", ok_http, tmp_path)
    spec_for("trellis2", {"command": "echo {out}", "faces": 20000, "model": "TRELLIS.2", "model_version": "4B"}, tmp_path)
    for provider, args, msg in [
        ("hunyuan", {}, "needs provider_args.url"),
        ("hunyuan", {"url": "ftp://x"}, "http:// or https://"),
        ("http", {"url": "http://x", "response": "xml"}, "response must be"),
        ("http", {"url": "http://x", "typo_key": 1}, "unknown provider_args"),
        ("trellis2", {}, "needs provider_args.command"),
        ("trellis2", {"command": "x", "resource": "GPU 1"}, "resource must look like"),
        ("trellis2", {"command": "x", "timeout_s": -1}, "positive"),
        ("hunyuan", {"url": "http://x", "headers": ["a"]}, "mappings"),
    ]:
        with pytest.raises(ValueError, match=msg):
            spec_for(provider, args, tmp_path)
    with pytest.raises(ValueError, match="needs a sketch"):
        JobSpec(name="x", provider="hunyuan", provider_args=ok_http).validate()


def test_resource_tags():
    assert resource_of("trellis2", {}) == "gpu" and resource_of("hunyuan", {}) == "gpu"
    assert resource_of("http", {}) == "remote"
    assert resource_of("command", {}) is None and resource_of("procedural", {}) is None
    assert resource_of("command", {"resource": "gpu"}) == "gpu"
    assert resource_of("trellis2", {"resource": "gpu_b"}) == "gpu_b"


def test_request_shape_follows_provider_args(tmp_path):
    s = spec_for("hunyuan", {"url": "http://h:8081/", "payload": {"texture": True, "octree_resolution": 256}}, tmp_path, seed=5)
    url, pl = build_http_request(s, s.provider_args, SKETCH)
    assert url == "http://h:8081/generate"
    assert pl["image"] == base64.b64encode(SKETCH.read_bytes()).decode() and pl["seed"] == 5
    assert pl["texture"] is True and pl["octree_resolution"] == 256
    s2 = spec_for("http", {"url": "http://h", "endpoint": "v2/make", "image_field": "img", "seed_field": "s",
                           "side_image_field": "img_side"}, tmp_path, sketch_side=str(ROOT / "examples/sketches/hero_side.png"))
    url, pl = build_http_request(s2, s2.provider_args, SKETCH)
    assert url == "http://h/v2/make" and set(pl) == {"img", "s", "img_side"}


# ---------------------------------------------------------------- mock server
class Mock:
    def __init__(self):
        self.requests: list[dict] = []
        self.mode = "bytes"
        self.glb = make_glb(30)
        self.delay = 0.0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(self.rfile.read(n))})
                time.sleep(outer.delay)
                m = outer.mode
                if m.startswith("status:"):
                    code = int(m.split(":")[1]); body = b'{"error": "nope"}'
                    self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
                if m == "bytes":
                    body, ctype = outer.glb, "model/gltf-binary"
                elif m == "garbage":
                    body, ctype = b"<html>queue is full</html>", "text/html"
                elif m == "json":
                    body, ctype = json.dumps({"result": {"model_b64": base64.b64encode(outer.glb).decode()}}).encode(), "application/json"
                elif m == "json_badkey":
                    body, ctype = json.dumps({"other": 1}).encode(), "application/json"
                elif m == "json_badb64":
                    body, ctype = json.dumps({"result": {"model_b64": "!!!"}}).encode(), "application/json"
                self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


@pytest.fixture
def mock():
    m = Mock()
    yield m
    m.close()


def run_http(tmp_path, mock, provider="hunyuan", **args):
    s = spec_for(provider, {"url": mock.url, **args}, tmp_path)
    return provide_glb(s, tmp_path / "cache")


def test_hunyuan_bytes_roundtrip_and_provenance(tmp_path, mock):
    glb, prov = run_http(tmp_path, mock, model="Hunyuan3D-2.1", model_version="2.1.0", payload={"texture": True})
    assert glb.read_bytes() == mock.glb
    assert prov["provider"] == "hunyuan" and prov["model"] == "Hunyuan3D-2.1" and prov["model_version"] == "2.1.0"
    assert prov["glb_info"]["triangles"] == 30 and prov["payload_keys"] == ["image", "seed", "texture"]
    assert prov["url"].endswith("/generate") and "command" not in prov
    req = mock.requests[0]
    assert req["path"] == "/generate" and req["body"]["texture"] is True and req["body"]["seed"] == 0


def test_json_base64_response_with_dotted_key(tmp_path, mock):
    mock.mode = "json"
    glb, prov = run_http(tmp_path, mock, provider="http", response="json_base64", response_key="result.model_b64")
    assert glb.read_bytes() == mock.glb
    glb2, _ = run_http(tmp_path, mock, provider="http", response_key="result.model_b64")        # auto detects JSON too
    assert glb2.exists()


@pytest.mark.parametrize("mode,code,msg", [("json_badkey", "bad_response", "no 'result.model_b64'"),
                                           ("json_badb64", "bad_response", "not valid base64"),
                                           ("garbage", "bad_response", "expected JSON")])
def test_bad_http_bodies(tmp_path, mock, mode, code, msg):
    mock.mode = mode
    with pytest.raises(ProviderError, match=msg) as e:
        run_http(tmp_path, mock, provider="http", response_key="result.model_b64")
    assert e.value.code == code and not e.value.retryable


def test_json_without_response_key_is_a_config_error(tmp_path, mock):
    mock.mode = "json"
    with pytest.raises(ProviderError) as e:
        run_http(tmp_path, mock, provider="http")
    assert e.value.code == "config" and "response_key" in str(e.value)


def test_non_glb_bytes_are_rejected_and_never_cached(tmp_path, mock):
    mock.glb = b"glTF" + b"\0" * 40                         # right magic, nonsense inside
    with pytest.raises(ProviderError) as e:
        run_http(tmp_path, mock)
    assert e.value.code == "invalid_glb" and not list((tmp_path / "cache").glob("*.glb"))


@pytest.mark.parametrize("status,retry", [(503, True), (429, True), (500, True), (400, False), (404, False), (401, False)])
def test_http_status_retryability(tmp_path, mock, status, retry):
    mock.mode = f"status:{status}"
    with pytest.raises(ProviderError) as e:
        run_http(tmp_path, mock)
    assert e.value.code == "http_error" and e.value.retryable is retry and str(status) in str(e.value)


def test_unreachable_and_timeout_are_retryable(tmp_path, mock):
    s = spec_for("hunyuan", {"url": "http://127.0.0.1:9"}, tmp_path)         # discard port: connection refused
    with pytest.raises(ProviderError) as e:
        provide_glb(s, tmp_path / "c1")
    assert e.value.code == "unreachable" and e.value.retryable
    mock.delay = 2.0
    with pytest.raises(ProviderError) as e:
        run_http(tmp_path, mock, timeout_s=0.5)
    assert e.value.code == "timeout" and e.value.retryable


def test_header_env_expansion_and_secrets_stay_out_of_provenance(tmp_path, mock, monkeypatch):
    monkeypatch.setenv("GEN_TOKEN", "s3cret")
    glb, prov = run_http(tmp_path, mock, headers={"Authorization": "Bearer ${GEN_TOKEN}"})
    assert mock.requests[0]["headers"]["Authorization"] == "Bearer s3cret"
    assert "s3cret" not in json.dumps(prov) and prov["header_names"] == ["Authorization"]
    monkeypatch.delenv("GEN_TOKEN")
    with pytest.raises(ProviderError) as e:
        run_http(tmp_path / "other", mock, headers={"Authorization": "Bearer ${GEN_TOKEN}"})
    assert e.value.code == "config" and "GEN_TOKEN" in str(e.value)


def test_cache_prevents_second_request_and_keeps_provenance_stable(tmp_path, mock):
    g1, p1 = run_http(tmp_path, mock)
    g2, p2 = run_http(tmp_path, mock)
    assert len(mock.requests) == 1 and g1 == g2 and p1["cached"] is False and p2["cached"] is True
    strip = lambda p: {k: v for k, v in p.items() if k not in ("cached", "seconds")}   # noqa: E731
    assert strip(p1) == strip(p2)                      # what gets embedded in the glb must not depend on cache state
    mock.glb = make_glb(31)
    run_http(tmp_path, mock, payload={"x": 1})          # different request -> different cache entry
    assert len(mock.requests) == 2


# ---------------------------------------------------------------- trellis2 (command based)
def write_wrapper(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "wrap.py"; p.write_text(body); return p


def test_trellis2_placeholders_and_provenance(tmp_path):
    glb = tmp_path / "model.glb"; glb.write_bytes(make_glb(50))
    w = write_wrapper(tmp_path, "import sys, shutil, json\nimage, out, seed, faces, tex, side = sys.argv[1:7]\n"
                      f"json.dump(dict(image=image, seed=seed, faces=faces, tex=tex, side=side), open(r'{tmp_path}/args.json','w'))\n"
                      f"shutil.copy(r'{glb}', out)\n")
    s = spec_for("trellis2", {"command": "{python} wrap.py {image} {out} {seed} {faces} {texture_size} {side_image}",
                              "faces": 20000, "texture_size": 1024, "model": "TRELLIS.2", "model_version": "4B"}, tmp_path,
                 seed=9, sketch_side=str(ROOT / "examples/sketches/hero_side.png"))
    out, prov = provide_glb(s, tmp_path / "cache")
    a = json.loads((tmp_path / "args.json").read_text())
    assert a["seed"] == "9" and a["faces"] == "20000" and a["tex"] == "1024" and a["side"].endswith("hero_side.png")
    assert prov["provider"] == "trellis2" and prov["model"] == "TRELLIS.2" and prov["faces"] == 20000
    assert prov["command"].startswith("{python} wrap.py") and prov["glb_info"]["triangles"] == 50
    assert s.resource() == "gpu"


@pytest.mark.parametrize("script,code,retry", [
    ("import sys; sys.exit(3)", "exit_nonzero", False),
    ("pass", "no_output", False),
    ("import time; time.sleep(30)", "timeout", True),
    ("import sys; open(sys.argv[1], 'wb').write(b'oops')", "invalid_glb", False),
])
def test_command_failures_are_structured(tmp_path, script, code, retry):
    w = write_wrapper(tmp_path, script)
    s = spec_for("trellis2", {"command": "{python} wrap.py {out}", "timeout_s": 1}, tmp_path)
    with pytest.raises(ProviderError) as e:
        provide_glb(s, tmp_path / "cache")
    assert e.value.code == code and e.value.retryable is retry


def test_unknown_placeholder_is_a_config_error(tmp_path):
    s = spec_for("command", {"command": "{python} x.py {nonsense}"}, tmp_path)
    with pytest.raises(ProviderError) as e:
        provide_glb(s, tmp_path / "cache")
    assert e.value.code == "config" and not e.value.retryable


def test_import_glb_is_inspected(tmp_path):
    bad = tmp_path / "x.glb"; bad.write_bytes(b"nope")
    s = JobSpec(name="imp", provider="import_glb", provider_args={"glb": str(bad)}, base_dir=str(tmp_path)); s.validate()
    with pytest.raises(ProviderError) as e:
        provide_glb(s, tmp_path / "c")
    assert e.value.code == "invalid_glb"
    s2 = JobSpec(name="imp", provider="import_glb", provider_args={"glb": str(tmp_path / "missing.glb")}, base_dir=str(tmp_path)); s2.validate()
    with pytest.raises(ProviderError) as e:
        provide_glb(s2, tmp_path / "c")
    assert e.value.code == "missing_input"


# ---------------------------------------------------------------- through the whole pipeline (Blender)
@pytest.fixture(scope="module")
def generated_glb(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen") / "fake.glb"
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run([sys.executable, str(ROOT / "examples" / "fake_generator.py"), str(SKETCH), str(out)], check=True, env=env, timeout=300)
    return out.read_bytes()


def job_file(tmp_path, url, name="gen_http", **extra):
    d = {"name": name, "sketch": str(SKETCH), "style": "stylized", "provider": "hunyuan",
         "provider_args": {"url": url, "model": "mock-hunyuan", "model_version": "0.0"}, "height_m": 1.75,
         "poly_budget": 1200, "naming": "simple", "max_iterations": 1, **extra}
    p = tmp_path / f"{name}.yaml"; p.write_text(yaml.safe_dump(d)); return p


def test_hunyuan_provider_full_build_embeds_provenance(tmp_path, mock, generated_glb):
    mock.glb = generated_glb
    spec = load_job(job_file(tmp_path, mock.url))
    m = build_one(spec, tmp_path / "out", Backend(kind="bpy-module"))
    assert m["status"] in ("passed", "needs_review"), m.get("error")
    assert m["provider"]["model"] == "mock-hunyuan" and m["provider"]["glb_info"]["triangles"] > 100
    from vgbuilder.bl.export import read_embedded
    emb = read_embedded(tmp_path / "out" / "gen_http" / "final" / "gen_http_LOD0.glb")
    assert emb["provider"]["model"] == "mock-hunyuan" and "seconds" not in emb["provider"] and "cached" not in emb["provider"]
    # rebuilding from the provider cache must give byte-identical files
    again = build_one(spec, tmp_path / "out2", Backend(kind="bpy-module"))
    assert again["provider"]["cached"] is False                      # new out_root, new cache
    a = (tmp_path / "out" / "gen_http" / "final" / "gen_http_LOD0.glb").read_bytes()
    b = (tmp_path / "out2" / "gen_http" / "final" / "gen_http_LOD0.glb").read_bytes()
    assert a == b and len(mock.requests) == 2


def test_queue_retries_only_retryable_provider_errors(tmp_path, mock):
    lib = Library(tmp_path / "lib.db")
    mock.mode = "status:503"
    lib.enqueue(job_file(tmp_path, mock.url, "flaky"))
    work(lib, tmp_path / "out", Backend(kind="bpy-module"), once=True, log=lambda s: None)
    row = lib.queue_rows()[0]
    assert row["state"] == "failed" and row["attempts"] == 3 and len(mock.requests) == 3 and "503" in row["error"]
    assert lib.get("flaky")["status"] == "error"                       # the final failure is visible in the library
    mock.requests.clear(); mock.mode = "status:400"
    lib.enqueue(job_file(tmp_path, mock.url, "wrongrequest"))
    work(lib, tmp_path / "out", Backend(kind="bpy-module"), once=True, log=lambda s: None)
    row = [r for r in lib.queue_rows() if r["name"] == "wrongrequest"][0]
    assert row["state"] == "failed" and row["attempts"] == 1 and len(mock.requests) == 1


# ---------------------------------------------------------------- QA gate for generated meshes
def _qa_metrics(iou):
    from vgbuilder import rigdefs
    return {"lod_tris": [1000, 500], "height_m": 1.8, "symmetry": 0.001, "zero_weight_frac": 0.0, "loose_parts": 20,
            "bone_roles": rigdefs.ROLES, "has_uv": True, "stretch_max": 1.2, "roundtrip": {"ok": True}, "silhouette_iou": iou}


@pytest.mark.parametrize("provider,iou,codes", [
    ("procedural", 0.70, ["silhouette_low"]),
    ("trellis2", 0.70, ["silhouette_low"]),               # same default gate: nothing was calibrated to justify loosening it
    ("trellis2", 0.45, ["silhouette_low"]),
    ("hunyuan", 0.10, ["generated_mesh_collapsed"]),      # does not resemble the sketch at all: its own code
    ("trellis2", 0.90, []),
])
def test_generated_meshes_have_their_own_silhouette_gate_and_a_collapse_gate(provider, iou, codes):
    from vgbuilder import qa
    s = JobSpec(name="a", sketch="x.png", poly_budget=1200, provider=provider,
                provider_args={} if provider == "procedural" else {"command": "x"} if provider == "trellis2" else {"url": "http://x"})
    assert [f["code"] for f in qa.evaluate(_qa_metrics(iou), s)] == codes
    s.qa = {"silhouette_iou_min_generated": 0.6}
    if provider == "trellis2" and iou == 0.70:
        assert qa.evaluate(_qa_metrics(iou), s) == []          # overridable per job, once you have calibrated it


# ---------------------------------------------------------------- shipped files stay valid
@pytest.mark.parametrize("p", sorted((ROOT / "examples" / "providers").glob("*.yaml")))
def test_example_provider_jobs_validate(p):
    assert load_job(p).resource() == "gpu"


@pytest.mark.parametrize("p", sorted((ROOT / "jobs").glob("*.yaml")))
def test_every_file_in_jobs_is_a_job(p):
    load_job(p)          # `vgbuilder run jobs` validates the whole folder up front, so one stray file breaks it


@pytest.mark.parametrize("p", sorted((ROOT / "farms").glob("*.yaml")))
def test_shipped_farms_expand_to_valid_jobs(p, tmp_path):
    assert len(make_variants(p, tmp_path / "o")) >= 2
