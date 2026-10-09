"""The no-GPU tranche, pure side: animation-set spec rules, texture defaults, and the GPU
handoff kit's wrapper contract (stub-tested; no GPU involved). Blender end-to-end coverage
lives in test_upgrades_bpy.py."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vgbuilder.spec import ANIM_KINDS, ANIMATIONS, JobSpec, load_job  # noqa: E402

ALL_ANIMS = ["idle", "walk", "attack_light", "hit", "cast", "death"]


def test_animation_kinds_cover_the_set():
    assert set(ANIM_KINDS) == set(ANIMATIONS) == set(ALL_ANIMS)
    assert ANIM_KINDS["death"] == "oneshot_fall"
    assert ANIM_KINDS["attack_light"] == "oneshot_recover"


def test_new_job_fields_validate():
    JobSpec(name="a", animations=ALL_ANIMS, root_motion=True, sprite_sheet=True, sprite_size=128).validate()
    with pytest.raises(ValueError, match="sprite_sheet"):
        JobSpec(name="a", sprite_sheet=True).validate()                      # needs an animation
    with pytest.raises(ValueError, match="sprite_size"):
        JobSpec(name="a", animations=["idle"], sprite_sheet=True, sprite_size=8).validate()
    with pytest.raises(ValueError, match="character-only"):
        JobSpec(name="a", kind="effect", effect={"template": "fireball"}, sprite_sheet=True).validate()
    with pytest.raises(ValueError, match="character-only"):
        JobSpec(name="s", kind="structure", structure={"floors": 1}, root_motion=True).validate()


def test_texture_default_is_1k():
    assert JobSpec(name="a").texture_size == 1024


# ------------------------------------------------------------------ GPU handoff wrapper (no GPU)

WRAPPER = ROOT / "examples" / "gpu_handoff" / "trellis2_local.py"


def _run_wrapper(tmp_path: Path, stub: str | None):
    img = tmp_path / "sketch.png"
    img.write_bytes(b"\x89PNG")
    out = tmp_path / "out.glb"
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRELLIS2")}
    if stub is not None:
        s = tmp_path / "stub_gen.py"
        s.write_text(stub)
        env["TRELLIS2_CMD"] = f"{sys.executable} {s} {{image}} {{out}} {{seed}}"
    r = subprocess.run([sys.executable, str(WRAPPER), str(img), str(out), "7"],
                       capture_output=True, text=True, env=env, timeout=60)
    return r, out


def test_wrapper_cmd_mode_success(tmp_path):
    stub = "import sys\nopen(sys.argv[2], 'wb').write(b'glTF' + bytes(64))\n"
    r, out = _run_wrapper(tmp_path, stub)
    assert r.returncode == 0 and out.exists() and out.read_bytes()[:4] == b"glTF"


def test_wrapper_cmd_mode_propagates_failure(tmp_path):
    r, out = _run_wrapper(tmp_path, "import sys; sys.exit(3)\n")
    assert r.returncode == 3 and not out.exists()


def test_wrapper_without_env_is_a_clean_error(tmp_path):
    r, _ = _run_wrapper(tmp_path, None)
    assert r.returncode == 2 and "TRELLIS2_CMD" in r.stderr


def test_gpu_kit_jobs_validate():
    for name in ("hero_trellis", "hero_fake"):
        spec = load_job(ROOT / "examples" / "gpu_handoff" / f"{name}.yaml")
        assert spec.provider == "command"
