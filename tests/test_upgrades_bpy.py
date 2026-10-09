"""The no-GPU tranche, end-to-end: full combat animation set + root motion, sprite-sheet
export, and the texture QA gates firing on real build metrics. Needs bpy."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

bpy = pytest.importorskip("bpy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vgbuilder import qa  # noqa: E402
from vgbuilder.orchestrator import Backend, build_one  # noqa: E402
from vgbuilder.spec import load_job  # noqa: E402

pytestmark = pytest.mark.bpy

ALL_ANIMS = ["idle", "walk", "attack_light", "hit", "cast", "death"]


@pytest.fixture(scope="module")
def combat(tmp_path_factory):
    d = tmp_path_factory.mktemp("combat")
    job = d / "combat.yaml"
    job.write_text(
        "name: combat\nstyle: lowpoly\nstyle_template: souls_like\ntexture_size: 512\n"
        f"animations: {ALL_ANIMS}\nroot_motion: true\nsprite_sheet: true\nsprite_size: 128\n")
    spec = load_job(job)
    out = tmp_path_factory.mktemp("combat_out")
    return build_one(spec, out, Backend()), out, spec


def test_combat_build_passes_with_full_animation_set(combat):
    manifest, _, _ = combat
    assert manifest["status"] == "passed", manifest["iterations"][0]["failures"]
    an = manifest["iterations"][0]["metrics"]["animations"]
    assert set(an) == set(ALL_ANIMS)
    assert an["walk"]["kind"] == "loop" and an["walk"]["root_distance_m"] > 0.3
    assert an["death"]["kind"] == "oneshot_fall" and an["death"]["pelvis_drop_m"] > 0.4
    assert an["death"]["floor_lift_m"] >= 0.0 and an["death"]["min_z_m"] >= -0.036
    for name in ("attack_light", "hit", "cast"):
        assert an[name]["kind"] == "oneshot_recover" and an[name]["end_delta_rad"] < 0.08
    content = manifest["iterations"][0]["metrics"]["glb_content"][0]
    assert set(ALL_ANIMS) <= set(content["animations"])


def test_sprite_sheet_outputs(combat):
    manifest, out, _ = combat
    final = out / "combat" / "final"
    meta = json.loads((final / "combat_sprites.json").read_text())
    assert meta["format"] == "vgbuilder-sprites-1"
    assert set(meta["animations"]) == set(ALL_ANIMS)
    assert meta["animations"]["death"]["loop"] is False
    assert meta["animations"]["walk"]["loop"] is True
    from PIL import Image
    sheet = Image.open(final / "combat_spritesheet.png")
    assert list(sheet.size) == meta["sheet_size"]
    assert sheet.convert("RGBA").getpixel((0, 0))[3] == 0          # transparent background
    assert manifest["iterations"][0]["metrics"]["sprites"]["ok"] is True


def test_texture_gates_fire_on_mutated_metrics(combat):
    manifest, _, spec = combat
    metrics = copy.deepcopy(manifest["iterations"][0]["metrics"])

    def codes(m):
        return {f["code"] for f in qa.evaluate(m, spec)}

    assert "texture_too_saturated" not in codes(metrics)
    metrics["texture"]["analysis"]["mean_saturation"] = 0.99
    assert "texture_too_saturated" in codes(metrics)
    metrics["texture"]["analysis"]["mean_saturation"] = 0.1
    metrics["texture"]["analysis"]["value_p02"] = 0.50
    metrics["texture"]["analysis"]["value_p98"] = 0.55
    assert "texture_flat" in codes(metrics)
    metrics["texture"]["analysis"]["value_p98"] = 0.95
    metrics["texture"]["size"] = 64
    assert "texture_resolution_low" in codes(metrics)
