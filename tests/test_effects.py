"""Flipbook effects: rendering, image-space QA, auto-fix, outputs (no Blender needed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vgbuilder import effects
from vgbuilder.orchestrator import Backend, build_one
from vgbuilder.spec import EFFECT_TEMPLATES, JobSpec, load_job

ROOT = Path(__file__).resolve().parents[1]


def spec_for(tpl, **eff):
    s = JobSpec(name=f"fx_{tpl}", kind="effect", effect={"template": tpl, **eff})
    s.validate()
    return s


def test_templates_in_spec_and_module_agree():
    assert set(effects.TEMPLATES) == EFFECT_TEMPLATES


@pytest.mark.parametrize("tpl", sorted(EFFECT_TEMPLATES))
def test_default_templates_pass_their_own_qa(tpl, tmp_path):
    m = effects.build_effect(spec_for(tpl), tmp_path)
    assert m["status"] == "passed", m["final_failures"]
    assert m["kind"] == "effect" and m["iterations"][0]["metrics"]["frames"] == 16


def test_outputs_are_consistent(tmp_path):
    m = effects.build_effect(spec_for("explosion", frames=12, size=64), tmp_path)
    final = tmp_path / "fx_explosion" / "final"
    names = {o["file"] for o in m["outputs"]}
    assert names == {"fx_explosion_flipbook.png", "fx_explosion_particles.json",
                     "fx_explosion_contact_sheet.png", "fx_explosion_preview.gif"}
    sheet = Image.open(final / "fx_explosion_flipbook.png")
    assert sheet.mode == "RGBA" and sheet.size == (4 * 64, 3 * 64)            # 12 frames -> 4 x 3
    p = json.loads((final / "fx_explosion_particles.json").read_text())
    assert p["flipbook"] == {"texture": "fx_explosion_flipbook.png", "columns": 4, "rows": 3, "frames": 12,
                             "frame_size_px": 64, "fps": 15.0}
    assert len(p["curves"]["alpha_over_life"]) == 12 and max(p["curves"]["alpha_over_life"]) == 1.0
    assert Image.open(final / "fx_explosion_preview.gif").n_frames == 12
    for o in m["outputs"]:
        assert (final / o["file"]).stat().st_size == o["bytes"] > 0


def test_same_spec_gives_identical_files(tmp_path):
    a = effects.build_effect(spec_for("smoke"), tmp_path / "a")
    b = effects.build_effect(spec_for("smoke"), tmp_path / "b")
    assert {o["file"]: o["sha256"] for o in a["outputs"]} == {o["file"]: o["sha256"] for o in b["outputs"]}
    c = effects.build_effect(JobSpec(name="fx_smoke", kind="effect", effect={"template": "smoke"}, seed=5), tmp_path / "c")
    assert {o["file"]: o["sha256"] for o in a["outputs"]}["fx_smoke_flipbook.png"] != \
        {o["file"]: o["sha256"] for o in c["outputs"]}["fx_smoke_flipbook.png"]


def test_clipped_effect_is_shrunk_until_it_fits(tmp_path):
    m = effects.build_effect(spec_for("shockwave", radius=0.95, fade=0.0), tmp_path)
    first = m["iterations"][0]
    assert {f["code"] for f in first["failures"]} >= {"effect_clipped", "effect_no_fade"}
    assert m["status"] == "passed", m["final_failures"]
    last = m["iterations"][-1]["params"]
    assert last["radius"] < 0.95 and last["fade"] == 1.0


def test_invisible_effect_is_grown(tmp_path):
    m = effects.build_effect(spec_for("fireball", radius=0.15), tmp_path)
    assert m["iterations"][0]["failures"][0]["code"] == "effect_empty"
    assert m["status"] == "passed" and m["iterations"][-1]["params"]["radius"] > 0.15


def test_unfixable_problem_goes_to_needs_review(tmp_path):
    # a shockwave that never expands: timing can't be repaired by radius/fade
    m = effects.build_effect(spec_for("smoke"), tmp_path)
    cfg = effects.resolved(spec_for("smoke"))
    bad = dict(m["iterations"][0]["metrics"], peak_fraction=0.99)
    fails = effects.check(bad, cfg)
    assert [f["code"] for f in fails] == ["effect_timing"] and not fails[0]["fixable"]
    assert effects.adjust({"radius": .5, "fade": 1., "fade_start": .5, "turbulence": .5}, fails)[1] == []


def test_pop_between_frames_is_detected():
    cfg = effects.resolved(spec_for("fireball"))
    frames = effects.render_frames(cfg, {k: float(cfg[k]) for k in ("radius", "turbulence", "fade", "fade_start")}, 0)
    frames[7] = frames[7] * 0 + np.array([1, 1, 1, 1], np.float32)       # one frame suddenly fills the whole sprite
    m = effects.analyze_frames(frames, cfg, 4, 4)
    assert "effect_pop" in {f["code"] for f in effects.check(m, cfg)}


def test_style_limits_the_colour_ramp(tmp_path):
    plain = effects.build_effect(spec_for("fireball"), tmp_path / "p")
    s = JobSpec(name="fx_fireball", kind="effect", effect={"template": "fireball"}, style_template="souls_like")
    s.validate()
    styled = effects.build_effect(s, tmp_path / "s")
    sp = plain["iterations"][0]["metrics"]["mean_saturation"]
    ss = styled["iterations"][0]["metrics"]["mean_saturation"]
    assert ss < sp and ss <= 0.34 + 0.03
    assert styled["style_resolved"]["name"] == "souls_like"


def test_effect_spec_validation():
    for bad in ({"template": "tornado"}, {"template": "fireball", "size": 4}, {"template": "fireball", "frames": 1},
                {"template": "fireball", "radius": 5}, {"template": "fireball", "sparkle": 1},
                {"template": "fireball", "blend": "multiply"}, {"template": "fireball", "ramp": ["#fff"]},
                {"template": "fireball", "ramp": ["#ffffff"]}):
        with pytest.raises(ValueError):
            JobSpec(name="a", kind="effect", effect=bad).validate()


def test_orchestrator_runs_effect_jobs_without_blender(tmp_path):
    spec = load_job(ROOT / "jobs" / "fx_shockwave.yaml")
    m = build_one(spec, tmp_path, Backend(kind="blender", exe="/nonexistent/blender"))
    assert m["status"] == "passed"
    assert json.loads((tmp_path / "fx_shockwave" / "manifest.json").read_text())["fingerprint"] == spec.fingerprint()


def test_shipped_effect_jobs_load():
    for n in ("fx_fireball", "fx_shockwave"):
        assert load_job(ROOT / "jobs" / f"{n}.yaml").kind == "effect"
