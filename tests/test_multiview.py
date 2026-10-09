"""Front + side sketch fitting and proportion reporting (no Blender needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from vgbuilder import fit, fixer, proportions, providers, qa, sketch as sk  # noqa: E402
from vgbuilder.bodygen import build_body, silhouette_side  # noqa: E402
from vgbuilder.spec import DEFAULT_BODY_PARAMS, JobSpec  # noqa: E402
from make_sketch import draw, draw_side  # noqa: E402


@pytest.fixture(scope="module")
def sk_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("mv")
    draw(d / "front.png")
    draw(d / "stocky_front.png", shoulder=135, hip=115, head_r=58, leg_len=360, stocky=1.12)
    draw_side(d / "side.png")
    draw_side(d / "side_thin.png", depth=0.8)
    draw_side(d / "side_deep.png", depth=1.3)
    draw_side(d / "side_slim_legs.png", leg_thick=0.6)
    draw_side(d / "side_fat_legs.png", leg_thick=1.5)
    return d


def params(**kw):
    return {**DEFAULT_BODY_PARAMS, "segments": 8.0, **kw}


def test_side_silhouette_faces_left():
    m = silhouette_side(build_body(params(), 1.8))
    h, w = m.shape
    foot_rows = m[int(h * 0.96):]
    xs = np.nonzero(foot_rows)[1]
    body_cx = np.nonzero(m[int(h * 0.3): int(h * 0.4)])[1].mean()
    assert xs.mean() < body_cx - 5, "toes (front, -Y) should be on the image's left"


def test_side_sketch_orientation_matches_the_body(sk_dir):
    target = sk.load_side_mask(sk_dir / "side.png")
    body = silhouette_side(build_body(params(torso_depth=1.0, head_scale=0.9, leg_frac=0.47), 1.8))
    flipped = body[:, ::-1]
    assert sk.silhouette_iou(body, target) > sk.silhouette_iou(flipped, target) + 0.05


def test_facing_right_is_mirrored(sk_dir):
    left = sk.load_side_mask(sk_dir / "side.png", "left")
    right = sk.load_side_mask(sk_dir / "side.png", "right")
    assert np.array_equal(left[:, ::-1], right)


def test_side_seed_orders_torso_depth(sk_dir):
    thin = sk.fit_params_side(sk.load_side_mask(sk_dir / "side_thin.png"))["torso_depth"]
    deep = sk.fit_params_side(sk.load_side_mask(sk_dir / "side_deep.png"))["torso_depth"]
    assert thin < 1.0 < deep


def _fit_free(sk_dir, side_name, free):
    front = sk.load_mask(sk_dir / "front.png")
    side = sk.load_side_mask(sk_dir / side_name)
    p = params(**sk.fit_params(front))
    pinned = set(fit.BOUNDS) - set(free)
    best, _, _ = fit.optimize(p, 1.8, front, pinned=pinned, side=side)
    return best


def test_side_view_recovers_torso_depth(sk_dir):
    thin = _fit_free(sk_dir, "side_thin.png", {"torso_depth"})["torso_depth"]
    deep = _fit_free(sk_dir, "side_deep.png", {"torso_depth"})["torso_depth"]
    assert deep > thin + 0.3
    assert abs(thin - 0.8) < 0.2 and abs(deep - 1.3) < 0.25


def test_side_view_separates_thick_legs_from_thin_legs(sk_dir):
    """Same front sketch, different side sketches -> different leg thickness. Front alone cannot do this."""
    slim = _fit_free(sk_dir, "side_slim_legs.png", {"leg_thickness"})["leg_thickness"]
    fat = _fit_free(sk_dir, "side_fat_legs.png", {"leg_thickness"})["leg_thickness"]
    assert fat > slim + 0.3


def test_torso_depth_is_not_free_without_a_side_sketch(sk_dir):
    front = sk.load_mask(sk_dir / "front.png")
    p = params(**sk.fit_params(front), torso_depth=1.3)
    best, _, _ = fit.optimize(p, 1.8, front, max_rounds=3)
    assert best["torso_depth"] == 1.3


def test_joint_fit_reports_both_views(sk_dir):
    front = sk.load_mask(sk_dir / "front.png")
    side = sk.load_side_mask(sk_dir / "side.png")
    spec = JobSpec(name="a", sketch=str(sk_dir / "front.png"), sketch_side=str(sk_dir / "side.png"), poly_budget=2500)
    p, info = fixer.initial_params(spec, front, side)
    f, s = fit.view_scores(p, 1.8, front, side)
    assert info["fit"]["predicted_iou"] == pytest.approx(f, abs=1e-4)
    assert info["fit"]["predicted_side_iou"] == pytest.approx(s, abs=1e-4)
    assert f > 0.78 and s > 0.8
    assert "ratios" in info["proportions"]


def test_leg_frac_stays_near_the_observed_crotch(sk_dir):
    front = sk.load_mask(sk_dir / "front.png")
    crotch = sk.crotch_fraction(front)
    spec = JobSpec(name="a", sketch="x", poly_budget=2500)
    p, _ = fixer.initial_params(spec, front)
    assert abs(p["leg_frac"] - crotch) < 0.04


def test_pinned_param_is_never_moved_by_the_joint_fit(sk_dir):
    front = sk.load_mask(sk_dir / "front.png")
    side = sk.load_side_mask(sk_dir / "side.png")
    spec = JobSpec(name="a", sketch="x", poly_budget=2500, params={"torso_depth": 1.4})
    p, _ = fixer.initial_params(spec, front, side)
    assert p["torso_depth"] == 1.4


# -------------------------------------------------------------- proportions
def test_default_body_has_no_proportion_warnings():
    r = proportions.report(params(), 1.8)
    assert r["warnings"] == []
    assert 6 < r["ratios"]["heads_tall"] < 8


def test_extreme_limbs_are_flagged():
    r = proportions.report(params(leg_thickness=2.2), 1.8)
    assert [w["ratio"] for w in r["warnings"]] == ["thigh_slenderness"]
    assert r["warnings"][0]["side"] == "above"
    assert any(w["ratio"] == "heads_tall" for w in proportions.report(params(head_scale=0.6), 1.8)["warnings"])


def test_realistic_alternative_is_offered_when_a_fit_looks_odd(sk_dir):
    front = sk.load_mask(sk_dir / "stocky_front.png")
    spec = JobSpec(name="a", sketch="x", poly_budget=2500)
    p, info = fixer.initial_params(spec, front)
    if not info["proportions"]["warnings"]:
        pytest.skip("this sketch fits with plausible proportions")
    alt = info["realistic_alternative"]
    assert fit.prior_penalty(alt["params"] | {"leg_frac": 0.5}) <= fit.prior_penalty(p) + 1e-9
    assert alt["front_iou_change"] <= 1e-6      # realism costs overlap, and the report says how much


# ----------------------------------------------------------------- qa/spec
def _metrics(**kw):
    from vgbuilder import rigdefs
    return {"lod_tris": [1000, 500], "height_m": 1.8, "symmetry": 0.001, "zero_weight_frac": 0.0,
            "loose_parts": 20, "bone_roles": rigdefs.ROLES, "has_uv": True, "stretch_max": 1.2,
            "roundtrip": {"ok": True}, "silhouette_iou": 0.9, **kw}


def test_side_iou_gate_only_applies_with_a_side_sketch():
    low = _metrics(silhouette_side_iou=0.3)
    assert qa.evaluate(low, JobSpec(name="a", sketch="x.png", poly_budget=1200)) == []
    fails = qa.evaluate(low, JobSpec(name="a", sketch="x.png", sketch_side="y.png", poly_budget=1200))
    assert [f["code"] for f in fails] == ["silhouette_side_low"] and not fails[0]["fixable"]
    assert qa.evaluate(_metrics(silhouette_side_iou=0.9), JobSpec(name="a", sketch="x.png", sketch_side="y.png", poly_budget=1200)) == []


def test_qa_score_blends_the_side_view():
    assert qa.score(_metrics(silhouette_side_iou=0.5), []) == pytest.approx(0.7)


def test_spec_validation_and_fingerprint(sk_dir):
    with pytest.raises(ValueError):
        JobSpec(name="a", sketch_side="y.png").validate()
    with pytest.raises(ValueError):
        JobSpec(name="a", sketch="x.png", sketch_side="y.png", side_facing="up").validate()
    a = JobSpec(name="a", sketch=str(sk_dir / "front.png"), sketch_side=str(sk_dir / "side_thin.png"))
    b = JobSpec(name="a", sketch=str(sk_dir / "front.png"), sketch_side=str(sk_dir / "side_deep.png"))
    assert a.fingerprint() != b.fingerprint()


def test_command_provider_receives_the_side_image(tmp_path, sk_dir):
    from glbs import make_glb
    (tmp_path / "ok.glb").write_bytes(make_glb())
    cmd = (f"{{python}} -c \"import sys, shutil; shutil.copy(sys.argv[1], sys.argv[2]); open(sys.argv[2] + '.side', 'w').write(sys.argv[3])\" "
           f"{tmp_path / 'ok.glb'} {{out}} {{side_image}}")
    spec = JobSpec(name="gen", sketch=str(sk_dir / "front.png"), sketch_side=str(sk_dir / "side.png"),
                   provider="command", provider_args={"command": cmd}, base_dir=str(sk_dir))
    out, prov = providers.provide_glb(spec, tmp_path)
    assert Path(str(out) + ".side").read_text() == str(sk_dir / "side.png")
    # changing only the side sketch must not reuse a cached mesh
    spec2 = JobSpec(name="gen", sketch=str(sk_dir / "front.png"), sketch_side=str(sk_dir / "side_deep.png"),
                    provider="command", provider_args={"command": cmd}, base_dir=str(sk_dir))
    out2, prov2 = providers.provide_glb(spec2, tmp_path)
    assert out2 != out and not prov2["cached"]
