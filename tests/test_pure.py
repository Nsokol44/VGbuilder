"""Tests that need no Blender."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from charforge import fit, fixer, qa, rigdefs, sketch as sk  # noqa: E402
from charforge.bodygen import build_body, choose_segments, silhouette  # noqa: E402
from charforge.spec import DEFAULT_BODY_PARAMS, JobSpec, load_job  # noqa: E402
from make_sketch import draw  # noqa: E402


@pytest.fixture(scope="module")
def sketches(tmp_path_factory):
    d = tmp_path_factory.mktemp("sk")
    draw(d / "hero.png")
    draw(d / "stocky.png", shoulder=135, hip=115, head_r=58, leg_len=360, stocky=1.12)
    return d


def body_params(**kw):
    return {**DEFAULT_BODY_PARAMS, "segments": 8.0, **kw}


# ------------------------------------------------------------------ spec
def test_job_file_loads_and_validates():
    spec = load_job(ROOT / "jobs" / "hero.yaml")
    assert spec.name == "hero" and spec.resolved_budget() == 2500
    assert spec.resolve_path(spec.sketch).exists()


@pytest.mark.parametrize("bad", [
    {"name": "has space"}, {"style": "photoreal"}, {"lods": [0.5, 1.0]}, {"lods": [0.9, 0.5]},
    {"height_m": 0.01}, {"params": {"nonsense": 1}}, {"qa": {"nonsense": 1}},
    {"provider": "import_glb"}, {"provider": "command", "provider_args": {"command": "x"}},
])
def test_invalid_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        JobSpec(**{"name": "x", **bad}).validate()


def test_unknown_yaml_key_rejected(tmp_path):
    p = tmp_path / "j.yaml"
    p.write_text("name: a\nstyel: lowpoly\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_job(p)


def test_fingerprint_changes_with_spec_and_sketch(tmp_path, sketches):
    a = JobSpec(name="a", sketch=str(sketches / "hero.png"))
    b = JobSpec(name="a", sketch=str(sketches / "stocky.png"))
    c = JobSpec(name="a", sketch=str(sketches / "hero.png"), height_m=2.0)
    assert len({a.fingerprint(), b.fingerprint(), c.fingerprint()}) == 3
    assert a.fingerprint() == JobSpec(name="a", sketch=str(sketches / "hero.png")).fingerprint()


# ---------------------------------------------------------------- sketch
def test_outline_sketch_becomes_filled_silhouette(sketches):
    m = sk.load_mask(sketches / "hero.png")
    h, w = m.shape
    assert m[h // 2, w // 2]  # torso interior is filled even though only the outline was drawn
    assert 0.12 < m.mean() < 0.30


def test_crotch_detected_from_leg_gap(sketches):
    cf = sk.crotch_fraction(sk.load_mask(sketches / "hero.png"))
    assert cf is not None and 0.40 < cf < 0.52


def test_crotch_is_none_when_legs_are_together():
    solid = np.zeros((400, 200), bool)
    solid[10:390, 60:140] = True
    assert sk.crotch_fraction(solid) is None


def test_iou_identity_and_scale_invariance(sketches):
    m = sk.load_mask(sketches / "hero.png")
    assert sk.silhouette_iou(m, m) == pytest.approx(1.0)
    small = np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray((m * 255).astype(np.uint8)).resize((350, 500))) > 127
    assert sk.silhouette_iou(m, small) > 0.95  # nearest-neighbour half-res loses thin edge pixels


def test_iou_distinguishes_different_figures(sketches):
    a, b = sk.load_mask(sketches / "hero.png"), sk.load_mask(sketches / "stocky.png")
    assert sk.silhouette_iou(a, b) < 0.95


# ---------------------------------------------------------------- bodygen
def test_body_is_deterministic_and_mirror_symmetric():
    p = body_params()
    a, b = build_body(p, 1.8), build_body(p, 1.8)
    assert np.array_equal(a.verts, b.verts)
    v = a.verts
    mirrored = v * np.array([-1, 1, 1])
    d = np.linalg.norm(mirrored[:, None, :] - v[None, :, :], axis=2).min(axis=1)
    assert d.max() < 1e-6


def test_body_height_and_grounding():
    v = build_body(body_params(), 1.8).verts
    assert v[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert v[:, 2].max() == pytest.approx(1.8, abs=0.02)


def test_every_vertex_has_weights_summing_to_one():
    b = build_body(body_params(), 1.8)
    assert set(b.weights) == set(range(len(b.verts)))
    for ws in b.weights.values():
        assert sum(w for _, w in ws) == pytest.approx(1.0)
        assert all(r in rigdefs.ROLES for r, _ in ws)


def test_tri_count_grows_with_segments_and_budget_is_respected():
    t = [build_body(body_params(segments=float(s)), 1.8).tri_count for s in (4, 8, 16)]
    assert t[0] < t[1] < t[2]
    for budget in (600, 1500, 5000):
        s = choose_segments(body_params(), 1.8, budget)
        assert build_body(body_params(segments=float(s)), 1.8).tri_count <= budget or s == 4


# -------------------------------------------------------------------- fit
def test_fit_beats_seed_and_respects_pins(sketches):
    target = sk.load_mask(sketches / "stocky.png")
    p = body_params(**sk.fit_params(target))
    seed = fit.score(p, 1.8, target)
    best, iou, evals = fit.optimize(p, 1.8, target, pinned={"arm_angle_deg"})
    assert iou > seed + 0.05 and iou > 0.8
    assert best["arm_angle_deg"] == p["arm_angle_deg"]
    assert evals < 600
    for k, (lo, hi) in fit.BOUNDS.items():
        assert lo <= best[k] <= hi


def test_prior_trades_iou_for_human_proportions(sketches):
    target = sk.load_mask(sketches / "stocky.png")
    p = body_params(**sk.fit_params(target))
    free, iou0, _ = fit.optimize(p, 1.8, target, prior=0.0)
    reg, iou1, _ = fit.optimize(p, 1.8, target, prior=2.0)
    assert fit.prior_penalty(reg) < fit.prior_penalty(free)
    assert iou1 <= iou0 + 1e-6


def test_body_vertices_are_float32_exact():
    v = build_body(body_params(), 1.8).verts
    assert np.array_equal(v, v.astype(np.float32).astype(np.float64))


def test_initial_params_pins_win_and_budget_chooses_segments(sketches):
    spec = JobSpec(name="a", sketch=str(sketches / "hero.png"), poly_budget=800, params={"shoulder_half": 0.12})
    mask = sk.load_mask(sketches / "hero.png")
    p, info = fixer.initial_params(spec, mask)
    assert p["shoulder_half"] == 0.12
    assert build_body(p, spec.height_m).tri_count <= 800
    assert info["fit"]["predicted_iou"] > info["fit"]["seed_iou"]


# --------------------------------------------------------------------- qa
def _good_metrics():
    return {
        "lod_tris": [1000, 500], "height_m": 1.8, "symmetry": 0.001, "zero_weight_frac": 0.0,
        "loose_parts": 20, "bone_roles": rigdefs.ROLES, "has_uv": True, "stretch_max": 1.2,
        "roundtrip": {"ok": True}, "silhouette_iou": 0.9,
    }


def test_qa_passes_good_metrics():
    spec = JobSpec(name="a", sketch="x.png", poly_budget=1200)
    assert qa.evaluate(_good_metrics(), spec) == []


@pytest.mark.parametrize("patch,code", [
    ({"lod_tris": [5000, 2000]}, "over_budget"),
    ({"height_m": 2.5}, "height_off"),
    ({"symmetry": 0.2}, "asymmetric"),
    ({"zero_weight_frac": 0.2}, "unweighted"),
    ({"loose_parts": 500}, "fragmented"),
    ({"bone_roles": ["root"]}, "missing_bones"),
    ({"has_uv": False}, "no_uv"),
    ({"stretch_max": 9.0}, "bad_deformation"),
    ({"stretch_max": float("nan")}, "bad_deformation"),
    ({"roundtrip": {"ok": False}}, "roundtrip_failed"),
    ({"silhouette_iou": 0.3}, "silhouette_low"),
])
def test_qa_flags_each_failure(patch, code):
    spec = JobSpec(name="a", sketch="x.png", poly_budget=1200)
    codes = [f["code"] for f in qa.evaluate({**_good_metrics(), **patch}, spec)]
    assert code in codes


def test_only_budget_failures_are_auto_fixable():
    spec = JobSpec(name="a", poly_budget=1200)
    fails = qa.evaluate({**_good_metrics(), "lod_tris": [5000, 1], "symmetry": 0.2}, spec)
    assert {f["code"]: f["fixable"] for f in fails} == {"over_budget": True, "asymmetric": False}
    new, changes = fixer.next_params(spec, {**DEFAULT_BODY_PARAMS, "segments": 10.0}, fails)
    assert new["segments"] == 9.0 and changes


# ----------------------------------------------------------------- rigdefs
def test_naming_schemes_cover_all_roles_and_roundtrip():
    for naming, table in rigdefs.NAMING.items():
        assert set(table) == set(rigdefs.ROLES)
        assert len(set(table.values())) == len(rigdefs.ROLES)  # no duplicate bone names
        names = [rigdefs.bone_name(r, naming) for r in rigdefs.ROLES]
        assert rigdefs.roles_from_names(names, naming) == set(rigdefs.ROLES)


def test_hierarchy_parents_come_first():
    seen = set()
    for role, parent in rigdefs.HIERARCHY.items():
        assert parent is None or parent in seen
        seen.add(role)


def test_joints_from_vertices_follows_t_pose_arms():
    p = body_params(arm_angle_deg=80.0, arm_frac=0.45)
    v = build_body(p, 1.8).verts
    j = rigdefs.joints_from_vertices(v)
    wrist_x = j["hand_l"][0][0]
    default_wrist_x = rigdefs.joints_from_params({**DEFAULT_BODY_PARAMS}, 1.8)["hand_l"][0][0]
    assert wrist_x > default_wrist_x + 0.2
    assert j["hand_r"][0][0] < 0 < j["hand_l"][0][0]


# ---------------------------------------------------------------- engines
def test_engine_presets_and_auto_naming():
    from charforge import engines

    assert set(engines.PRESETS) == {"generic", "unity", "unreal", "godot"}
    assert JobSpec(name="a", target_engine="unity").resolved_naming() == "humanoid"
    assert JobSpec(name="a", target_engine="godot").resolved_naming() == "humanoid"
    assert JobSpec(name="a", target_engine="unreal").resolved_naming() == "ue"
    assert JobSpec(name="a").resolved_naming() == "ue"                       # generic keeps the old default
    assert JobSpec(name="a", target_engine="unity", naming="simple").resolved_naming() == "simple"   # explicit wins
    assert all(not p.verified_in_engine for p in engines.PRESETS.values())   # we never imported into a real engine


def test_humanoid_names_follow_unity_godot_convention():
    n = rigdefs.NAMING["humanoid"]
    assert n["pelvis"] == "Hips" and n["thigh_l"] == "LeftUpperLeg" and n["forearm_r"] == "RightLowerArm"
    assert len(set(n.values())) == len(rigdefs.ROLES)


def test_qa_flags_too_many_skin_influences():
    spec = JobSpec(name="a", poly_budget=1200, target_engine="unity")
    assert "too_many_influences" in [f["code"] for f in qa.evaluate({**_good_metrics(), "max_influences": 6}, spec)]
    assert "too_many_influences" not in [f["code"] for f in qa.evaluate({**_good_metrics(), "max_influences": 4}, spec)]
    ue = JobSpec(name="a", poly_budget=1200, target_engine="unreal")
    assert "too_many_influences" not in [f["code"] for f in qa.evaluate({**_good_metrics(), "max_influences": 6}, ue)]


# ---------------------------------------------------- LOD validation / provenance
def _lod(k, **kw):
    base = {"lod": k, "tris": int(1000 * [1.0, 0.5, 0.25][k]), "target_ratio": [1.0, 0.5, 0.25][k],
            "ratio": [1.0, 0.5, 0.25][k], "height_m": 1.8, "armature_ok": True, "groups_match_lod0": True,
            "groups_are_bones": True, "zero_weight_frac": 0.0, "has_uv": True, "has_material": True,
            "silhouette_front_vs_lod0": 0.97, "silhouette_side_vs_lod0": 0.97}
    return {**base, **kw}


def _lod_metrics(**lod2):
    return {**_good_metrics(), "lods": [_lod(0), _lod(1), _lod(2, **lod2)]}


def _codes(metrics, **spec_kw):
    return [f["code"] for f in qa.evaluate(metrics, JobSpec(name="a", poly_budget=1200, **spec_kw))]


def test_good_lod_chain_passes():
    assert _codes(_lod_metrics()) == []


@pytest.mark.parametrize("patch,code", [
    ({"ratio": 0.05}, "lod_ratio_off"),
    ({"silhouette_front_vs_lod0": 0.6}, "lod_silhouette_drift"),
    ({"silhouette_side_vs_lod0": 0.6}, "lod_silhouette_drift"),
    ({"height_m": 1.5}, "lod_inconsistent"),
    ({"armature_ok": False}, "lod_inconsistent"),
    ({"groups_match_lod0": False}, "lod_inconsistent"),
    ({"has_uv": False}, "lod_inconsistent"),
    ({"zero_weight_frac": 0.3}, "lod_inconsistent"),
])
def test_each_lod_problem_is_flagged_with_its_lod_index(patch, code):
    fails = qa.evaluate(_lod_metrics(**patch), JobSpec(name="a", poly_budget=1200))
    hit = [f for f in fails if f["code"] == code]
    assert hit and hit[0]["lod"] == 2 and not hit[0]["fixable"]


def test_embedded_provenance_must_match_the_job():
    spec = JobSpec(name="a", poly_budget=1200)
    fp = spec.fingerprint()
    ok = [{"fingerprint": fp, "lod": 0}, {"fingerprint": fp, "lod": 1}]
    assert "provenance_missing" not in _codes({**_good_metrics(), "embedded": ok})
    for bad in ([None, ok[1]], [ok[0], {"fingerprint": "other", "lod": 1}], [ok[0], {"fingerprint": fp, "lod": 0}]):
        assert "provenance_missing" in _codes({**_good_metrics(), "embedded": bad})
