"""Accessories and animation cycles: the parts that need no Blender."""
from __future__ import annotations

import math

import numpy as np
import pytest

from vgbuilder import accessories, fixer, qa, rigdefs
from vgbuilder.bodygen import build_body
from vgbuilder.spec import DEFAULT_BODY_PARAMS, JobSpec

H = 1.8
P = {**DEFAULT_BODY_PARAMS, "segments": 8.0}


def parts(names):
    return {p.name: p for p in accessories.build_parts(names, P, H)}


def test_which_parts_join_the_body_and_which_stay_separate():
    ps = parts(["cape", "pauldrons", "horns"])
    assert {n for n, p in ps.items() if p.joined} == {"pauldrons", "horns"}
    assert not ps["cape"].joined and ps["cape"].tags["vgbuilder_physics"] == "cloth"


def test_parts_are_x_mirror_symmetric():
    for name in ("pauldrons", "horns", "cape"):
        v = parts([name])[name].body.verts
        mirrored = v * np.array([-1, 1, 1])
        d = np.linalg.norm(mirrored[:, None, :] - v[None, :, :], axis=2).min(axis=1)
        assert d.max() < 1e-5, name


def test_horns_sit_on_the_head_and_pauldrons_on_the_shoulders():
    j = rigdefs.joints_from_params(P, H)
    horns = parts(["horns"])["horns"].body.verts
    assert horns[:, 2].max() > H and horns[:, 2].min() > H - 0.15 * H
    assert {r for ws in parts(["horns"])["horns"].body.weights.values() for r, _ in ws} == {"head"}
    pl = parts(["pauldrons"])["pauldrons"].body
    assert {r for ws in pl.weights.values() for r, _ in ws} == {"upperarm_l", "upperarm_r"}
    sh = j["upperarm_l"][0]
    left = pl.verts[pl.verts[:, 0] > 0]
    assert np.linalg.norm(left.mean(axis=0) - np.array(sh)) < 0.05 * H


def test_cape_hangs_behind_the_body_with_only_the_top_row_pinned():
    cape = parts(["cape"])["cape"]
    v, pin = cape.body.verts, cape.point_attrs["_CLOTH_PIN"]
    assert len(pin) == len(v) and pin.sum() == 9 and set(pin) == {0.0, 1.0}
    assert (v[:, 1] > 0.05 * H).all()                                   # +Y is behind a character that faces -Y
    top = v[pin == 1.0]
    assert np.ptp(top[:, 2]) < 1e-6 and v[:, 2].min() < top[0, 2] - 0.3 * H
    assert {r for ws in cape.body.weights.values() for r, _ in ws} <= {"chest", "spine"}


def test_cape_faces_outward_triangles_are_valid():
    cape = parts(["cape"])["cape"].body
    t = cape.triangles()
    n = np.cross(cape.verts[t[:, 1]] - cape.verts[t[:, 0]], cape.verts[t[:, 2]] - cape.verts[t[:, 0]])
    assert (np.linalg.norm(n, axis=1) > 1e-9).all()
    assert (n[:, 1] > 0).mean() > 0.95                                   # all facing away from the body


def test_triangle_accounting():
    names = ["cape", "pauldrons", "horns"]
    joined, sep = accessories.tri_split(names, P, H)
    assert joined + sep == accessories.tri_total(names, P, H) and sep == 128
    assert accessories.tri_total([], P, H) == 0


def test_budget_reserves_room_for_accessories(tmp_path):
    spec = JobSpec(name="a", poly_budget=2500, accessories=["pauldrons", "horns", "cape"])
    p, _ = fixer.initial_params(spec, None)
    total = build_body(p, H).tri_count + accessories.tri_total(spec.accessories, p, H)
    assert total <= 2500
    plain, _ = fixer.initial_params(JobSpec(name="a", poly_budget=2500), None)
    assert p["segments"] <= plain["segments"]


def test_accessories_need_the_procedural_body():
    with pytest.raises(ValueError, match="accessories"):
        JobSpec(name="a", provider="import_glb", provider_args={"glb": "x.glb"}, accessories=["cape"]).validate()


# --------------------------------------------------------------- animation poses
def anim():
    from vgbuilder.bl import anim as a  # needs bpy only for import; pose_for itself is pure
    return a


bpy = pytest.importorskip("bpy")


@pytest.mark.parametrize("name", ["idle", "walk"])
def test_cycles_are_periodic(name):
    a = anim()
    p0, p1 = a.pose_for(name, 0.0, H), a.pose_for(name, 1.0, H)
    assert p0.keys() == p1.keys()
    for role in p0:
        assert np.allclose(p0[role][0], p1[role][0], atol=1e-9) and abs(p0[role][1] - p1[role][1]) < 1e-9


def test_walk_swings_arms_against_legs_and_flexes_the_knee_of_the_swinging_leg():
    a = anim()
    q = a.pose_for("walk", 0.25, H)             # sin = 1: left leg fully forward
    assert q["thigh_l"][0][0] < 0 < q["thigh_r"][0][0]                  # forward swing is negative X for down-pointing bones
    assert q["upperarm_l"][0][0] > 0 > q["upperarm_r"][0][0]            # arms oppose the legs
    mid = a.pose_for("walk", 0.0, H)            # cos = 1: left leg is passing through, knee bent
    assert mid["shin_l"][0][0] > 0.5 and mid["shin_r"][0][0] == 0


def test_idle_stays_small():
    a = anim()
    worst = max(abs(x) for ph in np.linspace(0, 1, 25) for eul, _ in a.pose_for("idle", ph, H).values() for x in eul)
    assert worst < 0.1


def test_sign_conventions_match_the_rig():
    """The docstring's claim, checked on the real armature: forward = -Y."""
    from vgbuilder.bl import anim as a, condition, rig

    condition.reset_scene()
    body = build_body(P, H)
    arm = rig.build_armature(body.joints, "simple", "t")
    a.reset_pose(arm)
    rest_l, rest_r = (np.array(arm.pose.bones[n].tail) for n in ("shin_l", "shin_r"))
    rest_hand = np.array(arm.pose.bones["forearm_l"].tail)
    a._apply(arm, "simple", "walk", 0.25, H)
    bpy.context.view_layer.update()
    foot_l, foot_r = (np.array(arm.pose.bones[n].tail) for n in ("shin_l", "shin_r"))
    hand_l = np.array(arm.pose.bones["forearm_l"].tail)
    assert foot_l[1] < rest_l[1] - 0.05 and foot_r[1] > rest_r[1] + 0.05     # left foot forward (-Y), right back
    assert hand_l[1] > rest_hand[1] + 0.03                                     # left hand back while left foot is forward


# -------------------------------------------------------------------------- qa
def _metrics(**kw):
    return {"lod_tris": [1000], "height_m": 1.8, "symmetry": 0.0, "zero_weight_frac": 0.0, "loose_parts": 1,
            "bone_roles": rigdefs.ROLES, "bone_names": ["Hips", "Spine"], "has_uv": True, "stretch_max": 1.0,
            "roundtrip": {"ok": True}, **kw}


GOOD_ANIM = {"frames": 25, "fps": 24, "duration_s": 1.0, "loops": True, "max_rotation_rad": 0.7, "max_stretch": 1.1,
             "min_z_m": -0.01, "bones_keyed": 11}


def _codes(metrics, **spec_kw):
    return [f["code"] for f in qa.evaluate(metrics, JobSpec(name="a", poly_budget=1200, **spec_kw))]


def _content(anim_channels=10, targets=("Hips",), cape=True):
    nodes = {"a_cape": {"vgbuilder_physics": "cloth"}} if cape else {}
    return {"nodes": nodes, "mesh_attributes": {"a_cape": ["POSITION", "_CLOTH_PIN"]} if cape else {},
            "animations": {"walk": {"channels": anim_channels, "targets": list(targets)}}}


def test_good_animation_and_cape_pass():
    m = _metrics(animations={"walk": GOOD_ANIM}, accessories={"cape": {"tris": 128}}, glb_content=[_content()])
    assert _codes(m, animations=["walk"], accessories=["cape"]) == []


@pytest.mark.parametrize("patch,code", [
    ({"loops": False}, "anim_not_looping"),
    ({"max_stretch": 3.0}, "anim_bad_deformation"),
    ({"min_z_m": -0.2}, "anim_floor_penetration"),
    ({"max_rotation_rad": 2.5}, "anim_range"),
])
def test_each_animation_problem_is_flagged(patch, code):
    m = _metrics(animations={"walk": {**GOOD_ANIM, **patch}}, glb_content=[_content(cape=False)])
    assert code in _codes(m, animations=["walk"])


def test_missing_animation_everywhere_is_flagged():
    assert "anim_missing" in _codes(_metrics(), animations=["walk"])
    m = _metrics(animations={"walk": GOOD_ANIM}, glb_content=[_content(anim_channels=0, cape=False)])
    assert "anim_missing" in _codes(m, animations=["walk"])
    m = _metrics(animations={"walk": GOOD_ANIM}, glb_content=[_content(targets=("Mystery",), cape=False)])
    assert "anim_missing" in _codes(m, animations=["walk"])


def test_cape_without_cloth_tag_or_pin_is_flagged():
    base = dict(animations={}, accessories={"cape": {"tris": 1}})
    assert "accessory_missing" in _codes(_metrics(**base, glb_content=[{"nodes": {"a_cape": {}}, "mesh_attributes": {"a_cape": ["_CLOTH_PIN"]}, "animations": {}}]), accessories=["cape"])
    assert "accessory_missing" in _codes(_metrics(**base, glb_content=[{"nodes": {"a_cape": {"vgbuilder_physics": "cloth"}}, "mesh_attributes": {"a_cape": ["POSITION"]}, "animations": {}}]), accessories=["cape"])
    assert "accessory_missing" in _codes(_metrics(), accessories=["horns"])
