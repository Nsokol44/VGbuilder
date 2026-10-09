"""Wear baking and image-based texture QA (no Blender needed)."""
from __future__ import annotations

import numpy as np
import pytest

from vgbuilder import fixer, lookdev, qa, styles
from vgbuilder.bodygen import _Builder, build_body
from vgbuilder.spec import DEFAULT_BODY_PARAMS, JobSpec

H = 1.8


@pytest.fixture(scope="module")
def mesh():
    b = build_body({**DEFAULT_BODY_PARAMS, "segments": 8.0}, H)
    tris = b.triangles()
    return b.verts, tris, lookdev.atlas_uv(len(tris))


def bake(mesh, style="souls_like", **kw):
    v, t, uv = mesh
    st = styles.load_style(style) if isinstance(style, str) else style
    kw.setdefault("size", 512)
    kw.setdefault("seed", 1)
    return st, lookdev.bake(v, t, uv, st, H, **kw)


def measure(st, bk):
    return lookdev.analyze(bk.base_rgb, bk.covered, bk.height01, st)


# ---------------------------------------------------------------- building blocks
def test_hsv_roundtrip():
    rng = np.random.default_rng(0)
    c = rng.random((500, 3))
    assert np.allclose(lookdev.hsv_to_rgb(lookdev.rgb_to_hsv(c)), c, atol=1e-6)


def test_noise_is_deterministic_bounded_and_seeded():
    p = np.random.default_rng(1).random((2000, 3)) * 20
    a, b = lookdev.fbm(p, 3), lookdev.fbm(p, 3)
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() <= 1 and a.std() > 0.05
    assert not np.allclose(a, lookdev.fbm(p, 4))


def test_signed_volume_of_the_body_is_positive_so_winding_is_outward(mesh):
    v, t, _ = mesh
    assert (v[t[:, 0]] * np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6 > 0


def test_convexity_positive_on_a_ball_negative_on_a_dent():
    b = _Builder()
    b.sphere((0, 0, 0), (0.05, 0.05, 0.05), 16, 9)
    v = np.asarray(b.v)
    t = np.array([(f[0], f[1], f[2]) for f in b.f if len(f) == 3] +
                 [(f[0], f[1], f[2]) for f in b.f if len(f) == 4] + [(f[0], f[2], f[3]) for f in b.f if len(f) == 4])
    n = lookdev.vertex_normals(v, t)
    c = lookdev.convexity(v, t, n, feature=0.05)
    assert np.median(c) > 0.5
    assert np.median(lookdev.convexity(v, t, -n, feature=0.05)) < -0.5


def test_rasterize_interpolates_corner_values():
    uv = np.array([[[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]]])
    attrs = np.array([[[0.0], [1.0], [0.0]]], np.float32)
    img, cov = lookdev.rasterize_attributes(uv, attrs, 100)
    assert abs(cov.mean() - 0.32) < 0.02                         # triangle area 0.5 * 0.8 * 0.8
    assert img[85, 85, 0] > 0.9                                  # v = 0.15 row, near the "1.0" corner on the right
    assert img[85, 15, 0] < 0.1
    assert not cov[5, 5] and cov[85, 15]                         # uv origin is bottom-left


# ------------------------------------------------------------------------ bake
def test_bake_is_deterministic_and_seeded(mesh):
    _, a = bake(mesh)
    _, b = bake(mesh)
    _, c = bake(mesh, seed=2)
    assert np.array_equal(a.base_rgb, b.base_rgb) and np.array_equal(a.roughness, b.roughness)
    assert not np.array_equal(a.base_rgb, c.base_rgb)


@pytest.mark.parametrize("name", ["souls_like", "clean_lowpoly", "semi_photoreal"])
def test_builtin_styles_pass_their_own_checks(mesh, name):
    st, bk = bake(mesh, name)
    m = measure(st, bk)
    assert lookdev.check(m, st) == [], m


def test_roughness_stays_in_the_styles_range(mesh):
    st, bk = bake(mesh)
    r = bk.roughness[bk.covered] / 255.0
    assert r.min() >= st.roughness[0] - 0.01 and r.max() <= st.roughness[1] + 0.01


def test_wear_collects_low_on_the_body(mesh):
    st, bk = bake(mesh)
    m = measure(st, bk)
    assert m["dirty_lower_third"] > m["dirty_upper_third"] + 0.3


def test_more_wear_scale_means_more_rot_and_less_clean_surface(mesh):
    rot, clean = [], []
    for sc in (0.4, 1.0, 1.8):
        st, bk = bake(mesh, wear_scale=sc)
        m = measure(st, bk)
        rot.append(m["rot_fraction"]); clean.append(m["clean_fraction"])
    assert rot[0] < rot[1] < rot[2] and clean[0] > clean[1] > clean[2]


def test_hue_shift_override_changes_the_palette(mesh):
    st = styles.load_style("clean_lowpoly")
    _, a = bake(mesh, st)
    _, b = bake(mesh, st.with_overrides(overrides={"hue_shift_deg": 120}))
    ha = lookdev.rgb_to_hsv(a.base_rgb[a.covered] / 255.0)[:, 0].mean()
    hb = lookdev.rgb_to_hsv(b.base_rgb[b.covered] / 255.0)[:, 0].mean()
    assert abs(ha - hb) > 0.05


# --------------------------------------------------------------------- checks
def test_a_garish_texture_fails_saturation_and_value():
    st = styles.load_style("souls_like")
    red = np.zeros((64, 64, 3), np.uint8); red[..., 0] = 255
    cov = np.ones((64, 64), bool)
    m = lookdev.analyze(red, cov, np.linspace(0, 1, 64 * 64).reshape(64, 64), st)
    codes = {f["code"] for f in lookdev.check(m, st)}
    assert {"texture_saturation", "texture_value_range"} <= codes


def test_too_little_and_too_much_wear_are_reported_with_a_direction(mesh):
    st, lo = bake(mesh, wear_scale=0.1)
    f = lookdev.check(measure(st, lo), st)
    assert [(x["code"], x["direction"]) for x in f] == [("texture_wear_coverage", "more")]
    st, hi = bake(mesh, wear_scale=3.0)
    f = lookdev.check(measure(st, hi), st)
    assert f and f[0]["code"] == "texture_wear_coverage" and f[0]["direction"] == "less"


def test_empty_texture_is_reported():
    st = styles.load_style("souls_like")
    m = lookdev.analyze(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), bool), np.zeros((8, 8)), st)
    assert lookdev.check(m, st)[0]["code"] == "texture_missing"


# ------------------------------------------------ closed loop through qa + fixer
def _metrics_for(mesh, scale, style="souls_like"):
    st, bk = bake(mesh, style, wear_scale=scale)
    from vgbuilder import rigdefs
    return {"lod_tris": [1000], "height_m": H, "symmetry": 0.0, "zero_weight_frac": 0.0, "loose_parts": 1,
            "bone_roles": rigdefs.ROLES, "has_uv": True, "stretch_max": 1.0, "roundtrip": {"ok": True},
            "texture": {"mode": "replace", "analysis": measure(st, bk)}}


def test_wear_coverage_is_auto_fixed_by_rescaling(mesh):
    spec = JobSpec(name="a", poly_budget=1200, style_template="souls_like")
    params = {"wear_scale": 0.1}
    for _ in range(6):
        fails = qa.evaluate(_metrics_for(mesh, params["wear_scale"]), spec)
        if not fails:
            break
        assert [f["code"] for f in fails] == ["texture_wear_coverage"] and fails[0]["fixable"]
        params, changes = fixer.next_params(spec, params, fails)
        assert changes and "wear_scale" in changes[0]
    assert fails == [] and params["wear_scale"] > 0.1


def test_kept_foreign_textures_are_flagged_but_never_auto_rescaled(mesh):
    spec = JobSpec(name="a", poly_budget=1200, style_template="souls_like")
    m = _metrics_for(mesh, 0.1)
    m["texture"]["mode"] = "keep"
    fails = qa.evaluate(m, spec)
    assert fails and not fails[0]["fixable"]
    assert fixer.next_params(spec, {}, fails)[1] == []


def test_good_texture_passes_qa(mesh):
    spec = JobSpec(name="a", poly_budget=1200, style_template="souls_like")
    assert qa.evaluate(_metrics_for(mesh, 1.0), spec) == []


# ---------------------------------------------------------------------- spec
def test_spec_validates_the_look_fields(tmp_path):
    with pytest.raises(ValueError, match="unknown style_template"):
        JobSpec(name="a", style_template="nope").validate()
    with pytest.raises(ValueError, match="needs a style_template"):
        JobSpec(name="a", texture="replace").validate()
    with pytest.raises(ValueError, match="texture_size"):
        JobSpec(name="a", texture_size=8).validate()
    with pytest.raises(ValueError, match="unknown wear"):
        JobSpec(name="a", style_template="souls_like", wear={"sparkle": 1}).validate()
    JobSpec(name="a", style_template="souls_like", wear={"rot": 0.1}).validate()


def test_project_styles_override_builtin_and_overrides_apply(tmp_path):
    (tmp_path / "styles").mkdir()
    (tmp_path / "styles" / "mine.yaml").write_text(
        "name: mine\npalette:\n  base: ['#808080']\n  grime: '#101010'\nmax_saturation: 0.2\n")
    spec = JobSpec(name="a", style_template="mine", wear={"intensity": 0.9}, base_dir=str(tmp_path))
    spec.validate()
    st = spec.load_style()
    assert st.name == "mine" and st.wear["intensity"] == 0.9 and st.max_saturation == 0.2
