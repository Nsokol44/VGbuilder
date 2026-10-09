"""End-to-end tests. Need the bpy module (headless Blender); skipped when it's missing."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

bpy = pytest.importorskip("bpy")

ROOT = Path(__file__).resolve().parents[1]
from vgbuilder.orchestrator import Backend, build_one  # noqa: E402
from vgbuilder.spec import load_job  # noqa: E402

pytestmark = pytest.mark.bpy


@pytest.fixture(scope="module")
def jobs(tmp_path_factory):
    """Copy jobs + examples so tests never write into the repo."""
    d = tmp_path_factory.mktemp("proj")
    shutil.copytree(ROOT / "jobs", d / "jobs")
    shutil.copytree(ROOT / "examples", d / "examples")
    return d


def run(jobs_dir: Path, name: str, out: Path):
    spec = load_job(jobs_dir / "jobs" / f"{name}.yaml")
    return build_one(spec, out, Backend())


@pytest.fixture(scope="module")
def hero(jobs, tmp_path_factory):
    out = tmp_path_factory.mktemp("out_hero")
    return run(jobs, "hero", out), out


def test_hero_passes_and_outputs_exist(hero):
    m, out = hero
    assert m["status"] == "passed", m.get("final_failures")
    names = {o["file"] for o in m["outputs"]}
    assert {"hero_LOD0.glb", "hero_LOD1.glb", "hero_LOD2.glb", "hero_contact_sheet.png", "hero.blend"} <= names
    for o in m["outputs"]:
        assert (out / "hero" / "final" / o["file"]).stat().st_size == o["bytes"] > 0


def test_manifest_records_provenance(hero):
    m, out = hero
    on_disk = json.loads((out / "hero" / "manifest.json").read_text())
    assert on_disk["fingerprint"] == m["fingerprint"]
    assert on_disk["sketch_sha256"] and on_disk["provider"] == {"provider": "procedural"}
    it = on_disk["iterations"][0]
    assert it["blender_version"] and it["params"]["segments"] >= 4
    assert on_disk["fit"]["fit"]["predicted_iou"] >= on_disk["fit"]["fit"]["seed_iou"]


def test_budget_height_and_silhouette(hero):
    m, _ = hero
    me = m["iterations"][m["best_iteration"] - 1]["metrics"]
    assert me["lod_tris"][0] <= 2500 * 1.05
    assert me["lod_tris"][0] > me["lod_tris"][1] > me["lod_tris"][2]
    assert me["height_m"] == pytest.approx(1.8, abs=0.01)
    assert me["silhouette_iou"] >= 0.78
    assert me["zero_weight_frac"] == 0 and me["stretch_max"] < 2.0


def test_blender_silhouette_matches_the_numpy_prediction(hero):
    """The fast pure-numpy fit is only trustworthy if the real Blender mesh agrees with it."""
    m, _ = hero
    predicted = m["fit"]["fit"]["predicted_iou"]
    measured = m["iterations"][m["best_iteration"] - 1]["metrics"]["silhouette_iou"]
    assert abs(predicted - measured) < 0.01


def test_contact_sheet_is_a_real_render(hero):
    m, out = hero
    me = m["iterations"][m["best_iteration"] - 1]["metrics"]
    assert me["render"]["mode"] == "workbench"
    from PIL import Image
    im = Image.open(out / "hero" / "final" / "hero_contact_sheet.png")
    assert im.width > 1000 and len(im.convert("L").getcolors(maxcolors=70000)) > 50


def test_same_job_gives_byte_identical_glb(jobs, hero, tmp_path):
    """Script is the source of truth: rebuilding from nothing reproduces the same file."""
    m, out = hero
    m2 = run(jobs, "hero", tmp_path)
    a = {o["file"]: o["sha256"] for o in m["outputs"] if o["file"].endswith(".glb")}
    b = {o["file"]: o["sha256"] for o in m2["outputs"] if o["file"].endswith(".glb")}
    assert a == b and len(a) == 3


def test_every_exported_lod_reimports_skinned(hero):
    m, out = hero
    me = m["iterations"][m["best_iteration"] - 1]["metrics"]
    from vgbuilder.bl import export
    for k, tris in enumerate(me["lod_tris"]):
        rt = export.roundtrip_glb(out / "hero" / "final" / f"hero_LOD{k}.glb", tris, 1.8, 18)
        assert rt["ok"], (k, rt)


def test_lowpoly_job_exports_fbx_too(jobs, tmp_path):
    m = run(jobs, "stocky", tmp_path)
    assert m["status"] == "passed"
    names = {o["file"] for o in m["outputs"]}
    assert {"stocky_LOD0.fbx", "stocky_LOD0.glb", "stocky_LOD1.fbx"} <= names
    assert all(m_["bytes"] > 1000 for m_ in m["outputs"] if m_["file"].endswith(".fbx"))
    # 6-segment low-poly is the strict case for "what the fitter optimised is what shipped"
    measured = m["iterations"][m["best_iteration"] - 1]["metrics"]["silhouette_iou"]
    assert abs(m["fit"]["fit"]["predicted_iou"] - measured) < 0.01


def test_external_generator_path_normalises_decimates_and_rigs(jobs, tmp_path):
    m = run(jobs, "generated_example", tmp_path) if (jobs / "jobs" / "generated_example.yaml").exists() else run(jobs, "generated", tmp_path)
    assert m["provider"]["provider"] == "command" and not m["provider"]["cached"]
    me = m["iterations"][m["best_iteration"] - 1]["metrics"]
    w = me["weights"]
    assert w["normalize"]["source_height"] == pytest.approx(37.0, rel=0.01)   # generator used silly units
    assert me["height_m"] == pytest.approx(1.75, abs=0.01)
    assert w["budget"]["tris_before"] > 1200 >= w["budget"]["tris_after"] * 0.95
    assert me["zero_weight_frac"] == 0 and me["roundtrip"]["ok"]
    # the fake generator ignores the sketch, so QA must notice and not wave it through
    assert m["status"] == "needs_review"
    assert "silhouette_low" in [f["code"] for f in m["final_failures"]]


def test_generator_result_is_cached_across_runs(jobs, tmp_path):
    name = "generated_example" if (jobs / "jobs" / "generated_example.yaml").exists() else "generated"
    run(jobs, name, tmp_path)
    again = run(jobs, name, tmp_path)
    assert again["provider"]["cached"] is True


def test_failing_generator_is_reported_not_raised(jobs, tmp_path):
    p = jobs / "jobs" / "broken.yaml"
    p.write_text(
        "name: broken\nsketch: ../examples/sketches/hero_front.png\nprovider: command\n"
        "provider_args:\n  command: \"{python} -c 'import sys; sys.exit(3)'\"\n"
    )
    m = run(jobs, "broken", tmp_path)
    assert m["status"] == "error" and "exit 3" in m["error"]
    assert (tmp_path / "broken" / "manifest.json").exists()


def test_unweighted_islands_get_nearest_bone_fallback(jobs, tmp_path):
    """A floating shell far from any bone must still end up skinned (QA 'unweighted' stays at 0)."""
    from vgbuilder.bl import condition, rig
    from vgbuilder.bodygen import build_body
    from vgbuilder.bl.procedural import mesh_from_body
    from vgbuilder.spec import DEFAULT_BODY_PARAMS

    condition.reset_scene()
    p = {**DEFAULT_BODY_PARAMS, "segments": 8.0}
    body = build_body(p, 1.8)
    obj = mesh_from_body(body, "m", smooth=False)
    obj.vertex_groups.clear()
    bpy.ops.mesh.primitive_cube_add(size=0.05, location=(0.9, 0.0, 1.0))  # detached, far outside the body
    cube = bpy.context.active_object
    with bpy.context.temp_override(active_object=obj, object=obj, selected_objects=[obj, cube], selected_editable_objects=[obj, cube]):
        bpy.ops.object.join()
    arm = rig.build_armature(body.joints, "simple", "t")
    report = rig.auto_weights(obj, arm, body.joints, "simple")
    from vgbuilder.bl.measure import zero_weight_fraction
    assert zero_weight_fraction(obj) == 0.0
    assert report["verts"] == len(obj.data.vertices)


# ------------------------------------------------------------ front + side sketch
@pytest.fixture(scope="module")
def hero2(jobs, tmp_path_factory):
    out = tmp_path_factory.mktemp("out_hero2")
    return run(jobs, "hero_2view", out), out


def test_two_view_job_passes_and_side_prediction_matches_blender(hero2):
    m, _ = hero2
    assert m["status"] == "passed", m.get("final_failures")
    assert m["sketch_side_sha256"]
    it = m["iterations"][m["best_iteration"] - 1]
    assert it["metrics"]["silhouette_side_iou"] >= 0.70
    assert it["metrics"]["silhouette_side_iou"] == pytest.approx(m["fit"]["fit"]["predicted_side_iou"], abs=0.005)
    assert it["metrics"]["silhouette_iou"] == pytest.approx(m["fit"]["fit"]["predicted_iou"], abs=0.005)


def test_proportions_are_recorded_in_the_metrics(hero2):
    m, _ = hero2
    r = m["iterations"][0]["metrics"]["proportions"]
    assert set(r["ratios"]) >= {"heads_tall", "leg_to_torso", "thigh_slenderness"}
    assert m["fit"]["proportions"]["ratios"] == r["ratios"]


# ------------------------------------------------------------------ engine presets
def test_unity_job_uses_humanoid_names_and_records_the_preset(hero2):
    m, _ = hero2
    met = m["iterations"][m["best_iteration"] - 1]["metrics"]
    assert "Hips" in met["bone_names"] and "LeftUpperLeg" in met["bone_names"]
    assert met["engine"]["naming_used"] == "humanoid" and met["engine"]["verified_in_engine"] is False
    assert m["engine_preset"]["max_influences"] == 4


def test_skin_influence_limit_caps_and_renormalises():
    from vgbuilder.bl import condition, rig

    condition.reset_scene()
    bpy.ops.mesh.primitive_cube_add()
    o = bpy.context.active_object
    groups = [o.vertex_groups.new(name=f"g{i}") for i in range(6)]
    for v in o.data.vertices:
        for i, g in enumerate(groups):
            g.add([v.index], (1.0 + i) / 6, "REPLACE")    # 6 influences everywhere, heaviest is g5
    rep = rig.limit_influences(o, 3)
    assert rep == {"limited": True, "max_before": 6, "limit": 3}
    assert rig.influence_counts(o).max() == 3
    for v in o.data.vertices:
        assert sum(g.weight for g in v.groups) == pytest.approx(1.0, abs=1e-4)
        assert {o.vertex_groups[g.group].name for g in v.groups if g.weight > 1e-4} == {"g5", "g4", "g3"}
    assert rig.limit_influences(o, 3)["limited"] is False


@pytest.mark.parametrize("engine", ["generic", "unity", "unreal", "godot"])
def test_every_preset_exports_glb_and_fbx(engine, tmp_path):
    from vgbuilder.bl import condition, export, procedural, rig
    from vgbuilder.bodygen import build_body
    from vgbuilder.spec import DEFAULT_BODY_PARAMS
    from vgbuilder.engines import preset

    condition.reset_scene()
    body = build_body({**DEFAULT_BODY_PARAMS, "segments": 6.0}, 1.8)
    naming = preset(engine).naming
    obj = procedural.mesh_from_body(body, "t", smooth=False)
    arm = rig.build_armature(body.joints, naming, "t")
    rig.rename_groups_to_bones(obj, naming)
    rig.bind(obj, arm)
    files = export.export_all([obj], arm, tmp_path, "t", ["glb", "fbx"], engine)
    assert {f.suffix for f in files} == {".glb", ".fbx"} and all(f.stat().st_size > 1000 for f in files)


# ----------------------------------------------------------------------- look-dev
@pytest.fixture(scope="module")
def souls(jobs, tmp_path_factory):
    out = tmp_path_factory.mktemp("out_souls")
    return run(jobs, "hero_souls", out), out


def test_look_dev_bakes_textures_and_passes_image_qa(souls):
    m, out = souls
    assert m["status"] == "passed", m.get("final_failures")
    names = {o["file"] for o in m["outputs"]}
    assert {"hero_souls_basecolor.png", "hero_souls_roughness.png"} <= names
    tex = m["iterations"][m["best_iteration"] - 1]["metrics"]["texture"]
    a = tex["analysis"]
    assert tex["mode"] == "replace" and tex["style"] == "souls_like"
    assert a["mean_saturation"] <= 0.36 and 0.03 <= a["rot_fraction"] <= 0.45 and a["dirty_lower_third"] > a["dirty_upper_third"]
    assert m["style_resolved"]["name"] == "souls_like" and m["style_resolved"]["wear"]["rot"] == 0.30


def _glb_json(path: Path) -> dict:
    import struct
    d = path.read_bytes()
    n = struct.unpack("<I", d[12:16])[0]
    return json.loads(d[20:20 + n])


def test_exported_glb_carries_the_baked_textures(souls):
    _, out = souls
    for lod in (0, 1, 2):
        j = _glb_json(out / "hero_souls" / "final" / f"hero_souls_LOD{lod}.glb")
        pbr = j["materials"][0]["pbrMetallicRoughness"]
        assert "baseColorTexture" in pbr and "metallicRoughnessTexture" in pbr
        assert len(j["images"]) == 2


def test_existing_texture_is_analysed_not_replaced(souls, jobs, tmp_path):
    _, out = souls
    glb = out / "hero_souls" / "final" / "hero_souls_LOD0.glb"
    (jobs / "jobs" / "keep.yaml").write_text(
        f"name: keep\nprovider: import_glb\nprovider_args: {{glb: {glb}}}\nstyle: stylized\n"
        "style_template: souls_like\ntexture_size: 256\nmax_iterations: 1\n")
    m = run(jobs, "keep", tmp_path)
    assert m["status"] != "error", m.get("error")
    tex = m["iterations"][0]["metrics"]["texture"]
    assert tex["mode"] == "keep"                       # auto-detected: the glb already has a base-colour image
    assert tex["analysis"]["pixels"] > 1000 and "soil_fraction" in tex["analysis"]
    assert "keep_basecolor.png" not in {Path(f).name for f in m["iterations"][0]["files"]}


def test_wear_coverage_failure_is_fixed_by_rebuilding_with_a_new_scale(jobs, tmp_path):
    (jobs / "jobs" / "loopy.yaml").write_text(
        "name: loopy\nsketch: ../examples/sketches/hero_front.png\nstyle: lowpoly\nstyle_template: souls_like\n"
        "wear: {intensity: 0.3, rot: 0.02, rust: 0.2}\ntexture_size: 256\npoly_budget: 1500\nmax_iterations: 5\n")
    m = run(jobs, "loopy", tmp_path)
    assert m["status"] == "passed", m.get("final_failures")
    scales = [it["metrics"]["texture"]["wear_scale"] for it in m["iterations"]]
    assert len(scales) >= 2 and scales == sorted(scales) and scales[-1] > scales[0]
    assert m["iterations"][0]["failures"][0]["code"] == "texture_wear_coverage"


# ---------------------------------------------- LOD validation + embedded provenance
def test_lod_chain_is_measured_and_consistent(hero):
    m, _ = hero
    lods = m["iterations"][m["best_iteration"] - 1]["metrics"]["lods"]
    assert [e["lod"] for e in lods] == [0, 1, 2]
    for e in lods[1:]:
        assert abs(e["ratio"] - e["target_ratio"]) < 0.1
        assert e["silhouette_front_vs_lod0"] > 0.9 and e["silhouette_side_vs_lod0"] > 0.9
        assert e["armature_ok"] and e["groups_match_lod0"] and e["has_uv"] and e["has_material"]


def test_every_glb_carries_its_own_recipe(hero2):
    from vgbuilder.bl.export import read_embedded

    m, out = hero2
    it = m["iterations"][m["best_iteration"] - 1]
    for k in range(3):
        blob = read_embedded(out / "hero_2view" / "final" / f"hero_2view_LOD{k}.glb")
        assert blob["lod"] == k and blob["fingerprint"] == m["fingerprint"]
        assert blob["sketch_sha256"] == m["sketch_sha256"] and blob["sketch_side_sha256"] == m["sketch_side_sha256"]
        assert blob["params"] == {k_: round(v, 6) for k_, v in sorted(it["params"].items())}
        assert blob["engine"] == "unity" and blob["blender"] and blob["numpy"] and blob["naming"] == "humanoid"
        assert blob["spec"]["poly_budget"] == 2500


def test_embedded_style_hash_matches_the_manifest(souls):
    from vgbuilder.bl.export import read_embedded

    m, out = souls
    blob = read_embedded(out / "hero_souls" / "final" / "hero_souls_LOD0.glb")
    assert blob["style"] == {"name": "souls_like", "sha256": m["style_resolved"]["sha256"]}
    assert blob["texture"]["mode"] == "replace"


def test_a_glb_without_the_block_reads_as_none(tmp_path):
    from vgbuilder.bl import condition, export, procedural, rig
    from vgbuilder.bodygen import build_body
    from vgbuilder.spec import DEFAULT_BODY_PARAMS

    condition.reset_scene()
    body = build_body({**DEFAULT_BODY_PARAMS, "segments": 6.0}, 1.8)
    obj = procedural.mesh_from_body(body, "t", smooth=False)
    arm = rig.build_armature(body.joints, "simple", "t")
    rig.rename_groups_to_bones(obj, "simple")
    rig.bind(obj, arm)
    (a,) = export.export_all([obj], arm, tmp_path, "plain", ["glb"])
    assert export.read_embedded(a) is None
    (b,) = export.export_all([obj], arm, tmp_path, "tagged", ["glb"], "generic", {"fingerprint": "abc"})
    assert export.read_embedded(b) == {"fingerprint": "abc", "lod": 0, "lod_ratio": 1.0}


def test_a_crushed_lod_is_caught(jobs, tmp_path):
    (jobs / "jobs" / "crushed.yaml").write_text(
        "name: crushed\nsketch: ../examples/sketches/hero_front.png\nstyle: lowpoly\npoly_budget: 1500\n"
        "lods: [1.0, 0.03]\nmax_iterations: 1\n")
    m = run(jobs, "crushed", tmp_path)
    codes = {f["code"] for f in m["final_failures"]}
    assert codes & {"lod_silhouette_drift", "lod_ratio_off"}, m["final_failures"]
    assert m["status"] == "needs_review"


# ------------------------------------------------------ accessories + animations
@pytest.fixture(scope="module")
def gear(jobs, tmp_path_factory):
    out = tmp_path_factory.mktemp("out_gear")
    return run(jobs, "hero_gear", out), out


def test_gear_job_passes_with_sketch_judged_on_the_base_body(gear):
    m, _ = gear
    assert m["status"] == "passed", m.get("final_failures")
    met = m["iterations"][m["best_iteration"] - 1]["metrics"]
    assert abs(met["height_m"] - 1.8) < 0.04 and met["height_m_total"] > met["height_m"] + 0.1     # horns excluded from character height
    assert met["silhouette_iou"] == pytest.approx(m["fit"]["fit"]["predicted_iou"], abs=0.005)    # unchanged by accessories
    assert met["lod_tris"][0] <= 3000 * 1.05
    assert set(met["accessories"]) >= {"pauldrons", "horns", "cape"}


def test_cape_is_a_separate_cloth_tagged_mesh_with_pin_weights(gear):
    from vgbuilder.bl.export import read_summary

    _, out = gear
    for lod in (0, 1, 2):
        s = read_summary(out / "hero_gear" / "final" / f"hero_gear_LOD{lod}.glb")
        assert s["nodes"]["hero_gear_cape"]["vgbuilder_physics"] == "cloth"
        assert any("_CLOTH_PIN" in attrs for attrs in s["mesh_attributes"].values())


def test_animations_are_in_every_glb_and_loop(gear):
    from vgbuilder.bl.export import read_summary

    m, out = gear
    met = m["iterations"][m["best_iteration"] - 1]["metrics"]
    for name in ("idle", "walk"):
        a = met["animations"][name]
        assert a["loops"] and a["max_stretch"] < 1.5 and a["bones_keyed"] >= 8
    assert met["animations"]["walk"]["duration_s"] == 1.0 and met["animations"]["walk"]["frames"] == 25
    bones = set(met["bone_names"])
    for lod in (0, 1, 2):
        s = read_summary(out / "hero_gear" / "final" / f"hero_gear_LOD{lod}.glb")
        assert set(s["animations"]) == {"idle", "walk"}
        assert all(a["channels"] > 0 and set(a["targets"]) <= bones for a in s["animations"].values())


def test_fbx_with_animations_exports(gear):
    _, out = gear
    fbx = sorted((out / "hero_gear" / "final").glob("*.fbx"))
    assert len(fbx) == 3 and all(f.stat().st_size > 5000 for f in fbx)


def test_animation_survives_reimport(gear):
    from vgbuilder.bl import condition

    _, out = gear
    condition.reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(out / "hero_gear" / "final" / "hero_gear_LOD0.glb"))
    names = {a.name.split("|")[-1] for a in bpy.data.actions}
    assert any("walk" in n for n in names) and any("idle" in n for n in names)
