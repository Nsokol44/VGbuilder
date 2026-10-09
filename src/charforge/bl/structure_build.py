"""Blender side of the `structure` job kind: numpy building -> LOD meshes, UVs, textures, collision, export, re-import.

Invoked from charforge.bl.entry.run() when spec.kind == "structure". Same contract as the character build: start from an
empty scene, return {"metrics", "files", ...}; nothing here reads state from an earlier run."""
from __future__ import annotations

import time
from pathlib import Path

import bmesh
import bpy
import numpy as np

from .. import engines, structuregen, structuremeasure as sm
from . import condition, export, looks, render
from .measure import triangles_proj  # noqa: F401  (kept for parity with the character build)
from ..spec import JobSpec


def mesh_object(v: np.ndarray, t: np.ndarray, name: str, flat: bool = True):
    """Closed shells -> Blender object. Coplanar triangle pairs are re-joined into quads (better UV islands and a more
    natural lightmap pack); the triangle count after triangulation is unchanged."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(v.tolist(), [], t.tolist())
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.join_triangles(bm, faces=bm.faces, angle_face_threshold=1e-3, angle_shape_threshold=1e-3)
    bm.to_mesh(mesh)
    bm.free()
    for p in mesh.polygons:
        p.use_smooth = not flat
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def arrays(obj) -> tuple[np.ndarray, np.ndarray]:
    """(verts float32, tris int32) as the exporter will see them (after triangulation)."""
    m = obj.data
    m.calc_loop_triangles()
    v = condition.vertex_array(obj).astype(np.float32)
    t = np.empty(len(m.loop_triangles) * 3, np.int32)
    m.loop_triangles.foreach_get("vertices", t)
    return v, t.reshape(-1, 3)


def add_lightmap_uv(obj) -> None:
    """Second UV layer 'lightmap' (non-overlapping, one island per face); the first layer stays active for the texture bake."""
    me = obj.data
    lm = me.uv_layers.new(name="lightmap")
    me.uv_layers.active = lm
    condition._deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.lightmap_pack(PREF_CONTEXT="ALL_FACES", PREF_PACK_IN_ONE=True, PREF_BOX_DIV=48, PREF_MARGIN_DIV=0.3)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    me.uv_layers.active = me.uv_layers[0]
    me.uv_layers[0].active_render = True


def lightmap_report(obj) -> dict:
    me = obj.data
    me.calc_loop_triangles()
    if "lightmap" not in me.uv_layers:
        return {"present": False}
    layer = me.uv_layers["lightmap"]
    uv_all = np.empty(len(layer.data) * 2)
    layer.data.foreach_get("uv", uv_all)
    loops = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("loops", loops)
    uv = uv_all.reshape(-1, 2)[loops.reshape(-1, 3)]
    return {"present": True, **sm.uv_overlap(uv)}


def collision_objects(d: dict, name: str, seed: int) -> list:
    objs = []
    for i, (v, t) in enumerate(structuregen.collision(d, seed), start=1):
        o = mesh_object(v, t, f"UCX_{name}_{i:02d}")        # Unreal's convex-collision naming convention
        o.hide_render = True
        objs.append(o)
    return objs


def _gltf(objs, path: Path) -> None:
    export._select_only(objs)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True, export_apply=False,
                              export_skins=False, export_animations=False, export_attributes=True, export_yup=True,
                              export_extras=True)


def _fbx(objs, path: Path, engine: str) -> None:
    export._select_only(objs)
    pr = engines.preset(engine)
    bpy.ops.export_scene.fbx(filepath=str(path), use_selection=True, object_types={"MESH"}, add_leaf_bones=False,
                             bake_anim=False, use_custom_props=True, path_mode="COPY", embed_textures=True,
                             **{"axis_forward": "-Z", "axis_up": "Y", **pr.fbx})


def _embed(objs, blob: dict | None) -> None:
    import json
    if blob is None:
        return
    text = json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str)
    bpy.context.scene[export.EMBED_KEY] = text
    for o in objs:
        o[export.EMBED_KEY] = text


def roundtrip(path: Path, expect_tris: int, expect_size: list[float]) -> dict:
    """Wipe the scene and re-import a glb: triangle count and overall size must survive. Call LAST."""
    condition.reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        return {"ok": False, "reason": "no mesh after import"}
    tris = sum(condition.tri_count(m) for m in meshes)
    co = np.concatenate([condition.vertex_array(m) @ np.array(m.matrix_world.to_3x3()).T + np.array(m.matrix_world.translation) for m in meshes])
    size = (co.max(0) - co.min(0)).tolist()
    # glTF is Y-up: Blender's importer converts back to Z-up, so sizes compare directly
    ok = abs(tris - expect_tris) <= max(2, 0.01 * expect_tris) and all(abs(a - b) <= 0.01 * max(b, 1e-6) + 1e-4 for a, b in zip(size, expect_size))
    return {"ok": bool(ok), "tris": tris, "size": size}


def run(job: dict) -> dict:
    t0 = time.time()
    spec = JobSpec(**job["spec"])
    params = job["params"]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = spec.name
    d = structuregen.resolve(spec.structure)
    seed = spec.seed
    steps: list[str] = []

    # LOD k is detail level (L0 + k); levels that would repeat the previous one are not built
    level0 = int(params.get("detail_level", 0))
    levels: list[int] = []
    for k in range(len(spec.lods)):
        lv = min(2, level0 + k)
        if not levels or lv > levels[-1]:
            levels.append(lv)

    condition.reset_scene()
    lods, built, metas = [], [], []
    for k, lv in enumerate(levels):
        v, t, roles, meta = structuregen.build(d, lv, seed)
        o = mesh_object(v, t, f"{name}_LOD{k}")
        o.data.name = f"{name}_LOD{k}_mesh"
        condition.ensure_material(o, spec.style, f"{name}_LOD{k}")
        condition.smart_uv(o)
        add_lightmap_uv(o)
        lods.append(o)
        metas.append(meta)
    steps += ["build", "uv", "lightmap_uv"]
    H = structuregen.expected_height(d)

    style = spec.load_style()
    texture_report, extra_files = None, []
    if style is not None and spec.texture not in ("none",):
        for k, o in enumerate(lods):
            rep, files = looks.process(o, style, "replace", max(128, spec.texture_size >> k), seed,
                                       float(params.get("wear_scale", 1.0)), H, out_dir, f"{name}_LOD{k}")
            extra_files += files
            if k == 0:
                texture_report = rep
        steps.append("look_replace")

    # ---- measure what Blender actually holds
    geo = [arrays(o) for o in lods]
    v0, t0_ = geo[0]
    bb = sm.bbox(v0)
    frame = {"min": bb["min"], "max": bb["max"]}
    sil0 = sm.silhouettes(v0, t0_, frame)
    lod_rows = []
    for k, (o, (v, t)) in enumerate(zip(lods, geo)):
        b = sm.bbox(v)
        row = {"lod": k, "level": levels[k], "tris": int(len(t)), "ratio": len(t) / max(len(t0_), 1), "size": b["size"],
               "has_uv": len(o.data.uv_layers) > 0, "has_material": len(o.data.materials) > 0}
        if k == 0:
            row.update(silhouette_front_vs_lod0=1.0, silhouette_side_vs_lod0=1.0, silhouette_top_vs_lod0=1.0)
        else:
            s = sm.silhouettes(v, t, frame)
            row.update({f"silhouette_{n}_vs_lod0": sm.iou(s[n], sil0[n]) for n in ("front", "side", "top")})
        lod_rows.append(row)
    W, D = d["footprint"]
    metrics: dict = {
        "kind": "structure",
        "lod_tris": [r["tris"] for r in lod_rows],
        "bbox": bb,
        "height_m": bb["size"][2],
        "expected": {"height": H, "footprint": [W, D], "max_protrusion": structuregen.max_protrusion(d),
                     "envelope_area": sm.envelope_area((W, D), structuregen.wall_top(d))},
        "degenerate_frac": sm.degenerate_fraction(v0, t0_),
        "has_uv": len(lods[0].data.uv_layers) > 0,
        "has_material": len(lods[0].data.materials) > 0,
        "lightmap": lightmap_report(lods[0]),
        "lods": lod_rows,
        "lods_requested": len(spec.lods),
        "structure": {"style": d["style"], "levels": levels, **{k: metas[0][k] for k in ("windows", "doors", "parts") if k in metas[0]},
                      "door": metas[0].get("door"), "resolved": d},
        "engine": {**engines.preset(spec.target_engine).describe()},
    }
    if texture_report is not None:
        metrics["texture"] = texture_report
    steps.append("measure")

    coll = collision_objects(d, name, seed)
    cv = [arrays(o) for o in coll]
    allv = np.concatenate([x[0] for x in cv])
    metrics["collision"] = {"pieces": len(coll), "tris": int(sum(len(x[1]) for x in cv)), "names": [o.name for o in coll],
                            "bbox": sm.bbox(allv)}
    steps.append("collision")

    sheet = out_dir / f"{name}_contact_sheet.png"
    metrics["render"] = render.structure_sheet(lods, coll, bb["size"], sheet)
    steps.append("contact_sheet")

    blend = out_dir / f"{name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    embed = {
        **(job.get("provenance") or {}),
        "params": {k: round(float(x), 6) for k, x in sorted(params.items())},
        "blender": bpy.app.version_string,
        "texture": ({k: texture_report[k] for k in ("mode", "style", "size", "wear_scale")} if texture_report else None),
    }
    files: list[Path] = []
    for k, o in enumerate(lods):
        _embed([o], {**embed, "lod": k, "level": levels[k]})
        for fmt in spec.formats:
            p = out_dir / f"{name}_LOD{k}.{fmt}"
            (_gltf if fmt == "glb" else lambda ob, pa: _fbx(ob, pa, spec.target_engine))([o], p)
            files.append(p)
    _embed(coll, {**embed, "collision": True})
    cpath = out_dir / f"{name}_collision.glb"
    _gltf(coll, cpath)
    files.append(cpath)
    steps.append("export")

    glbs = [f for f in files if f.suffix == ".glb"]
    metrics["embedded"] = [export.read_embedded(g) for g in glbs]
    metrics["glb_content"] = [export.read_summary(g) for g in glbs]
    metrics["roundtrip"] = roundtrip(next(g for g in glbs if "LOD0" in g.name), metrics["lod_tris"][0], bb["size"])
    steps.append("roundtrip")
    return {
        "metrics": metrics, "profiles": {}, "crotch": {},
        "files": [str(p) for p in files] + [str(sheet), str(blend)] + [str(p) for p in extra_files],
        "steps": steps, "blender_version": bpy.app.version_string, "seconds": round(time.time() - t0, 2),
    }
