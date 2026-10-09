"""Scene reset, import, normalisation, decimation, UV and material helpers."""
from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np

STYLE_COLORS = {
    "lowpoly": (0.45, 0.55, 0.75, 1.0),
    "stylized": (0.80, 0.62, 0.50, 1.0),
    "realistic": (0.72, 0.58, 0.48, 1.0),
}


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.unit_settings.system = "METRIC"
    s.unit_settings.scale_length = 1.0


def tri_count(obj) -> int:
    m = obj.data
    m.calc_loop_triangles()
    return len(m.loop_triangles)


def vertex_array(obj) -> np.ndarray:
    m = obj.data
    a = np.empty(len(m.vertices) * 3)
    m.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def _deselect_all() -> None:
    for o in bpy.context.view_layer.objects:
        o.select_set(False)


def import_glb(path: Path):
    """Import a glb and flatten it to ONE mesh object in world space, no parents, no rig."""
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"{path} contains no mesh")
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.data.transform(mw)
        o.matrix_world.identity()
        for mod in list(o.modifiers):  # imported armature modifiers refer to objects we delete
            o.modifiers.remove(mod)
    for o in list(bpy.context.scene.objects):
        if o.type != "MESH":
            bpy.data.objects.remove(o)
    active = meshes[0]
    if len(meshes) > 1:
        with bpy.context.temp_override(
            active_object=active, object=active, selected_objects=meshes, selected_editable_objects=meshes
        ):
            bpy.ops.object.join()
    active.vertex_groups.clear()
    return active


def normalize(obj, height: float) -> dict:
    """Uniform scale to `height`, feet on z=0, centred on x/y. Applied to the vertices."""
    co = vertex_array(obj)
    lo, hi = co.min(axis=0), co.max(axis=0)
    h = float(hi[2] - lo[2])
    if h <= 1e-9:
        raise RuntimeError("mesh has zero height")
    s = height / h
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    co = (co - np.array([cx, cy, lo[2]])) * s
    obj.data.vertices.foreach_set("co", co.ravel())
    obj.data.update()
    return {"scale": s, "source_height": h}


def decimate_to(obj, ratio: float) -> None:
    """Replace obj's mesh with a decimated copy (collapse). Works on skinned meshes too."""
    tmp = bpy.data.objects.new("_dec", obj.data.copy())
    bpy.context.scene.collection.objects.link(tmp)
    mod = tmp.modifiers.new("dec", "DECIMATE")
    mod.ratio = float(max(0.01, min(1.0, ratio)))
    deps = bpy.context.evaluated_depsgraph_get()
    new_mesh = bpy.data.meshes.new_from_object(tmp.evaluated_get(deps))
    bpy.data.objects.remove(tmp)
    old = obj.data
    obj.data = new_mesh
    new_mesh.name = old.name
    if old.users == 0:
        bpy.data.meshes.remove(old)


def fit_budget(obj, budget: int) -> dict:
    before = tri_count(obj)
    if before > budget:
        decimate_to(obj, budget / before * 1.02)
    return {"tris_before": before, "tris_after": tri_count(obj)}


def smart_uv(obj) -> None:
    _deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)


def ensure_uv(obj) -> bool:
    if len(obj.data.uv_layers) == 0:
        smart_uv(obj)
        return True
    return False


def ensure_material(obj, style: str, name: str) -> None:
    if len(obj.data.materials) and obj.data.materials[0] is not None:
        return
    mat = bpy.data.materials.new(f"{name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = STYLE_COLORS[style]
        bsdf.inputs["Roughness"].default_value = 0.8
    mat.diffuse_color = STYLE_COLORS[style]
    obj.data.materials.append(mat)
