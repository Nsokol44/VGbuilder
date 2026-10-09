"""Measurements taken from the built character. Everything QA looks at comes from here."""
from __future__ import annotations

import math

import bmesh
import bpy
import numpy as np

from ..rigdefs import roles_from_names
from ..sketch import rasterize_triangles
from .condition import tri_count, vertex_array
from .rig import influence_counts


def symmetry_error(co: np.ndarray, height: float, max_samples: int = 2500) -> float:
    """Mean distance from each (sampled) vertex's X-mirror to its nearest vertex, / height."""
    rng = np.random.default_rng(0)
    idx = np.arange(len(co))
    if len(idx) > max_samples:
        idx = rng.choice(idx, max_samples, replace=False)
    mir = co[idx] * np.array([-1.0, 1.0, 1.0])
    best = np.full(len(idx), np.inf)
    for s in range(0, len(co), 20000):
        chunk = co[s: s + 20000]
        d = np.linalg.norm(mir[:, None, :] - chunk[None, :, :], axis=2).min(axis=1)
        best = np.minimum(best, d)
    return float(best.mean() / height)


def loose_parts(mesh) -> int:
    n = len(mesh.vertices)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in mesh.edges:
        ra, rb = find(e.vertices[0]), find(e.vertices[1])
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n)})


def non_manifold_edges(mesh) -> int:
    bm = bmesh.new()
    bm.from_mesh(mesh)
    n = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    bm.free()
    return n


def zero_weight_fraction(obj) -> float:
    verts = obj.data.vertices
    if len(verts) == 0:
        return 1.0
    zero = sum(1 for v in verts if sum(g.weight for g in v.groups) <= 1e-6)
    return zero / len(verts)


def triangles_proj(obj, axes: tuple[int, int]) -> np.ndarray:
    m = obj.data
    m.calc_loop_triangles()
    co = vertex_array(obj)
    tri = np.empty(len(m.loop_triangles) * 3, dtype=np.int32)
    m.loop_triangles.foreach_get("vertices", tri)
    return co[tri.reshape(-1, 3)][:, :, list(axes)]


def triangles_xz(obj) -> np.ndarray:
    return triangles_proj(obj, (0, 2))


def front_mask(obj) -> np.ndarray:
    return rasterize_triangles(triangles_xz(obj))


def side_mask(obj) -> np.ndarray:
    """Side silhouette in (y, z): the character faces -Y, i.e. screen-left."""
    return rasterize_triangles(triangles_proj(obj, (1, 2)))


def stretch_test(mesh_obj, arm, naming: str) -> float:
    """Pose arms and legs, return the largest edge stretch factor (posed / rest)."""
    from ..rigdefs import bone_name

    mesh = mesh_obj.data
    edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)
    rest = vertex_array(mesh_obj)
    rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)

    pose = {
        bone_name("upperarm_l", naming): 0.9, bone_name("forearm_l", naming): 1.0,
        bone_name("thigh_r", naming): 0.8, bone_name("shin_r", naming): -0.9,
        bone_name("head", naming): 0.4, bone_name("spine", naming): 0.3,
    }
    for n, ang in pose.items():
        pb = arm.pose.bones.get(n)
        if pb is not None:
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = (ang, 0.0, 0.0)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    ev = mesh_obj.evaluated_get(deps)
    tmp = ev.to_mesh()
    posed = np.empty(len(tmp.vertices) * 3)
    tmp.vertices.foreach_get("co", posed)
    posed = posed.reshape(-1, 3)
    ev.to_mesh_clear()
    for n in pose:
        pb = arm.pose.bones.get(n)
        if pb is not None:
            pb.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    if len(posed) != len(rest):
        return float("nan")
    posed_len = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)
    ok = rest_len > 1e-6
    return float((posed_len[ok] / rest_len[ok]).max())


def lod_report(lods, arm, ratios: list[float]) -> list[dict]:
    """Per-LOD facts QA needs: size, how much it shrank, silhouette drift from LOD0, rig/UV/weight consistency."""
    from ..sketch import silhouette_iou

    base_f, base_s = front_mask(lods[0]), side_mask(lods[0])
    base_groups = sorted(g.name for g in lods[0].vertex_groups)
    base_tris = tri_count(lods[0])
    bones = {b.name for b in arm.data.bones}
    out = []
    for k, o in enumerate(lods):
        co = vertex_array(o)
        mod = next((m for m in o.modifiers if m.type == "ARMATURE"), None)
        names = sorted(g.name for g in o.vertex_groups)
        entry = {
            "lod": k, "tris": tri_count(o), "target_ratio": ratios[k],
            "ratio": tri_count(o) / max(base_tris, 1),
            "height_m": float(co[:, 2].max() - co[:, 2].min()),
            "armature_ok": bool(mod is not None and mod.object == arm and o.parent == arm),
            "groups_match_lod0": names == base_groups,
            "groups_are_bones": set(names) <= bones,
            "zero_weight_frac": zero_weight_fraction(o),
            "has_uv": len(o.data.uv_layers) > 0,
            "has_material": len(o.data.materials) > 0,
        }
        if k == 0:
            entry["silhouette_front_vs_lod0"] = entry["silhouette_side_vs_lod0"] = 1.0
        else:
            entry["silhouette_front_vs_lod0"] = silhouette_iou(front_mask(o), base_f)
            entry["silhouette_side_vs_lod0"] = silhouette_iou(side_mask(o), base_s)
        out.append(entry)
    return out


def collect(lod0, lods, arm, naming: str, height_target: float, ratios: list[float] | None = None) -> dict:
    ratios = ratios or [1.0] * len(lods)
    co = vertex_array(lod0)
    h = float(co[:, 2].max() - co[:, 2].min())
    w = float(co[:, 0].max() - co[:, 0].min())
    bone_names = [b.name for b in arm.data.bones]
    return {
        "lod_tris": [tri_count(o) for o in lods],
        "height_m": h,
        "width_m": w,
        "symmetry": symmetry_error(co, h),
        "zero_weight_frac": zero_weight_fraction(lod0),
        "loose_parts": loose_parts(lod0.data),
        "non_manifold_edges": non_manifold_edges(lod0.data),
        "bone_names": bone_names,
        "bone_roles": sorted(roles_from_names(bone_names, naming)),
        "has_uv": len(lod0.data.uv_layers) > 0,
        "has_material": len(lod0.data.materials) > 0,
        "vertex_groups": len(lod0.vertex_groups),
        "stretch_max": stretch_test(lod0, arm, naming),
        "lods": lod_report(lods, arm, ratios),
        "max_influences": max(int(influence_counts(o).max()) if len(o.data.vertices) else 0 for o in lods),
    }
