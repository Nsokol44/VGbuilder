"""Armature creation and skinning."""
from __future__ import annotations

import bpy
import numpy as np
from mathutils import Vector

from ..rigdefs import HIERARCHY, bone_name, segment_distance


def build_armature(joints: dict[str, tuple[tuple, tuple]], naming: str, name: str):
    arm_data = bpy.data.armatures.new(f"{name}_rig")
    arm = bpy.data.objects.new(f"{name}_rig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    for role, parent in HIERARCHY.items():
        head, tail = joints[role]
        b = eb.new(bone_name(role, naming))
        b.head, b.tail = Vector(head), Vector(tail)
        if (np.linalg.norm(np.array(tail) - np.array(head))) < 1e-4:
            b.tail = Vector((head[0], head[1], head[2] + 0.01))
    for role, parent in HIERARCHY.items():
        if parent:
            b = eb[bone_name(role, naming)]
            b.parent = eb[bone_name(parent, naming)]
            b.use_connect = bool(np.allclose(b.head, b.parent.tail, atol=1e-5))
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.select_set(False)
    return arm


def rename_groups_to_bones(mesh_obj, naming: str) -> None:
    from ..rigdefs import NAMING

    for vg in mesh_obj.vertex_groups:
        if vg.name in NAMING[naming]:          # leave non-bone groups (e.g. cloth_pin) alone
            vg.name = bone_name(vg.name, naming)


def bind(mesh_obj, arm) -> None:
    """Parent the mesh to the armature with an Armature modifier (weights already exist)."""
    mesh_obj.parent = arm
    mesh_obj.matrix_parent_inverse = arm.matrix_world.inverted()
    mod = mesh_obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm


def _zero_weight_mask(mesh_obj) -> np.ndarray:
    mesh = mesh_obj.data
    total = np.zeros(len(mesh.vertices))
    for i, v in enumerate(mesh.vertices):
        total[i] = sum(g.weight for g in v.groups)
    return total <= 1e-6


def auto_weights(mesh_obj, arm, joints, naming: str) -> dict:
    """Blender heat-map weights, then a nearest-bone fallback for any vertex it missed.

    Heat weighting often fails on generated meshes (non-manifold, disconnected shells),
    so the fallback is not optional. Returns a small report for the manifest.
    """
    for o in bpy.context.selected_objects:
        o.select_set(False)
    mesh_obj.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    method = "heat"
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError:
        method = "heat_failed"
        bpy.ops.object.parent_set(type="ARMATURE_NAME")
    mesh_obj.select_set(False)
    arm.select_set(False)
    zero = _zero_weight_mask(mesh_obj)
    n_zero_before = int(zero.sum())
    if n_zero_before:
        _nearest_bone_weights(mesh_obj, joints, naming, np.nonzero(zero)[0])
    return {"method": method, "unweighted_after_heat": n_zero_before, "verts": len(mesh_obj.data.vertices)}


def _nearest_bone_weights(mesh_obj, joints, naming: str, vert_idx: np.ndarray) -> None:
    mesh = mesh_obj.data
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)[vert_idx]
    roles = list(HIERARCHY)
    dists = np.stack(
        [segment_distance(co, np.array(joints[r][0]), np.array(joints[r][1])) for r in roles], axis=1
    )
    best = dists.argmin(axis=1)
    for r in roles:
        if bone_name(r, naming) not in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.new(name=bone_name(r, naming))
    for vi, bi in zip(vert_idx, best):
        mesh_obj.vertex_groups[bone_name(roles[bi], naming)].add([int(vi)], 1.0, "REPLACE")


def influence_counts(mesh_obj, eps: float = 1e-4) -> np.ndarray:
    """Number of bone weights above eps on each vertex."""
    return np.array([sum(1 for g in v.groups if g.weight > eps) for v in mesh_obj.data.vertices], dtype=int)


def limit_influences(mesh_obj, limit: int) -> dict:
    """Cap skin weights per vertex at `limit` (keeps the largest, renormalises). Returns a small report."""
    before = int(influence_counts(mesh_obj).max()) if len(mesh_obj.data.vertices) else 0
    if before <= limit:
        return {"limited": False, "max_before": before, "limit": limit}
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=limit)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
    mesh_obj.select_set(False)
    return {"limited": True, "max_before": before, "limit": limit}
