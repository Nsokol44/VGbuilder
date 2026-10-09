"""LOD generation for skinned meshes."""
from __future__ import annotations

import bpy

from .condition import decimate_to, tri_count


def _skin(obj, arm) -> None:
    obj.parent = arm
    obj.matrix_parent_inverse = arm.matrix_world.inverted()
    for m in list(obj.modifiers):
        obj.modifiers.remove(m)
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm


def make_lods(lod0, arm, ratios: list[float], name: str) -> list:
    """Return [LOD0, LOD1, ...]. LOD k keeps `ratio_k` of LOD0's triangles.

    Each LOD is a separate object parented to the same armature, so each exports as its
    own file with identical skeleton and bone names.
    """
    lod0.name = f"{name}_LOD0"
    lod0.data.name = f"{name}_LOD0_mesh"
    out = [lod0]
    for k, r in enumerate(ratios[1:], start=1):
        o = bpy.data.objects.new(f"{name}_LOD{k}", lod0.data.copy())
        bpy.context.scene.collection.objects.link(o)
        o.data.name = f"{name}_LOD{k}_mesh"
        decimate_to(o, r)
        _skin(o, arm)
        out.append(o)
    return out


def lod_tris(lods) -> list[int]:
    return [tri_count(o) for o in lods]
