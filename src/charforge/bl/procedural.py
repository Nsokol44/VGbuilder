"""Turn the numpy body (charforge.bodygen) into a Blender mesh with skin weights.

The mesh is all triangles, using the exact triangulation the fitter evaluated."""
from __future__ import annotations

import bmesh
import bpy

from ..bodygen import Body


def mesh_from_body(body: Body, name: str, smooth: bool):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(body.verts.tolist(), [], body.triangles().tolist())
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = smooth
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    # skin weights -> vertex groups (named by role here, renamed to bone names by the rig stage)
    roles = sorted({r for ws in body.weights.values() for r, _ in ws})
    for r in roles:
        obj.vertex_groups.new(name=r)
    for vi, ws in body.weights.items():
        for r, w in ws:
            obj.vertex_groups[r].add([vi], w, "REPLACE")
    return obj
