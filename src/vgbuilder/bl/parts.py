"""Blender side of accessories: turn the numpy parts into mesh objects."""
from __future__ import annotations

import bpy

from ..accessories import Part
from . import procedural
from .condition import tri_count


def _join(target, objs) -> None:
    with bpy.context.temp_override(active_object=target, object=target, selected_objects=[target, *objs],
                                   selected_editable_objects=[target, *objs]):
        bpy.ops.object.join()


def add_parts(obj, parts: list[Part], name: str, smooth: bool) -> tuple[dict, list[tuple[str, object]]]:
    """Join the joined parts into `obj`; create separate objects for the rest.

    Returns (report for the manifest, [(part name, object)] for the separate ones).
    """
    report: dict = {}
    separate: list[tuple[str, object]] = []
    joined = []
    for part in parts:
        pobj = procedural.mesh_from_body(part.body, f"{name}_{part.name}", smooth)
        report[part.name] = {"tris": part.body.tri_count, "verts": len(part.body.verts), "joined": part.joined,
                             "tags": dict(part.tags)}
        if part.joined:
            joined.append(pobj)
            continue
        for attr, values in part.point_attrs.items():
            a = pobj.data.attributes.new(name=attr, type="FLOAT", domain="POINT")
            a.data.foreach_set("value", [float(v) for v in values])
        if "_CLOTH_PIN" in part.point_attrs:
            g = pobj.vertex_groups.new(name="cloth_pin")           # for Blender-side cloth sims
            for i, v in enumerate(part.point_attrs["_CLOTH_PIN"]):
                if v > 0:
                    g.add([i], float(v), "REPLACE")
        for k, v in part.tags.items():
            pobj[k] = v                                            # exported as glTF extras on the node
        separate.append((part.name, pobj))
    if joined:
        before = tri_count(obj)
        _join(obj, joined)
        report["_joined_tris_added"] = tri_count(obj) - before
    return report, separate
