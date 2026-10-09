"""Stand-in for an image-to-3D model (TRELLIS.2, Hunyuan3D, a hosted API...).

It ignores the image and writes an *unrigged*, fused, wrongly-scaled, off-centre GLB, which is
the kind of thing generators hand you. Used to test the import -> normalise -> decimate -> rig
path without a GPU. A real wrapper has the same contract:

    python your_wrapper.py <image> <out.glb>

and must exit 0 after writing the glb.
"""
from __future__ import annotations

import sys

import bpy

from charforge.bl import condition
from charforge.bl.procedural import mesh_from_body
from charforge.bodygen import build_body
from charforge.spec import DEFAULT_BODY_PARAMS


def main(image: str, out: str) -> None:
    condition.reset_scene()
    p = dict(DEFAULT_BODY_PARAMS, segments=14.0, arm_angle_deg=30.0)
    body = build_body(p, 1.0)
    obj = mesh_from_body(body, "generated", smooth=True)
    obj.vertex_groups.clear()  # generators give you no skin weights
    obj.scale = (37.0, 37.0, 37.0)  # arbitrary units
    obj.location = (5.0, -2.0, 11.0)  # off the floor, off-centre
    bpy.context.view_layer.update()
    for o in bpy.context.view_layer.objects:
        o.select_set(o is obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
