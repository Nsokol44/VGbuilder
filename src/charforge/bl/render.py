"""Contact-sheet renders for human/VLM review. Uses the Workbench engine (fast, CPU-friendly).

If Workbench can't render in this environment, we fall back to rasterised silhouettes so the
pipeline still produces a review image instead of failing.
"""
from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw

from ..rigdefs import bone_name
from ..sketch import rasterize_triangles
from .measure import triangles_xz

VIEWS = {
    "front": Vector((0, -1, 0)),
    "side": Vector((1, 0, 0)),
    "back": Vector((0, 1, 0)),
    "three_quarter": Vector((0.7, -0.7, 0.35)),
}


def _setup(height: float, size: int):
    s = bpy.context.scene
    s.render.engine = "BLENDER_WORKBENCH"
    s.render.resolution_x = s.render.resolution_y = size
    s.render.image_settings.file_format = "PNG"
    s.display.shading.light = "STUDIO"
    # show baked textures when a material has them; plain material colour otherwise
    s.display.shading.color_type = "TEXTURE" if any(i.users and i.size[0] > 0 for i in bpy.data.images) else "MATERIAL"
    s.world = s.world or bpy.data.worlds.new("w")
    s.world.color = (0.18, 0.18, 0.2)
    cam_data = bpy.data.cameras.new("review_cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = height * 1.2
    cam = bpy.data.objects.new("review_cam", cam_data)
    s.collection.objects.link(cam)
    s.camera = cam
    return s, cam


def _shoot(scene, cam, direction: Vector, height: float, path: Path, center: Vector | None = None) -> None:
    target = center if center is not None else Vector((0, 0, height / 2))
    cam.location = target + direction.normalized() * (height * 4)
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def _pose(arm, naming: str, on: bool) -> None:
    pose = {"upperarm_l": (0, 0, 0.9), "upperarm_r": (0, 0, -0.9), "thigh_l": (-0.6, 0, 0), "thigh_r": (0.4, 0, 0)}
    for role, rot in pose.items():
        pb = arm.pose.bones.get(bone_name(role, naming))
        if pb is not None:
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = rot if on else (0, 0, 0)
    bpy.context.view_layer.update()


def contact_sheet(lod_objs, arm, height: float, naming: str, out_png: Path, size: int = 320) -> dict:
    tmp = out_png.parent / "_views"
    tmp.mkdir(parents=True, exist_ok=True)
    labels: list[str] = []
    tiles: list[Image.Image] = []
    mode = "workbench"
    try:
        for o in lod_objs[1:]:
            o.hide_render = True
        arm.hide_render = True
        scene, cam = _setup(height, size)
        for name, d in VIEWS.items():
            p = tmp / f"{name}.png"
            _shoot(scene, cam, d, height, p)
            tiles.append(Image.open(p).convert("RGB")); labels.append(name)
        _pose(arm, naming, True)
        p = tmp / "front_posed.png"
        _shoot(scene, cam, VIEWS["front"], height, p)
        _pose(arm, naming, False)
        tiles.append(Image.open(p).convert("RGB")); labels.append("front_posed")
    except Exception as e:  # noqa: BLE001 - any GPU/EGL problem must not kill the job
        mode = f"silhouette_fallback ({type(e).__name__})"
        tiles, labels = [], []
        xz = triangles_xz(lod_objs[0])
        m = rasterize_triangles(xz, height_px=size)
        tiles.append(Image.fromarray((m * 255).astype(np.uint8)).convert("RGB")); labels.append("front_silhouette")
    finally:
        for o in lod_objs[1:]:
            o.hide_render = False
    w = sum(t.width for t in tiles)
    sheet = Image.new("RGB", (w, size + 18), (24, 24, 28))
    x = 0
    d = ImageDraw.Draw(sheet)
    for t, lab in zip(tiles, labels):
        sheet.paste(t, (x, 18)); d.text((x + 6, 3), lab, fill=(230, 230, 230)); x += t.width
    sheet.save(out_png)
    return {"mode": mode, "views": labels}


def structure_sheet(lods, hidden, size_xyz, out_png: Path, size: int = 320) -> dict:
    """Front / side / back / three-quarter / top renders of a building's LOD0 (other LODs and collision hidden)."""
    tmp = out_png.parent / "_views"
    tmp.mkdir(parents=True, exist_ok=True)
    big = float(max(size_xyz))
    center = Vector((0, 0, size_xyz[2] / 2))
    views = {**{k: VIEWS[k] for k in ("front", "side", "back", "three_quarter")}, "top": Vector((0, -0.02, 1))}
    tiles, labels, mode = [], [], "workbench"
    try:
        for o in [*lods[1:], *hidden]:
            o.hide_render = True
        scene, cam = _setup(big, size)
        for label, d in views.items():
            p = tmp / f"{label}.png"
            _shoot(scene, cam, d, big, p, center)
            tiles.append(Image.open(p).convert("RGB")); labels.append(label)
    except Exception as e:  # noqa: BLE001 - a render problem must not kill the job
        mode = f"no_render ({type(e).__name__})"
        tiles, labels = [Image.new("RGB", (size, size), (40, 40, 44))], ["unavailable"]
    finally:
        for o in lods[1:]:
            o.hide_render = False
    sheet = Image.new("RGB", (sum(t.width for t in tiles), size + 18), (24, 24, 28))
    x, dr = 0, ImageDraw.Draw(sheet)
    for t, lab in zip(tiles, labels):
        sheet.paste(t, (x, 18)); dr.text((x + 6, 3), lab, fill=(230, 230, 230)); x += t.width
    sheet.save(out_png)
    return {"mode": mode, "views": labels}
