"""Blender side of look-dev: pull the mesh out, push baked textures back in as a material."""
from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np
from PIL import Image

from .. import lookdev
from ..styles import Style
from .condition import vertex_array


def extract(obj) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(verts (N,3), tris (T,3), uv (T,3,2) per corner, vertex normals (N,3)) of the rest-pose mesh."""
    m = obj.data
    m.calc_loop_triangles()
    verts = vertex_array(obj)
    T = len(m.loop_triangles)
    tris = np.empty(T * 3, np.int32)
    loops = np.empty(T * 3, np.int32)
    m.loop_triangles.foreach_get("vertices", tris)
    m.loop_triangles.foreach_get("loops", loops)
    layer = m.uv_layers.active
    uv_all = np.empty(len(layer.data) * 2)
    layer.data.foreach_get("uv", uv_all)
    uv = uv_all.reshape(-1, 2)[loops.reshape(-1, 3)]
    nrm = np.empty(len(m.vertices) * 3)
    m.vertex_normals.foreach_get("vector", nrm)
    return verts, tris.reshape(-1, 3), uv, nrm.reshape(-1, 3)


def _principled(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF") or next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return nt, bsdf


def existing_base_image(obj):
    """The image wired into Base Color of the first material, if any."""
    if not obj.data.materials or obj.data.materials[0] is None or not obj.data.materials[0].use_nodes:
        return None
    nt = obj.data.materials[0].node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return None
    link = next((l for l in nt.links if l.to_node == bsdf and l.to_socket.name == "Base Color"), None)
    if link is not None and link.from_node.type == "TEX_IMAGE" and link.from_node.image is not None:
        return link.from_node.image
    return None


def image_to_rgb(img, size: int, tmp_dir: Path) -> np.ndarray:
    """Base-colour image as (size, size, 3) uint8, row 0 = top, as stored in the file (sRGB)."""
    tmp = tmp_dir / "_existing_basecolor.png"
    old_path, old_fmt = img.filepath_raw, img.file_format
    img.filepath_raw, img.file_format = str(tmp), "PNG"
    img.save()
    img.filepath_raw, img.file_format = old_path, old_fmt
    arr = Image.open(tmp).convert("RGB").resize((size, size), Image.BILINEAR)
    tmp.unlink(missing_ok=True)
    return np.asarray(arr)


def apply_textures(obj, bake: lookdev.Bake, out_dir: Path, name: str) -> list[Path]:
    """Save the baked maps as PNGs and wire them into the object's (first) material."""
    base_p, rough_p = out_dir / f"{name}_basecolor.png", out_dir / f"{name}_roughness.png"
    Image.fromarray(bake.base_rgb, "RGB").save(base_p)
    Image.fromarray(bake.roughness, "L").save(rough_p)
    if not obj.data.materials or obj.data.materials[0] is None:
        obj.data.materials.append(bpy.data.materials.new(f"{name}_mat"))
    mat = obj.data.materials[0]
    nt, bsdf = _principled(mat)
    for l in [l for l in nt.links if l.to_node == bsdf and l.to_socket.name in ("Base Color", "Roughness")]:
        nt.links.remove(l)
    for path, socket, cs in ((base_p, "Base Color", "sRGB"), (rough_p, "Roughness", "Non-Color")):
        img = bpy.data.images.load(str(path), check_existing=False)
        img.colorspace_settings.name = cs
        img.pack()
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.label = socket
        nt.links.new(tex.outputs["Color"], bsdf.inputs[socket])
    return [base_p, rough_p]


def process(obj, style: Style, mode: str, size: int, seed: int, wear_scale: float, height: float,
            out_dir: Path, name: str) -> tuple[dict, list[Path]]:
    """Run the look-dev step for one object. Returns (report for metrics['texture'], new files)."""
    verts, tris, uv, nrm = extract(obj)
    report: dict = {"mode": mode, "style": style.name, "size": size, "wear_scale": wear_scale}
    files: list[Path] = []
    if mode == "replace":
        bake = lookdev.bake(verts, tris, uv, style, height, size=size, seed=seed, wear_scale=wear_scale, normals=nrm)
        files = apply_textures(obj, bake, out_dir, name)
        report["analysis"] = lookdev.analyze(bake.base_rgb, bake.covered, bake.height01, style)
    elif mode == "keep":
        img = existing_base_image(obj)
        if img is None:
            report["analysis"] = {"ok": False, "reason": "no base-colour image to analyse (flat material colour)"}
        else:
            covered, h01 = lookdev.uv_coverage(verts, tris, uv, size)
            rgb = image_to_rgb(img, size, out_dir)
            report["analysis"] = lookdev.analyze(rgb, covered, h01, style)
    return report, files
