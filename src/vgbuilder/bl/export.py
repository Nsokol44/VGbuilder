"""Export each LOD (mesh + armature) and verify the exported file by re-importing it."""
from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np

from .condition import reset_scene, tri_count, vertex_array


def _select_only(objs) -> None:
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


EMBED_KEY = "vgbuilder"


def read_embedded(glb: Path) -> dict | None:
    """The provenance blob stored in a glb's scene extras (None if absent). Pure python: reads the file itself,
    which is what any downstream tool would do."""
    import json
    import struct

    d = Path(glb).read_bytes()
    if d[:4] != b"glTF":
        return None
    n = struct.unpack("<I", d[12:16])[0]
    j = json.loads(d[20:20 + n])
    for sc in j.get("scenes", []):
        raw = (sc.get("extras") or {}).get(EMBED_KEY)
        if raw:
            return json.loads(raw) if isinstance(raw, str) else raw
    return None


def read_summary(glb: Path) -> dict:
    """What a downstream tool sees in a glb: node names + extras, mesh attributes, animations and their targets."""
    import json
    import struct

    d = Path(glb).read_bytes()
    n = struct.unpack("<I", d[12:16])[0]
    j = json.loads(d[20:20 + n])
    nodes = j.get("nodes", [])
    meshes = j.get("meshes", [])
    return {
        "nodes": {nd.get("name", ""): nd.get("extras") or {} for nd in nodes},
        "mesh_attributes": {m.get("name", ""): sorted({a for p in m["primitives"] for a in p["attributes"]}) for m in meshes},
        "animations": {a.get("name", ""): {
            "channels": len(a["channels"]),
            "targets": sorted({nodes[c["target"]["node"]].get("name", "") for c in a["channels"] if "node" in c["target"]}),
        } for a in j.get("animations", [])},
    }


def _embed(mesh_obj, arm, blob: dict | None) -> None:
    import json

    if blob is None:
        return
    text = json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str)
    bpy.context.scene[EMBED_KEY] = text
    for o in (mesh_obj, arm):
        o[EMBED_KEY] = text


def export_lod(mesh_obj, arm, out_path: Path, fmt: str, engine: str = "generic", embed: dict | None = None,
               extras: list | None = None, animations: bool = False) -> None:
    from ..engines import preset

    pr = preset(engine)
    _embed(mesh_obj, arm, embed)
    _select_only([arm, mesh_obj, *(extras or [])])
    if fmt == "glb":
        bpy.ops.export_scene.gltf(
            filepath=str(out_path), export_format="GLB", use_selection=True,
            export_apply=False, export_skins=True, export_animations=animations, export_animation_mode="ACTIONS",
            export_attributes=True, export_yup=True,
            export_influence_nb=pr.max_influences, export_extras=True,
        )
    elif fmt == "fbx":
        bpy.ops.export_scene.fbx(
            filepath=str(out_path), use_selection=True, object_types={"ARMATURE", "MESH"},
            add_leaf_bones=False, bake_anim=animations, bake_anim_use_all_actions=True, bake_anim_use_nla_strips=False,
            use_custom_props=True, path_mode="COPY", embed_textures=True,
            **{"axis_forward": "-Z", "axis_up": "Y", **pr.fbx},
        )
    else:
        raise ValueError(fmt)


def export_all(lods, arm, out_dir: Path, name: str, formats: list[str], engine: str = "generic",
               embed: dict | None = None, ratios: list[float] | None = None, extras: list | None = None,
               animations: bool = False) -> list[Path]:
    files: list[Path] = []
    for k, o in enumerate(lods):
        for fmt in formats:
            p = out_dir / f"{name}_LOD{k}.{fmt}"
            blob = None if embed is None else {**embed, "lod": k, "lod_ratio": (ratios or [1.0] * len(lods))[k]}
            export_lod(o, arm, p, fmt, engine, blob, extras, animations)
            files.append(p)
    return files


def roundtrip_glb(path: Path, expect_tris: int, expect_height: float, expect_bones: int) -> dict:
    """Wipe the scene and re-import the exported glb. Call this LAST: it clears everything."""
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(path))
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    # the glTF importer also adds helper meshes (bone shapes); only count skinned character meshes
    meshes = [
        o for o in bpy.context.scene.objects
        if o.type == "MESH" and any(mod.type == "ARMATURE" for mod in o.modifiers)
    ]
    if not meshes or not arms:
        return {"ok": False, "reason": "missing mesh or armature after import"}
    tris = sum(tri_count(m) for m in meshes)
    allco = np.concatenate([vertex_array(m) @ np.array(m.matrix_world.to_3x3()).T + np.array(m.matrix_world.translation) for m in meshes])
    height = float(allco[:, 2].max() - allco[:, 2].min())
    bones = len(arms[0].data.bones)
    skinned = any(len(m.vertex_groups) > 0 for m in meshes)
    ok = (
        abs(tris - expect_tris) <= max(2, 0.01 * expect_tris)
        and abs(height - expect_height) <= 0.02 * expect_height
        and bones == expect_bones
        and skinned
    )
    return {"ok": bool(ok), "tris": tris, "height_m": height, "bones": bones, "skinned": skinned}
