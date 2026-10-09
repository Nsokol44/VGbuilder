"""Runs ONE build iteration inside Blender, starting from an empty scene.

    python -m vgbuilder.bl.entry job.json result.json          # with the bpy module
    blender -b -P entry.py -- job.json result.json              # with a Blender binary

The scene is rebuilt from scratch on every call, so the same job + params always give the
same result. The .blend written at the end is a build artifact, never an input.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # launched as a script by `blender -P`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from vgbuilder import accessories, engines, proportions, sketch as sk  # noqa: E402
from vgbuilder.bl import anim, condition, export, looks, lod, measure, parts as parts_bl, procedural, render, rig, sprites  # noqa: E402
from vgbuilder.bodygen import build_body  # noqa: E402
from vgbuilder.rigdefs import joints_from_vertices  # noqa: E402
from vgbuilder.spec import JobSpec  # noqa: E402


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def run(job: dict) -> dict:
    if job["spec"].get("kind") == "structure":
        from vgbuilder.bl import structure_build
        return structure_build.run(job)
    t0 = time.time()
    spec = JobSpec(**job["spec"])
    params = job["params"]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name, naming, H = spec.name, spec.resolved_naming(), spec.height_m
    pr = engines.preset(spec.target_engine)
    steps: list[str] = []

    condition.reset_scene()
    parts_report: dict = {}
    base_height = None
    extra_objs: list[tuple[str, object]] = []
    if spec.provider == "procedural":
        body = build_body(params, H)
        joints = body.joints
        smooth = spec.style != "lowpoly"
        obj = procedural.mesh_from_body(body, name, smooth=smooth)
        # the sketch is compared against the base body; accessories are added on top afterwards
        base_masks = (measure.front_mask(obj), measure.side_mask(obj))
        base_height = float(body.verts[:, 2].max() - body.verts[:, 2].min())
        if spec.accessories:
            parts_report, extra_objs = parts_bl.add_parts(
                obj, accessories.build_parts(list(spec.accessories), params, H, shape=spec.accessory_shape), name, smooth)
            steps.append("accessories")
        condition.ensure_material(obj, spec.style, name)
        condition.smart_uv(obj)
        for _, eo in extra_objs:
            condition.ensure_material(eo, spec.style, eo.name)
            condition.smart_uv(eo)
        arm = rig.build_armature(joints, naming, name)
        for o in (obj, *[eo for _, eo in extra_objs]):
            rig.rename_groups_to_bones(o, naming)
            rig.bind(o, arm)
        weights_report = {"method": "rigid-procedural"}
        steps += ["build_procedural", "uv", "rig"]
    else:
        obj = condition.import_glb(Path(job["imported_glb"]))
        norm = condition.normalize(obj, H)
        fit = condition.fit_budget(obj, spec.resolved_budget())
        joints = joints_from_vertices(condition.vertex_array(obj))
        arm = rig.build_armature(joints, naming, name)
        weights_report = rig.auto_weights(obj, arm, joints, naming)
        weights_report.update(normalize=norm, budget=fit)
        condition.ensure_uv(obj)
        condition.ensure_material(obj, spec.style, name)
        base_masks = (measure.front_mask(obj), measure.side_mask(obj))
        steps += ["import", "normalize", "fit_budget", "rig", "uv"]

    style = spec.load_style()
    texture_report = None
    extra_files: list[Path] = []
    if style is not None and spec.texture != "none":
        mode = spec.texture
        if mode == "auto":
            mode = "keep" if looks.existing_base_image(obj) is not None else "replace"
        texture_report, extra_files = looks.process(
            obj, style, mode, spec.texture_size, spec.seed, float(params.get("wear_scale", 1.0)), H, out_dir, name)
        steps.append(f"look_{mode}")
        for pname, eo in extra_objs:                    # e.g. the cape gets its own bake (own UV layout)
            _, f = looks.process(eo, style, "replace", max(128, spec.texture_size // 2), spec.seed + 7,
                                 float(params.get("wear_scale", 1.0)), H, out_dir, f"{name}_{pname}")
            extra_files += f

    influence = rig.limit_influences(obj, pr.max_influences)
    if influence["limited"]:
        steps.append("limit_influences")

    lods = lod.make_lods(obj, arm, spec.lods, name)
    steps.append("lods")
    metrics = measure.collect(lods[0], lods, arm, naming, H, list(spec.lods))
    metrics["weights"] = weights_report
    if parts_report:
        metrics["height_m_total"] = metrics["height_m"]       # horns stick out above the head
        metrics["height_m"] = base_height if base_height else metrics["height_m"]
        extra_tris = sum(parts_report[n]["tris"] for n, _ in extra_objs)
        metrics["accessories"] = parts_report
        metrics["lod_tris"] = [t + extra_tris for t in metrics["lod_tris"]]   # separate meshes ship in every LOD file
    if texture_report is not None:
        metrics["texture"] = texture_report
    metrics["engine"] = {**pr.describe(), "naming_used": naming, "influence_limit": influence}

    # ---- compare against the sketch (same measurement applied to both sides)
    mask = base_masks[0]
    profiles: dict = {"actual": sk.width_profile(mask).tolist()}
    crotch = {"actual": sk.crotch_fraction(mask)}
    sketch_path = spec.resolve_path(spec.sketch)
    if sketch_path is not None:
        target = sk.load_mask(sketch_path)
        metrics["silhouette_iou"] = sk.silhouette_iou(mask, target)
        profiles["target"] = sk.width_profile(target).tolist()
        crotch["target"] = sk.crotch_fraction(target)

    side_path = spec.resolve_path(spec.sketch_side)
    if side_path is not None:
        smask = base_masks[1]
        starget = sk.load_side_mask(side_path, spec.side_facing)
        metrics["silhouette_side_iou"] = sk.silhouette_iou(smask, starget)
        profiles["side_actual"] = sk.width_profile(smask).tolist()
        profiles["side_target"] = sk.width_profile(starget).tolist()
    if spec.provider == "procedural":
        metrics["proportions"] = proportions.report(params, H)

    actions = anim.create_actions(arm, naming, list(spec.animations), H, root_motion=spec.root_motion) if spec.animations else {}
    if actions:
        lifts = anim.clamp_floor(arm, lods[0], actions, naming)
        metrics["animations"] = anim.measure(arm, lods[0], actions, naming, H, root_motion=spec.root_motion)
        for a, v in lifts.items():
            if a in metrics["animations"]:
                metrics["animations"][a]["floor_lift_m"] = v
        steps.append("animations")

    sil_files = []
    for tag, m_ in (("front", base_masks[0]), ("side", base_masks[1])):    # base-body silhouettes: library + uniqueness checks use them
        sp = out_dir / f"{name}_silhouette_{tag}.png"
        Image.fromarray(np.where(m_, 20, 245).astype(np.uint8), "L").save(sp)
        sil_files.append(sp)

    sheet = out_dir / f"{name}_contact_sheet.png"
    metrics["render"] = render.contact_sheet(lods, arm, H, naming, sheet)
    steps.append("contact_sheet")

    sprite_files: list[Path] = []
    if spec.sprite_sheet and actions:
        srep = sprites.render_sprites(lods, arm, actions, naming, H, out_dir, name, spec.sprite_size)
        metrics["sprites"] = srep
        if srep.get("ok"):
            sprite_files = [out_dir / srep["sheet"], out_dir / srep["metadata"]]
        steps.append("sprites")

    blend = out_dir / f"{name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    embed = {
        **(job.get("provenance") or {}),
        "params": {k: round(float(v), 6) for k, v in sorted(params.items())},
        "blender": bpy.app.version_string,
        "naming": naming,
        "texture": ({k: texture_report[k] for k in ("mode", "style", "size", "wear_scale")} if texture_report else None),
    }
    files = export.export_all(lods, arm, out_dir, name, spec.formats, spec.target_engine, embed, spec.lods,
                              [eo for _, eo in extra_objs], bool(actions))
    steps.append("export")

    glbs = [f for f in files if f.suffix == ".glb"]
    if glbs:
        metrics["embedded"] = [export.read_embedded(g) for g in glbs]
        metrics["glb_content"] = [export.read_summary(g) for g in glbs]
        metrics["roundtrip"] = export.roundtrip_glb(
            glbs[0], metrics["lod_tris"][0], metrics.get("height_m_total", metrics["height_m"]), len(metrics["bone_names"])
        )
        steps.append("roundtrip")

    return {
        "metrics": metrics,
        "profiles": profiles,
        "crotch": crotch,
        "files": [str(p) for p in files] + [str(sheet), str(blend)] + [str(p) for p in extra_files] + [str(p) for p in sil_files] + [str(p) for p in sprite_files],
        "steps": steps,
        "blender_version": bpy.app.version_string,
        "seconds": round(time.time() - t0, 2),
    }


def main(argv: list[str]) -> int:
    job = json.loads(Path(argv[0]).read_text())
    try:
        result = run(job)
        result["ok"] = True
    except Exception as e:  # noqa: BLE001
        import traceback

        result = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    Path(argv[1]).write_text(json.dumps(result, default=_jsonable, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(args))
