"""Rule-based QA. Pure python: takes the metrics measured inside Blender and returns failures.

Every failure has a stable `code`, so the fixer (and a human reading the manifest)
can act on it. Anything that cannot be fixed automatically ends up as `needs_review`.
"""
from __future__ import annotations

from typing import Any

from . import lookdev
from .spec import JobSpec

# Bones every humanoid must have, by logical role. Names per naming scheme live in bl/rig.py
REQUIRED_ROLES = [
    "root", "pelvis", "spine", "chest", "neck", "head",
    "upperarm_l", "forearm_l", "hand_l", "upperarm_r", "forearm_r", "hand_r",
    "thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r",
]

FIXABLE = {"over_budget"}


def evaluate_structure(metrics: dict[str, Any], spec: JobSpec) -> list[dict[str, Any]]:
    """QA for buildings: scale against the numbers they were built from, density, UVs (texture + lightmap), LODs,
    collision proxy, re-import, embedded provenance, and the same texture checks as characters."""
    qa = spec.resolved_qa()
    fails: list[dict[str, Any]] = []

    def fail(code: str, detail: str, **extra: Any) -> None:
        fails.append({"code": code, "detail": detail, "fixable": extra.pop("fixable", code in FIXABLE), **extra})

    ex, bb = metrics["expected"], metrics["bbox"]
    W, D = ex["footprint"]
    pr = ex["max_protrusion"]
    lod0 = metrics["lod_tris"][0]
    if lod0 > spec.resolved_budget() * 1.05:
        fail("over_budget", f"LOD0 has {lod0} tris, budget {spec.resolved_budget()}")
    tol = qa["height_tolerance"]
    if abs(metrics["height_m"] - ex["height"]) > tol * ex["height"]:
        fail("height_off", f"height {metrics['height_m']:.2f} m, expected {ex['height']:.2f} m from the structure numbers")
    sx, sy = bb["size"][0], bb["size"][1]
    for axis, size, want in (("width", sx, W), ("depth", sy, D)):
        if size < want * (1 - tol) or size > want + 2 * pr + tol * want:
            fail("footprint_off", f"{axis} {size:.2f} m, expected {want:.2f} m (+ at most {pr:.2f} m of trim/roof on each side)")
    if abs(bb["min"][2]) > 0.01:
        fail("off_ground", f"lowest point is {bb['min'][2]:.3f} m, should be 0")
    for axis, (lo, hi), want in (("x", (bb["min"][0], bb["max"][0]), W), ("y", (bb["min"][1], bb["max"][1]), D)):
        if abs((lo + hi) / 2) > 0.5 * pr + tol * want:
            fail("off_center", f"bounding box centre is {(lo + hi) / 2:.2f} m off the origin along {axis}")
    dens = lod0 / ex["envelope_area"]
    if dens > qa["tri_density_max"]:
        fail("tri_density_high", f"{dens:.1f} tris per m2 of envelope > {qa['tri_density_max']}")
    if metrics["degenerate_frac"] > 0.001:
        fail("degenerate_faces", f"{metrics['degenerate_frac']:.2%} of triangles have (almost) zero area")
    if not metrics.get("has_uv"):
        fail("no_uv", "no UV map")
    if not metrics.get("has_material"):
        fail("no_material", "no material")
    lm = metrics.get("lightmap", {})
    if not lm.get("present"):
        fail("lightmap_missing", "no second UV set named 'lightmap'")
    else:
        if lm["overlap_fraction"] > qa["lightmap_overlap_max"]:
            fail("lightmap_overlap", f"{lm['overlap_fraction']:.2%} of lightmap texels are covered more than once")
        if lm["out_of_range"] > 0:
            fail("lightmap_out_of_range", f"{lm['out_of_range']:.1%} of triangles leave the 0..1 UV square")
    lods = metrics["lods"]
    for k in range(1, len(lods)):
        row = lods[k]
        if row["tris"] >= lods[k - 1]["tris"]:
            fail("lod_not_decreasing", f"LOD{k} has {row['tris']} tris, LOD{k - 1} has {lods[k - 1]['tris']}", lod=k)
        worst = min(row["silhouette_front_vs_lod0"], row["silhouette_side_vs_lod0"], row["silhouette_top_vs_lod0"])
        if worst < qa["lod_silhouette_min"]:
            fail("lod_silhouette_drift", f"LOD{k} silhouette IoU vs LOD0 {worst:.3f} < {qa['lod_silhouette_min']}", lod=k)
        for ax, i in (("width", 0), ("depth", 1)):
            if abs(row["size"][i] - lods[0]["size"][i]) > qa["lod_footprint_tolerance"] * lods[0]["size"][i]:
                fail("lod_footprint_drift", f"LOD{k} {ax} {row['size'][i]:.2f} m vs LOD0 {lods[0]['size'][i]:.2f} m", lod=k)
        if abs(row["size"][2] - lods[0]["size"][2]) > qa["lod_height_tolerance"] * lods[0]["size"][2]:
            fail("lod_height_drift", f"LOD{k} height {row['size'][2]:.2f} m vs LOD0 {lods[0]['size'][2]:.2f} m", lod=k)
        if not (row["has_uv"] and row["has_material"]):
            fail("lod_inconsistent", f"LOD{k} lacks a UV map or a material", lod=k)
    col = metrics.get("collision", {})
    if not col or col.get("pieces", 0) == 0:
        fail("collision_missing", "no collision proxy was produced")
    else:
        if col["tris"] > qa["collision_max_tris"]:
            fail("collision_heavy", f"collision proxy has {col['tris']} tris > {qa['collision_max_tris']}")
        ct = qa["collision_bbox_tolerance"]
        cs = col["bbox"]["size"]
        if cs[0] < W * (1 - ct) or cs[1] < D * (1 - ct) or cs[2] < ex["height"] * (1 - ct):
            fail("collision_bbox_off", f"collision box {cs[0]:.2f} x {cs[1]:.2f} x {cs[2]:.2f} m does not cover the building")
    rt = metrics.get("roundtrip")
    if rt is not None and not rt.get("ok", False):
        fail("roundtrip_failed", f"exported file did not re-import cleanly: {rt}")
    emb = metrics.get("embedded")
    if emb is not None and (not emb or any(e is None for e in emb)):
        fail("provenance_missing", "an exported glb carries no embedded provenance")
    tex = metrics.get("texture")
    if tex and tex.get("analysis") is not None and "analysis" in tex:
        style = spec.load_style()
        if style is not None and tex["analysis"].get("ok", True):
            for f in lookdev.check(tex["analysis"], style):
                fixable = f["code"] == "texture_wear_coverage" and f["direction"] in ("more", "less") and tex["mode"] == "replace"
                fail(f["code"], f["detail"], fixable=fixable, direction=f["direction"])
    return fails


def evaluate(metrics: dict[str, Any], spec: JobSpec) -> list[dict[str, Any]]:
    if spec.kind == "structure":
        return evaluate_structure(metrics, spec)
    qa = spec.resolved_qa()
    fails: list[dict[str, Any]] = []

    def fail(code: str, detail: str, **extra: Any) -> None:
        fails.append({"code": code, "detail": detail, "fixable": code in FIXABLE, **extra})

    budget = spec.resolved_budget()
    lod0 = metrics["lod_tris"][0]
    if lod0 > budget * 1.05:
        fail("over_budget", f"LOD0 has {lod0} tris, budget {budget}")

    want_h = spec.height_m
    if abs(metrics["height_m"] - want_h) > qa["height_tolerance"] * want_h:
        fail("height_off", f"height {metrics['height_m']:.3f} m, wanted {want_h} m")

    if metrics["symmetry"] > qa["symmetry_max"]:
        fail("asymmetric", f"mirror error {metrics['symmetry']:.4f} of height (max {qa['symmetry_max']})")

    if metrics["zero_weight_frac"] > qa["max_zero_weight_frac"]:
        fail("unweighted", f"{metrics['zero_weight_frac']:.2%} of vertices have no skin weight")

    if metrics["loose_parts"] > qa["max_loose_parts"]:
        fail("fragmented", f"{metrics['loose_parts']} disconnected mesh islands")

    st = metrics.get("stretch_max")
    if st is not None and (st != st or st > qa["max_edge_stretch"]):  # NaN counts as a failure
        fail("bad_deformation", f"posed edge stretch {st:.2f}x (max {qa['max_edge_stretch']}x)")

    from .engines import preset

    limit = preset(spec.target_engine).max_influences
    if metrics.get("max_influences", 0) > limit:
        fail("too_many_influences", f"{metrics['max_influences']} skin weights on a vertex, {spec.target_engine} keeps {limit}")

    missing = [r for r in REQUIRED_ROLES if r not in metrics["bone_roles"]]
    if missing:
        fail("missing_bones", f"missing roles: {missing}")

    if not metrics["has_uv"]:
        fail("no_uv", "mesh has no UV map")

    rt = metrics.get("roundtrip")
    if rt is not None and not rt.get("ok", False):
        fail("roundtrip_failed", f"exported file did not re-import cleanly: {rt}")

    if spec.sketch and "silhouette_iou" in metrics:
        iou = metrics["silhouette_iou"]
        if spec.provider == "procedural":
            if iou < qa["silhouette_iou_min"]:
                fail("silhouette_low", f"silhouette IoU {iou:.3f} < {qa['silhouette_iou_min']}")
        else:
            # An image-to-3D mesh re-interprets the sketch, so its gate has its own key (`silhouette_iou_min_generated`)
            # and a separate "does not resemble it at all" code (usually a wrong facing axis or a failed generation).
            # Defaults are NOT loosened: no real generator output was available to calibrate anything softer.
            if iou < qa["silhouette_collapse_iou"]:
                fail("generated_mesh_collapsed",
                     f"silhouette IoU {iou:.3f} < {qa['silhouette_collapse_iou']}: the generated mesh barely resembles the sketch "
                     "(check facing axis, or regenerate with another seed)")
            elif iou < qa["silhouette_iou_min_generated"]:
                fail("silhouette_low", f"silhouette IoU {iou:.3f} < {qa['silhouette_iou_min_generated']} (generated-mesh threshold)")
    if spec.sketch_side and "silhouette_side_iou" in metrics:
        if metrics["silhouette_side_iou"] < qa["silhouette_side_iou_min"]:
            fail(
                "silhouette_side_low",
                f"side silhouette IoU {metrics['silhouette_side_iou']:.3f} < {qa['silhouette_side_iou_min']}",
            )
    lods = metrics.get("lods")
    if lods:
        base_h = lods[0]["height_m"]
        for e in lods:
            k = e["lod"]
            if k > 0:
                if abs(e["ratio"] - e["target_ratio"]) > qa["lod_ratio_tolerance"] * e["target_ratio"]:
                    fail("lod_ratio_off", f"LOD{k} has {e['ratio']:.2f} of LOD0's triangles, asked for {e['target_ratio']}", lod=k)
                worst = min(e["silhouette_front_vs_lod0"], e["silhouette_side_vs_lod0"])
                if worst < qa["lod_silhouette_min"]:
                    fail("lod_silhouette_drift", f"LOD{k} silhouette IoU vs LOD0 {worst:.3f} < {qa['lod_silhouette_min']}", lod=k)
                if abs(e["height_m"] - base_h) > qa["lod_height_tolerance"] * base_h:
                    fail("lod_inconsistent", f"LOD{k} height {e['height_m']:.3f} vs LOD0 {base_h:.3f}", lod=k)
            problems = [name for name, ok in (
                ("not bound to the shared armature", e["armature_ok"]),
                ("vertex groups differ from LOD0", e["groups_match_lod0"]),
                ("vertex groups are not bones", e["groups_are_bones"]),
                ("no UV map", e["has_uv"]), ("no material", e["has_material"]),
                ("unweighted vertices", e["zero_weight_frac"] <= qa["max_zero_weight_frac"])) if not ok]
            if problems:
                fail("lod_inconsistent", f"LOD{k}: {', '.join(problems)}", lod=k)

    emb = metrics.get("embedded")
    if emb is not None:
        fp = spec.fingerprint()
        bad = [k for k, b in enumerate(emb) if not b or b.get("fingerprint") != fp or b.get("lod") != k]
        if bad:
            fail("provenance_missing", f"exported glb for LOD{bad} lacks a matching embedded provenance block")

    contents = metrics.get("glb_content") or []
    bones = set(metrics.get("bone_names", []))
    for acc in spec.accessories:
        rep = (metrics.get("accessories") or {}).get(acc)
        if rep is None:
            fail("accessory_missing", f"accessory {acc} was not built", accessory=acc)
        elif acc == "cape":
            node = f"{spec.name}_cape"
            for k, gc in enumerate(contents):
                tagged = gc["nodes"].get(node, {}).get("vgbuilder_physics") == "cloth"
                pinned = any("_CLOTH_PIN" in a for a in gc["mesh_attributes"].values())
                if not (tagged and pinned):
                    fail("accessory_missing", f"cape in LOD{k} glb lacks its cloth tag or _CLOTH_PIN attribute", accessory=acc)
    for anim in spec.animations:
        info = (metrics.get("animations") or {}).get(anim)
        if info is None:
            fail("anim_missing", f"animation {anim} was not built", animation=anim)
            continue
        if not info["loops"]:
            fail("anim_not_looping", f"{anim}: last frame differs from the first", animation=anim)
        if info["max_stretch"] > qa["max_edge_stretch"]:
            fail("anim_bad_deformation", f"{anim}: edge stretch {info['max_stretch']:.2f}x (max {qa['max_edge_stretch']}x)", animation=anim)
        if info["min_z_m"] < -qa["anim_floor_penetration"] * spec.height_m:
            fail("anim_floor_penetration", f"{anim}: lowest point {info['min_z_m']:.3f} m below the floor", animation=anim)
        if info["max_rotation_rad"] > qa["anim_max_rotation"]:
            fail("anim_range", f"{anim}: a joint rotates {info['max_rotation_rad']:.2f} rad", animation=anim)
        for k, gc in enumerate(contents):
            a = gc["animations"].get(anim)
            if not a or a["channels"] == 0 or not set(a["targets"]) <= bones:
                fail("anim_missing", f"{anim} is absent or targets non-bone nodes in the LOD{k} glb", animation=anim)

    tex = metrics.get("texture")
    if tex and tex.get("analysis") is not None and "analysis" in tex:
        style = spec.load_style()
        if style is not None and tex["analysis"].get("ok", True):
            for f in lookdev.check(tex["analysis"], style):
                # wear coverage can be corrected by rebuilding with a different wear_scale, but only for textures we bake
                fixable = f["code"] == "texture_wear_coverage" and f["direction"] in ("more", "less") and tex["mode"] == "replace"
                fail(f["code"], f["detail"], fixable=fixable, direction=f["direction"])
    return fails


def score(metrics: dict[str, Any], fails: list[dict[str, Any]]) -> float:
    """Higher is better. Used to pick the best iteration when none fully passes."""
    s = metrics.get("silhouette_iou", 1.0)
    if metrics.get("kind") == "structure":
        s = min([r["silhouette_front_vs_lod0"] for r in metrics["lods"]] + [1.0])
    if "silhouette_side_iou" in metrics:
        s = (s + metrics["silhouette_side_iou"]) / 2
    return s - 0.25 * len(fails)
