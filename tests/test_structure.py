"""Structures: pure geometry (no Blender), QA rules, and full builds through Blender."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from charforge import qa, structuregen as sg, structuremeasure as sm
from charforge.bl.export import read_embedded
from charforge.farm import make_variants
from charforge.library import Library, summarize, work
from charforge.orchestrator import Backend, build_one
from charforge.spec import JobSpec, load_job

ROOT = Path(__file__).resolve().parents[1]


def volume(v, t):
    tri = v[t].astype(np.float64)
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)


def ray_hits(v, t, o, d):
    """Möller–Trumbore against every triangle; returns sorted hit distances along d (d unit)."""
    tri = v[t].astype(np.float64)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    s = np.array(o) - tri[:, 0]
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    w = (q @ d) * inv
    tt = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (tt > 1e-9)
    return np.sort(tt[hit])


# ------------------------------------------------------------------ resolve / validation
@pytest.mark.parametrize("raw,msg", [
    ({"style": "castle"}, "style must be"),
    ({"bogus": 1}, "unknown structure keys"),
    ({"footprint": [2, 9]}, "footprint"),
    ({"footprint": [10]}, "footprint"),
    ({"floors": 0}, "floors"),
    ({"floors": 2.5}, "whole number"),
    ({"floor_height": 9}, "floor_height"),
    ({"roof": "dome"}, "roof must"),
    ({"parapet": "spikes"}, "parapet must"),
    ({"windows": "yes"}, "true or false"),
    ({"damage": 1.5}, "damage"),
    ({"wall_t": 1.4, "footprint": [3, 3]}, "too thick"),
    ({"window_w": 3.0, "bay_width": 3.0}, "leaves less than"),
    ({"window_h": 2.9, "sill_h": 0.5, "floor_height": 3.2}, "sill_h \\+ window_h"),
    ({"door_h": 3.1, "floor_height": 3.2}, "door does not fit"),
    ({"floors": True}, "floors"),
])
def test_resolve_rejects(raw, msg):
    with pytest.raises(ValueError, match=msg):
        sg.resolve(raw)


@pytest.mark.parametrize("style", sorted(sg.STYLES))
def test_style_defaults_are_valid_and_overridable(style):
    d = sg.resolve({"style": style})
    assert d["style"] == style
    assert sg.resolve({"style": style, "floors": 1, "roof": "hip"})["roof"] == "hip"


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("style", sorted(sg.STYLES))
@pytest.mark.parametrize("level", [0, 1, 2])
def test_meshes_are_closed_positive_and_match_the_numbers(style, level):
    d = sg.resolve({"style": style})
    v, t, r, meta = sg.build(d, level)
    assert volume(v, t) > 0 and len(t) == len(r) == meta["tris"]
    lo, hi = v.min(0), v.max(0)
    assert lo[2] == pytest.approx(0, abs=1e-6)
    assert hi[2] == pytest.approx(sg.expected_height(d), abs=1e-3)          # same top at every level of detail
    W, D = d["footprint"]
    pr = sg.max_protrusion(d)
    assert W - 1e-4 <= hi[0] - lo[0] <= W + 2 * pr + 1e-4 and D - 1e-4 <= hi[1] - lo[1] <= D + 2 * pr + 1e-4
    assert abs((hi[0] + lo[0]) / 2) < 0.5 * pr + 1e-3 and abs((hi[1] + lo[1]) / 2) < 0.5 * pr + 1e-3
    assert not np.isnan(v).any() and t.min() >= 0 and t.max() < len(v)


def test_levels_shrink_and_are_deterministic():
    for style in sg.STYLES:
        d = sg.resolve({"style": style})
        n = [len(sg.build(d, lv)[1]) for lv in (0, 1, 2)]
        assert n[0] > n[1] > n[2]
        a, b = sg.build(d, 0, 4), sg.build(d, 0, 4)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_opening_counts_follow_the_bays():
    d = sg.resolve({"style": "hall"})                     # 12 x 7, bay 3 -> front/back 4 bays, sides (7-0.7)/3 -> 2 bays
    meta = sg.build(d, 0)[3]
    assert meta["doors"] == 1 and meta["windows"] == 2 * 4 * 2 - 1 + 2 * 2 * 2
    assert sg.build(sg.resolve({"style": "hall", "windows": False}), 0)[3]["windows"] == 0


def test_openings_are_really_open_and_piers_are_solid():
    d = sg.resolve({"style": "hall", "floors": 1, "string_courses": False})
    v, t, r, meta = sg.build(d, 1)
    W, D = d["footprint"]
    tw = d["wall_t"]
    bays = round(W / d["bay_width"]); bw = W / bays
    uc = -W / 2 + 0.5 * bw                                # bay 0: a window (the door is bay n//2)
    z = d["sill_h"] + d["window_h"] / 2
    through_window = ray_hits(v, t, (uc, -D / 2 - 1.0, z), np.array([0, 1.0, 0]))
    assert not (through_window < 1.0 + tw + 1e-6).any()   # nothing in the wall thickness: the window is open
    pier_x = uc + d["window_w"] / 2 + 0.15
    hit = ray_hits(v, t, (pier_x, -D / 2 - 1.0, z), np.array([0, 1.0, 0]))
    assert len(hit) and hit[0] == pytest.approx(1.0, abs=1e-6)       # a pier is solid at the outer face
    door_u = -W / 2 + (bays // 2 + 0.5) * bw
    door = ray_hits(v, t, (door_u, -D / 2 - 1.0, 1.0), np.array([0, 1.0, 0]))
    assert not (door < 1.0 + tw + 1e-6).any()


def test_hip_spire_flat_and_gable_roofs():
    for roof in sg.ROOFS:
        d = sg.resolve({"style": "hall", "roof": roof})
        v, t, r, meta = sg.build(d, 0)
        assert (r == sg.ROLE_ID["roof"]).any() and volume(v, t) > 0
        assert v[:, 2].max() == pytest.approx(sg.expected_height(d), abs=1e-3)
    wide = sg.build(sg.resolve({"style": "hall", "roof": "hip", "footprint": [7, 12]}), 0)
    assert wide[0][:, 2].max() == pytest.approx(sg.expected_height(sg.resolve({"style": "hall", "roof": "hip", "footprint": [7, 12]})), abs=1e-3)


def test_trim_and_buttresses_only_at_level_zero():
    d = sg.resolve({"style": "gothic"})
    roles = lambda lv: set(sg.build(d, lv)[2].tolist())      # noqa: E731
    assert sg.ROLE_ID["buttress"] in roles(0) and sg.ROLE_ID["buttress"] not in roles(1)
    t = sg.resolve({"style": "tower"})
    r0, r1 = sg.build(t, 0)[2], sg.build(t, 1)[2]
    assert (r1 == sg.ROLE_ID["parapet"]).sum() < (r0 == sg.ROLE_ID["parapet"]).sum()


# ------------------------------------------------------------------ damage
def test_damage_is_seeded_hits_upper_floors_and_is_consistent_across_lods():
    base = sg.resolve({"style": "gothic"})
    d = sg.resolve({"style": "gothic", "damage": 0.5})
    n0 = len(sg.build(base, 0)[1])
    a, b, c = (len(sg.build(d, 0, s)[1]) for s in (7, 7, 8))
    assert a == b and a < n0 and c != a
    v0, t0, r0, _ = sg.build(d, 0, 7)
    # ground-floor wall pieces are never damaged (trim, buttresses, roof and upper floors may be)
    def ground_walls(dd):
        v, t, r, _ = sg.build(dd, 0, 7)
        z = v[t][:, :, 2]
        low = (z.min(1) < base["floor_height"] - 1e-6) & (z.max(1) <= base["floor_height"] + 1e-6)   # not the bottom faces of floor 1
        return int(((r == sg.ROLE_ID["wall"]) & low).sum())
    assert ground_walls(d) == ground_walls(base)
    upper = lambda dd: int((sg.build(dd, 0, 7)[2] == sg.ROLE_ID["wall"]).sum()) - ground_walls(dd)   # noqa: E731
    assert upper(d) < upper(base)
    # the surviving gable segments are the same at every level of detail
    roofs = [int((sg.build(d, lv, 7)[2] == sg.ROLE_ID["roof"]).sum()) for lv in (0, 1, 2)]
    assert roofs[0] == roofs[1] == roofs[2] and 0 < roofs[0] < int((sg.build(base, 0)[2] == sg.ROLE_ID["roof"]).sum())


def test_full_damage_still_leaves_the_ground_floor_standing():
    d = sg.resolve({"style": "hall", "damage": 1.0})
    v, t, r, _ = sg.build(d, 0, 0)
    assert len(t) > 0 and not (r == sg.ROLE_ID["roof"]).any() and v[:, 2].min() == 0
    assert sg.build(d, 2, 0)[0][:, 2].min() == 0


# ------------------------------------------------------------------ collision
@pytest.mark.parametrize("style", sorted(sg.STYLES))
def test_collision_is_small_convex_and_covers_the_building(style):
    d = sg.resolve({"style": style})
    pieces = sg.collision(d)
    assert sum(len(t) for _, t in pieces) <= 64
    allv = np.concatenate([p[0] for p in pieces])
    v = sg.build(d, 0)[0]
    W, D = d["footprint"]
    assert allv.max(0)[2] == pytest.approx(v.max(0)[2], abs=1e-3)
    assert allv.max(0)[0] - allv.min(0)[0] >= W - 1e-4 and allv.max(0)[1] - allv.min(0)[1] >= D - 1e-4
    for pv, pt in pieces:
        assert volume(pv, pt) > 0


def test_collision_drops_the_roof_when_all_of_it_is_gone():
    d = sg.resolve({"style": "hall", "damage": 1.0})
    assert len(sg.collision(d, 0)) == 1


# ------------------------------------------------------------------ measurement helpers
def tri_uv(*pts):
    return np.array([pts], float)


def test_uv_overlap_detects_real_overlap_but_not_shared_edges():
    a = [(0.1, 0.1), (0.4, 0.1), (0.1, 0.4)]
    b = [(0.4, 0.1), (0.4, 0.4), (0.1, 0.4)]           # the other half of the same quad: shares the diagonal edge
    far = [(0.6, 0.6), (0.9, 0.6), (0.6, 0.9)]
    ok = sm.uv_overlap(np.array([a, b, far]))
    assert ok["overlap_fraction"] == 0 and ok["out_of_range"] == 0 and ok["covered"] > 0.1
    bad = sm.uv_overlap(np.array([a, a, far]))
    assert bad["overlap_fraction"] > 0.4
    partly = sm.uv_overlap(np.array([a, [(0.2, 0.1), (0.5, 0.1), (0.2, 0.4)]]))
    assert 0.05 < partly["overlap_fraction"] < 0.9
    assert sm.uv_overlap(np.array([[(0.9, 0.9), (1.2, 0.9), (0.9, 1.2)]]))["out_of_range"] == 1


def test_silhouette_fill_holes_ignores_windows_but_not_open_gaps():
    box = lambda x0, x1: [(x0, 0, 0), (x1, 0, 0), (x1, 0, 1), (x0, 0, 0), (x1, 0, 1), (x0, 0, 1)]   # noqa: E731
    # a frame: four strips around a hole, vs the same frame with a gap in the top strip (open to the outside)
    def strips(gap):
        v = [(0, 0, 0), (1, 0, 0), (1, 0, 0.2), (0, 0, 0.2),            # bottom
             (0, 0, 0.8), (1, 0, 0.8), (1, 0, 1), (0, 0, 1),            # top
             (0, 0, 0.2), (0.2, 0, 0.2), (0.2, 0, 0.8), (0, 0, 0.8),    # left
             (0.8, 0, 0.2), (1, 0, 0.2), (1, 0, 0.8), (0.8, 0, 0.8)]    # right
        t = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (8, 9, 10), (8, 10, 11), (12, 13, 14), (12, 14, 15)]
        if gap:
            t = [x for x in t if x[0] != 4]
        return np.array(v, np.float32), np.array(t, np.int32)
    frame = {"min": [0, 0, 0], "max": [1, 0.001, 1]}
    solid = np.zeros(1)
    closed = sm.silhouettes(*strips(False), frame)["front"]
    opened = sm.silhouettes(*strips(True), frame)["front"]
    raw = sm.silhouettes(*strips(False), frame, fill_holes=False)["front"]
    assert closed.sum() > raw.sum() * 1.3 and opened.sum() < closed.sum()
    assert sm.iou(closed, closed) == 1.0


def test_degenerate_fraction():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], np.float32)
    assert sm.degenerate_fraction(v, np.array([[0, 1, 2], [3, 4, 5]], np.int32)) == 0.5


# ------------------------------------------------------------------ spec
def sjob(**kw):
    s = JobSpec(name="b", kind="structure", structure={"style": "hall"}, **kw)
    s.validate()
    return s


@pytest.mark.parametrize("kw,msg", [
    ({"sketch": "x.png"}, "sketch is not supported"),
    ({"accessories": ["cape"]}, "character-only"),
    ({"params": {"head_scale": 1.0}}, "detail_level"),
    ({"params": {"detail_level": 5}}, "detail_level"),
    ({"lods": [1.0, 0.5, 0.25, 0.1]}, "1 to 3"),
    ({"qa": {"nonsense": 1}}, "unknown qa"),
    ({"texture": "replace"}, "needs a style_template"),
    ({"provider": "trellis2", "provider_args": {"command": "x"}}, "procedurally"),
])
def test_structure_spec_rules(kw, msg):
    with pytest.raises(ValueError, match=msg):
        sjob(**kw)


def test_structure_key_only_valid_for_structures():
    with pytest.raises(ValueError, match="only valid when kind is 'structure'"):
        JobSpec(name="x", structure={"style": "hall"}).validate()
    with pytest.raises(ValueError, match="unknown structure keys"):
        JobSpec(name="x", kind="structure", structure={"colour": "red"}).validate()
    assert sjob().resolved_budget() == 8000 and sjob(poly_budget=3000).resolved_budget() == 3000


# ------------------------------------------------------------------ QA rules (synthetic metrics)
def good_metrics(**kw):
    m = {"kind": "structure", "lod_tris": [1900, 1200, 60], "height_m": 9.5, "bbox": {"min": [-6.5, -4, 0], "max": [6.5, 4, 9.5], "size": [13, 8, 9.5]},
         "expected": {"height": 9.5, "footprint": [12, 7], "max_protrusion": 0.5, "envelope_area": 327.0},
         "degenerate_frac": 0.0, "has_uv": True, "has_material": True,
         "lightmap": {"present": True, "overlap_fraction": 0.0, "out_of_range": 0.0},
         "lods": [{"lod": 0, "tris": 1900, "size": [13, 8, 9.5], "has_uv": True, "has_material": True, "silhouette_front_vs_lod0": 1, "silhouette_side_vs_lod0": 1, "silhouette_top_vs_lod0": 1},
                  {"lod": 1, "tris": 1200, "size": [13, 8, 9.5], "has_uv": True, "has_material": True, "silhouette_front_vs_lod0": 0.99, "silhouette_side_vs_lod0": 0.99, "silhouette_top_vs_lod0": 1},
                  {"lod": 2, "tris": 60, "size": [13, 8, 9.5], "has_uv": True, "has_material": True, "silhouette_front_vs_lod0": 0.95, "silhouette_side_vs_lod0": 0.9, "silhouette_top_vs_lod0": 1}],
         "collision": {"pieces": 2, "tris": 20, "bbox": {"size": [13, 8, 9.5]}}, "roundtrip": {"ok": True}, "embedded": [{"a": 1}, {"a": 1}]}
    m.update(kw)
    return m


def codes(m, **spec):
    return [f["code"] for f in qa.evaluate(m, JobSpec(name="b", kind="structure", structure={"style": "hall"}, **spec))]


def test_good_structure_metrics_pass():
    assert codes(good_metrics()) == []


@pytest.mark.parametrize("patch,code", [
    ({"lod_tris": [9000, 1200, 60]}, "over_budget"),
    ({"height_m": 8.0}, "height_off"),
    ({"bbox": {"min": [-5, -4, 0], "max": [5, 4, 9.5], "size": [10, 8, 9.5]}}, "footprint_off"),
    ({"bbox": {"min": [-6.5, -4, 0.3], "max": [6.5, 4, 9.5], "size": [13, 8, 9.5]}}, "off_ground"),
    ({"bbox": {"min": [-3, -4, 0], "max": [10, 4, 9.5], "size": [13, 8, 9.5]}}, "off_center"),
    ({"expected": {"height": 9.5, "footprint": [12, 7], "max_protrusion": 0.5, "envelope_area": 20.0}}, "tri_density_high"),
    ({"degenerate_frac": 0.01}, "degenerate_faces"),
    ({"has_uv": False}, "no_uv"),
    ({"has_material": False}, "no_material"),
    ({"lightmap": {"present": False}}, "lightmap_missing"),
    ({"lightmap": {"present": True, "overlap_fraction": 0.05, "out_of_range": 0.0}}, "lightmap_overlap"),
    ({"lightmap": {"present": True, "overlap_fraction": 0.0, "out_of_range": 0.2}}, "lightmap_out_of_range"),
    ({"collision": {"pieces": 0, "tris": 0}}, "collision_missing"),
    ({"collision": {"pieces": 3, "tris": 500, "bbox": {"size": [13, 8, 9.5]}}}, "collision_heavy"),
    ({"collision": {"pieces": 1, "tris": 12, "bbox": {"size": [10, 8, 9.5]}}}, "collision_bbox_off"),
    ({"roundtrip": {"ok": False}}, "roundtrip_failed"),
    ({"embedded": [None]}, "provenance_missing"),
])
def test_structure_qa_codes(patch, code):
    assert code in codes(good_metrics(**patch))


def test_structure_lod_rules():
    m = good_metrics()
    m["lods"][1]["tris"] = 1900
    assert "lod_not_decreasing" in codes(m)
    m = good_metrics(); m["lods"][2]["silhouette_top_vs_lod0"] = 0.5
    assert "lod_silhouette_drift" in codes(m)
    m = good_metrics(); m["lods"][2]["size"] = [9, 8, 9.5]
    assert "lod_footprint_drift" in codes(m)
    m = good_metrics(); m["lods"][1]["size"] = [13, 8, 8.0]
    assert "lod_height_drift" in codes(m)
    m = good_metrics(); m["lods"][1]["has_uv"] = False
    assert "lod_inconsistent" in codes(m)
    assert "lod_not_decreasing" not in codes(good_metrics(), lods=[1.0])      # a single LOD has nothing to compare


def test_over_budget_is_fixable_by_dropping_detail_not_by_segments():
    from charforge import fixer
    spec = sjob(poly_budget=1500)
    fails = qa.evaluate(good_metrics(lod_tris=[1900, 1200, 60]), spec)
    assert [f["code"] for f in fails] == ["over_budget"] and fails[0]["fixable"]
    p, ch = fixer.next_params(spec, {"detail_level": 0.0}, fails)
    assert p["detail_level"] == 1.0 and ch == ["detail_level 0 -> 1"]
    p, ch = fixer.next_params(spec, {"detail_level": 2.0}, fails)
    assert ch == []                                         # nothing left to drop -> needs_review
    pinned = sjob(poly_budget=1500, params={"detail_level": 0})
    assert fixer.next_params(pinned, {"detail_level": 0.0}, fails)[1] == []


# ------------------------------------------------------------------ farm
def test_farm_varies_structure_numbers(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"name": "b", "kind": "structure", "structure": {"style": "hall"}}))
    (tmp_path / "farm.yaml").write_text(yaml.safe_dump({"base": "base.yaml", "count": 6, "seed": 2, "vary": {
        "structure": {"footprint": [[9, 16], [6, 9]], "floors": [1, 4], "damage": [0, 0.5]}, "seed": True}}))
    specs = [load_job(p) for p in make_variants(tmp_path / "farm.yaml", tmp_path / "o")]
    assert len({tuple(s.structure["footprint"]) for s in specs}) > 3 and {s.structure["floors"] for s in specs} <= {1, 2, 3, 4}
    assert all(isinstance(s.structure["floors"], int) and 9 <= s.structure["footprint"][0] <= 16 for s in specs)


# ------------------------------------------------------------------ through Blender
@pytest.fixture(scope="module")
def hall(tmp_path_factory):
    out = tmp_path_factory.mktemp("st")
    spec = load_job(ROOT / "jobs" / "struct_hall.yaml")
    m = build_one(spec, out, Backend(kind="bpy-module"))
    return spec, m, out


def test_hall_builds_passes_and_matches_the_numbers(hall):
    spec, m, out = hall
    assert m["status"] == "passed", m.get("final_failures")
    it = m["iterations"][0]["metrics"]
    d = sg.resolve(spec.structure)
    assert it["height_m"] == pytest.approx(sg.expected_height(d), abs=0.01)
    assert it["lod_tris"] == [1908, 1272, 60]
    assert it["lightmap"]["overlap_fraction"] == 0 and it["roundtrip"]["ok"]
    assert it["collision"]["tris"] <= 64 and it["collision"]["names"][0].startswith("UCX_")
    assert it["structure"]["windows"] == 23 and it["structure"]["doors"] == 1


def test_hall_files_have_two_uv_sets_provenance_and_collision(hall):
    spec, m, out = hall
    final = out / "struct_hall" / "final"
    names = {p.name for p in final.iterdir()}
    assert {"struct_hall_LOD0.glb", "struct_hall_LOD1.glb", "struct_hall_LOD2.glb", "struct_hall_collision.glb",
            "struct_hall_LOD0_basecolor.png", "struct_hall_contact_sheet.png", "struct_hall.blend"} <= names
    it = m["iterations"][0]["metrics"]
    attrs = it["glb_content"][0]["mesh_attributes"]
    assert {"TEXCOORD_0", "TEXCOORD_1"} <= set(next(iter(attrs.values())))
    emb = read_embedded(final / "struct_hall_LOD0.glb")
    assert emb["name"] == "struct_hall" and emb["fingerprint"] == spec.fingerprint() and emb["lod"] == 0 and emb["level"] == 0
    assert read_embedded(final / "struct_hall_LOD2.glb")["level"] == 2
    assert read_embedded(final / "struct_hall_collision.glb")["collision"] is True


def test_hall_rebuild_is_byte_identical(hall, tmp_path):
    spec, m, out = hall
    build_one(spec, tmp_path, Backend(kind="bpy-module"))
    for f in ("struct_hall_LOD0.glb", "struct_hall_LOD2.glb", "struct_hall_collision.glb"):
        assert (out / "struct_hall" / "final" / f).read_bytes() == (tmp_path / "struct_hall" / "final" / f).read_bytes()


def test_tight_budget_drops_detail_and_records_it(tmp_path):
    spec = load_job(ROOT / "jobs" / "struct_hall.yaml")
    spec.poly_budget = 1500
    m = build_one(spec, tmp_path, Backend(kind="bpy-module"))
    its = m["iterations"]
    assert [i["params"]["detail_level"] for i in its][:2] == [0.0, 1.0] and its[0]["next_changes"] == ["detail_level 0 -> 1"]
    assert m["status"] == "passed" and its[-1]["metrics"]["lod_tris"][0] <= 1500 * 1.05
    assert len(its[-1]["metrics"]["lod_tris"]) == 2        # LOD1 would repeat level 2 -> not built, and said so
    assert its[-1]["metrics"]["lods_requested"] == 3


def test_ruin_and_tower_pass_and_go_into_the_library(tmp_path):
    lib = Library(tmp_path / "lib.db")
    for j in ("struct_ruin", "struct_tower"):
        assert "queued" in lib.enqueue(ROOT / "jobs" / f"{j}.yaml")
    work(lib, tmp_path / "out", Backend(kind="bpy-module"), once=True, workers=2, log=lambda s: None)
    for j in ("struct_ruin", "struct_tower"):
        a = lib.get(j)
        assert a["kind"] == "structure" and a["status"] == "passed", a["failures"]
        assert a["thumb"] == f"{j}_contact_sheet.png" and (lib.resolve(a["final_dir"]) / a["thumb"]).exists()
        assert a["summary"]["collision_tris"] <= 64 and a["summary"]["windows"] > 0 and len(a["summary"]["footprint_m"]) == 2
    assert lib.get("struct_tower")["summary"]["style"] == "tower"
