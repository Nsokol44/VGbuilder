"""Concept stage: scoring, diverse selection, emitted jobs (no Blender needed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from charforge import concepts, sketch as sk, styles
from charforge.spec import load_job

ROOT = Path(__file__).resolve().parents[1]


def rect(h=100, w=40, y0=0, y1=None, x0=0, x1=None, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[y0:y1 or h, x0:x1 or w] = True
    return m


# ---------------------------------------------------------------- scorers
def test_solidity_of_simple_shapes():
    assert concepts.solidity(rect(60, 30, shape=(100, 100))) > 0.97
    plus = np.zeros((100, 100), bool)
    plus[40:60, :] = True
    plus[:, 40:60] = True
    assert 0.4 < concepts.solidity(plus) < 0.6
    spiky = np.zeros((100, 100), bool)
    spiky[20:80, 45:55] = True
    for y in range(20, 80, 10):
        spiky[y:y + 2, 10:90] = True
    assert concepts.solidity(spiky) < concepts.solidity(plus) or concepts.solidity(spiky) < 0.6


def test_components_and_lobes():
    base = np.zeros((100, 100), bool); base[20:90, 40:60] = True
    full = base.copy()
    full[25:40, 20:40] = True       # left pauldron-like lump touching the body
    full[25:40, 60:80] = True
    full[2:14, 45:55] = True        # a horn-like piece above
    assert concepts.lobes(full, base) == 3
    assert concepts.lobes(base, base) == 0
    tiny = base.copy(); tiny[0:1, 0:2] = True
    assert concepts.lobes(tiny, base) == 0           # specks are not lobes


def test_limb_separation_sees_arm_gaps():
    sep = np.zeros((100, 60), bool)
    sep[:, 25:35] = True; sep[:, 5:12] = True; sep[:, 48:55] = True      # torso + two clear arms
    assert concepts.limb_separation(sep) == 1.0
    merged = np.zeros((100, 60), bool); merged[:, 5:55] = True
    assert concepts.limb_separation(merged) == 0.0


def test_identical_silhouettes_have_zero_uniqueness():
    c = concepts.sample_variant(np.random.default_rng(1), "c1", ["horns", "cape", "pauldrons"])
    concepts.render(c, 1.8)
    st = styles.load_style("souls_like")
    concepts.score_variant(c, st, [c.front])
    assert c.metrics["uniqueness"] == pytest.approx(0.0, abs=1e-9) and any("uniqueness" in f for f in c.failures)


def test_rendered_variant_frames_agree():
    c = concepts.sample_variant(np.random.default_rng(2), "c1", ["horns", "cape", "pauldrons"])
    concepts.render(c, 1.8)
    assert c.front.shape == c.base_front.shape and (c.base_front & ~c.front).sum() == 0     # base is inside the full silhouette
    assert c.side.any()


# ------------------------------------------------------------------- run
def concept_file(tmp_path, **kw):
    cfg = {"name": "kn", "style_template": "souls_like", "count": 30, "select": 3, "seed": 1,
           "job": {"style": "stylized", "poly_budget": 3000, "animations": ["idle"]}, **kw}
    p = tmp_path / "kn.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_run_emits_valid_diverse_jobs(tmp_path):
    m = concepts.run_concepts(concept_file(tmp_path), tmp_path / "out", log=lambda s: None)
    out = tmp_path / "out" / "concepts" / "kn"
    assert len(m["selected"]) == 3 and (out / "sheet.png").exists()
    st = styles.load_style("souls_like")
    chosen = [c for c in m["candidates"] if c["selected"]]
    for c in chosen:                                         # winners satisfy every rule
        assert not [f for f in c["failures"] if not f.startswith("uniqueness")], c
        assert c["metrics"]["uniqueness"] >= st.silhouette["min_uniqueness"]
    for jp in m["jobs"]:
        spec = load_job(jp)
        assert spec.provider == "procedural" and spec.sketch is None and spec.style_template == "souls_like"
        assert set(spec.params) == set(concepts.DEFAULT_BODY_PARAMS) and "concept" in spec.tags
    assert (out / "silhouettes" / f"kn_{m['selected'][0]}_front.png").exists()
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["fingerprint"] == m["fingerprint"] and on_disk["rejected_by_rule"] > 0


def test_same_seed_same_winners_other_seed_other_winners(tmp_path):
    a = concepts.run_concepts(concept_file(tmp_path), tmp_path / "a", log=lambda s: None)
    b = concepts.run_concepts(concept_file(tmp_path), tmp_path / "b", log=lambda s: None)
    c = concepts.run_concepts(concept_file(tmp_path, seed=2), tmp_path / "c", log=lambda s: None)
    assert a["selected"] == b["selected"] and a["fingerprint"] == b["fingerprint"]
    assert a["fingerprint"] != c["fingerprint"]
    assert [x["params"] for x in a["candidates"]] != [x["params"] for x in c["candidates"]]


def test_emitted_job_reproduces_its_silhouette(tmp_path):
    m = concepts.run_concepts(concept_file(tmp_path), tmp_path / "out", log=lambda s: None)
    jp = Path(m["jobs"][0])
    spec = load_job(jp)
    v = concepts.Variant("x", dict(spec.params), list(spec.accessories), dict(spec.accessory_shape))
    concepts.render(v, spec.height_m, segments=spec.resolved_segments())
    saved = sk.load_mask(tmp_path / "out" / "concepts" / "kn" / "silhouettes" / f"{spec.name}_front.png")
    assert sk.silhouette_iou(v.front, saved) > 0.9


def test_library_silhouettes_are_avoided(tmp_path):
    first = concepts.run_concepts(concept_file(tmp_path), tmp_path / "o1", log=lambda s: None)
    lib = tmp_path / "lib"
    lib.mkdir()
    sil = tmp_path / "o1" / "concepts" / "kn" / "silhouettes"
    for w in first["selected"]:
        (lib / f"{w}.png").write_bytes((sil / f"kn_{w}_front.png").read_bytes())
    second = concepts.run_concepts(concept_file(tmp_path, exclude_dir="lib"), tmp_path / "o2", log=lambda s: None)
    assert second["library_silhouettes"] == 3
    assert not set(second["selected"]) & set(first["selected"])
    for c in second["candidates"]:
        if c["selected"]:
            assert c["metrics"]["uniqueness"] >= 0.10


def test_pool_limits_accessories(tmp_path):
    m = concepts.run_concepts(concept_file(tmp_path, accessories_pool=["cape"], select=2), tmp_path / "o", log=lambda s: None)
    assert all(set(c["accessories"]) <= {"cape"} for c in m["candidates"])


def test_bad_concept_files_are_rejected(tmp_path):
    for kw in ({"bogus": 1}, {"select": 99, "count": 5}, {"accessories_pool": ["sword"]}):
        with pytest.raises(ValueError):
            concepts.load_concept(concept_file(tmp_path, **kw))
    (tmp_path / "x.yaml").write_text("name: a\n")
    with pytest.raises(ValueError, match="style_template"):
        concepts.load_concept(tmp_path / "x.yaml")


def test_style_rejects_unknown_silhouette_rules(tmp_path):
    d = tmp_path / "styles"; d.mkdir()
    (d / "bad.yaml").write_text("name: bad\npalette: {base: ['#808080']}\nsilhouette: {shininess: 3}\n")
    with pytest.raises(ValueError, match="silhouette rules"):
        styles.load_style("bad", [d])
