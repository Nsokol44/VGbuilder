"""Variant farm: one base job -> N validated variant jobs (colour, wear, proportions, accessory shape, VFX knobs).

Deterministic: the same farm file always yields the same variants. Ranges are [lo, hi] and are sampled
uniformly; every emitted job is loaded through the normal validator, so a bad range fails here, not
halfway through a batch. Variants are tagged `variant_of:<base name>` so the library can group them.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .spec import load_job

FARM_KEYS = {"base", "count", "seed", "name", "vary", "priority"}
VARY_KEYS = {"seed", "hue_shift_deg", "wear", "params", "accessory_shape", "effect", "max_saturation", "structure"}
PATH_KEYS = ("sketch", "sketch_side")


def _rng_range(rng: np.random.Generator, r: Any, what: str) -> float:
    if not (isinstance(r, (list, tuple)) and len(r) == 2 and r[0] <= r[1]):
        raise ValueError(f"vary.{what}: expected [lo, hi], got {r!r}")
    return round(float(rng.uniform(r[0], r[1])), 4)


def load_farm(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or "base" not in cfg:
        raise ValueError(f"{path}: farm file needs a base job")
    extra = set(cfg) - FARM_KEYS
    if extra:
        raise ValueError(f"{path}: unknown keys {sorted(extra)}")
    vary = cfg.setdefault("vary", {})
    bad = set(vary) - VARY_KEYS
    if bad:
        raise ValueError(f"{path}: unknown vary keys {sorted(bad)}")
    cfg.setdefault("count", 8)
    cfg.setdefault("seed", 0)
    if not (1 <= int(cfg["count"]) <= 500):
        raise ValueError("count must be in [1, 500]")
    return cfg


def make_variants(path: Path, out_dir: Path) -> list[Path]:
    path = Path(path)
    cfg = load_farm(path)
    base_path = (path.parent / cfg["base"]).resolve()
    base_spec = load_job(base_path)                  # the base itself must be valid
    raw = yaml.safe_load(base_path.read_text())
    prefix = cfg.get("name") or base_spec.name
    vary = cfg["vary"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = base_path.parent / "styles"
    if styles.is_dir():                              # project styles resolve relative to the job file
        shutil.copytree(styles, out_dir / "styles", dirs_exist_ok=True)

    rng = np.random.default_rng(int(cfg["seed"]))
    paths = []
    for i in range(int(cfg["count"])):
        job = yaml.safe_load(yaml.safe_dump(raw))    # deep copy
        for k in PATH_KEYS:
            if job.get(k):
                job[k] = str((base_path.parent / job[k]).resolve())
        if job.get("provider_args", {}).get("glb"):
            job["provider_args"]["glb"] = str((base_path.parent / job["provider_args"]["glb"]).resolve())
        job["name"] = f"{prefix}_v{i + 1:02d}"
        job["tags"] = [*raw.get("tags", []), f"variant_of:{base_spec.name}", "variant"]
        if "priority" in cfg:
            job["priority"] = int(cfg["priority"])
        if vary.get("seed"):
            job["seed"] = int(rng.integers(0, 2**31 - 1))
        so = dict(job.get("style_overrides", {}))
        if "hue_shift_deg" in vary:
            so["hue_shift_deg"] = _rng_range(rng, vary["hue_shift_deg"], "hue_shift_deg")
        if "max_saturation" in vary:
            so["max_saturation"] = _rng_range(rng, vary["max_saturation"], "max_saturation")
        if so:
            job["style_overrides"] = so
        if "structure" in vary:
            cur = dict(job.get("structure", {}))
            for k, r in vary["structure"].items():
                if k == "footprint":                       # [[w_lo, w_hi], [d_lo, d_hi]]
                    if not (isinstance(r, (list, tuple)) and len(r) == 2):
                        raise ValueError("vary.structure.footprint: expected [[w_lo, w_hi], [d_lo, d_hi]]")
                    cur[k] = [_rng_range(rng, r[0], "structure.footprint.width"), _rng_range(rng, r[1], "structure.footprint.depth")]
                elif k == "floors":
                    cur[k] = int(round(_rng_range(rng, r, "structure.floors")))
                else:
                    cur[k] = _rng_range(rng, r, f"structure.{k}")
            job["structure"] = cur
        for field, key in (("wear", "wear"), ("params", "params"), ("accessory_shape", "accessory_shape"), ("effect", "effect")):
            if field in vary:
                cur = dict(job.get(key, {}))
                for k, r in vary[field].items():
                    cur[k] = _rng_range(rng, r, f"{field}.{k}")
                job[key] = cur
        note = f"variant {i + 1} of {base_spec.name}"
        job["notes"] = (raw.get("notes", "") + " " + note).strip()
        jp = out_dir / f"{job['name']}.yaml"
        jp.write_text(yaml.safe_dump(job, sort_keys=False))
        load_job(jp)                                 # fail now, not mid-batch
        paths.append(jp)
    return paths
