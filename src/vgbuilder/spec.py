"""Job specification: the declarative input for one character.

A job is a YAML file. The script/params are the source of truth; the .blend that
comes out is a build artifact and is rebuilt from an empty scene every time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

STYLES = {"lowpoly", "stylized", "realistic"}
PROVIDERS = {"procedural", "import_glb", "command", "trellis2", "hunyuan", "http"}
ENGINES = {"unity", "unreal", "godot", "generic"}
NAMINGS = {"auto", "ue", "simple", "humanoid"}
FORMATS = {"glb", "fbx"}
KINDS = {"character", "effect", "structure"}
ACCESSORIES = {"cape", "pauldrons", "horns"}
ANIMATIONS = {"idle", "walk"}
TEXTURE_MODES = {"auto", "replace", "keep", "none"}
SIDE_FACINGS = {"left", "right"}
EFFECT_TEMPLATES = {"fireball", "explosion", "shockwave", "smoke"}

# Sensible defaults per style. Users override in the job file.
STYLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "lowpoly": {"poly_budget": 1500, "segments": 6},
    "stylized": {"poly_budget": 6000, "segments": 10},
    "realistic": {"poly_budget": 30000, "segments": 20},
}

# Parameters of the procedural humanoid (all lengths are fractions of height).
DEFAULT_BODY_PARAMS: dict[str, float] = {
    "head_scale": 1.0,       # 1.0 -> head radius 0.065 * height
    "shoulder_half": 0.115,  # half shoulder width at shoulder joint
    "waist_half": 0.085,
    "hip_half": 0.10,
    "leg_frac": 0.47,        # crotch height / height
    "arm_frac": 0.40,        # shoulder->wrist length / height
    "arm_thickness": 1.0,    # multiplier on arm radii
    "leg_thickness": 1.0,    # multiplier on leg radii
    "torso_depth": 1.0,      # multiplier on front-back thickness
    "arm_angle_deg": 15.0,   # A-pose angle from vertical
}

DEFAULT_QA: dict[str, float] = {
    "silhouette_iou_min": 0.78,   # only enforced when a sketch is provided
    "symmetry_max": 0.01,         # mean mirrored-vertex distance / height
    "height_tolerance": 0.02,     # fraction of requested height
    "max_zero_weight_frac": 0.001,
    "max_loose_parts": 40,        # procedural body is ~20 separate shells by design (tubes + joint caps)
    "max_edge_stretch": 2.0,      # worst edge stretch when posed vs rest
    "lod_ratio_tolerance": 0.35,      # |actual - target| / target for each LOD's triangle ratio
    "lod_silhouette_min": 0.85,       # front and side silhouette IoU of every LOD against LOD0
    "lod_height_tolerance": 0.01,     # LOD height vs LOD0 height, fraction
    "anim_floor_penetration": 0.02,   # lowest vertex during an animation may dip this fraction of height below z=0
    "anim_max_rotation": 1.5,         # radians, any bone
    "silhouette_iou_min_generated": 0.78,  # front IoU gate for generated / imported meshes; same as procedural until
                                           # real generator output exists to calibrate a looser one (override per job)
    "silhouette_collapse_iou": 0.30,       # below this a generated mesh is reported as unrelated to the sketch
    "tri_density_max": 40.0,          # structures: triangles per m2 of envelope (walls + roof footprint)
    "lightmap_overlap_max": 0.001,    # structures: share of lit lightmap texels covered by more than one triangle
    "collision_max_tris": 64,         # structures: triangles in the collision proxy
    "collision_bbox_tolerance": 0.03, # structures: collision box vs wall footprint / height, fraction
    "lod_footprint_tolerance": 0.15,  # structures: LOD bbox x/y vs LOD0 (trim and buttresses are dropped in far LODs)
    "silhouette_side_iou_min": 0.70,  # only enforced when a side sketch is provided (hand-drawn side views are looser)
}


@dataclass
class JobSpec:
    name: str
    kind: str = "character"                 # character | effect
    sketch: str | None = None               # front view
    sketch_side: str | None = None          # optional side view: separates thickness from length
    side_facing: str = "left"               # which way the character faces in the side sketch: left | right
    style: str = "stylized"
    provider: str = "procedural"
    provider_args: dict[str, Any] = field(default_factory=dict)
    target_engine: str = "generic"
    height_m: float = 1.8
    poly_budget: int | None = None          # triangles, LOD0
    lods: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    naming: str = "auto"                    # auto = the target_engine preset's naming (see engines.py)
    formats: list[str] = field(default_factory=lambda: ["glb"])
    params: dict[str, float] = field(default_factory=dict)   # procedural body overrides
    qa: dict[str, float] = field(default_factory=dict)
    seed: int = 0
    fit_prior: float = 0.0    # >0 pulls the sketch fit toward human proportions, trading silhouette match for realism
    style_template: str | None = None       # look: palette, wear, silhouette rules (see styles/)
    style_overrides: dict[str, Any] = field(default_factory=dict)  # e.g. {"hue_shift_deg": 20, "wear": {...}}
    wear: dict[str, float] = field(default_factory=dict)           # per-job wear overrides
    texture: str = "auto"                   # auto | replace | keep | none   (needs style_template for anything but none)
    texture_size: int = 512                 # baked texture edge in pixels
    accessories: list[str] = field(default_factory=list)           # cape | pauldrons | horns
    accessory_shape: dict[str, float] = field(default_factory=dict)  # pauldron_scale, horn_len, horn_spread, cape_len, cape_flare
    animations: list[str] = field(default_factory=list)            # idle | walk
    effect: dict[str, Any] = field(default_factory=dict)           # for kind: effect
    structure: dict[str, Any] = field(default_factory=dict)        # for kind: structure (see structuregen.resolve)
    priority: int = 0                       # queue order, higher first
    tags: list[str] = field(default_factory=list)
    critic: dict[str, Any] = field(default_factory=dict)           # {"command": "...", "min_score": 0.6}
    max_iterations: int = 4
    notes: str = ""

    # filled by load(): where relative paths resolve from
    base_dir: str = "."

    def validate(self) -> None:
        if not self.name or not all(c.isalnum() or c in "-_" for c in self.name):
            raise ValueError(f"name must be [A-Za-z0-9_-]+, got {self.name!r}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        from .styles import WEAR_KEYS
        if set(self.wear) - WEAR_KEYS:
            raise ValueError(f"unknown wear keys: {sorted(set(self.wear) - WEAR_KEYS)}")
        if self.kind == "effect":
            tpl = self.effect.get("template")
            if tpl not in EFFECT_TEMPLATES:
                raise ValueError(f"effect.template must be one of {sorted(EFFECT_TEMPLATES)}")
            from .effects import validate_effect
            validate_effect(self.effect)
            if self.style_template:
                self.load_style()
            return  # character-only checks below do not apply to effects
        if self.effect:
            raise ValueError("effect: is only valid when kind is 'effect'")
        if self.kind == "structure":
            self._validate_structure()
            return
        if self.structure:
            raise ValueError("structure: is only valid when kind is 'structure'")
        if self.side_facing not in SIDE_FACINGS:
            raise ValueError(f"side_facing must be one of {sorted(SIDE_FACINGS)}")
        if self.sketch_side and not self.sketch:
            raise ValueError("sketch_side needs a front sketch as well")
        if not set(self.accessories) <= ACCESSORIES:
            raise ValueError(f"accessories must be a subset of {sorted(ACCESSORIES)}")
        if self.accessories and self.provider != "procedural":
            raise ValueError("accessories are built for the procedural body; imported meshes cannot take them yet")
        from .accessories import validate_shape
        validate_shape(self.accessory_shape)
        if not set(self.animations) <= ANIMATIONS:
            raise ValueError(f"animations must be a subset of {sorted(ANIMATIONS)}")
        if self.texture not in TEXTURE_MODES:
            raise ValueError(f"texture must be one of {sorted(TEXTURE_MODES)}")
        if not (64 <= self.texture_size <= 4096):
            raise ValueError("texture_size must be in [64, 4096]")
        if self.style_template:
            self.load_style()                # unknown name / bad wear keys raise here, at validate time
        elif self.texture in ("replace", "keep"):
            raise ValueError(f"texture: {self.texture} needs a style_template")
        if self.critic and "command" not in self.critic:
            raise ValueError("critic needs a command")
        if self.style not in STYLES:
            raise ValueError(f"style must be one of {sorted(STYLES)}")
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
        if self.target_engine not in ENGINES:
            raise ValueError(f"target_engine must be one of {sorted(ENGINES)}")
        if self.naming not in NAMINGS:
            raise ValueError(f"naming must be one of {sorted(NAMINGS)}")
        if not set(self.formats) <= FORMATS:
            raise ValueError(f"formats must be a subset of {sorted(FORMATS)}")
        if not (0.2 <= self.height_m <= 30):
            raise ValueError("height_m out of range")
        if not self.lods or self.lods[0] != 1.0 or any(not (0.02 <= x <= 1.0) for x in self.lods):
            raise ValueError("lods must start at 1.0 and each be in [0.02, 1.0]")
        if not (0.0 <= self.fit_prior <= 5.0):
            raise ValueError("fit_prior must be in [0, 5]")
        if self.lods != sorted(self.lods, reverse=True):
            raise ValueError("lods must be descending")
        unknown = set(self.params) - set(DEFAULT_BODY_PARAMS) - {"segments"}
        if unknown:
            raise ValueError(f"unknown params: {sorted(unknown)}")
        unknown_qa = set(self.qa) - set(DEFAULT_QA)
        if unknown_qa:
            raise ValueError(f"unknown qa keys: {sorted(unknown_qa)}")
        from .providers import validate_args
        validate_args(self.provider, self.provider_args, bool(self.sketch))

    def _validate_structure(self) -> None:
        from . import structuregen
        structuregen.resolve(self.structure)
        if self.provider != "procedural":
            raise ValueError("structures are built procedurally; image-to-3D providers only feed the character path")
        for k in ("sketch", "sketch_side"):
            if getattr(self, k):
                raise ValueError(f"{k} is not supported for structures yet (the footprint comes from structure.footprint)")
        if self.accessories or self.animations:
            raise ValueError("accessories and animations are character-only")
        bad = set(self.params) - {"detail_level"}
        if bad:
            raise ValueError(f"structure params may only contain detail_level, got {sorted(bad)}")
        if "detail_level" in self.params and self.params["detail_level"] not in (0, 1, 2):
            raise ValueError("params.detail_level must be 0, 1 or 2")
        if not (1 <= len(self.lods) <= 3) or self.lods[0] != 1.0:
            raise ValueError("structure lods: 1 to 3 entries, first 1.0 (LOD k is detail level k, not a decimation ratio)")
        if self.target_engine not in ENGINES or not set(self.formats) <= FORMATS:
            raise ValueError("bad target_engine or formats")
        if self.texture not in TEXTURE_MODES or not (64 <= self.texture_size <= 4096):
            raise ValueError("bad texture / texture_size")
        if self.style_template:
            self.load_style()
        elif self.texture in ("replace", "keep"):
            raise ValueError(f"texture: {self.texture} needs a style_template")
        unknown_qa = set(self.qa) - set(DEFAULT_QA)
        if unknown_qa:
            raise ValueError(f"unknown qa keys: {sorted(unknown_qa)}")

    # ---- derived values -------------------------------------------------
    def resource(self) -> str | None:
        """Queue resource tag (e.g. 'gpu'): the worker limits how many jobs with the same tag run at once."""
        from .providers import resource_of
        return resource_of(self.provider, self.provider_args)

    def resolved_budget(self) -> int:
        if self.kind == "structure":
            return int(self.poly_budget or 8000)
        return int(self.poly_budget or STYLE_DEFAULTS[self.style]["poly_budget"])

    def resolved_segments(self) -> int:
        return int(self.params.get("segments", STYLE_DEFAULTS[self.style]["segments"]))

    def style_dirs(self) -> list[Path]:
        return [Path(self.base_dir) / "styles"]          # project styles override the built-in ones

    def load_style(self):
        """The resolved look (template + style_overrides + wear overrides), or None."""
        if not self.style_template:
            return None
        from .styles import load_style
        return load_style(self.style_template, self.style_dirs()).with_overrides(self.wear, self.style_overrides)

    def resolved_naming(self) -> str:
        from .engines import resolve_naming
        return resolve_naming(self.naming, self.target_engine)

    def resolved_qa(self) -> dict[str, float]:
        return {**DEFAULT_QA, **self.qa}

    def resolve_path(self, p: str | None) -> Path | None:
        if p is None:
            return None
        path = Path(p)
        return path if path.is_absolute() else (Path(self.base_dir) / path).resolve()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Hash of everything that should change the output (spec + sketch bytes)."""
        h = hashlib.sha256()
        d = self.to_dict()
        d.pop("base_dir", None)
        h.update(json.dumps(d, sort_keys=True, default=str).encode())
        for rel in (self.sketch, self.sketch_side):
            sk = self.resolve_path(rel)
            if sk and sk.exists():
                h.update(sk.read_bytes())
        return h.hexdigest()[:16]


def load_job(path: str | Path, overrides: dict[str, Any] | None = None) -> JobSpec:
    """Read and validate a job file. `overrides` ({"structure.floors": 3, ...}) are laid over the file's contents
    before validation (the live-edit link uses this; the file on disk is not touched). They count in the fingerprint."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping")
    if overrides:
        from .live import apply_overrides
        raw = apply_overrides(raw, overrides)
    allowed = set(JobSpec.__dataclass_fields__) - {"base_dir"}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"{path}: unknown keys {sorted(extra)}")
    spec = JobSpec(**raw, base_dir=str(path.parent.resolve()))
    spec.validate()
    return spec
