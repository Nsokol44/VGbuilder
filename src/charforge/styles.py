"""Style templates: a look (palette, wear, rot, silhouette rules) as data, with measurable checks.

A style is NOT a prompt. Every field is either an input to the look-dev stage (palette, wear
amounts) or a threshold that QA measures on the baked textures / silhouettes (saturation,
coverage, solidity, lobes, uniqueness). `checklist` is the human-readable list handed to a critic.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

BUILTIN_DIR = Path(__file__).parent / "styles_builtin"

WEAR_KEYS = {"intensity", "edge_wear", "lower_dirt", "rust", "rot", "blood", "noise_scale"}
SILHOUETTE_KEYS = {"solidity", "min_lobes", "min_uniqueness", "min_limb_separation"}
PALETTE_KEYS = {"base", "accent", "rot", "grime", "rust", "blood"}
ALLOWED_TOP = {
    "name", "description", "palette", "max_saturation", "value_range", "wear", "roughness",
    "coverage", "rot_fraction", "min_clean_fraction", "silhouette", "checklist",
}

DEFAULT_WEAR: dict[str, float] = {
    "intensity": 0.5, "edge_wear": 0.5, "lower_dirt": 0.5, "rust": 0.2, "rot": 0.0, "blood": 0.0,
    "noise_scale": 6.0,
}


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"bad colour {h!r}")
    return tuple(int(h[i: i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(c: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(round(max(0, min(1, v)) * 255)):02x}" for v in c)


def hue_shift(rgb: tuple[float, float, float], deg: float) -> tuple[float, float, float]:
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return colorsys.hsv_to_rgb((h + deg / 360.0) % 1.0, s, v)


@dataclass(frozen=True)
class Style:
    name: str
    description: str = ""
    palette: dict[str, list[str]] = field(default_factory=dict)   # role -> list of hex colours
    max_saturation: float = 0.40       # mean HSV saturation of baked base colour
    value_range: tuple[float, float] = (0.05, 0.75)
    wear: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEAR))
    roughness: tuple[float, float] = (0.5, 0.95)
    coverage: dict[str, tuple[float, float]] = field(default_factory=dict)  # "grime": allowed share of grime+rust+blood
    rot_fraction: tuple[float, float] = (0.0, 1.0)
    min_clean_fraction: float = 0.0
    silhouette: dict[str, Any] = field(default_factory=dict)
    checklist: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def colors(self, role: str) -> list[tuple[float, float, float]]:
        return [hex_to_rgb(h) for h in self.palette.get(role, [])]

    def with_overrides(self, wear: dict[str, float] | None = None, overrides: dict[str, Any] | None = None) -> "Style":
        """Apply job-level wear overrides and style_overrides ({hue_shift_deg, wear, max_saturation})."""
        overrides = dict(overrides or {})
        st = self
        w = dict(st.wear)
        w.update(overrides.pop("wear", {}) or {})
        w.update(wear or {})
        bad = set(w) - WEAR_KEYS
        if bad:
            raise ValueError(f"unknown wear keys: {sorted(bad)}")
        st = replace(st, wear=w)
        shift = overrides.pop("hue_shift_deg", None)
        if shift:
            pal = {r: [rgb_to_hex(hue_shift(hex_to_rgb(h), float(shift))) for h in cols] if isinstance(cols, list)
                   else rgb_to_hex(hue_shift(hex_to_rgb(cols), float(shift))) for r, cols in st.palette.items()}
            st = replace(st, palette=pal)
        if "max_saturation" in overrides:
            st = replace(st, max_saturation=float(overrides.pop("max_saturation")))
        if overrides:
            raise ValueError(f"unknown style_overrides: {sorted(overrides)}")
        return st


def _parse(raw: dict[str, Any], origin: str) -> Style:
    extra = set(raw) - ALLOWED_TOP
    if extra:
        raise ValueError(f"{origin}: unknown style keys {sorted(extra)}")
    pal = raw.get("palette", {})
    badp = set(pal) - PALETTE_KEYS
    if badp:
        raise ValueError(f"{origin}: unknown palette roles {sorted(badp)}")
    norm: dict[str, list[str]] = {}
    for role, v in pal.items():
        vals = v if isinstance(v, list) else [v]
        for c in vals:
            hex_to_rgb(c)  # validates
        norm[role] = list(vals)
    if "base" not in norm:
        raise ValueError(f"{origin}: palette.base is required")
    bads = set(raw.get("silhouette", {})) - SILHOUETTE_KEYS
    if bads:
        raise ValueError(f"{origin}: unknown silhouette rules {sorted(bads)}")
    wear = dict(DEFAULT_WEAR)
    wear.update(raw.get("wear", {}))
    badw = set(wear) - WEAR_KEYS
    if badw:
        raise ValueError(f"{origin}: unknown wear keys {sorted(badw)}")
    return Style(
        name=raw["name"],
        description=raw.get("description", ""),
        palette=norm,
        max_saturation=float(raw.get("max_saturation", 0.40)),
        value_range=tuple(raw.get("value_range", (0.05, 0.75))),  # type: ignore[arg-type]
        wear=wear,
        roughness=tuple(raw.get("roughness", (0.5, 0.95))),  # type: ignore[arg-type]
        coverage={k: tuple(v) for k, v in raw.get("coverage", {}).items()},  # type: ignore[misc]
        rot_fraction=tuple(raw.get("rot_fraction", (0.0, 1.0))),  # type: ignore[arg-type]
        min_clean_fraction=float(raw.get("min_clean_fraction", 0.0)),
        silhouette=dict(raw.get("silhouette", {})),
        checklist=list(raw.get("checklist", [])),
    )


def list_styles(extra_dirs: list[Path] | None = None) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for d in [BUILTIN_DIR, *(extra_dirs or [])]:
        if d.is_dir():
            for p in sorted(d.glob("*.yaml")):
                found[p.stem] = p  # later dirs (project) override built-ins
    return found


def load_style(name: str, extra_dirs: list[Path] | None = None) -> Style:
    table = list_styles(extra_dirs)
    if name not in table:
        raise ValueError(f"unknown style_template {name!r}; available: {sorted(table)}")
    raw = yaml.safe_load(table[name].read_text())
    if raw.get("name") != name:
        raise ValueError(f"{table[name]}: name field must equal file stem {name!r}")
    return _parse(raw, str(table[name]))
