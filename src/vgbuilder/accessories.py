"""Procedural accessories: pauldrons, horns (joined into the body mesh) and a cape (separate, cloth-tagged).

Pure numpy, built with the same helpers as the body so triangle counts are known BEFORE Blender runs
(the budget fit reserves them) and the Blender stage just turns the arrays into meshes.

They are added after the sketch fit and are not part of the silhouette comparison: the fit and the
silhouette QA judge the base body, accessories ride on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bodygen import Body, _Builder, _frame
from .rigdefs import joints_from_params

SHAPE_DEFAULTS = {"pauldron_scale": 1.0, "horn_len": 1.0, "horn_spread": 1.0, "cape_len": 1.0, "cape_flare": 0.35}
SHAPE_LIMITS = {"pauldron_scale": (0.5, 2.5), "horn_len": (0.4, 3.0), "horn_spread": (0.4, 2.5),
                "cape_len": (0.4, 2.0), "cape_flare": (0.0, 1.5)}

JOINED = ("pauldrons", "horns")     # become part of the body mesh
SEPARATE = ("cape",)                # own mesh object, with a physics tag


@dataclass
class Part:
    name: str
    body: Body
    joined: bool
    tags: dict[str, str] = field(default_factory=dict)             # exported as glTF extras on the node
    point_attrs: dict[str, np.ndarray] = field(default_factory=dict)  # per-vertex floats (e.g. _CLOTH_PIN)


def _pauldrons(p, H, seg, shp) -> Body:
    j = joints_from_params(p, H)
    b = _Builder()
    hs = max(3, seg // 2 + 1)
    for side, sx in (("l", 1.0), ("r", -1.0)):
        sh = np.array(j[f"upperarm_{side}"][0])
        c = sh + np.array([sx * 0.012 * H, 0.0, 0.014 * H])
        n0 = len(b.v)
        k = shp["pauldron_scale"]
        b.sphere(c, (0.058 * H * k, 0.052 * H * k, 0.034 * H * k), seg, hs)
        b.rigid(n0, f"upperarm_{side}")
    return _body(b, j)


def _horns(p, H, seg, shp) -> Body:
    j = joints_from_params(p, H)
    b = _Builder()
    head_r = 0.065 * p["head_scale"] * H
    rings_n = 6
    for sx in (1.0, -1.0):
        p0 = np.array([sx * 0.88 * head_r, 0.0, H - 0.75 * head_r])
        p1 = p0 + np.array([sx * 0.075 * H * shp["horn_spread"], 0.0, 0.015 * H])
        p2 = p0 + np.array([sx * 0.095 * H * shp["horn_spread"], 0.0, 0.115 * H * shp["horn_len"]])
        ts = np.linspace(0, 1, rings_n)
        pts = [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2 for t in ts]
        rings = []
        for i, c in enumerate(pts):
            d = pts[min(i + 1, rings_n - 1)] - pts[max(i - 1, 0)]
            u, v = _frame(d)
            r = 0.016 * H * (1 - 0.92 * ts[i])
            rings.append((c, u, v, r, r))
        n0 = len(b.v)
        b.loft(rings, seg)
        b.rigid(n0, "head")
    return _body(b, j)


def _cape(p, H, seg, shp, cols: int = 8, rows: int = 8) -> tuple[Body, np.ndarray]:
    """A hanging sheet behind the torso. Top row is pinned (cloth pin 1.0), the rest is free."""
    j = joints_from_params(p, H)
    sh_z = j["upperarm_l"][0][2]
    half = p["shoulder_half"] * H * 0.98
    back_y = 0.056 * H * p["torso_depth"] + 0.012 * H
    length = 0.42 * H * shp["cape_len"]
    b = _Builder()
    grid = []
    pin = []
    for r in range(rows + 1):
        t = r / rows
        row = []
        for c in range(cols + 1):
            s = c / cols * 2 - 1
            x = s * half * (1 + shp["cape_flare"] * t)
            y = back_y + 0.05 * H * t * t
            z = sh_z - 0.01 * H - length * t
            row.append(b.add((x, y, z)))
            pin.append(1.0 if r == 0 else 0.0)
            role = "chest" if t < 0.5 else "spine"
            b.w[row[-1]] = [(role, 1.0)]
        grid.append(row)
    for r in range(rows):
        for c in range(cols):
            # wound so the front face looks away from the body (+Y)
            b.f.append((grid[r][c], grid[r][c + 1], grid[r + 1][c + 1], grid[r + 1][c]))
    return _body(b, j), np.array(pin)


def _body(b: _Builder, j) -> Body:
    return Body(np.asarray(b.v).astype(np.float32).astype(np.float64), b.f, b.w, j)


def validate_shape(shape: dict[str, float]) -> None:
    bad = set(shape) - set(SHAPE_DEFAULTS)
    if bad:
        raise ValueError(f"unknown accessory_shape keys: {sorted(bad)}")
    for k, v in shape.items():
        lo, hi = SHAPE_LIMITS[k]
        if not (lo <= float(v) <= hi):
            raise ValueError(f"accessory_shape.{k} must be in [{lo}, {hi}]")


def build_parts(names: list[str], p: dict[str, float], H: float, seg: int | None = None,
                shape: dict[str, float] | None = None) -> list[Part]:
    shp = {**SHAPE_DEFAULTS, **(shape or {})}
    seg = int(seg if seg is not None else max(4, int(p.get("segments", 8))))
    seg += seg % 2
    parts: list[Part] = []
    if "pauldrons" in names:
        parts.append(Part("pauldrons", _pauldrons(p, H, seg, shp), joined=True))
    if "horns" in names:
        parts.append(Part("horns", _horns(p, H, seg, shp), joined=True))
    if "cape" in names:
        body, pin = _cape(p, H, seg, shp)
        parts.append(Part("cape", body, joined=False, tags={"vgbuilder_physics": "cloth", "cloth_pin_attribute": "_CLOTH_PIN"},
                          point_attrs={"_CLOTH_PIN": pin}))
    return parts


def tri_total(names: list[str], p: dict[str, float], H: float, shape: dict[str, float] | None = None) -> int:
    return sum(part.body.tri_count for part in build_parts(names, p, H, shape=shape))


def tri_split(names: list[str], p: dict[str, float], H: float) -> tuple[int, int]:
    """(triangles joined into the body, triangles in separate meshes)."""
    parts = build_parts(names, p, H)
    return (sum(x.body.tri_count for x in parts if x.joined), sum(x.body.tri_count for x in parts if not x.joined))
