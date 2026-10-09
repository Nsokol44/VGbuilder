"""Procedural buildings in pure numpy (no Blender): the numbers -> a triangle mesh, deterministically.

A building is kit-built from closed boxes and prisms: wall runs split into bays (piers, sills, lintels around real
window / door openings), floor slabs, a plinth, optional trim (string courses, window hoods, door frame, buttresses,
crenellations) and a roof (gable in segments, hip, flat, spire). Damage removes pieces (upper-floor wall pieces,
merlons, gable-roof segments) chosen by a hash of (seed, piece key), so a given seed always breaks the same way and
every level of detail breaks identically.

This is NOT a Blender Geometry Nodes setup; it is Python-generated geometry, which is what makes it testable
headless and byte-reproducible. Walls are separate boxes that abut (no boolean union), so the mesh is a stack of closed
shells with hidden coplanar faces between neighbours: fine for game kits, not for watertight printing.

Levels of detail are built, not decimated: 0 = everything, 1 = openings kept but no trim, 2 = massing (one box + roof).

Conventions: x = width, y = depth, z up; centred on the origin in x/y, ground at z = 0; the door is in the front
wall (-Y), like characters facing -Y.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

STYLES = {"hall", "gothic", "tower"}
ROOFS = {"gable", "hip", "flat", "spire"}
PARAPETS = {"none", "crenellated"}
ROLES = ["wall", "slab", "plinth", "trim", "buttress", "parapet", "roof"]
ROLE_ID = {r: i for i, r in enumerate(ROLES)}

DEFAULTS: dict[str, dict[str, Any]] = {
    "hall": dict(footprint=[12.0, 7.0], floors=2, floor_height=3.2, bay_width=3.0, roof="gable", roof_pitch_deg=38.0,
                 overhang=0.5, wall_t=0.35, window_w=0.9, window_h=1.3, sill_h=0.9, door_w=1.5, door_h=2.3, plinth_h=0.4,
                 buttresses=False, parapet="none", string_courses=True, windows=True, damage=0.0),
    "gothic": dict(footprint=[9.0, 16.0], floors=3, floor_height=4.0, bay_width=3.5, roof="gable", roof_pitch_deg=52.0,
                   overhang=0.2, wall_t=0.7, window_w=1.1, window_h=2.6, sill_h=1.0, door_w=1.8, door_h=3.2, plinth_h=0.6,
                   buttresses=True, parapet="none", string_courses=False, windows=True, damage=0.0),
    "tower": dict(footprint=[6.0, 6.0], floors=5, floor_height=3.5, bay_width=3.0, roof="flat", roof_pitch_deg=60.0,
                  overhang=0.0, wall_t=0.8, window_w=0.6, window_h=1.2, sill_h=1.2, door_w=1.4, door_h=2.4, plinth_h=0.5,
                  buttresses=False, parapet="crenellated", string_courses=True, windows=True, damage=0.0),
}
KEYS = {"style", *DEFAULTS["hall"]}
# numeric keys and their allowed ranges (also what the live-edit sliders offer)
RANGES = {"floors": (1, 12), "floor_height": (2.2, 6.0), "bay_width": (1.5, 8.0), "roof_pitch_deg": (10, 70),
          "overhang": (0, 1.5), "wall_t": (0.15, 1.5), "window_w": (0.3, 3.0), "window_h": (0.3, 5.0),
          "sill_h": (0.0, 3.0), "door_w": (0.8, 4.0), "door_h": (1.9, 5.0), "plinth_h": (0, 1.2), "damage": (0, 1)}
PIER_MIN = 0.3            # solid wall left on each side of an opening, m
SLAB_T = 0.2
TRIM_OUT = 0.08           # how far trim protrudes
PLINTH_OUT = 0.12
BUTTRESS_OUT = 0.5
PARAPET_H = 0.9
FLAT_ROOF_T = 0.25


# ------------------------------------------------------------------ spec
def resolve(raw: dict[str, Any]) -> dict[str, Any]:
    """Style defaults + overrides. Raises ValueError with the offending key."""
    bad = set(raw) - KEYS
    if bad:
        raise ValueError(f"unknown structure keys {sorted(bad)}; allowed: {sorted(KEYS)}")
    style = raw.get("style", "hall")
    if style not in STYLES:
        raise ValueError(f"structure.style must be one of {sorted(STYLES)}")
    d = {**DEFAULTS[style], **{k: v for k, v in raw.items() if k != "style"}, "style": style}
    fp = d["footprint"]
    if not (isinstance(fp, (list, tuple)) and len(fp) == 2 and all(isinstance(x, (int, float)) and 3 <= x <= 60 for x in fp)):
        raise ValueError("structure.footprint must be [width, depth] in metres, each 3..60")
    for k, (lo, hi) in RANGES.items():
        v = d[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= v <= hi):
            raise ValueError(f"structure.{k} must be a number in [{lo}, {hi}], got {v!r}")
    if int(d["floors"]) != d["floors"]:
        raise ValueError("structure.floors must be a whole number")
    if d["roof"] not in ROOFS:
        raise ValueError(f"structure.roof must be one of {sorted(ROOFS)}")
    if d["parapet"] not in PARAPETS:
        raise ValueError(f"structure.parapet must be one of {sorted(PARAPETS)}")
    for k in ("buttresses", "string_courses", "windows"):
        if not isinstance(d[k], bool):
            raise ValueError(f"structure.{k} must be true or false")
    W, D = fp
    if d["wall_t"] * 3 > min(W, D):
        raise ValueError("structure.wall_t is too thick for the footprint")
    if d["windows"] and d["window_w"] + 2 * PIER_MIN > d["bay_width"]:
        raise ValueError(f"window_w {d['window_w']} leaves less than {PIER_MIN} m of wall in a {d['bay_width']} m bay")
    if d["windows"] and d["sill_h"] + d["window_h"] + 0.3 > d["floor_height"]:
        raise ValueError("sill_h + window_h must leave 0.3 m below the next floor (lower the window or raise floor_height)")
    if d["door_w"] + 2 * PIER_MIN > d["bay_width"] or d["door_h"] + 0.3 > d["floor_height"]:
        raise ValueError("the door does not fit in a bay / a storey (door_w + 0.6 <= bay_width, door_h + 0.3 <= floor_height)")
    d["footprint"] = [float(W), float(D)]
    d["floors"] = int(d["floors"])
    return d


# ------------------------------------------------------------------ mesh builder
class _B:
    def __init__(self) -> None:
        self.v: list[tuple[float, float, float]] = []
        self.t: list[tuple[int, int, int]] = []
        self.r: list[int] = []
        self.parts = 0

    def box(self, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, role: str) -> None:
        if x1 - x0 < 1e-6 or y1 - y0 < 1e-6 or z1 - z0 < 1e-6:
            return
        i = len(self.v)
        self.v += [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        for a, b, c in ((0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
                        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)):
            self.t.append((i + a, i + b, i + c))
        self.r += [ROLE_ID[role]] * 12
        self.parts += 1

    def solid(self, verts: list[tuple[float, float, float]], faces: list[tuple[int, ...]], role: str) -> None:
        """Convex solid from polygon faces; each face is flipped to point away from the centroid."""
        P = np.array(verts, float)
        ctr = P.mean(0)
        i = len(self.v)
        self.v += [tuple(p) for p in verts]
        for f in faces:
            a, b, c = P[f[0]], P[f[1]], P[f[2]]
            order = list(f)
            if np.dot(np.cross(b - a, c - a), P[list(f)].mean(0) - ctr) < 0:
                order = order[::-1]
            for k in range(1, len(order) - 1):
                self.t.append((i + order[0], i + order[k], i + order[k + 1]))
                self.r.append(ROLE_ID[role])
        self.parts += 1

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array(self.v, np.float32), np.array(self.t, np.int32).reshape(-1, 3), np.array(self.r, np.int8)


def _gone(seed: int, key: str, p: float) -> bool:
    """Deterministic per-piece coin flip; independent of build order and of the level of detail."""
    if p <= 0:
        return False
    return float(np.random.default_rng(zlib.crc32(f"{seed}:{key}".encode())).random()) < p


# ------------------------------------------------------------------ derived dimensions
def wall_top(d: dict[str, Any]) -> float:
    return d["floors"] * d["floor_height"]


def roof_rise(d: dict[str, Any]) -> float:
    W, D = d["footprint"]
    if d["roof"] == "flat":
        return FLAT_ROOF_T
    return (min(W, D) / 2 + d["overhang"]) * math.tan(math.radians(d["roof_pitch_deg"]))


def expected_height(d: dict[str, Any], level: int = 0) -> float:
    """Height of the finished mesh, from the numbers alone (the QA compares the built mesh against this)."""
    h = wall_top(d) + roof_rise(d)
    if d["roof"] == "flat" and d["parapet"] == "crenellated":
        h += PARAPET_H
    return h


def max_protrusion(d: dict[str, Any]) -> float:
    """Furthest anything may stick out past the wall footprint (plinth, trim, buttresses, roof overhang)."""
    ro = d["overhang"] if d["roof"] != "flat" else 0.0
    return max(PLINTH_OUT if d["plinth_h"] > 0 else 0.0, TRIM_OUT if (d["string_courses"] or d["windows"]) else 0.0,
               BUTTRESS_OUT if d["buttresses"] else 0.0, ro)


# ------------------------------------------------------------------ the build
def _map(side: str, W: float, D: float, t: float):
    """Local wall frame (u along the wall, v inward from the outer face, z) -> world box ranges."""
    def f(u0, u1, v0, v1, z0, z1):
        if side == "front":
            return (u0, u1, -D / 2 + v0, -D / 2 + v1, z0, z1)
        if side == "back":
            return (u0, u1, D / 2 - v1, D / 2 - v0, z0, z1)
        if side == "left":
            return (-W / 2 + v0, -W / 2 + v1, u0, u1, z0, z1)
        return (W / 2 - v1, W / 2 - v0, u0, u1, z0, z1)
    return f


def build(d: dict[str, Any], level: int = 0, seed: int = 0, roof_segment_len: float = 2.0
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """(verts float32, tris int32, role per tri int8, meta). `d` comes from resolve()."""
    if level not in (0, 1, 2):
        raise ValueError("level must be 0, 1 or 2")
    b = _B()
    W, D = d["footprint"]
    t, fh, nf, bw = d["wall_t"], d["floor_height"], d["floors"], d["bay_width"]
    H = wall_top(d)
    dmg = d["damage"]
    meta: dict[str, Any] = {"windows": 0, "doors": 0, "level": level, "wall_top": H, "roof_rise": roof_rise(d),
                            "expected_height": expected_height(d), "max_protrusion": max_protrusion(d)}

    # --- plinth (ring, split around the door) -------------------------------------------------------------
    ph = d["plinth_h"]
    door_bay_u = None
    if level < 2 and ph > 0:
        o = PLINTH_OUT
        nfront = max(1, round(W / bw))
        door_bay = nfront // 2
        bwf = W / nfront
        uc = -W / 2 + (door_bay + 0.5) * bwf
        gap0, gap1 = uc - d["door_w"] / 2, uc + d["door_w"] / 2
        b.box(-W / 2 - o, gap0, -D / 2 - o, -D / 2 + t, 0, ph, "plinth")
        b.box(gap1, W / 2 + o, -D / 2 - o, -D / 2 + t, 0, ph, "plinth")
        b.box(-W / 2 - o, W / 2 + o, D / 2 - t, D / 2 + o, 0, ph, "plinth")
        b.box(-W / 2 - o, -W / 2 + t, -D / 2 + t, D / 2 - t, 0, ph, "plinth")
        b.box(W / 2 - t, W / 2 + o, -D / 2 + t, D / 2 - t, 0, ph, "plinth")

    # --- walls ----------------------------------------------------------------------------------------------
    if level == 2 and dmg <= 0:
        b.box(-W / 2, W / 2, -D / 2, D / 2, 0, H, "wall")
    elif level == 2:
        # damaged massing: one box per bay, kept unless LOD0 lost every piece of that bay, so the far LOD
        # is not a pristine box around a ruin
        for side in ("front", "back", "left", "right"):
            L = W if side in ("front", "back") else D - 2 * t
            mp = _map(side, W, D, t)
            n = max(1, round(L / bw))
            bayw = L / n
            for f in range(nf):
                wgt = 0.0 if f == 0 or nf == 1 else (f / (nf - 1)) ** 1.5
                for i in range(n):
                    is_door = side == "front" and f == 0 and i == n // 2
                    names = ["pl", "pr", "lintel"] + (["sill"] if (not is_door and d["windows"] and d["sill_h"] > 0) else []) \
                        if (is_door or d["windows"]) else ["full"]
                    if all(_gone(seed, f"wall_{side}_{f}_{i}_{nm}", dmg * wgt) for nm in names):
                        continue
                    uc = -L / 2 + (i + 0.5) * bayw
                    b.box(*mp(uc - bayw / 2, uc + bayw / 2, 0, t, f * fh, (f + 1) * fh), "wall")
        b.box(-W / 2 + t, W / 2 - t, -D / 2 + t, D / 2 - t, (nf - 1) * fh, (nf - 1) * fh + SLAB_T, "slab")   # what you see through roof gaps
    else:
        for side in ("front", "back", "left", "right"):
            L = W if side in ("front", "back") else D - 2 * t
            mp = _map(side, W, D, t)
            n = max(1, round(L / bw))
            bayw = L / n
            for f in range(nf):
                z0, z1 = f * fh, (f + 1) * fh
                wgt = 0.0 if f == 0 or nf == 1 else (f / (nf - 1)) ** 1.5
                for i in range(n):
                    uc = -L / 2 + (i + 0.5) * bayw
                    ua, ub = uc - bayw / 2, uc + bayw / 2
                    is_door = side == "front" and f == 0 and i == n // 2
                    if is_door:
                        ow, oh, sill = d["door_w"], d["door_h"], 0.0
                    elif d["windows"]:
                        ow, oh, sill = d["window_w"], d["window_h"], d["sill_h"]
                    else:
                        ow = 0.0
                    pieces: list[tuple[str, tuple[float, ...]]] = []
                    if ow == 0.0:
                        pieces.append(("full", (ua, ub, 0, t, z0, z1)))
                    else:
                        o0, o1 = uc - ow / 2, uc + ow / 2
                        pieces += [("pl", (ua, o0, 0, t, z0, z1)), ("pr", (o1, ub, 0, t, z0, z1)),
                                   ("lintel", (o0, o1, 0, t, z0 + sill + oh, z1))]
                        if sill > 0:
                            pieces.append(("sill", (o0, o1, 0, t, z0, z0 + sill)))
                    for name, box in pieces:
                        if _gone(seed, f"wall_{side}_{f}_{i}_{name}", dmg * wgt):
                            continue
                        b.box(*mp(*box), "wall")
                    if ow:
                        meta["doors" if is_door else "windows"] += 1
                        if is_door:
                            meta["door"] = {"w": ow, "h": oh}
                    # trim (level 0 only): hood above and ledge below the opening, frame around the door
                    if level == 0 and ow and not _gone(seed, f"trim_{side}_{f}_{i}", dmg * wgt):
                        o0, o1 = uc - ow / 2, uc + ow / 2
                        if is_door:
                            b.box(*mp(o0 - 0.12, o0, -TRIM_OUT, 0, 0, oh + 0.12), "trim")
                            b.box(*mp(o1, o1 + 0.12, -TRIM_OUT, 0, 0, oh + 0.12), "trim")
                            b.box(*mp(o0 - 0.12, o1 + 0.12, -TRIM_OUT, 0, oh, oh + 0.12), "trim")
                        else:
                            b.box(*mp(o0 - 0.1, o1 + 0.1, -TRIM_OUT, 0, z0 + sill + oh, z0 + sill + oh + 0.1), "trim")
                            b.box(*mp(o0 - 0.1, o1 + 0.1, -TRIM_OUT, 0, z0 + sill - 0.08, z0 + sill), "trim")

    # --- floor slabs ------------------------------------------------------------------------------------------
    if level < 2:
        for f in range(nf):
            z0 = 0.0 if f == 0 else f * fh - SLAB_T / 2
            b.box(-W / 2 + t, W / 2 - t, -D / 2 + t, D / 2 - t, z0, z0 + SLAB_T, "slab")

    # --- string courses / buttresses (level 0) ---------------------------------------------------------------
    if level == 0:
        p = TRIM_OUT
        if d["string_courses"]:
            for f in range(1, nf):
                z = f * fh
                b.box(-W / 2 - p, W / 2 + p, -D / 2 - p, -D / 2, z - 0.08, z + 0.08, "trim")
                b.box(-W / 2 - p, W / 2 + p, D / 2, D / 2 + p, z - 0.08, z + 0.08, "trim")
                b.box(-W / 2 - p, -W / 2, -D / 2, D / 2, z - 0.08, z + 0.08, "trim")
                b.box(W / 2, W / 2 + p, -D / 2, D / 2, z - 0.08, z + 0.08, "trim")
        if d["buttresses"]:
            bo, bwid = BUTTRESS_OUT, 0.5
            nb = max(1, round((D - 2 * t) / bw))
            ys = [-D / 2 + t + (k + 1) * (D - 2 * t) / nb for k in range(nb - 1)] + [-D / 2, D / 2]
            for sx in (-1, 1):
                for y in ys:
                    if _gone(seed, f"butt_{sx}_{y:.2f}", dmg):
                        continue
                    xo, xi = (sx * W / 2, sx * (W / 2 + bo)) if sx > 0 else (sx * (W / 2 + bo), sx * W / 2)
                    b.box(xo, xi, y - bwid / 2, y + bwid / 2, 0, H * 0.7, "buttress")
                    b.box(xo + (0.12 if sx > 0 else 0.0) * 1, xi - (0.0 if sx > 0 else 0.12), y - bwid / 2 + 0.08, y + bwid / 2 - 0.08,
                          H * 0.7, H * 0.88, "buttress")

    # --- roof -------------------------------------------------------------------------------------------------
    ov = d["overhang"] if d["roof"] != "flat" else 0.0
    rw, rd = W / 2 + ov, D / 2 + ov
    rise = roof_rise(d)
    kind = d["roof"]
    if kind == "gable":
        along_x = W >= D
        half = rd if along_x else rw                    # half-span across the ridge
        length = (W if along_x else D) + 2 * ov
        nseg = max(1, round(length / roof_segment_len))
        seg = length / nseg
        for k in range(nseg):
            if _gone(seed, f"roof_{k}", dmg):
                continue
            a0, a1 = -length / 2 + k * seg, -length / 2 + (k + 1) * seg
            if along_x:
                vs = [(a0, -half, H), (a0, half, H), (a0, 0, H + rise), (a1, -half, H), (a1, half, H), (a1, 0, H + rise)]
            else:
                vs = [(-half, a0, H), (half, a0, H), (0, a0, H + rise), (-half, a1, H), (half, a1, H), (0, a1, H + rise)]
            b.solid(vs, [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (0, 2, 5, 3), (1, 2, 5, 4)], "roof")
    elif kind == "hip":
        along_x = rw >= rd
        rl = abs(rw - rd)                               # half ridge length
        if along_x:
            vs = [(-rw, -rd, H), (rw, -rd, H), (rw, rd, H), (-rw, rd, H), (-rl, 0, H + rise), (rl, 0, H + rise)]
        else:
            vs = [(-rw, -rd, H), (rw, -rd, H), (rw, rd, H), (-rw, rd, H), (0, -rl, H + rise), (0, rl, H + rise)]
        faces = ([(0, 1, 2, 3), (0, 1, 5, 4), (1, 2, 5), (2, 3, 4, 5), (3, 0, 4)] if along_x else
                 [(0, 1, 2, 3), (0, 1, 4), (1, 2, 5, 4), (2, 3, 5), (3, 0, 4, 5)])
        b.solid(vs, faces, "roof")
    elif kind == "spire":
        vs = [(-rw, -rd, H), (rw, -rd, H), (rw, rd, H), (-rw, rd, H), (0, 0, H + rise)]
        b.solid(vs, [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], "roof")
    else:  # flat
        b.box(-W / 2, W / 2, -D / 2, D / 2, H, H + FLAT_ROOF_T, "roof")
        if d["parapet"] == "crenellated" and level < 2:
            z0, z1 = H + FLAT_ROOF_T, H + FLAT_ROOF_T + PARAPET_H
            pt = max(0.2, t * 0.5)
            if level == 1:                              # continuous band, same top as LOD0 so the height does not change
                b.box(-W / 2, W / 2, -D / 2, -D / 2 + pt, z0, z1, "parapet")
                b.box(-W / 2, W / 2, D / 2 - pt, D / 2, z0, z1, "parapet")
                b.box(-W / 2, -W / 2 + pt, -D / 2 + pt, D / 2 - pt, z0, z1, "parapet")
                b.box(W / 2 - pt, W / 2, -D / 2 + pt, D / 2 - pt, z0, z1, "parapet")
            else:                                       # base course + merlons
                zb = z0 + PARAPET_H * 0.45
                b.box(-W / 2, W / 2, -D / 2, -D / 2 + pt, z0, zb, "parapet")
                b.box(-W / 2, W / 2, D / 2 - pt, D / 2, z0, zb, "parapet")
                b.box(-W / 2, -W / 2 + pt, -D / 2 + pt, D / 2 - pt, z0, zb, "parapet")
                b.box(W / 2 - pt, W / 2, -D / 2 + pt, D / 2 - pt, z0, zb, "parapet")
                mw, gap = 0.7, 0.5
                for side in ("front", "back", "left", "right"):
                    Ls = W if side in ("front", "back") else D
                    m = max(1, int((Ls + gap) // (mw + gap)))
                    used = m * mw + (m - 1) * gap
                    for k in range(m):
                        if _gone(seed, f"merlon_{side}_{k}", min(1.0, dmg * 1.2)):
                            continue
                        u0 = -used / 2 + k * (mw + gap)
                        if side == "front":
                            b.box(u0, u0 + mw, -D / 2, -D / 2 + pt, zb, z1, "parapet")
                        elif side == "back":
                            b.box(u0, u0 + mw, D / 2 - pt, D / 2, zb, z1, "parapet")
                        elif side == "left":
                            b.box(-W / 2, -W / 2 + pt, u0, u0 + mw, zb, z1, "parapet")
                        else:
                            b.box(W / 2 - pt, W / 2, u0, u0 + mw, zb, z1, "parapet")
    if level == 2 and kind == "flat" and d["parapet"] == "crenellated":
        b.box(-W / 2, W / 2, -D / 2, D / 2, H + FLAT_ROOF_T, H + FLAT_ROOF_T + PARAPET_H, "parapet")   # solid band, same top as LOD0

    v, tr, ro = b.result()
    if len(tr) == 0:
        raise ValueError("nothing left of the building (damage removed every piece)")
    # centre in x/y is by construction; ground is z = 0 (plinth / slab / pier bottoms)
    meta["tris"] = int(len(tr))
    meta["parts"] = b.parts
    return v, tr, ro, meta


def collision(d: dict[str, Any], seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convex pieces for a physics engine: the wall volume as ONE solid box (so: no walking inside, doors and windows
    are ignored; a flat roof and its parapet are folded into the box), plus a pitched roof as one convex piece unless
    damage removed every roof segment."""
    W, D = d["footprint"]
    H = wall_top(d)
    flat = d["roof"] == "flat"
    bb = _B()
    bb.box(-W / 2, W / 2, -D / 2, D / 2, 0, expected_height(d) if flat else H, "wall")    # flat roof + parapet fold into the box
    v, t, _ = bb.result()
    out = [(v, t)]
    if not flat and _roof_survives(d, seed):
        rv, rt, rr, _ = build(dict(d, damage=0.0), 2, seed, roof_segment_len=1e9)
        keep = rr == ROLE_ID["roof"]
        used = np.unique(rt[keep])
        remap = np.full(len(rv), -1, np.int64)
        remap[used] = np.arange(len(used))
        out.append((rv[used], remap[rt[keep]].astype(np.int32)))
    return out


def _roof_survives(d: dict[str, Any], seed: int) -> bool:
    if d["roof"] != "gable":
        return True
    W, D = d["footprint"]
    length = (W if W >= D else D) + 2 * d["overhang"]
    return any(not _gone(seed, f"roof_{k}", d["damage"]) for k in range(max(1, round(length / 2.0))))
