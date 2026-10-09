"""Procedural humanoid body as plain numpy arrays (no Blender).

Same geometry the Blender stage builds, but available anywhere, so the sketch-fitting
optimiser can evaluate thousands of candidate bodies per second against the sketch
silhouette. Blender only turns the final arrays into a mesh.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rigdefs import joints_from_params
from .sketch import SIL_PX, rasterize_triangles


@dataclass
class Body:
    verts: np.ndarray                              # (N, 3)
    faces: list[tuple[int, ...]]
    weights: dict[int, list[tuple[str, float]]] = field(default_factory=dict)
    joints: dict = field(default_factory=dict)

    @property
    def tri_count(self) -> int:
        return sum(len(f) - 2 for f in self.faces)

    def triangles(self) -> np.ndarray:
        """Fixed triangulation shared by the numpy silhouette AND the Blender mesh.

        Quads split along the shorter diagonal, bigger polygons (the flat end caps) are fanned.
        Blender is handed these triangles directly, so what the fitter optimises is exactly
        the mesh that ships, instead of a slightly different triangulation of the same quads.
        """
        v = self.verts
        out: list[tuple[int, int, int]] = []
        for f in self.faces:
            if len(f) == 3:
                out.append((f[0], f[1], f[2]))
            elif len(f) == 4:
                a, b, c, d = f
                if np.linalg.norm(v[a] - v[c]) <= np.linalg.norm(v[b] - v[d]):
                    out += [(a, b, c), (a, c, d)]
                else:
                    out += [(a, b, d), (b, c, d)]
            else:
                out += [(f[0], f[k], f[k + 1]) for k in range(1, len(f) - 1)]
        return np.asarray(out, dtype=np.int32)


class _Builder:
    def __init__(self) -> None:
        self.v: list[np.ndarray] = []
        self.f: list[tuple[int, ...]] = []
        self.w: dict[int, list[tuple[str, float]]] = {}

    def add(self, p) -> int:
        self.v.append(np.asarray(p, float))
        return len(self.v) - 1

    def rigid(self, start: int, role: str) -> None:
        for i in range(start, len(self.v)):
            self.w[i] = [(role, 1.0)]

    def ring(self, c, u, v, rx, ry, n) -> list[int]:
        return [self.add(c + u * rx * np.cos(2 * np.pi * i / n) + v * ry * np.sin(2 * np.pi * i / n)) for i in range(n)]

    def bridge(self, a: list[int], b: list[int]) -> None:
        n = len(a)
        for i in range(n):
            j = (i + 1) % n
            self.f.append((a[i], a[j], b[j], b[i]))

    def loft(self, rings, n, caps=True) -> list[list[int]]:
        vr = [self.ring(c, u, v, rx, ry, n) for c, u, v, rx, ry in rings]
        for a, b in zip(vr, vr[1:]):
            self.bridge(a, b)
        if caps:
            self.f.append(tuple(vr[0][::-1]))
            self.f.append(tuple(vr[-1]))
        return vr

    def tube(self, p0, p1, r0, r1, n) -> None:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        u, v = _frame(p1 - p0)
        self.loft([(p0, u, v, r0, r0), (p1, u, v, r1, r1)], n)

    def sphere(self, c, radii, n, rings) -> None:
        c = np.asarray(c, float)
        rx, ry, rz = radii
        bottom = self.add((c[0], c[1], c[2] - rz))
        top = self.add((c[0], c[1], c[2] + rz))
        rows = []
        for k in range(1, rings):
            t = np.pi * k / rings
            s = np.sin(t)
            z = c[2] - rz * np.cos(t)
            rows.append([self.add((c[0] + rx * s * np.cos(2 * np.pi * i / n),
                                   c[1] + ry * s * np.sin(2 * np.pi * i / n), z)) for i in range(n)])
        for i in range(n):
            j = (i + 1) % n
            self.f.append((bottom, rows[0][j], rows[0][i]))
            self.f.append((top, rows[-1][i], rows[-1][j]))
        for a, b in zip(rows, rows[1:]):
            self.bridge(a, b)

    def box(self, c, size) -> None:
        c, s = np.asarray(c, float), np.asarray(size, float)
        idx = [self.add(c + s * np.array(o) * 0.5) for o in
               [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]]
        for q in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
            self.f.append(tuple(idx[i] for i in q))


def _frame(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = d / np.linalg.norm(d)
    ref = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def build_body(p: dict[str, float], H: float) -> Body:
    seg = max(4, int(p["segments"]))
    seg += seg % 2                      # even ring counts keep every part exactly X-mirror symmetric
    ring_n = max(8, seg * 2)
    hs = max(3, seg // 2 + 1)
    j = joints_from_params(p, H)
    b = _Builder()

    crotch = p["leg_frac"] * H
    hip_z, waist_z, chest_z = j["pelvis"][0][2], j["spine"][0][2], j["chest"][0][2]
    shoulder_z, neck_z = j["upperarm_l"][0][2], j["neck"][0][2]
    td = p["torso_depth"]
    prof = [
        (crotch + 0.005 * H, p["hip_half"] * H * 0.88, 0.058 * H * td),
        (crotch + 0.055 * H, p["hip_half"] * H, 0.062 * H * td),
        (waist_z, p["waist_half"] * H, 0.054 * H * td),
        (chest_z, p["shoulder_half"] * H * 0.93, 0.066 * H * td),
        (shoulder_z - 0.004 * H, p["shoulder_half"] * H * 0.95, 0.056 * H * td),
        (neck_z, 0.045 * H, 0.042 * H),
    ]
    ex, ey = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    rings = b.loft([(np.array([0, 0, z]), ex, ey, hw, hd) for z, hw, hd in prof], ring_n)
    for (z, _, _), ring in zip(prof, rings):
        if z <= hip_z:
            w = [("pelvis", 1.0)]
        elif z <= waist_z:
            t = (z - hip_z) / (waist_z - hip_z)
            w = [("pelvis", 1 - t), ("spine", t)]
        elif z <= chest_z:
            t = (z - waist_z) / (chest_z - waist_z)
            w = [("spine", 1 - t), ("chest", t)]
        else:
            t = min(1.0, (z - chest_z) / max(1e-6, neck_z - chest_z))
            w = [("chest", 1 - 0.5 * t), ("neck", 0.5 * t)]
        for vi in ring:
            b.w[vi] = [(r, x) for r, x in w if x > 1e-4]

    head_r = 0.065 * p["head_scale"] * H
    n0 = len(b.v)
    b.sphere((0, 0, H - head_r), (head_r * 0.92, head_r * 0.98, head_r), seg + 2, hs + 1)
    b.rigid(n0, "head")
    n0 = len(b.v)
    nr = 0.034 * H
    b.tube(j["neck"][0], j["neck"][1], nr, nr * 0.95, seg)
    b.rigid(n0, "neck")

    at, lt = p["arm_thickness"], p["leg_thickness"]
    for side in ("l", "r"):
        ua, fa, ha = j[f"upperarm_{side}"], j[f"forearm_{side}"], j[f"hand_{side}"]
        r_up, r_fo = 0.027 * H * at, 0.021 * H * at
        n0 = len(b.v); b.tube(ua[0], ua[1], r_up, r_up * 0.82, seg); b.rigid(n0, f"upperarm_{side}")
        n0 = len(b.v); b.sphere(ua[0], (r_up * 1.15,) * 3, seg, hs); b.rigid(n0, f"upperarm_{side}")
        n0 = len(b.v); b.tube(fa[0], fa[1], r_fo * 1.05, r_fo * 0.8, seg); b.rigid(n0, f"forearm_{side}")
        n0 = len(b.v); b.sphere(fa[0], (r_fo * 1.12,) * 3, seg, hs); b.rigid(n0, f"forearm_{side}")
        n0 = len(b.v)
        hc = (np.array(ha[0]) + np.array(ha[1])) / 2
        b.sphere(hc, (0.018 * H * at, 0.012 * H * at, 0.032 * H), seg, hs)
        b.rigid(n0, f"hand_{side}")

        th, sh = j[f"thigh_{side}"], j[f"shin_{side}"]
        r_th, r_sh = 0.045 * H * lt, 0.032 * H * lt
        n0 = len(b.v); b.tube(th[0], th[1], r_th, r_th * 0.78, seg); b.rigid(n0, f"thigh_{side}")
        n0 = len(b.v); b.sphere(th[1], (r_th * 0.8,) * 3, seg, hs); b.rigid(n0, f"shin_{side}")
        n0 = len(b.v); b.tube(sh[0], sh[1], r_sh * 1.05, r_sh * 0.72, seg); b.rigid(n0, f"shin_{side}")
        n0 = len(b.v)
        b.box((th[0][0], -0.035 * H, 0.022 * H), (0.055 * H * lt, 0.15 * H, 0.044 * H))
        b.rigid(n0, f"foot_{side}")

    # Blender stores float32; quantise here so the fitter and Blender see identical coordinates
    return Body(np.asarray(b.v).astype(np.float32).astype(np.float64), b.f, b.w, j)


def silhouette(body: Body, height_px: int = SIL_PX) -> np.ndarray:
    tris = body.verts[body.triangles()][:, :, [0, 2]]
    return rasterize_triangles(tris, height_px=height_px)


def silhouette_side(body: Body, height_px: int = SIL_PX) -> np.ndarray:
    """Side view in (y, z): the character faces -Y, so its front is on the left of the image."""
    tris = body.verts[body.triangles()][:, :, [1, 2]]
    return rasterize_triangles(tris, height_px=height_px)


def choose_segments(p: dict[str, float], H: float, budget: int, lo: int = 4, hi: int = 40, extra=None) -> int:
    """Largest `segments` whose LOD0 stays within the triangle budget.

    `extra(q)` -> triangles that ride on top of the body (accessories) and count against the same budget."""
    best = lo
    for s in range(lo, hi + 1):
        q = dict(p, segments=float(s))
        if build_body(q, H).tri_count + (extra(q) if extra else 0) <= budget:
            best = s
        else:
            break
    return best
