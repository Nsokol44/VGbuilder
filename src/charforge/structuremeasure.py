"""Measurements on structure meshes, in numpy only (the Blender side hands over plain arrays)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

VIEWS = {"front": (0, 2), "side": (1, 2), "top": (0, 1)}


def bbox(v: np.ndarray) -> dict[str, list[float]]:
    lo, hi = v.min(0), v.max(0)
    return {"min": [float(x) for x in lo], "max": [float(x) for x in hi], "size": [float(x) for x in hi - lo]}


def degenerate_fraction(v: np.ndarray, t: np.ndarray, eps: float = 1e-8) -> float:
    tri = v[t].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return float((area < eps).mean()) if len(t) else 0.0


def silhouettes(v: np.ndarray, t: np.ndarray, frame: dict[str, list[float]], px: int = 256,
                fill_holes: bool = True) -> dict[str, np.ndarray]:
    """Front (x,z), side (y,z), top (x,y) silhouettes drawn in `frame` (the LOD0 bbox + a margin), so that
    different LODs are comparable pixel for pixel. With fill_holes the OUTLINE is compared: windows you can see
    straight through are enclosed holes and do not count, because a far LOD is an opaque stand-in by design;
    gaps that open to the outside (a collapsed wall, a missing roof section) still do."""
    out = {}
    lo, hi = np.array(frame["min"]), np.array(frame["max"])
    pad = 0.02 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    for name, (a, b) in VIEWS.items():
        scale = (px - 1) / max(hi[a] - lo[a], hi[b] - lo[b])
        w, h = int((hi[a] - lo[a]) * scale) + 1, int((hi[b] - lo[b]) * scale) + 1
        img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        for tri in v[t][:, :, [a, b]]:
            d.polygon([((x - lo[a]) * scale, (hi[b] - y) * scale) for x, y in tri], fill=255)
        if fill_holes:
            ImageDraw.floodfill(img, (0, 0), 128, thresh=0)        # background = what the border can reach
            out[name] = np.asarray(img) != 128
        else:
            out[name] = np.asarray(img) > 127
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def uv_overlap(uv: np.ndarray, res: int = 512) -> dict[str, float]:
    """uv: (T, 3, 2) per triangle corner. Rasterise every triangle at texel centres and report how much of the
    covered area is covered more than once (a lightmap must be 0), and how many triangles leave the [0,1] square.
    Texel centres that sit exactly on a shared edge count for neither triangle, so adjacent triangles of one island
    do not register as an overlap."""
    cnt = np.zeros((res, res), np.int16)
    P = uv.astype(np.float64) * res
    out_of_range = 0
    for tri in P:
        if (tri < -1e-6).any() or (tri > res + 1e-6).any():
            out_of_range += 1
        x0, y0 = np.floor(tri.min(0)).astype(int).clip(0, res - 1)
        x1, y1 = np.ceil(tri.max(0)).astype(int).clip(0, res - 1)
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = tri
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        l3 = 1 - l1 - l2
        inside = (l1 > 1e-9) & (l2 > 1e-9) & (l3 > 1e-9)
        cnt[y0:y1 + 1, x0:x1 + 1] += inside
    covered = int((cnt > 0).sum())
    return {"covered": covered / (res * res), "overlap_fraction": float((cnt > 1).sum() / max(covered, 1)),
            "out_of_range": out_of_range / max(len(uv), 1), "res": res}


def envelope_area(footprint: tuple[float, float], wall_top: float) -> float:
    W, D = footprint
    return 2 * (W + D) * wall_top + W * D
