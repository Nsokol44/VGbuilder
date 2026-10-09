"""Look-dev: palette-driven wear baked into textures, and image-based checks on the result.

Pure numpy, no Blender. The Blender stage only hands in the mesh (rest-pose vertices, triangles,
per-corner UVs, vertex normals) and loads the PNGs back into a material.

The bake is procedural, not simulated:
  * curvature proxy per vertex (how much the normal changes along each edge): convex = edges that
    wear and chip, concave = crevices that collect grime
  * height on the character: lower = dirtier
  * 3D value noise sampled at the surface *position* (not the UV), so it is seamless across UV islands
These masks are painted into base-colour and roughness textures through the UVs, using the palette
and wear numbers of a style template. Nothing here "looks at" the result: `analyze` measures the baked
(or any other) texture in image space so QA can gate it with the style's numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .styles import Style


# ------------------------------------------------------------------ small helpers
def smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-9
    rc = np.where(nz & (mx == r), ((g - b) / np.where(nz, d, 1)) % 6, 0)
    gc = np.where(nz & (mx == g) & (mx != r), (b - r) / np.where(nz, d, 1) + 2, 0)
    bc = np.where(nz & (mx == b) & (mx != r) & (mx != g), (r - g) / np.where(nz, d, 1) + 4, 0)
    h = (rc + gc + bc) / 6.0 % 1.0
    s = np.where(mx > 1e-9, d / np.where(mx > 1e-9, mx, 1), 0.0)
    return np.stack([h, s, mx], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0] % 1.0, hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    choices_r = [v, q, p, p, t, v]
    choices_g = [t, v, v, q, p, p]
    choices_b = [p, p, t, v, v, q]
    r = np.choose(i, choices_r)
    g = np.choose(i, choices_g)
    b = np.choose(i, choices_b)
    return np.stack([r, g, b], axis=-1)


def _hash01(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761 & 0x7FFFFFFF)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float64) / 65535.0


def value_noise(p: np.ndarray, seed: int) -> np.ndarray:
    """Smooth 3D value noise in [0, 1] at points p (N, 3)."""
    i = np.floor(p).astype(np.int64)
    f = p - i
    f = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * _hash01(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out


def fbm(p: np.ndarray, seed: int, octaves: int = 3) -> np.ndarray:
    total, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        total = total + amp * value_noise(p * (2.0 ** o), seed + 101 * o)
        norm += amp
        amp *= 0.5
    return total / norm


# ------------------------------------------------------------- mesh signals
def vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)           # length = 2 * area, so this is area-weighted
    n = np.zeros_like(verts)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln > 1e-12, ln, 1.0)


def convexity(verts: np.ndarray, tris: np.ndarray, normals: np.ndarray, feature: float) -> np.ndarray:
    """Per-vertex curvature proxy in about [-1, 1]: >0 convex (edges), <0 concave (crevices).

    For each edge, (n_j - n_i) . (p_j - p_i) / |p_j - p_i|^2 is 1/R on a sphere of radius R (negative
    for a dent). Averaged over a vertex's edges and scaled by `feature` (a length: a bump of that radius
    scores 1).
    """
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    d = verts[e[:, 1]] - verts[e[:, 0]]
    dn = normals[e[:, 1]] - normals[e[:, 0]]
    L2 = (d * d).sum(axis=1)
    ok = L2 > 1e-12
    c = np.zeros(len(e))
    c[ok] = (dn[ok] * d[ok]).sum(axis=1) / L2[ok]
    acc = np.zeros(len(verts))
    cnt = np.zeros(len(verts))
    for k in range(2):
        np.add.at(acc, e[:, k], c)
        np.add.at(cnt, e[:, k], 1.0)
    return np.clip(acc / np.maximum(cnt, 1.0) * feature, -1.0, 1.0)


def atlas_uv(n_tris: int) -> np.ndarray:
    """A trivially valid unwrap: every triangle gets its own grid cell. Wasteful, but needs no Blender.

    Used by tests and as a stand-in when no UV tool is available. Returns (T, 3, 2).
    """
    g = int(np.ceil(np.sqrt(max(n_tris, 1))))
    out = np.zeros((n_tris, 3, 2))
    cell = 1.0 / g
    pad = 0.12 * cell
    local = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    for t in range(n_tris):
        cx, cy = (t % g) * cell, (t // g) * cell
        out[t] = np.array([cx, cy]) + pad + local * (cell - 2 * pad) * 0.9
    return out


# --------------------------------------------------------------- rasterising
def rasterize_attributes(uv: np.ndarray, attrs: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate per-corner attributes (T, 3, K) over UV triangles (T, 3, 2) into a (size, size, K) image.

    UV (0,0) is the bottom-left, as in Blender, so row 0 of the image is v = 1.
    """
    K = attrs.shape[2]
    img = np.zeros((size, size, K), np.float32)
    covered = np.zeros((size, size), bool)
    px = uv[:, :, 0] * size
    py = (1.0 - uv[:, :, 1]) * size
    for t in range(len(uv)):
        x, y = px[t], py[t]
        x0, x1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())) + 1, size)
        y0, y1 = max(int(np.floor(y.min())), 0), min(int(np.ceil(y.max())) + 1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(den) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        w0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
        w1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        val = w0[..., None] * attrs[t, 0] + w1[..., None] * attrs[t, 1] + w2[..., None] * attrs[t, 2]
        sub = img[y0:y1, x0:x1]
        sub[inside] = val[inside]
        covered[y0:y1, x0:x1] |= inside
    return img, covered


def dilate(img: np.ndarray, covered: np.ndarray, iters: int = 4) -> np.ndarray:
    """Bleed colours outward from covered pixels so mip-mapping does not pull in black at UV seams."""
    out = img.copy()
    cov = covered.copy()
    for _ in range(iters):
        acc = np.zeros_like(out, dtype=np.float64)
        cnt = np.zeros(cov.shape, np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sh_c = np.roll(np.roll(cov, dy, 0), dx, 1)
                sh_i = np.roll(np.roll(out, dy, 0), dx, 1)
                acc += sh_i * sh_c[..., None]
                cnt += sh_c
        grow = (~cov) & (cnt > 0)
        out[grow] = (acc[grow] / cnt[grow][:, None]).astype(out.dtype)
        cov = cov | grow
    return out


def uv_coverage(verts: np.ndarray, tris: np.ndarray, uv: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Which texture pixels belong to the mesh, and their height on the character. For analysing foreign textures."""
    h01 = np.clip((verts[:, 2] - verts[:, 2].min()) / max(float(np.ptp(verts[:, 2])), 1e-9), 0, 1)
    img, covered = rasterize_attributes(uv, h01[tris][..., None].astype(np.float32), size)
    return covered, img[..., 0]


# ------------------------------------------------------------------- baking
@dataclass
class Bake:
    base_rgb: np.ndarray       # (S, S, 3) uint8
    roughness: np.ndarray      # (S, S) uint8
    covered: np.ndarray        # (S, S) bool, pixels that belong to a UV island
    height01: np.ndarray       # (S, S) float32, height on the character (0 feet .. 1 head) per pixel
    masks: dict[str, np.ndarray]


def _pick(colors: list[tuple[float, float, float]], t: np.ndarray) -> np.ndarray:
    """Blend through a palette list with a 0..1 control field."""
    arr = np.array(colors)
    if len(arr) == 1:
        return np.broadcast_to(arr[0], t.shape + (3,)).copy()
    x = np.clip(t, 0, 0.9999) * (len(arr) - 1)
    i = np.floor(x).astype(int)
    f = smoothstep(0.35, 0.65, x - i)[..., None]
    return arr[i] * (1 - f) + arr[i + 1] * f


def _cover(field: np.ndarray, cov: float) -> np.ndarray:
    """Soft mask that selects the top `cov` fraction of `field`."""
    cov = float(np.clip(cov, 0.0, 1.0))
    if cov <= 0.0:
        return np.zeros_like(field)
    rank = np.empty(len(field))
    rank[np.argsort(field, kind="stable")] = np.linspace(0.0, 1.0, len(field))
    return smoothstep(1.0 - cov - 0.04, 1.0 - cov + 0.04, rank)


def bake(verts: np.ndarray, tris: np.ndarray, uv: np.ndarray, style: Style, height: float,
         size: int = 512, seed: int = 0, wear_scale: float = 1.0,
         normals: np.ndarray | None = None) -> Bake:
    """Paint `style` onto a mesh through its UVs. uv is (T, 3, 2) per triangle corner."""
    if normals is None:
        normals = vertex_normals(verts, tris)
    w = style.wear
    I = float(w["intensity"]) * wear_scale
    conv_v = convexity(verts, tris, normals, feature=0.012 * height)
    h01_v = np.clip((verts[:, 2] - verts[:, 2].min()) / max(float(np.ptp(verts[:, 2])), 1e-9), 0, 1)
    corner = np.concatenate([verts[tris], conv_v[tris][..., None], h01_v[tris][..., None]], axis=2)  # (T,3,5)
    img, covered = rasterize_attributes(uv, corner.astype(np.float32), size)
    ys, xs = np.nonzero(covered)
    pos = img[ys, xs, :3].astype(np.float64)
    conv = img[ys, xs, 3].astype(np.float64)
    h01 = img[ys, xs, 4].astype(np.float64)

    # noise at three scales, sampled in world space so island borders show no seams
    k = float(w["noise_scale"]) / max(height, 1e-9)
    p = pos * k
    n_lo = fbm(p * 0.35, seed + 1, 2)
    n_mid = fbm(p, seed + 2, 3)
    n_hi = fbm(p * 3.1, seed + 3, 3)

    low = (1.0 - h01) ** 1.5
    edge = smoothstep(0.10, 0.60, conv) * (0.45 + 0.9 * n_mid) * float(w["edge_wear"]) * min(1.0, 1.6 * I)
    cavity = smoothstep(0.02, 0.45, -conv)

    # Wear is controlled as COVERAGE: each layer takes the top `cov` share of its field (rank-normalised),
    # so intensity 0.5 really means "about a third of the surface is grimy", whatever the mesh looks like.
    scale = float(wear_scale)
    grime = _cover(0.45 * cavity + 0.55 * low * float(w["lower_dirt"]) + 0.35 * n_mid + 0.15 * n_hi, 0.62 * I)
    rust = _cover(0.5 * np.clip(edge * 1.5 + cavity, 0, 1) + 0.5 * n_hi, 0.6 * float(w["rust"]) * scale)
    rot = _cover(0.5 * n_lo + 0.35 * low + 0.15 * n_mid, float(w["rot"]) * scale)
    blood = _cover(fbm(p * 1.7, seed + 4, 3), 0.5 * float(w["blood"]) * scale)

    pal = style.colors
    base = _pick(pal("base"), fbm(p * 0.5, seed + 5, 2))
    accent = pal("accent")
    bare = np.array(accent[0]) if accent else np.array([0.55, 0.55, 0.55]) * float(np.mean(pal("base")[0]) + 0.35)
    col = base
    if accent:
        trim = smoothstep(0.30, 0.65, conv)[..., None] * 0.85
        col = col * (1 - trim) + bare * trim
    e = np.clip(edge, 0, 1)[..., None]
    col = col * (1 - e) + np.clip(bare * 1.18 + 0.05, 0, 1) * e
    grime_col = np.array((style.colors("grime") or [(0.1, 0.1, 0.1)])[0])
    col = col * (1 - 0.88 * grime[..., None]) + grime_col * 0.88 * grime[..., None]
    rust_c = pal("rust")
    if rust_c:
        col = col * (1 - rust[..., None]) + np.array(rust_c[0]) * rust[..., None]
    rot_c = pal("rot")
    if rot_c:
        rc = _pick(rot_c, fbm(p * 0.8, seed + 6, 2))
        col = col * (1 - 0.92 * rot[..., None]) + rc * 0.92 * rot[..., None]
    blood_c = pal("blood")
    if blood_c:
        col = col * (1 - blood[..., None]) + np.array(blood_c[0]) * blood[..., None]

    # keep the look inside the style's saturation / value limits
    col = _clamp_look(col, style)

    r_lo, r_hi = style.roughness
    r_t = np.clip(0.45 + 0.40 * grime + 0.30 * rot - 0.55 * e[..., 0] + 0.12 * (n_hi - 0.5), 0, 1)
    rough = r_lo + (r_hi - r_lo) * r_t

    base_img = np.zeros((size, size, 3), np.float32)
    rough_img = np.zeros((size, size, 1), np.float32)
    h_img = np.zeros((size, size), np.float32)
    base_img[ys, xs] = col
    rough_img[ys, xs, 0] = rough
    h_img[ys, xs] = h01
    base_img = dilate(base_img, covered)
    rough_img = dilate(rough_img, covered)
    masks = {}
    for name, m in (("edge", e[..., 0]), ("grime", grime), ("rust", rust), ("rot", rot), ("blood", blood)):
        full = np.zeros((size, size), np.float32)
        full[ys, xs] = m
        masks[name] = full
    return Bake(
        base_rgb=(np.clip(base_img, 0, 1) * 255 + 0.5).astype(np.uint8),
        roughness=(np.clip(rough_img[..., 0], 0, 1) * 255 + 0.5).astype(np.uint8),
        covered=covered, height01=h_img, masks=masks,
    )


# ---------------------------------------------------------------- analysis
def _clamp_look(rgb: np.ndarray, style: Style) -> np.ndarray:
    """Apply the style's saturation and value limits (the bake does this to every pixel, so palette
    entries must be compared in their clamped form or a clamped rust looks like a clamped rot)."""
    hsv = rgb_to_hsv(np.clip(rgb, 0, 1))
    hsv[..., 1] = np.minimum(hsv[..., 1], style.max_saturation * 0.97)
    lo, hi = style.value_range
    hsv[..., 2] = np.clip(hsv[..., 2], lo + 0.01, hi - 0.01)
    return hsv_to_rgb(hsv)


def _classes(style: Style) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    cols: list[tuple[float, float, float]] = []
    for group, role in (("clean", "base"), ("clean", "accent"), ("grime", "grime"), ("rust", "rust"),
                        ("rot", "rot"), ("blood", "blood")):
        for c in style.colors(role):
            names.append(group)
            cols.append(c)
    return names, _clamp_look(np.array(cols), style)


def analyze(base_rgb: np.ndarray, covered: np.ndarray, height01: np.ndarray, style: Style) -> dict[str, Any]:
    """Image-space measurements of a base-colour texture (any source), restricted to UV-covered pixels."""
    px = base_rgb[covered].astype(np.float32) / 255.0
    if len(px) == 0:
        return {"ok": False, "reason": "texture has no UV-covered pixels"}
    hsv = rgb_to_hsv(px)
    names, palette = _classes(style)
    out: dict[str, Any] = {
        "pixels": int(len(px)),
        "mean_saturation": float(hsv[:, 1].mean()),
        "p95_saturation": float(np.percentile(hsv[:, 1], 95)),
        "value_p02": float(np.percentile(hsv[:, 2], 2)),
        "value_p98": float(np.percentile(hsv[:, 2], 98)),
    }
    # nearest palette entry per pixel, in chunks
    cls = np.empty(len(px), dtype=np.int32)
    for s in range(0, len(px), 200_000):
        d = ((px[s: s + 200_000, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        cls[s: s + 200_000] = d.argmin(axis=1)
    group = np.array(names)[cls]
    for g in ("clean", "grime", "rust", "rot", "blood"):
        out[f"{g}_fraction"] = float((group == g).mean())
    out["soil_fraction"] = out["grime_fraction"] + out["rust_fraction"] + out["blood_fraction"]
    z = height01[covered]
    dirty = group != "clean"
    lo_m, hi_m = z < 0.33, z > 0.66
    if lo_m.sum() > 50 and hi_m.sum() > 50:
        out["dirty_lower_third"] = float(dirty[lo_m].mean())
        out["dirty_upper_third"] = float(dirty[hi_m].mean())
    return out


def check(metrics: dict[str, Any], style: Style) -> list[dict[str, Any]]:
    """Compare texture measurements with a style's limits. Returns failure dicts (code, detail, direction)."""
    if metrics.get("ok") is False:
        return [{"code": "texture_missing", "detail": metrics.get("reason", "no texture"), "direction": None}]
    fails: list[dict[str, Any]] = []
    if metrics["mean_saturation"] > style.max_saturation + 0.02:
        fails.append({"code": "texture_saturation", "direction": None,
                      "detail": f"mean saturation {metrics['mean_saturation']:.2f} > {style.max_saturation:.2f}"})
    lo, hi = style.value_range
    if metrics["value_p02"] < lo - 0.03 or metrics["value_p98"] > hi + 0.05:
        fails.append({"code": "texture_value_range", "direction": None,
                      "detail": f"value {metrics['value_p02']:.2f}..{metrics['value_p98']:.2f} outside {lo}..{hi}"})
    wear: list[tuple[str, str]] = []
    overpowered = False   # rot swallowing the surface also hides soil and clean metal, so "less wear" is the cure

    def rng(name: str, val: float, lohi: tuple[float, float]) -> None:
        nonlocal overpowered
        if val < lohi[0]:
            wear.append((f"{name} {val:.2f} < {lohi[0]:.2f}", "more"))
        elif val > lohi[1]:
            wear.append((f"{name} {val:.2f} > {lohi[1]:.2f}", "less"))
            overpowered = overpowered or name == "rot"

    if "grime" in style.coverage:
        rng("soil (grime+rust+blood)", metrics["soil_fraction"], style.coverage["grime"])
    rng("rot", metrics["rot_fraction"], style.rot_fraction)
    if metrics["clean_fraction"] < style.min_clean_fraction:
        wear.append((f"clean {metrics['clean_fraction']:.2f} < {style.min_clean_fraction:.2f}", "less"))
        overpowered = True
    if wear:
        dirs = {"less"} if overpowered else {d for _, d in wear}
        fails.append({"code": "texture_wear_coverage", "direction": dirs.pop() if len(dirs) == 1 else "conflict",
                      "detail": "; ".join(t for t, _ in wear)})
    return fails
