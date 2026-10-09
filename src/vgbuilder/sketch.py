"""Sketch analysis and silhouette math. Pure numpy/PIL, no Blender needed.

The same functions are used on the *sketch* and on the *rendered silhouette of the
built character*, so the two are always measured the same way. That is what makes the
render-compare-fix loop deterministic and testable without a GPU or an LLM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROWS = 48  # vertical resolution of a width profile
SIL_PX = 256  # silhouette raster height used by BOTH the fitter and the QA check (they must measure alike)

# Row bands as fractions of figure height measured from the top (0 = top of head).
BANDS: dict[str, tuple[float, float]] = {
    "head": (0.00, 0.09),
    "shoulder": (0.15, 0.25),
    "waist": (0.36, 0.45),
    "hip": (0.47, 0.55),
    "leg": (0.70, 0.92),
}


# --------------------------------------------------------------------------- masks
def load_mask(path: str | Path) -> np.ndarray:
    """Return a filled boolean silhouette mask from a sketch image.

    Handles transparent PNGs (alpha), dark-on-light line art and filled shapes.
    Outlines are closed and filled so a line drawing becomes a solid silhouette.
    """
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") and np.asarray(im.getchannel("A")).min() < 250:
        fg = np.asarray(im.getchannel("A")) > 128
    else:
        g = np.asarray(im.convert("L")).astype(np.int16)
        corners = np.array([g[0, 0], g[0, -1], g[-1, 0], g[-1, -1]])
        bg = int(np.median(corners))
        fg = np.abs(g - bg) > 40
    return fill_holes(fg)


def load_side_mask(path: str | Path, facing: str = "left") -> np.ndarray:
    """Side-view silhouette, normalised so the character always faces screen-left (toward -Y).

    The side raster uses screen-right = +Y, so a sketch of a character facing right is mirrored.
    """
    m = load_mask(path)
    return m[:, ::-1].copy() if facing == "right" else m


def fill_holes(fg: np.ndarray, close_px: int = 3) -> np.ndarray:
    img = Image.fromarray((fg * 255).astype(np.uint8), "L")
    for _ in range(close_px):
        img = img.filter(ImageFilter.MaxFilter(3))
    padded = Image.new("L", (img.width + 2, img.height + 2), 0)
    padded.paste(img, (1, 1))
    ImageDraw.floodfill(padded, (0, 0), 128, thresh=0)
    arr = np.asarray(padded)[1:-1, 1:-1]
    filled = Image.fromarray(((arr != 128) * 255).astype(np.uint8), "L")
    for _ in range(close_px):
        filled = filled.filter(ImageFilter.MinFilter(3))
    return np.asarray(filled) > 127


def crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("empty silhouette mask")
    return mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


# ------------------------------------------------------------------------ profiles
def width_profile(mask: np.ndarray, rows: int = ROWS) -> np.ndarray:
    """Filled width per row, in units of figure height. Index 0 is the top."""
    m = crop_to_bbox(mask)
    h = m.shape[0]
    out = np.zeros(rows)
    for i in range(rows):
        y0 = int(i * h / rows)
        y1 = max(y0 + 1, int((i + 1) * h / rows))
        out[i] = m[y0:y1].sum(axis=1).mean() / h
    return out


def band_stat(profile: np.ndarray, band: str, how: str = "mean") -> float:
    lo, hi = BANDS[band]
    i0, i1 = int(lo * len(profile)), max(int(lo * len(profile)) + 1, int(hi * len(profile)))
    seg = profile[i0:i1]
    return float(seg.max() if how == "max" else seg.mean())


def crotch_fraction(mask: np.ndarray) -> float | None:
    """Height fraction (from the floor) where the legs split, or None if legs touch.

    Looks at the vertical midline in the lower 65% of the figure and takes the longest run of
    background rows: that is the gap between the legs, and its top is the crotch. Arm gaps are
    off-centre so they don't confuse it, and feet that touch at the bottom don't hide the gap.
    """
    m = crop_to_bbox(mask)
    h, w = m.shape
    cx = w // 2
    col = m[:, max(0, cx - 1): cx + 2].any(axis=1)
    start = int(h * 0.35)
    best_len, best_top, run_top = 0, None, None
    for y in range(start, h + 1):
        bg = y < h and not col[y]
        if bg and run_top is None:
            run_top = y
        if not bg and run_top is not None:
            if y - run_top > best_len:
                best_len, best_top = y - run_top, run_top
            run_top = None
    if best_top is None or best_len < 0.03 * h:
        return None
    frac = 1.0 - best_top / h
    return float(frac) if 0.25 < frac < 0.65 else None


def fit_params(mask: np.ndarray) -> dict[str, float]:
    """First-guess body parameters from a front-view silhouette."""
    p = width_profile(mask)
    out: dict[str, float] = {}
    head_w = band_stat(p, "head", "max")
    out["head_scale"] = float(np.clip(head_w / (2 * 0.065), 0.6, 1.8))
    out["shoulder_half"] = float(np.clip((band_stat(p, "shoulder", "max") - 0.06) / 2, 0.07, 0.2))
    out["waist_half"] = float(np.clip((band_stat(p, "waist") - 0.06) / 2, 0.05, 0.16))
    out["hip_half"] = float(np.clip((band_stat(p, "hip") - 0.06) / 2, 0.06, 0.18))
    cf = crotch_fraction(mask)
    if cf is not None:
        out["leg_frac"] = float(np.clip(cf, 0.35, 0.58))
    return out


def fit_params_side(mask: np.ndarray) -> dict[str, float]:
    """First-guess depth parameters from a side-view silhouette (character facing left)."""
    p = width_profile(mask)
    depth = band_stat(p, "waist")            # front-back thickness at waist height, in figure heights
    return {"torso_depth": float(np.clip(depth / (2 * 0.054), 0.5, 2.2))}


# --------------------------------------------------------------------- comparison
def _normalize(mask: np.ndarray, height_px: int = 256, width_px: int = 256) -> np.ndarray:
    m = crop_to_bbox(mask)
    h, w = m.shape
    new_w = max(1, int(round(w * height_px / h)))
    im = Image.fromarray((m * 255).astype(np.uint8), "L").resize((new_w, height_px), Image.NEAREST)
    canvas = Image.new("L", (max(width_px, new_w + 2), height_px), 0)
    canvas.paste(im, ((canvas.width - new_w) // 2, 0))
    return np.asarray(canvas) > 127


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU after both silhouettes are cropped, scaled to equal height and centred."""
    na, nb = _normalize(a), _normalize(b)
    w = max(na.shape[1], nb.shape[1])

    def pad(x: np.ndarray) -> np.ndarray:
        out = np.zeros((x.shape[0], w), bool)
        o = (w - x.shape[1]) // 2
        out[:, o: o + x.shape[1]] = x
        return out

    pa, pb = pad(na), pad(nb)
    inter = np.logical_and(pa, pb).sum()
    union = np.logical_or(pa, pb).sum()
    return float(inter / union) if union else 0.0


# ------------------------------------------------------------- mesh -> silhouette
def rasterize_triangles(tris_xz: np.ndarray, height_px: int = SIL_PX, margin: int = 4) -> np.ndarray:
    """Orthographic silhouette of triangles given as (N, 3, 2) arrays of (horizontal, z).

    Front view passes (x, z); side view passes (y, z), which puts the character's front (-Y) on the left."""
    pts = tris_xz.reshape(-1, 2)
    xmin, zmin = pts.min(axis=0)
    xmax, zmax = pts.max(axis=0)
    scale = (height_px - 2 * margin) / max(zmax - zmin, 1e-9)
    width_px = int((xmax - xmin) * scale) + 2 * margin
    img = Image.new("L", (width_px, height_px), 0)
    d = ImageDraw.Draw(img)
    for t in tris_xz:
        poly = [((x - xmin) * scale + margin, (zmax - z) * scale + margin) for x, z in t]
        d.polygon(poly, fill=255)
    return np.asarray(img) > 127
