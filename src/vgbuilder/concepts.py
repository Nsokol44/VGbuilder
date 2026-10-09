"""Concept stage: generate silhouette variants, score them with numbers, keep a diverse few, emit jobs.

Everything is parametric, no image model involved. A *variant* is a body-parameter set plus an accessory
set and accessory shape numbers. Its silhouette comes from the same numpy body/accessory meshes the
build uses, so a winner turns into a job (`params:` pinned, `accessories:`, `accessory_shape:`) that
rebuilds exactly that silhouette. What is scored is image-space geometry, so the same scorers apply to a
silhouette that came from somewhere else (an artist, a diffusion model) if you load it as a mask:

* solidity   - area / convex-hull area. Low = ragged and noisy, high = a blob.
* lobes      - separate protrusions beyond the base body (horns, pauldrons, cape wings), by connected components.
* limb separation - share of arm-height rows where arms are clear of the torso (readable in combat).
* uniqueness - 1 - best IoU against the other chosen variants and any silhouettes already in your library.

The style template's `silhouette:` block supplies the thresholds. "Looks good" is NOT measured here: these
numbers reject the obviously bad (shapeless, indistinguishable, unreadable) and keep the set diverse.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from . import __version__, accessories as acc
from .bodygen import build_body
from .sketch import SIL_PX, load_mask, silhouette_iou
from .spec import DEFAULT_BODY_PARAMS, load_job
from .styles import Style, load_style

RANGES: dict[str, tuple[float, float]] = {
    "head_scale": (0.8, 1.3), "shoulder_half": (0.09, 0.15), "waist_half": (0.06, 0.11), "hip_half": (0.07, 0.13),
    "leg_frac": (0.42, 0.52), "arm_frac": (0.34, 0.44), "arm_thickness": (0.8, 1.4), "leg_thickness": (0.8, 1.4),
    "torso_depth": (0.8, 1.3), "arm_angle_deg": (8.0, 38.0),
}
CHANCE = {"pauldrons": 0.7, "horns": 0.5, "cape": 0.6}
CONCEPT_KEYS = {"name", "style_template", "count", "select", "seed", "accessories_pool", "job", "exclude_dir", "max_limb_overlap"}


# --------------------------------------------------------------- silhouettes
@dataclass
class Variant:
    id: str
    params: dict[str, float]
    accessories: list[str]
    shape: dict[str, float]
    front: np.ndarray = field(repr=False, default=None)        # full silhouette (body + accessories)
    side: np.ndarray = field(repr=False, default=None)
    base_front: np.ndarray = field(repr=False, default=None)   # body only
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    selected: bool = False


def sample_variant(rng: np.random.Generator, vid: str, pool: list[str]) -> Variant:
    p = dict(DEFAULT_BODY_PARAMS)
    for k, (lo, hi) in RANGES.items():
        p[k] = round(float(rng.uniform(lo, hi)), 4)
    names = [a for a in ("pauldrons", "horns", "cape") if a in pool and rng.random() < CHANCE[a]]
    shape = {
        "pauldron_scale": round(float(rng.uniform(0.8, 1.9)), 3),
        "horn_len": round(float(rng.uniform(0.6, 2.2)), 3),
        "horn_spread": round(float(rng.uniform(0.6, 1.8)), 3),
        "cape_len": round(float(rng.uniform(0.6, 1.5)), 3),
        "cape_flare": round(float(rng.uniform(0.0, 1.2)), 3),
    }
    return Variant(vid, p, names, shape)


def render(v: Variant, height: float, segments: int = 8) -> None:
    """Fill v.front / v.side / v.base_front from the numpy meshes (orthographic, 256 px)."""
    q = {**v.params, "segments": float(segments)}
    body = build_body(q, height)
    tris = [body.verts[body.triangles()]]
    base = np.concatenate(tris)
    for part in acc.build_parts(v.accessories, q, height, seg=segments, shape=v.shape):
        tris.append(part.body.verts[part.body.triangles()])
    allt = np.concatenate(tris)

    # one shared frame (full figure) so base and full line up pixel for pixel
    lo, hi = allt[:, :, [0, 2]].reshape(-1, 2).min(axis=0), allt[:, :, [0, 2]].reshape(-1, 2).max(axis=0)
    v.front = _raster_in_frame(allt[:, :, [0, 2]], lo, hi)
    v.base_front = _raster_in_frame(base[:, :, [0, 2]], lo, hi)
    lo2, hi2 = allt[:, :, [1, 2]].reshape(-1, 2).min(axis=0), allt[:, :, [1, 2]].reshape(-1, 2).max(axis=0)
    v.side = _raster_in_frame(allt[:, :, [1, 2]], lo2, hi2)


def _raster_in_frame(tris: np.ndarray, lo: np.ndarray, hi: np.ndarray, height_px: int = SIL_PX, margin: int = 4) -> np.ndarray:
    scale = (height_px - 2 * margin) / max(hi[1] - lo[1], 1e-9)
    w = int((hi[0] - lo[0]) * scale) + 2 * margin
    img = Image.new("L", (w, height_px), 0)
    d = ImageDraw.Draw(img)
    for t in tris:
        d.polygon([((x - lo[0]) * scale + margin, (hi[1] - z) * scale + margin) for x, z in t], fill=255)
    return np.asarray(img) > 127


# ------------------------------------------------------------------ scorers
def _hull_area(mask: np.ndarray) -> float:
    ys = np.nonzero(mask.any(axis=1))[0]
    pts = []
    for y in ys:
        xs = np.nonzero(mask[y])[0]
        pts += [(xs.min(), y), (xs.max() + 1, y), (xs.min(), y + 1), (xs.max() + 1, y + 1)]
    pts = sorted(set(pts))

    def half(points):
        h: list[tuple[int, int]] = []
        for p in points:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h

    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    x, y = np.array([p[0] for p in hull], float), np.array([p[1] for p in hull], float)
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def solidity(mask: np.ndarray) -> float:
    return float(mask.sum() / max(_hull_area(mask), 1.0))


def components(mask: np.ndarray) -> list[int]:
    """Sizes of 8-connected components (small masks only: used on the thin leftover of full minus base)."""
    seen = np.zeros_like(mask, bool)
    h, w = mask.shape
    sizes = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack, n = [(y, x)], 0
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            n += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        sizes.append(n)
    return sizes


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        o = out.copy()
        o[1:] |= out[:-1]; o[:-1] |= out[1:]; o[:, 1:] |= out[:, :-1]; o[:, :-1] |= out[:, 1:]
        out = o
    return out


def lobes(full: np.ndarray, base: np.ndarray, min_frac: float = 0.004) -> int:
    """Protrusions beyond the base body: components of (full minus a slightly grown base) of meaningful size."""
    extra = full & ~_dilate(base, 2)
    return sum(1 for n in components(extra) if n >= min_frac * full.sum())


def limb_separation(mask: np.ndarray, band: tuple[float, float] = (0.28, 0.50)) -> float:
    """Share of rows in the arm band (fractions from the top) where arms and torso are separate runs (>= 3 runs)."""
    h = mask.shape[0]
    rows = range(int(band[0] * h), int(band[1] * h))
    ok = 0
    for y in rows:
        r = mask[y].astype(np.int8)
        runs = int(((r[1:] - r[:-1]) == 1).sum() + r[0])
        ok += runs >= 3
    return ok / max(len(rows), 1)


def score_variant(v: Variant, style: Style, others: list[np.ndarray]) -> None:
    rules = style.silhouette
    m = v.metrics
    m["solidity"] = round(solidity(v.front), 4)
    m["lobes"] = lobes(v.front, v.base_front)
    m["limb_separation"] = round(limb_separation(v.front), 4)
    best = max((silhouette_iou(v.front, o) for o in others), default=0.0)
    m["uniqueness"] = round(1.0 - best, 4)
    lo, hi = rules.get("solidity", (0.0, 1.0))
    if not lo <= m["solidity"] <= hi:
        v.failures.append(f"solidity {m['solidity']:.2f} outside {lo}..{hi}")
    if m["lobes"] < rules.get("min_lobes", 0):
        v.failures.append(f"{m['lobes']} lobes < {rules['min_lobes']}")
    if m["limb_separation"] < rules.get("min_limb_separation", 0.0):
        v.failures.append(f"limb separation {m['limb_separation']:.2f} < {rules['min_limb_separation']}")
    if m["uniqueness"] < rules.get("min_uniqueness", 0.0):
        v.failures.append(f"uniqueness {m['uniqueness']:.2f} < {rules['min_uniqueness']}")


def select_diverse(cands: list[Variant], k: int, style: Style, library: list[np.ndarray]) -> list[Variant]:
    """Greedy: highest-scoring valid variant first, then repeatedly the valid one furthest from everything chosen."""
    valid = [c for c in cands if not [f for f in c.failures if not f.startswith("uniqueness")]]
    chosen: list[Variant] = []
    min_u = style.silhouette.get("min_uniqueness", 0.0)
    pool = list(valid)
    while pool and len(chosen) < k:
        refs = library + [c.front for c in chosen]
        scored = []
        for c in pool:
            u = 1.0 - max((silhouette_iou(c.front, r) for r in refs), default=0.0)
            scored.append((u + 0.15 * c.metrics["limb_separation"] + 0.05 * min(c.metrics["lobes"], 4), u, c))
        scored.sort(key=lambda t: -t[0])
        total, u, best = scored[0]
        if u < min_u:
            break                                  # nothing left that is different enough
        best.metrics["uniqueness"] = round(u, 4)
        best.selected = True
        chosen.append(best)
        pool.remove(best)
    return chosen


# --------------------------------------------------------------------- run
def load_concept(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    extra = set(raw) - CONCEPT_KEYS
    if extra:
        raise ValueError(f"{path}: unknown keys {sorted(extra)}")
    raw.setdefault("count", 40)
    raw.setdefault("select", 4)
    raw.setdefault("seed", 0)
    raw.setdefault("accessories_pool", ["pauldrons", "horns", "cape"])
    raw.setdefault("job", {})
    if "name" not in raw or "style_template" not in raw:
        raise ValueError("concept file needs name and style_template")
    if not set(raw["accessories_pool"]) <= set(acc.JOINED) | set(acc.SEPARATE):
        raise ValueError("accessories_pool has unknown entries")
    if not (1 <= raw["select"] <= raw["count"] <= 500):
        raise ValueError("need 1 <= select <= count <= 500")
    return raw


def _png(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.where(mask, 20, 245).astype(np.uint8), "L").save(path)


def _sheet(cands: list[Variant], path: Path, cols: int = 8, cell: int = 128) -> None:
    rows = -(-len(cands) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), (30, 30, 34))
    d = ImageDraw.Draw(sheet)
    for i, v in enumerate(cands):
        x, y = (i % cols) * cell, (i // cols) * (cell + 16)
        im = Image.fromarray(np.where(v.front, 20, 235).astype(np.uint8), "L").convert("RGB")
        im.thumbnail((cell - 8, cell - 8))
        tint = (60, 140, 70) if v.selected else (150, 60, 60) if v.failures else (90, 90, 96)
        d.rectangle([x, y, x + cell - 1, y + cell + 15], outline=tint, width=3 if v.selected else 1)
        sheet.paste(im, (x + (cell - im.width) // 2, y + 4))
        d.text((x + 4, y + cell), f"{v.id} u{v.metrics.get('uniqueness', 0):.2f} L{v.metrics.get('lobes', 0)}", fill=tint)
    sheet.save(path)


def run_concepts(path: Path, out_root: Path, log=print) -> dict[str, Any]:
    path = Path(path)
    cfg = load_concept(path)
    style = load_style(cfg["style_template"], [path.parent / "styles"])
    job_defaults = dict(cfg["job"])
    height = float(job_defaults.get("height_m", 1.8))
    out = out_root / "concepts" / cfg["name"]
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    (out / "silhouettes").mkdir(exist_ok=True)

    library: list[np.ndarray] = []
    if cfg.get("exclude_dir"):
        d = (path.parent / cfg["exclude_dir"]).resolve()
        library = [load_mask(p) for p in sorted(d.glob("*.png"))]

    rng = np.random.default_rng(int(cfg["seed"]))
    cands = [sample_variant(rng, f"c{i + 1:02d}", cfg["accessories_pool"]) for i in range(cfg["count"])]
    for c in cands:
        render(c, height)
        score_variant(c, style, library)     # uniqueness vs library only; vs each other is decided at selection
    winners = select_diverse(cands, cfg["select"], style, library)

    _sheet(cands, out / "sheet.png")
    emitted = []
    for w in winners:
        base = f"{cfg['name']}_{w.id}"
        _png(w.front, out / "silhouettes" / f"{base}_front.png")
        _png(w.side, out / "silhouettes" / f"{base}_side.png")
        job = {
            "name": base,
            **{k: v for k, v in job_defaults.items() if k != "height_m"},
            "height_m": height,
            "style_template": cfg["style_template"],
            "provider": "procedural",
            "params": {k: v for k, v in w.params.items()},
            "accessories": w.accessories,
            "accessory_shape": w.shape if w.accessories else {},
            "tags": ["concept", cfg["name"]],
            "notes": f"concept {w.id}: " + ", ".join(f"{k}={v}" for k, v in w.metrics.items()),
        }
        if not w.accessories:
            job.pop("accessory_shape")
        jp = out / "jobs" / f"{base}.yaml"
        jp.write_text(yaml.safe_dump(job, sort_keys=False))
        load_job(jp)                         # every emitted job must validate
        emitted.append(str(jp))

    manifest = {
        "vgbuilder_version": __version__, "kind": "concepts", "name": cfg["name"],
        "fingerprint": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16],
        "config": cfg, "style": style.name, "library_silhouettes": len(library),
        "candidates": [{"id": c.id, "params": c.params, "accessories": c.accessories, "shape": c.shape,
                        "metrics": c.metrics, "failures": c.failures, "selected": c.selected} for c in cands],
        "selected": [w.id for w in winners], "jobs": emitted, "sheet": str(out / "sheet.png"),
        "rejected_by_rule": sum(1 for c in cands if c.failures),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"[{cfg['name']}] {len(cands)} candidates, {manifest['rejected_by_rule']} failed a rule, {len(winners)} selected")
    return manifest
