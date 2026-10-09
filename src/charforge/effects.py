"""Effects (fireball, explosion, shockwave, smoke) as flipbook sprite sheets. Pure numpy, no Blender.

This is NOT a Geometry Nodes / simulation path. Each template is a small procedural field
(radial profile + 3D value noise over x, y, time) rendered frame by frame into an RGBA flipbook,
plus an engine-neutral particle preset and a review GIF. It reuses the same job/QA/fix/manifest
loop as characters:

    spec -> render frames -> measure the frames (image space) -> failures -> adjust params -> re-render

What QA measures is on the rendered frames, not on the parameters: is anything visible, is it cut off
by the frame edge, does it fade out instead of popping, does the timing curve look right for the
template, does it stay inside the style's saturation limit.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from . import __version__
from .lookdev import _clamp_look, _pick, fbm, rgb_to_hsv, smoothstep
from .spec import JobSpec
from .styles import hex_to_rgb

TEMPLATES: dict[str, dict[str, Any]] = {
    "fireball": {
        "radius": 0.60, "turbulence": 0.55, "fade": 0.95, "fade_start": 0.55, "blend": "additive",
        "ramp": ["#1a0500", "#8a1c00", "#e65a00", "#ffc040", "#fff4d0"], "peak": (0.2, 0.8),
    },
    "explosion": {
        "radius": 0.75, "turbulence": 0.6, "fade": 1.0, "fade_start": 0.2, "blend": "additive",
        "ramp": ["#120a06", "#5a1a05", "#d04a00", "#ffb030", "#ffffff"], "peak": (0.0, 0.5),
    },
    "shockwave": {
        "radius": 0.85, "turbulence": 0.15, "fade": 1.0, "fade_start": 0.35, "blend": "additive",
        "ramp": ["#5a6a7a", "#b9c8d8", "#ffffff"], "peak": (0.0, 0.6),
    },
    "smoke": {
        "radius": 0.55, "turbulence": 0.7, "fade": 0.95, "fade_start": 0.5, "blend": "alpha",
        "ramp": ["#1c1c1c", "#4a4a4a", "#8c8c8c"], "peak": (0.15, 0.75),
    },
}
EFFECT_KEYS = {"template", "size", "frames", "duration_s", "radius", "turbulence", "fade", "fade_start",
               "ramp", "blend", "max_sheet_px"}
TUNABLE = ("radius", "fade", "fade_start")        # what the fixer may change between iterations
LIMITS = {"radius": (0.15, 0.95), "fade": (0.0, 1.0), "fade_start": (0.15, 0.9), "turbulence": (0.0, 1.5)}


def validate_effect(effect: dict[str, Any]) -> None:
    bad = set(effect) - EFFECT_KEYS
    if bad:
        raise ValueError(f"unknown effect keys: {sorted(bad)}")
    if not (16 <= int(effect.get("size", 128)) <= 512):
        raise ValueError("effect.size must be in [16, 512]")
    if not (4 <= int(effect.get("frames", 16)) <= 64):
        raise ValueError("effect.frames must be in [4, 64]")
    if not (0.1 <= float(effect.get("duration_s", 0.8)) <= 10):
        raise ValueError("effect.duration_s must be in [0.1, 10]")
    for k, (lo, hi) in LIMITS.items():
        if k in effect and not (lo <= float(effect[k]) <= hi):
            raise ValueError(f"effect.{k} must be in [{lo}, {hi}]")
    if effect.get("blend", "additive") not in ("additive", "alpha"):
        raise ValueError("effect.blend must be additive or alpha")
    for c in effect.get("ramp", []):
        hex_to_rgb(c)
    if "ramp" in effect and len(effect["ramp"]) < 2:
        raise ValueError("effect.ramp needs at least two colours")


def resolved(spec: JobSpec) -> dict[str, Any]:
    """Template defaults + job overrides."""
    e = spec.effect
    d = {**TEMPLATES[e["template"]]}
    d.update({k: v for k, v in e.items() if k != "template"})
    d["template"] = e["template"]
    d.setdefault("size", 128)
    d.setdefault("frames", 16)
    d.setdefault("duration_s", 0.8)
    d.setdefault("max_sheet_px", 2048)
    return d


# ------------------------------------------------------------------ rendering
def _ramp_colors(cfg: dict[str, Any], style) -> np.ndarray:
    cols = np.array([hex_to_rgb(h) for h in cfg["ramp"]])
    return _clamp_look(cols, style) if style is not None else cols


def render_frames(cfg: dict[str, Any], params: dict[str, float], seed: int, style=None) -> np.ndarray:
    """(frames, size, size, 4) float32 RGBA, straight alpha, y up."""
    S, N = int(cfg["size"]), int(cfg["frames"])
    xs = (np.arange(S) + 0.5) / S * 2 - 1
    X, Y = np.meshgrid(xs, -xs)                 # row 0 is the top of the image
    ramp = _ramp_colors(cfg, style)
    tpl = cfg["template"]
    R0, turb, fade, f0 = params["radius"], params["turbulence"], params["fade"], params["fade_start"]
    out = np.zeros((N, S, S, 4), np.float32)
    seed = seed + {"fireball": 11, "explosion": 23, "shockwave": 37, "smoke": 51}[tpl]

    def noise(x, y, t, scale, octaves=3, off=0):
        p = np.stack([x.ravel() * scale, y.ravel() * scale, np.full(x.size, t)], axis=1) + off
        return fbm(p, seed + off, octaves).reshape(x.shape)

    for k in range(N):
        u = k / (N - 1)
        env = 1.0 - fade * float(smoothstep(f0, 1.0, np.asarray(u)))     # fade-out
        if tpl == "fireball":
            cy = -0.25 + 0.4 * u
            grow = math.sin(math.pi / 2 * min(u / 0.6, 1.0))
            R = R0 * (0.35 + 0.65 * grow) * (1 - 0.35 * max(0.0, (u - 0.6) / 0.4))
            r = np.hypot(X, Y - cy)
            n = noise(X, Y - u * 1.4, u * 1.6, 2.2 + 2.0 * turb)
            d = smoothstep(0.0, 1.0, (R - r + turb * 0.9 * (n - 0.5) * R) / (0.38 * R + 1e-6))
            T = np.clip(d * (0.55 + 0.9 * n) * (0.5 + 0.5 * env), 0, 1)
            a = np.clip(d * env * 1.15, 0, 1)
        elif tpl == "explosion":
            ease = 1 - (1 - u) ** 3.5
            R = R0 * (0.12 + 0.88 * ease ** 0.7)
            r = np.hypot(X, Y)
            n = noise(X, Y, u * 2.0, 2.0 + 2.5 * turb)
            body = smoothstep(0.0, 1.0, (R - r + turb * 0.8 * (n - 0.5) * R) / (0.30 * R + 1e-6))
            core = smoothstep(0.0, 1.0, 1 - r / (R * 0.75 + 1e-6))
            flash = max(0.0, 1.0 - u / 0.18)
            T = np.clip((0.35 + 0.65 * core) * (1 - 0.75 * u) * (0.6 + 0.8 * n) + flash * core, 0, 1)
            a = np.clip(body * env * 1.2, 0, 1)
        elif tpl == "shockwave":
            R = R0 * (1 - (1 - u) ** 2.2)
            w = 0.05 + 0.11 * (1 - u)
            r = np.hypot(X, Y)
            n = noise(X, Y, u, 3.0, 2)
            rr = r + turb * 0.12 * (n - 0.5)
            g = np.exp(-(((rr - R) / w) ** 2))
            T = np.clip(g * (0.5 + 0.5 * (1 - u)), 0, 1)
            a = np.clip(g * env, 0, 1)
        else:  # smoke
            width = 0.18 + 0.55 * R0 * (0.35 + 0.65 * u)
            cy = -0.55 + 0.95 * u * R0 / 0.55
            n = noise(X * 1.0, Y - u * 1.1, u * 0.9, 1.8 + 1.6 * turb, 4)
            col = smoothstep(1.0, 0.25, np.abs(X) / width) * smoothstep(-1.0, cy + 0.05, Y) * smoothstep(1.0, cy + 0.9, Y)
            d = smoothstep(0.30, 0.72, n * (0.6 + 0.8 * col) - (1 - col) * 0.5) * col
            rise = float(smoothstep(0.0, 0.2, np.asarray(u)))
            T = np.clip(0.25 + 0.75 * n, 0, 1)
            a = np.clip(d * 0.85 * env * rise, 0, 1)
        rgb = _pick(list(map(tuple, ramp)), T)
        out[k, ..., :3] = np.where(a[..., None] > 1e-3, rgb, 0.0)
        out[k, ..., 3] = a
    return out


# ------------------------------------------------------------------ analysis
def analyze_frames(frames: np.ndarray, cfg: dict[str, Any], columns: int, rows: int) -> dict[str, Any]:
    N, S = frames.shape[0], frames.shape[1]
    a = frames[..., 3]
    xs = (np.arange(S) + 0.5) / S * 2 - 1
    X, Y = np.meshgrid(xs, -xs)
    R = np.hypot(X, Y)
    mean_a = a.reshape(N, -1).mean(axis=1)
    cover = (a > 0.05).reshape(N, -1).mean(axis=1)
    border = np.concatenate([a[:, 0, :], a[:, -1, :], a[:, :, 0], a[:, :, -1]], axis=1)
    w = a.reshape(N, -1)
    spread = np.sqrt((w * (R ** 2).ravel()).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9))
    diff = np.abs(np.diff(a, axis=0)).reshape(N - 1, -1).mean(axis=1)
    px = frames[a > 0.05]
    sat = float(rgb_to_hsv(np.clip(px[:, :3], 0, 1))[:, 1].mean()) if len(px) else 0.0
    peak = int(np.argmax(mean_a))
    return {
        "frames": N, "frame_px": S, "columns": columns, "rows": rows, "sheet_px": [columns * S, rows * S],
        "mean_alpha": [round(float(v), 5) for v in mean_a],
        "coverage": [round(float(v), 4) for v in cover],
        "spread": [round(float(v), 4) for v in spread],
        "peak_frame": peak, "peak_fraction": peak / (N - 1),
        "max_coverage": float(cover.max()),
        "first_over_peak": float(mean_a[0] / max(mean_a.max(), 1e-9)),
        "last_over_peak": float(mean_a[-1] / max(mean_a.max(), 1e-9)),
        "border_alpha_max": float(border.max()),
        "max_jump": float(diff.max() / max(diff.mean(), 1e-9)),
        "mean_saturation": sat,
    }


def check(m: dict[str, Any], cfg: dict[str, Any], style=None) -> list[dict[str, Any]]:
    fails: list[dict[str, Any]] = []

    def fail(code: str, detail: str, fix: str | None = None) -> None:
        fails.append({"code": code, "detail": detail, "fixable": fix is not None, "fix": fix})

    if m["max_coverage"] < 0.03:
        fail("effect_empty", f"peak coverage {m['max_coverage']:.3f} < 0.03 (nothing visible)", "grow")
    if m["border_alpha_max"] > 0.08:
        fail("effect_clipped", f"alpha {m['border_alpha_max']:.2f} reaches the frame border, the effect is cut off", "shrink")
    if m["last_over_peak"] > 0.25:
        fail("effect_no_fade", f"last frame is {m['last_over_peak']:.0%} of peak brightness, it will pop out", "fade")
    lo, hi = TEMPLATES[cfg["template"]]["peak"]
    if not (lo <= m["peak_fraction"] <= hi):
        fail("effect_timing", f"peak at {m['peak_fraction']:.2f} of the duration, expected {lo}..{hi}")
    if cfg["template"] == "shockwave":
        # frames that have faded to nothing carry no ring, so they say nothing about expansion
        keep = [v for v, a in zip(m["spread"], m["mean_alpha"]) if a > 0.02 * max(m["mean_alpha"])]
        sp = keep if len(keep) >= 3 else m["spread"]
        if any(b < a - 0.02 for a, b in zip(sp, sp[1:])) or sp[-1] < 1.5 * max(sp[0], 1e-3):
            fail("effect_timing", "shockwave ring does not expand steadily")
    if m["max_jump"] > 5.0:
        fail("effect_pop", f"one frame changes {m['max_jump']:.1f}x more than average (visible pop)")
    if style is not None and m["mean_saturation"] > style.max_saturation + 0.03:
        fail("effect_saturation", f"mean saturation {m['mean_saturation']:.2f} > style {style.max_saturation:.2f}")
    if max(m["sheet_px"]) > int(cfg["max_sheet_px"]):
        fail("effect_too_big", f"sheet {m['sheet_px']} exceeds {cfg['max_sheet_px']}px")
    return fails


def adjust(params: dict[str, float], fails: list[dict[str, Any]]) -> tuple[dict[str, float], list[str]]:
    new, changes = dict(params), []

    def setp(k: str, v: float) -> None:
        lo, hi = LIMITS[k]
        v = float(min(hi, max(lo, v)))
        if abs(v - new[k]) > 1e-9:
            changes.append(f"{k} {new[k]:.3g} -> {v:.3g}")
            new[k] = round(v, 4)

    fixes = {f["fix"] for f in fails if f.get("fix")}
    if "shrink" in fixes:
        setp("radius", new["radius"] * 0.85)
    elif "grow" in fixes:
        setp("radius", new["radius"] * 1.25)
    if "fade" in fixes:
        if new["fade"] < 1.0:
            setp("fade", 1.0)
        else:
            setp("fade_start", new["fade_start"] - 0.1)
    return new, changes


# --------------------------------------------------------------------- output
def sheet_layout(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    return cols, math.ceil(n / cols)


def pack_sheet(frames: np.ndarray, columns: int, rows: int) -> np.ndarray:
    N, S = frames.shape[0], frames.shape[1]
    sheet = np.zeros((rows * S, columns * S, 4), np.float32)
    for k in range(N):
        r, c = divmod(k, columns)
        sheet[r * S:(r + 1) * S, c * S:(c + 1) * S] = frames[k]
    return sheet


def to_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(path)


def composite(frame: np.ndarray, blend: str) -> np.ndarray:
    """RGB preview of one frame on a dark checker, the way an engine would show it."""
    S = frame.shape[0]
    chk = (((np.arange(S)[:, None] // 8) + (np.arange(S)[None, :] // 8)) % 2) * 0.04 + 0.10
    bg = np.repeat(chk[..., None], 3, axis=2)
    a = frame[..., 3:4]
    rgb = bg + frame[..., :3] * a if blend == "additive" else bg * (1 - a) + frame[..., :3] * a
    return np.clip(rgb, 0, 1)


def preview_files(frames: np.ndarray, cfg: dict[str, Any], out_dir: Path, name: str) -> list[Path]:
    comp = [composite(f, cfg["blend"]) for f in frames]
    strip = np.concatenate(comp, axis=1)
    contact = out_dir / f"{name}_contact_sheet.png"
    Image.fromarray((strip * 255 + 0.5).astype(np.uint8), "RGB").save(contact)
    gif = out_dir / f"{name}_preview.gif"
    ims = [Image.fromarray((c * 255 + 0.5).astype(np.uint8), "RGB").resize((256, 256), Image.NEAREST) for c in comp]
    ims[0].save(gif, save_all=True, append_images=ims[1:], duration=max(20, int(cfg["duration_s"] * 1000 / len(ims))), loop=0)
    return [contact, gif]


def particle_preset(name: str, cfg: dict[str, Any], m: dict[str, Any], sheet_file: str, ramp_hex: list[str]) -> dict[str, Any]:
    """Engine-neutral description of the effect. The curves are measured from the rendered frames."""
    ma = np.array(m["mean_alpha"])
    sp = np.array(m["spread"])
    return {
        "schema": "charforge.vfx_preset/1", "name": name, "template": cfg["template"],
        "duration_s": cfg["duration_s"], "blend": cfg["blend"],
        "flipbook": {"texture": sheet_file, "columns": m["columns"], "rows": m["rows"], "frames": m["frames"],
                     "frame_size_px": m["frame_px"], "fps": round(m["frames"] / cfg["duration_s"], 3)},
        "curves": {
            "alpha_over_life": [round(float(v), 4) for v in ma / max(ma.max(), 1e-9)],
            "size_over_life": [round(float(v), 4) for v in sp / max(sp.max(), 1e-9)],
        },
        "color_ramp_cold_to_hot": ramp_hex,
        "emitter": {"burst_count": 1, "lifetime_s": cfg["duration_s"], "loop": False},
        "notes": "Neutral description, not an engine asset: map it onto Niagara / VFX Graph / GPUParticles by hand or with your own importer.",
    }


# ------------------------------------------------------------------- the job
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_effect(spec: JobSpec, out_root: Path, log: Callable[[str], None] = lambda s: None,
                 max_iterations: int | None = None) -> dict[str, Any]:
    """Run one effect job with the same render -> measure -> fix loop and manifest as characters."""
    t0 = time.time()
    out_dir = out_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = resolved(spec)
    style = spec.load_style()
    manifest: dict[str, Any] = {
        "charforge_version": __version__, "name": spec.name, "kind": "effect", "fingerprint": spec.fingerprint(),
        "spec": {k: v for k, v in spec.to_dict().items() if k != "base_dir"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "effect_resolved": {k: v for k, v in cfg.items()},
        "iterations": [], "status": "error",
    }
    if style is not None:
        manifest["style_resolved"] = style.to_dict()
    params = {k: float(cfg[k]) for k in ("radius", "turbulence", "fade", "fade_start")}
    columns, rows = sheet_layout(int(cfg["frames"]))
    best: tuple[float, int] | None = None
    iters = max_iterations or spec.max_iterations
    for i in range(1, iters + 1):
        log(f"[{spec.name}] iteration {i}")
        it_dir = out_dir / f"iter_{i}"
        it_dir.mkdir(exist_ok=True)
        frames = render_frames(cfg, params, spec.seed, style)
        m = analyze_frames(frames, cfg, columns, rows)
        fails = check(m, cfg, style)
        sheet = it_dir / f"{spec.name}_flipbook.png"
        to_png(pack_sheet(frames, columns, rows), sheet)
        ramp_hex = cfg["ramp"]
        preset = it_dir / f"{spec.name}_particles.json"
        preset.write_text(json.dumps(particle_preset(spec.name, cfg, m, sheet.name, ramp_hex), indent=2))
        files = [sheet, preset, *preview_files(frames, cfg, it_dir, spec.name)]
        score = 1.0 - 0.25 * len(fails)
        entry = {"iteration": i, "params": dict(params), "metrics": m, "failures": fails, "score": score,
                 "files": [str(f) for f in files]}
        if best is None or score > best[0]:
            best = (score, i)
        manifest["iterations"].append(entry)
        if not fails:
            manifest["status"] = "passed"
            break
        params, changes = adjust(params, fails)
        entry["next_changes"] = changes
        if not changes:
            manifest["status"] = "needs_review"
            break
    else:
        manifest["status"] = "needs_review"

    assert best is not None
    manifest["best_iteration"] = best[1]
    final = out_dir / "final"
    if final.exists():
        shutil.rmtree(final)
    final.mkdir()
    outs = []
    for f in manifest["iterations"][best[1] - 1]["files"]:
        fp = Path(f)
        shutil.copy2(fp, final / fp.name)
        outs.append({"file": fp.name, "sha256": _sha(final / fp.name), "bytes": (final / fp.name).stat().st_size})
    manifest["outputs"] = outs
    manifest["final_failures"] = manifest["iterations"][best[1] - 1]["failures"]
    manifest["seconds"] = round(time.time() - t0, 2)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
