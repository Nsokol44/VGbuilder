"""Pre-rendered sprite sheets: the character's animations rendered from a fixed front camera.

This is the "pre-rendered from 3D" path for 2D games (Mage of Tharad Zur, Beat Knight) — NOT
hand-drawn pixel art, and the metadata says so. Frames are packed one animation per row with a
constant pivot (the character origin: feet centre) so an engine can anchor every frame the same
way. Output: <name>_spritesheet.png (RGBA, transparent background) + <name>_sprites.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
import numpy as np
from PIL import Image

from . import anim, render

MAX_FRAMES_PER_ANIM = 24


def _key_out_background(img: Image.Image) -> Image.Image:
    """Fallback if the renderer ignored film transparency: key out the world colour (0.18, 0.18, 0.2)."""
    a = np.asarray(img).astype(np.int16)
    bg = np.array([46, 46, 51], dtype=np.int16)
    dist = np.abs(a[..., :3] - bg).max(axis=2)
    a[..., 3] = np.where(dist <= 14, 0, a[..., 3])
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def render_sprites(lod_objs, arm, actions: dict, naming: str, height: float, out_dir: Path,
                   name: str, size: int) -> dict:
    """Render every action's frames and pack them into one sheet. Returns a report for metrics['sprites']."""
    tmp = out_dir / "_sprites"
    tmp.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / f"{name}_spritesheet.png"
    meta_path = out_dir / f"{name}_sprites.json"
    pivot = [size // 2, int(round(size * (0.5 + 0.5 / 1.2)))]     # world origin under the _setup framing
    rows: list[tuple[str, list[Image.Image], dict]] = []
    keyed: bool | None = None
    try:
        for o in lod_objs[1:]:
            o.hide_render = True
        arm.hide_render = True
        scene, cam = render._setup(height, size)
        if hasattr(scene.render, "film_transparent"):        # Blender >= 5 name
            scene.render.film_transparent = True
        elif hasattr(scene.render, "film_transparency"):      # Blender <= 4 name
            scene.render.film_transparency = True
        for anim_name, act in actions.items():
            n = anim.FRAMES[anim_name]
            loop = anim.KINDS[anim_name] == "loop"
            total = n if loop else n + 1                      # loops: frame N duplicates frame 0
            stride = max(1, math.ceil(total / MAX_FRAMES_PER_ANIM))
            src = list(range(0, total, stride))
            if not loop and src[-1] != n:
                src.append(n)                                 # a one-shot's end pose is the point of it
            arm.animation_data.action = act
            frames: list[Image.Image] = []
            for f in src:
                scene.frame_set(f)
                p = tmp / f"{anim_name}_{f:03d}.png"
                render._shoot(scene, cam, render.VIEWS["front"], height, p)
                img = Image.open(p).convert("RGBA")
                if keyed is None:
                    corner = np.asarray(img)[0, 0, 3]
                    keyed = bool(corner > 10)                 # renderer kept the backdrop: key it out
                if keyed:
                    img = _key_out_background(img)
                frames.append(img)
            rows.append((anim_name, frames, {
                "row": len(rows), "frames": len(frames), "fps": anim.FPS / stride,
                "loop": loop, "source_frames": src,
            }))
        arm.animation_data.action = None
        scene.frame_set(0)
        anim.reset_pose(arm)
    except Exception as e:  # noqa: BLE001 - report, QA turns it into sprites_missing
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        for o in lod_objs[1:]:
            o.hide_render = False
        arm.hide_render = False

    cols = max(len(frames) for _, frames, _ in rows)
    sheet = Image.new("RGBA", (cols * size, len(rows) * size), (0, 0, 0, 0))
    meta_anims = {}
    for anim_name, frames, info in rows:
        for i, img in enumerate(frames):
            sheet.paste(img, (i * size, info["row"] * size))
        meta_anims[anim_name] = info
    sheet.save(sheet_path)
    meta = {
        "format": "vgbuilder-sprites-1",
        "note": "pre-rendered from the 3D model (not hand-drawn pixel art); pivot is constant for all frames",
        "frame_size": size, "pivot_px": pivot, "sheet_size": [sheet.width, sheet.height],
        "animations": meta_anims,
    }
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    return {"ok": True, "sheet": sheet_path.name, "metadata": meta_path.name,
            "frame_size": size, "pivot_px": pivot, "animations": meta_anims}
