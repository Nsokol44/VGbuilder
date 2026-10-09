"""Draw a simple front-view line sketch of a character with PIL. Used by the tests and the
example job, so the repo is self-contained. Replace with your own sketches.

Outline only (dark strokes on white) on purpose: it exercises the hole-filling.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw


def draw(path: Path, *, shoulder=112, hip=92, head_r=52, leg_len=400, stocky=1.0, size=(700, 1000)) -> None:
    W, H = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    cx = W // 2
    top = 60
    ink = (20, 20, 20)
    lw = 5
    head_cy = top + head_r
    d.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], outline=ink, width=lw)
    neck_y = head_cy + head_r
    sh_y = neck_y + 30
    crotch_y = H - 70 - leg_len
    # torso (trapezoid-ish): shoulders -> waist -> hips
    waist_y = sh_y + (crotch_y - sh_y) * 0.55
    wh = int(hip * 0.85 * stocky)
    pts = [(cx - 22, neck_y), (cx - shoulder, sh_y), (cx - wh, waist_y), (cx - hip, crotch_y),
           (cx + hip, crotch_y), (cx + wh, waist_y), (cx + shoulder, sh_y), (cx + 22, neck_y)]
    d.line(pts + [pts[0]], fill=ink, width=lw, joint="curve")
    # arms hanging slightly out
    for s in (-1, 1):
        a0 = (cx + s * shoulder, sh_y)
        a1 = (cx + s * (shoulder + 88), sh_y + 330)
        d.line([a0, a1], fill=ink, width=34 + lw)
        d.line([a0, a1], fill="white", width=34 - lw)
    # legs
    foot_y = H - 70
    for s in (-1, 1):
        x_out, x_in = cx + s * (hip - 4), cx + s * 8
        leg = [(x_out, crotch_y), (cx + s * (hip - 24), foot_y), (cx + s * 20, foot_y), (x_in, crotch_y)]
        d.line(leg + [leg[0]], fill=ink, width=lw)
        fx0, fx1 = sorted([cx + s * (hip + 22), cx + s * 16])
        d.line([(fx0, foot_y), (fx0, foot_y + 22), (fx1, foot_y + 22), (fx1, foot_y)], fill=ink, width=lw)
    img.save(path)


def draw_side(path: Path, *, head_r=52, leg_len=400, depth=1.0, leg_thick=1.0, size=(700, 1000)) -> None:
    """Side view, character facing LEFT (toes point to the image's left). Same height/proportions as `draw`.

    depth scales the front-back thickness of the torso, leg_thick the thickness of the legs.
    """
    W, H = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    cx, top, ink, lw = W // 2, 60, (20, 20, 20), 5
    head_cy = top + head_r
    d.ellipse([cx - head_r * 0.95, head_cy - head_r, cx + head_r * 0.95, head_cy + head_r], outline=ink, width=lw)
    neck_y = head_cy + head_r
    sh_y = neck_y + 30
    crotch_y = H - 70 - leg_len
    waist_y = sh_y + (crotch_y - sh_y) * 0.55
    ch, wd, hd = 56 * depth, 48 * depth, 52 * depth
    torso = [(cx - 30, neck_y), (cx - ch, sh_y), (cx - wd, waist_y), (cx - hd, crotch_y),
             (cx + hd, crotch_y), (cx + wd, waist_y), (cx + ch, sh_y), (cx + 30, neck_y)]
    d.line(torso + [torso[0]], fill=ink, width=lw, joint="curve")
    arm = [(cx, sh_y), (cx, sh_y + 330)]            # arm hangs straight in the side view
    d.line(arm, fill=ink, width=34 + lw)
    d.line(arm, fill="white", width=34 - lw)
    foot_y = H - 70
    t = leg_thick
    leg = [(cx - 40 * t, crotch_y), (cx - 26 * t, foot_y), (cx - 98, foot_y), (cx - 98, foot_y + 22),
           (cx + 36, foot_y + 22), (cx + 36, foot_y), (cx + 26 * t, foot_y), (cx + 40 * t, crotch_y)]
    d.line(leg + [leg[0]], fill=ink, width=lw, joint="curve")
    img.save(path)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/sketches")
    out.mkdir(parents=True, exist_ok=True)
    draw(out / "hero_front.png")
    draw(out / "stocky_front.png", shoulder=135, hip=115, head_r=58, leg_len=360, stocky=1.12)
    draw(out / "lanky_front.png", shoulder=92, hip=72, head_r=46, leg_len=440, stocky=0.9)
    draw_side(out / "hero_side.png")
    draw_side(out / "stocky_side.png", head_r=58, leg_len=360, depth=1.2)
    draw_side(out / "lanky_side.png", head_r=46, leg_len=440, depth=0.85)
    print("wrote", sorted(p.name for p in out.glob("*.png")))
