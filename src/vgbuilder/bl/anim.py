"""Procedural animation on the rig, plus measurements QA needs (loops, stretch, ground, recovery).

Loops (idle, walk) are analytic sine poses. One-shots (attack_light, hit, cast, death) are
control poses interpolated with smoothstep: recover-type one-shots end back at the rest pose,
death ends DOWN. Poses are rotations of pose bones keyed per frame. Sign conventions (checked
numerically against the rig, see tests): bones that point DOWN (thighs, upper arms, shins,
forearms) swing BACKWARD for +X, so a forward swing is negative X; bones that point UP (spine,
chest, head, pelvis) lean FORWARD for +X. +X on a shin is knee flexion.

Pelvis location channels (probed on a built rig): [0] = world X (side), [1] = world Z (up,
the height bob), [2] = world -Y = FORWARD, the direction the character faces. Root motion
keys channel [2].
"""
from __future__ import annotations

import math

import bpy
import numpy as np

from ..rigdefs import bone_name
from ..spec import ANIM_KINDS as KINDS
from .condition import vertex_array

FPS = 24
CYCLES = {"idle": 48, "walk": 24}          # frames per loop (the loop closes on frame N, equal to frame 0)
ONESHOTS = {"attack_light": 16, "hit": 10, "cast": 24, "death": 30}   # frames per one-shot (frame N is the end pose)
FRAMES = {**CYCLES, **ONESHOTS}


def _pose(role: str, naming: str, arm):
    pb = arm.pose.bones.get(bone_name(role, naming))
    if pb is not None:
        pb.rotation_mode = "XYZ"
    return pb


def _ss(t: float) -> float:
    """Smoothstep on [0, 1]."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


# One-shot control poses: phase -> {role: (euler_xyz, pelvis_height_offset_m, pelvis_forward_m)}.
def _controls(anim: str, H: float) -> list[tuple[float, dict]]:
    h = H
    if anim == "attack_light":   # right-hand chop: wind up, strike through, recover; root motion keeps the step
        return [
            (0.00, {}),
            (0.30, {"upperarm_r": ((0.95, 0, -0.15), 0, 0), "forearm_r": ((-0.55, 0, 0), 0, 0),
                    "chest": ((0, 0.22, 0), 0, 0), "spine": ((-0.06, 0.10, 0), 0, 0),
                    "head": ((0, -0.15, 0), 0, 0), "upperarm_l": ((0.25, 0, 0.25), 0, 0),
                    "pelvis": ((0, 0, 0), -0.006 * h, 0.0)}),
            (0.50, {"upperarm_r": ((-1.45, 0, 0.05), 0, 0), "forearm_r": ((-0.10, 0, 0), 0, 0),
                    "chest": ((0.10, -0.25, 0), 0, 0), "spine": ((0.16, -0.10, 0), 0, 0),
                    "head": ((0.05, 0.10, 0), 0, 0), "upperarm_l": ((-0.35, 0, 0.45), 0, 0),
                    "pelvis": ((0, 0, 0), -0.018 * h, 0.050 * h)}),
            (0.70, {"upperarm_r": ((-1.10, 0, 0), 0, 0), "forearm_r": ((-0.25, 0, 0), 0, 0),
                    "chest": ((0.06, -0.12, 0), 0, 0),
                    "pelvis": ((0, 0, 0), -0.006 * h, 0.050 * h)}),
            (1.00, {"pelvis": ((0, 0, 0), 0.0, 0.050 * h)}),
        ]
    if anim == "hit":            # recoil snap, then recover
        return [
            (0.00, {}),
            (0.22, {"chest": ((-0.24, 0, 0), 0, 0), "spine": ((-0.14, 0, 0), 0, 0),
                    "head": ((-0.32, 0, 0.08), 0, 0), "upperarm_l": ((0.30, 0, 0.55), 0, 0),
                    "upperarm_r": ((0.30, 0, -0.55), 0, 0), "forearm_l": ((-0.50, 0, 0), 0, 0),
                    "forearm_r": ((-0.50, 0, 0), 0, 0), "pelvis": ((0, 0, 0), -0.020 * h, 0.0)}),
            (0.55, {"chest": ((-0.10, 0, 0), 0, 0), "head": ((-0.12, 0, 0), 0, 0),
                    "upperarm_l": ((0.12, 0, 0.25), 0, 0), "upperarm_r": ((0.12, 0, -0.25), 0, 0),
                    "pelvis": ((0, 0, 0), -0.008 * h, 0.0)}),
            (1.00, {}),
        ]
    if anim == "cast":           # raise both arms, charge, push forward, recover
        return [
            (0.00, {}),
            (0.35, {"upperarm_l": ((-1.85, 0, -0.15), 0, 0), "upperarm_r": ((-1.85, 0, 0.15), 0, 0),
                    "forearm_l": ((-0.35, 0, 0), 0, 0), "forearm_r": ((-0.35, 0, 0), 0, 0),
                    "chest": ((-0.14, 0, 0), 0, 0), "spine": ((-0.08, 0, 0), 0, 0),
                    "head": ((-0.18, 0, 0), 0, 0), "pelvis": ((0, 0, 0), 0.005 * h, 0.0)}),
            (0.55, {"upperarm_l": ((-1.90, 0, -0.15), 0, 0), "upperarm_r": ((-1.90, 0, 0.15), 0, 0),
                    "forearm_l": ((-0.40, 0, 0), 0, 0), "forearm_r": ((-0.40, 0, 0), 0, 0),
                    "chest": ((-0.18, 0, 0), 0, 0), "head": ((-0.22, 0, 0), 0, 0),
                    "pelvis": ((0, 0, 0), 0.005 * h, 0.0)}),
            (0.75, {"upperarm_l": ((-1.30, 0, -0.10), 0, 0), "upperarm_r": ((-1.30, 0, 0.10), 0, 0),
                    "forearm_l": ((0, 0, 0), 0, 0), "forearm_r": ((0, 0, 0), 0, 0),
                    "chest": ((0.14, 0, 0), 0, 0), "spine": ((0.08, 0, 0), 0, 0),
                    "head": ((0.05, 0, 0), 0, 0), "pelvis": ((0, 0, 0), -0.010 * h, 0.0)}),
            (1.00, {}),
        ]
    if anim == "death":          # stagger, stumble, forward fall, settle prone. Prone, pelvis+chest pitch
        # totals ~1.7 rad, so the arms use upperarm ~-0.2 to lie flat pointing back (rotations compose),
        # and the feet pitch -1.05 so the toes lie along the floor instead of digging through it.
        return [
            (0.00, {}),
            (0.18, {"chest": ((-0.16, 0, 0), 0, 0), "head": ((-0.20, 0, 0), 0, 0),
                    "upperarm_l": ((0.20, 0, 0.40), 0, 0), "upperarm_r": ((0.20, 0, -0.40), 0, 0),
                    "pelvis": ((0, 0, 0), -0.005 * h, 0.0)}),
            (0.45, {"thigh_l": ((-0.55, 0, 0), 0, 0), "thigh_r": ((0.30, 0, 0), 0, 0),
                    "shin_l": ((0.50, 0, 0), 0, 0), "shin_r": ((0.70, 0, 0), 0, 0),
                    "foot_l": ((-0.25, 0, 0), 0, 0), "foot_r": ((-0.25, 0, 0), 0, 0),
                    "chest": ((-0.10, 0, 0), 0, 0), "upperarm_l": ((0.50, 0, 0.50), 0, 0),
                    "upperarm_r": ((0.50, 0, -0.50), 0, 0),
                    "pelvis": ((0, 0, 0), -0.020 * h, 0.0)}),
            (0.58, {"thigh_l": ((0.15, 0, 0), 0, 0), "thigh_r": ((0.38, 0, 0), 0, 0),
                    "shin_l": ((0.90, 0, 0), 0, 0), "shin_r": ((0.60, 0, 0), 0, 0),
                    "foot_l": ((-0.55, 0, 0), 0, 0), "foot_r": ((-0.55, 0, 0), 0, 0),
                    "chest": ((0.05, 0, 0), 0, 0), "upperarm_l": ((-0.80, 0, 0.40), 0, 0),
                    "upperarm_r": ((-0.80, 0, -0.40), 0, 0),
                    "pelvis": ((0.95 * 0.45, 0, 0), -0.050 * h, 0.0)}),
            (0.70, {"pelvis": ((0.95, 0, 0), -0.080 * h, 0.0), "chest": ((0.25, 0, 0), 0, 0),
                    "upperarm_l": ((-1.20, 0, 0.30), 0, 0), "upperarm_r": ((-1.20, 0, -0.30), 0, 0),
                    "forearm_l": ((-0.20, 0, 0), 0, 0), "forearm_r": ((-0.20, 0, 0), 0, 0),
                    "foot_l": ((-0.30, 0, 0), 0, 0), "foot_r": ((-0.30, 0, 0), 0, 0),
                    "thigh_l": ((0.40, 0, 0), 0, 0), "thigh_r": ((0.35, 0, 0), 0, 0),
                    "shin_l": ((0.80, 0, 0), 0, 0), "shin_r": ((0.70, 0, 0), 0, 0)}),
            (0.85, {"pelvis": ((1.35, 0, 0), -0.300 * h, 0.0), "chest": ((0.28, 0, 0), 0, 0),
                    "upperarm_l": ((-0.60, 0, 0.20), 0, 0), "upperarm_r": ((-0.60, 0, -0.20), 0, 0),
                    "forearm_l": ((-0.15, 0, 0), 0, 0), "forearm_r": ((-0.15, 0, 0), 0, 0),
                    "foot_l": ((-0.70, 0, 0), 0, 0), "foot_r": ((-0.70, 0, 0), 0, 0),
                    "thigh_l": ((0.12, 0, 0), 0, 0), "thigh_r": ((0.10, 0, 0), 0, 0),
                    "shin_l": ((0.30, 0, 0), 0, 0), "shin_r": ((0.25, 0, 0), 0, 0)}),
            (1.00, {"pelvis": ((1.48, 0, 0), -0.400 * h, 0.0), "chest": ((0.22, 0, 0), 0, 0),
                    "head": ((-0.15, 0, 0), 0, 0), "upperarm_l": ((-0.21, 0, 0.15), 0, 0),
                    "upperarm_r": ((-0.21, 0, -0.15), 0, 0), "forearm_l": ((-0.10, 0, 0), 0, 0),
                    "forearm_r": ((-0.10, 0, 0), 0, 0),
                    "foot_l": ((-0.85, 0, 0), 0, 0), "foot_r": ((-0.85, 0, 0), 0, 0),
                    "thigh_l": ((0.0, 0, 0), 0, 0), "thigh_r": ((0.0, 0, 0), 0, 0),
                    "shin_l": ((0.08, 0, 0), 0, 0), "shin_r": ((0.05, 0, 0), 0, 0)}),
        ]
    raise ValueError(anim)


def _interp_controls(anim: str, phase: float, H: float) -> dict[str, tuple[tuple[float, float, float], float, float]]:
    ctrls = _controls(anim, H)
    # every phase returns EVERY role the animation ever touches (zeros where a control omits it),
    # so frames 0 and N key all of them: otherwise the end frames inherit stale pose values.
    roles: set[str] = set()
    for _, c in ctrls:
        roles |= set(c)
    zero = ((0.0, 0.0, 0.0), 0.0, 0.0)
    if phase <= ctrls[0][0]:
        return {r: ctrls[0][1].get(r, zero) for r in roles}
    if phase >= ctrls[-1][0]:
        return {r: ctrls[-1][1].get(r, zero) for r in roles}
    raw = {}
    for (p0, c0), (p1, c1) in zip(ctrls, ctrls[1:]):
        if p0 <= phase <= p1:
            t = _ss((phase - p0) / (p1 - p0)) if p1 > p0 else 0.0
            for role in roles:
                e0, y0, z0 = c0.get(role, zero)
                e1, y1, z1 = c1.get(role, zero)
                raw[role] = (tuple(a + (b - a) * t for a, b in zip(e0, e1)),
                             y0 + (y1 - y0) * t, z0 + (z1 - z0) * t)
            break
    return raw


def pose_for(anim: str, phase: float, H: float, root_motion: bool = False
             ) -> dict[str, tuple[tuple[float, float, float], float, float]]:
    """role -> (euler xyz, pelvis height offset m, pelvis forward m) at one point of the animation.
    phase in [0, 1]; loops wrap, one-shots clamp at their end pose. Only the pelvis uses the offsets."""
    if anim in ONESHOTS:
        P = _interp_controls(anim, phase, H)
        if not root_motion and "pelvis" in P:
            e, y, z = P["pelvis"]
            P["pelvis"] = (e, y, 0.0)
        return P
    f = 2 * math.pi * phase
    s, c = math.sin(f), math.cos(f)
    P: dict[str, tuple[tuple[float, float, float], float, float]] = {}
    if anim == "walk":
        A, K, B = 0.45, 0.75, 0.35
        P["thigh_l"] = ((-A * s, 0, 0), 0.0, 0.0)
        P["thigh_r"] = ((A * s, 0, 0), 0.0, 0.0)
        P["shin_l"] = ((K * max(0.0, c), 0, 0), 0.0, 0.0)
        P["shin_r"] = ((K * max(0.0, -c), 0, 0), 0.0, 0.0)
        P["upperarm_l"] = ((B * s, 0, 0), 0.0, 0.0)
        P["upperarm_r"] = ((-B * s, 0, 0), 0.0, 0.0)
        P["forearm_l"] = ((-(0.20 + 0.25 * max(0.0, -s)), 0, 0), 0.0, 0.0)
        P["forearm_r"] = ((-(0.20 + 0.25 * max(0.0, s)), 0, 0), 0.0, 0.0)
        P["spine"] = ((0, 0.06 * s, 0), 0.0, 0.0)
        P["chest"] = ((0, -0.04 * s, 0), 0.0, 0.0)
        # root travel ~0.55 * H per 1 s cycle (~1 m/s at human height), keyed on pelvis channel [2]
        P["pelvis"] = ((0, 0, 0), -0.010 * H * math.cos(2 * f), 0.55 * H * phase if root_motion else 0.0)
    elif anim == "idle":
        P["chest"] = ((-0.025 * s, 0, 0), 0.0, 0.0)
        P["spine"] = ((-0.015 * s, 0, 0), 0.0, 0.0)
        P["head"] = ((0.02 * math.sin(f + 0.8), 0, 0.015 * math.sin(f + 0.3)), 0.0, 0.0)
        P["upperarm_l"] = ((0.02 * s, 0, -0.02 * s), 0.0, 0.0)
        P["upperarm_r"] = ((0.02 * s, 0, 0.02 * s), 0.0, 0.0)
        P["forearm_l"] = ((-0.04 - 0.02 * s, 0, 0), 0.0, 0.0)
        P["forearm_r"] = ((-0.04 - 0.02 * s, 0, 0), 0.0, 0.0)
        P["pelvis"] = ((0, 0, 0), 0.003 * H * s, 0.0)
    else:
        raise ValueError(anim)
    return P


def _apply(arm, naming: str, anim: str, phase: float, H: float, key_frame: int | None = None,
           root_motion: bool = False) -> None:
    for role, (eul, ly, lz) in pose_for(anim, phase, H, root_motion).items():
        pb = _pose(role, naming, arm)
        if pb is None:
            continue
        pb.rotation_euler = eul
        pb.location = (0.0, ly, lz)
        if key_frame is not None:
            pb.keyframe_insert("rotation_euler", frame=key_frame)
            if role == "pelvis":                      # the only bone with a location channel; always key it
                pb.keyframe_insert("location", frame=key_frame)


def reset_pose(arm) -> None:
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()


def create_actions(arm, naming: str, names: list[str], H: float, root_motion: bool = False) -> dict[str, "bpy.types.Action"]:
    """Key every requested animation into its own action (fake user so it survives, unassigned afterwards).
    Loops close on frame N == frame 0; one-shots key their true end pose at frame N."""
    bpy.context.scene.render.fps = FPS
    arm.animation_data_create()
    actions = {}
    for anim in names:
        n = FRAMES[anim]
        act = bpy.data.actions.new(anim)
        act.use_fake_user = True
        arm.animation_data.action = act
        loop = KINDS[anim] == "loop"
        for k in range(n + 1):
            phase = ((k % n) / n) if loop else (k / n)
            _apply(arm, naming, anim, phase, H, key_frame=k, root_motion=root_motion)
        act.frame_start, act.frame_end = 0, n
        actions[anim] = act
    arm.animation_data.action = None
    reset_pose(arm)
    return actions


def _edges(mesh_obj):
    m = mesh_obj.data
    e = np.empty(len(m.edges) * 2, np.int32)
    m.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    rest = vertex_array(mesh_obj)
    return e, np.linalg.norm(rest[e[:, 0]] - rest[e[:, 1]], axis=1)


def _evaluated(mesh_obj) -> np.ndarray:
    bpy.context.view_layer.update()
    ev = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    tmp = ev.to_mesh()
    a = np.empty(len(tmp.vertices) * 3)
    tmp.vertices.foreach_get("co", a)
    ev.to_mesh_clear()
    return a.reshape(-1, 3)


def measure(arm, mesh_obj, actions: dict, naming: str, H: float, samples: int = 8,
            root_motion: bool = False) -> dict:
    """Per animation: kind, does it loop, how far the end pose sits from the start (rotations),
    pelvis drop / forward travel, worst edge stretch, lowest point, bones keyed."""
    edges, rest_len = _edges(mesh_obj)
    n_verts = len(vertex_array(mesh_obj))
    ok = rest_len > 1e-6
    pelvis_name = bone_name("pelvis", naming)
    out = {}
    for anim, act in actions.items():
        arm.animation_data.action = act
        n = FRAMES[anim]
        kind = KINDS[anim]
        bpy.context.scene.frame_set(0)
        first = _bone_state(arm)
        bpy.context.scene.frame_set(n)
        last = _bone_state(arm)

        def _loop_delta(exclude_root: bool) -> float:
            worst = 0.0
            for bn, a in first.items():
                b = last[bn]
                if exclude_root and bn == pelvis_name:
                    a = a.copy(); b = b.copy(); a[5] = b[5] = 0.0    # root travel is not a loop break
                worst = max(worst, float(np.abs(a - b).max()))
            return worst

        loops = _loop_delta(root_motion and kind == "loop") < 1e-4
        end_delta = max((float(np.abs(a[:3] - last[k][:3]).max()) for k, a in first.items()), default=0.0)
        pf, pl = first.get(pelvis_name), last.get(pelvis_name)
        pelvis_drop = float(pf[4] - pl[4]) if pf is not None and pl is not None else 0.0
        if kind == "loop" and root_motion and pf is not None:
            # frame N wraps to frame 0 by construction, so travel is read at frame N-1 and
            # extrapolated linearly to the full cycle (the walk's root channel is linear in phase)
            bpy.context.scene.frame_set(n - 1)
            prev = _bone_state(arm).get(pelvis_name)
            root_dist = float((prev[5] - pf[5]) * n / (n - 1)) if prev is not None else 0.0
        else:
            root_dist = float(pl[5] - pf[5]) if pf is not None and pl is not None else 0.0
        worst, min_z, max_rot = 1.0, 1e9, 0.0
        frames = np.linspace(0, n, samples, endpoint=(kind != "loop")).astype(int)
        for k in frames:
            bpy.context.scene.frame_set(int(k))
            co = _evaluated(mesh_obj)
            if len(co) == n_verts:
                d = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
                worst = max(worst, float((d[ok] / rest_len[ok]).max()))
            min_z = min(min_z, float(co[:, 2].min()))
            max_rot = max(max_rot, max(float(np.abs(pb.rotation_euler).max()) for pb in arm.pose.bones))
        keyed = sorted({fc.data_path.split('"')[1] for fc in _fcurves(act) if '"' in fc.data_path})
        out[anim] = {
            "kind": kind,
            "frames": n + 1, "fps": FPS, "duration_s": n / FPS, "loops": bool(loops),
            "end_delta_rad": round(end_delta, 4),
            "pelvis_drop_m": round(pelvis_drop, 4), "root_distance_m": round(root_dist, 4),
            "max_rotation_rad": round(max_rot, 4), "max_stretch": round(worst, 4),
            "min_z_m": round(min_z, 4), "bones_keyed": len(keyed),
        }
    arm.animation_data.action = None
    bpy.context.scene.frame_set(0)
    reset_pose(arm)
    return out


def clamp_floor(arm, mesh_obj, actions: dict, naming: str, max_lift: float = 0.25) -> dict[str, float]:
    """Fall one-shots only: lift the pelvis' height channel per frame so the body never sinks
    below z=0 (a rigid procedural fall cannot place a prone body exactly; this is the same
    ground-clamp games apply). Returns {anim: max lift applied}. QA still measures the result,
    so a fall that needs more than max_lift keeps failing honestly."""
    pelvis = _pose("pelvis", naming, arm)
    out: dict[str, float] = {}
    if pelvis is None:
        return out
    for anim_name, act in actions.items():
        if KINDS[anim_name] != "oneshot_fall":
            continue
        n = FRAMES[anim_name]
        arm.animation_data.action = act
        worst = 0.0
        for f in range(n + 1):
            bpy.context.scene.frame_set(f)
            mz = float(_evaluated(mesh_obj)[:, 2].min())
            if mz < 0.0:
                lift = min(-mz, max_lift)
                pelvis.location[1] += lift            # pelvis channel [1] is world up (probed on the rig)
                pelvis.keyframe_insert("location", frame=f)
                worst = max(worst, lift)
        out[anim_name] = round(worst, 4)
    arm.animation_data.action = None
    bpy.context.scene.frame_set(0)
    reset_pose(arm)
    return out


def _fcurves(act):
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []                                           # slotted actions (Blender 4.4+)
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for bag in strip.channelbags:
                out.extend(bag.fcurves)
    return out


def _bone_state(arm) -> dict[str, np.ndarray]:
    bpy.context.view_layer.update()
    return {pb.name: np.array([*pb.rotation_euler, *pb.location]) for pb in arm.pose.bones}
