"""Procedural idle / walk cycles on the rig, plus measurements QA needs (loops, stretch, ground).

Poses are rotations of pose bones keyed per frame. Sign conventions (checked numerically against the
rig, see tests): bones that point DOWN (thighs, upper arms, shins, forearms) swing BACKWARD for +X, so a
forward swing is negative X; bones that point UP (spine, chest, head) lean FORWARD for +X. +X on a shin is
knee flexion.
"""
from __future__ import annotations

import math

import bpy
import numpy as np

from ..rigdefs import bone_name
from .condition import vertex_array

FPS = 24
CYCLES = {"idle": 48, "walk": 24}          # frames per loop (the loop closes on frame N, equal to frame 0)


def _pose(role: str, naming: str, arm):
    pb = arm.pose.bones.get(bone_name(role, naming))
    if pb is not None:
        pb.rotation_mode = "XYZ"
    return pb


def pose_for(anim: str, phase: float, H: float) -> dict[str, tuple[tuple[float, float, float], float]]:
    """role -> (euler xyz, location y) for one point of the cycle. phase in [0, 1)."""
    f = 2 * math.pi * phase
    s, c = math.sin(f), math.cos(f)
    P: dict[str, tuple[tuple[float, float, float], float]] = {}
    if anim == "walk":
        A, K, B = 0.45, 0.75, 0.35
        P["thigh_l"] = ((-A * s, 0, 0), 0.0)
        P["thigh_r"] = ((A * s, 0, 0), 0.0)
        P["shin_l"] = ((K * max(0.0, c), 0, 0), 0.0)
        P["shin_r"] = ((K * max(0.0, -c), 0, 0), 0.0)
        P["upperarm_l"] = ((B * s, 0, 0), 0.0)
        P["upperarm_r"] = ((-B * s, 0, 0), 0.0)
        P["forearm_l"] = ((-(0.20 + 0.25 * max(0.0, -s)), 0, 0), 0.0)
        P["forearm_r"] = ((-(0.20 + 0.25 * max(0.0, s)), 0, 0), 0.0)
        P["spine"] = ((0, 0.06 * s, 0), 0.0)
        P["chest"] = ((0, -0.04 * s, 0), 0.0)
        P["pelvis"] = ((0, 0, 0), -0.010 * H * math.cos(2 * f))      # lowest at double support
    elif anim == "idle":
        P["chest"] = ((-0.025 * s, 0, 0), 0.0)
        P["spine"] = ((-0.015 * s, 0, 0), 0.0)
        P["head"] = ((0.02 * math.sin(f + 0.8), 0, 0.015 * math.sin(f + 0.3)), 0.0)
        P["upperarm_l"] = ((0.02 * s, 0, -0.02 * s), 0.0)
        P["upperarm_r"] = ((0.02 * s, 0, 0.02 * s), 0.0)
        P["forearm_l"] = ((-0.04 - 0.02 * s, 0, 0), 0.0)
        P["forearm_r"] = ((-0.04 - 0.02 * s, 0, 0), 0.0)
        P["pelvis"] = ((0, 0, 0), 0.003 * H * s)
    else:
        raise ValueError(anim)
    return P


def _apply(arm, naming: str, anim: str, phase: float, H: float, key_frame: int | None = None) -> None:
    for role, (eul, ly) in pose_for(anim, phase, H).items():
        pb = _pose(role, naming, arm)
        if pb is None:
            continue
        pb.rotation_euler = eul
        pb.location = (0.0, ly, 0.0)
        if key_frame is not None:
            pb.keyframe_insert("rotation_euler", frame=key_frame)
            if role == "pelvis":                      # the only bone with a location channel; always key it so the loop closes
                pb.keyframe_insert("location", frame=key_frame)


def reset_pose(arm) -> None:
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()


def create_actions(arm, naming: str, names: list[str], H: float) -> dict[str, "bpy.types.Action"]:
    """Key every requested cycle into its own action (fake user so it survives, unassigned afterwards)."""
    bpy.context.scene.render.fps = FPS
    arm.animation_data_create()
    actions = {}
    for anim in names:
        n = CYCLES[anim]
        act = bpy.data.actions.new(anim)
        act.use_fake_user = True
        arm.animation_data.action = act
        for k in range(n + 1):                       # frame n repeats frame 0 so the loop closes
            _apply(arm, naming, anim, (k % n) / n, H, key_frame=k)
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


def measure(arm, mesh_obj, actions: dict, naming: str, H: float, samples: int = 8) -> dict:
    """Per animation: does it loop, how far do joints rotate, worst edge stretch, lowest point, bones keyed."""
    edges, rest_len = _edges(mesh_obj)
    n_verts = len(vertex_array(mesh_obj))
    ok = rest_len > 1e-6
    out = {}
    for anim, act in actions.items():
        arm.animation_data.action = act
        n = CYCLES[anim]
        bpy.context.scene.frame_set(0)
        first = _bone_state(arm)
        bpy.context.scene.frame_set(n)
        last = _bone_state(arm)
        loops = max((np.abs(first[k] - last[k]).max() for k in first), default=0.0) < 1e-4
        worst, min_z, max_rot = 1.0, 1e9, 0.0
        for k in np.linspace(0, n, samples, endpoint=False).astype(int):
            bpy.context.scene.frame_set(int(k))
            co = _evaluated(mesh_obj)
            if len(co) == n_verts:
                d = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
                worst = max(worst, float((d[ok] / rest_len[ok]).max()))
            min_z = min(min_z, float(co[:, 2].min()))
            max_rot = max(max_rot, max(float(np.abs(pb.rotation_euler).max()) for pb in arm.pose.bones))
        keyed = sorted({fc.data_path.split('"')[1] for fc in _fcurves(act) if '"' in fc.data_path})
        out[anim] = {
            "frames": n + 1, "fps": FPS, "duration_s": n / FPS, "loops": bool(loops),
            "max_rotation_rad": round(max_rot, 4), "max_stretch": round(worst, 4),
            "min_z_m": round(min_z, 4), "bones_keyed": len(keyed),
        }
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
