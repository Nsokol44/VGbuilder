"""Skeleton definition: roles, per-engine naming, parenting, and joint placement.

Pure Python (no bpy) so it can be unit-tested anywhere. Coordinates: metres, Z up,
character faces -Y (Blender/glTF convention), "_l" is the character's left = +X.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

Vec = tuple[float, float, float]

# role -> parent role (None = root). Order matters: parents come before children.
HIERARCHY: dict[str, str | None] = {
    "root": None,
    "pelvis": "root",
    "spine": "pelvis",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    "upperarm_l": "chest", "forearm_l": "upperarm_l", "hand_l": "forearm_l",
    "upperarm_r": "chest", "forearm_r": "upperarm_r", "hand_r": "forearm_r",
    "thigh_l": "pelvis", "shin_l": "thigh_l", "foot_l": "shin_l",
    "thigh_r": "pelvis", "shin_r": "thigh_r", "foot_r": "shin_r",
}

ROLES = list(HIERARCHY)

# "ue": names in the style of the Unreal mannequin. This is NOT the full UE5 skeleton
# (no clavicles, extra spine/neck bones or fingers), so expect to map it with an IK Retargeter.
NAMING: dict[str, dict[str, str]] = {
    "simple": {r: r for r in ROLES},
    "ue": {
        **{r: r for r in ROLES},
        "spine": "spine_01", "chest": "spine_02", "neck": "neck_01",
        "forearm_l": "lowerarm_l", "forearm_r": "lowerarm_r",
        "shin_l": "calf_l", "shin_r": "calf_r",
    },
}
NAMING["humanoid"] = {
    # Unity Humanoid / Godot humanoid-profile style, so the engine's auto-mapping finds the bones.
    "root": "Root", "pelvis": "Hips", "spine": "Spine", "chest": "Chest", "neck": "Neck", "head": "Head",
    **{f"{part}_{side}": f"{'Left' if side == 'l' else 'Right'}{name}" for side in ("l", "r") for part, name in
       (("upperarm", "UpperArm"), ("forearm", "LowerArm"), ("hand", "Hand"),
        ("thigh", "UpperLeg"), ("shin", "LowerLeg"), ("foot", "Foot"))},
}


def bone_name(role: str, naming: str) -> str:
    return NAMING[naming][role]


def roles_from_names(names: Iterable[str], naming: str) -> set[str]:
    rev = {v: k for k, v in NAMING[naming].items()}
    return {rev[n] for n in names if n in rev}


# --------------------------------------------------------------------------- joints
def joints_from_params(p: dict[str, float], height: float) -> dict[str, tuple[Vec, Vec]]:
    """role -> (head, tail) for the procedural body. Shared by mesh builder and rig."""
    H = height
    crotch = p["leg_frac"] * H
    hip_z = crotch + 0.02 * H
    waist_z = max(0.58 * H, crotch + 0.09 * H)
    chest_z = max(0.70 * H, waist_z + 0.09 * H)
    shoulder_z = max(0.80 * H, chest_z + 0.07 * H)
    head_r = 0.065 * p["head_scale"] * H
    head_bottom = H - 2 * head_r
    neck_base = min(shoulder_z + 0.03 * H, head_bottom - 0.01 * H)
    ankle_z = 0.045 * H
    knee_z = ankle_z + (hip_z - ankle_z) * 0.5

    j: dict[str, tuple[Vec, Vec]] = {
        "root": ((0, 0, 0), (0, 0, 0.05 * H)),
        "pelvis": ((0, 0, hip_z), (0, 0, waist_z)),
        "spine": ((0, 0, waist_z), (0, 0, chest_z)),
        "chest": ((0, 0, chest_z), (0, 0, neck_base)),
        "neck": ((0, 0, neck_base), (0, 0, head_bottom)),
        "head": ((0, 0, head_bottom), (0, 0, H)),
    }
    ang = np.radians(p["arm_angle_deg"])
    d = np.array([np.sin(ang), 0.0, -np.cos(ang)])
    arm_len = p["arm_frac"] * H
    for side, sx in (("l", 1.0), ("r", -1.0)):
        sh = np.array([sx * p["shoulder_half"] * H, 0.0, shoulder_z])
        dd = d * np.array([sx, 1, 1])
        el = sh + dd * arm_len * 0.52
        wr = el + dd * arm_len * 0.48
        hd = wr + dd * 0.06 * H
        j[f"upperarm_{side}"] = (tuple(sh), tuple(el))
        j[f"forearm_{side}"] = (tuple(el), tuple(wr))
        j[f"hand_{side}"] = (tuple(wr), tuple(hd))
        hx = sx * p["hip_half"] * H * 0.55
        j[f"thigh_{side}"] = ((hx, 0, hip_z), (hx, 0, knee_z))
        j[f"shin_{side}"] = ((hx, 0, knee_z), (hx, 0, ankle_z))
        j[f"foot_{side}"] = ((hx, 0, ankle_z), (hx, -0.10 * H, 0.0))
    return {k: (tuple(map(float, h)), tuple(map(float, t))) for k, (h, t) in j.items()}


def joints_from_vertices(verts: np.ndarray) -> dict[str, tuple[Vec, Vec]]:
    """Heuristic joint placement for an arbitrary humanoid mesh (feet at z=0, facing -Y).

    Standard proportions scaled to the mesh height; arms follow the outermost vertices
    if they stick out (T/A-pose), otherwise hang in a default A-pose. Heuristic: it assumes
    a roughly upright, symmetric biped. Expect to eyeball the first few results.
    """
    H = float(verts[:, 2].max() - verts[:, 2].min())
    base = {
        "head_scale": 1.0, "shoulder_half": 0.115, "hip_half": 0.10, "leg_frac": 0.47,
        "arm_frac": 0.40, "arm_angle_deg": 22.0, "waist_half": 0.085,
    }
    j = joints_from_params(base, H)
    # follow the arms if they extend beyond the torso
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        sh = np.array(j[f"upperarm_{side}"][0])
        far = verts[(verts[:, 0] * sgn > 0.26 * H) & (verts[:, 2] > 0.45 * H)]
        if len(far) > 20:
            tip = far[np.argmax(far[:, 0] * sgn)]
            wrist = sh + (tip - sh) * 0.93
            elbow = sh + (wrist - sh) * 0.52
            hand_end = tip
            j[f"upperarm_{side}"] = (tuple(sh), tuple(map(float, elbow)))
            j[f"forearm_{side}"] = (tuple(map(float, elbow)), tuple(map(float, wrist)))
            j[f"hand_{side}"] = (tuple(map(float, wrist)), tuple(map(float, hand_end)))
    return j


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to segment ab."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1)
