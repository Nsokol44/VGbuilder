"""Limb-proportion report for the procedural body. Pure numpy/python.

A silhouette fit can win a few IoU points by quietly turning the character into something odd
(fat legs, tiny torso). This module puts the proportions next to the IoU so a human or agent can
see the trade-off. The ranges are generous "plausible biped" ranges, not rules: a stylised
brute is allowed to be outside them, it just gets listed under `warnings` for a decision.
"""
from __future__ import annotations

from .rigdefs import joints_from_params

# ratio name -> (low, high, what it measures)
RANGES: dict[str, tuple[float, float, str]] = {
    "heads_tall": (4.5, 9.5, "body height / head height"),
    "leg_to_torso": (0.85, 1.7, "crotch height / (crotch to neck base)"),
    "arm_to_leg": (0.65, 1.15, "arm length / leg length"),
    "thigh_slenderness": (0.10, 0.24, "thigh diameter / leg length"),
    "arm_slenderness": (0.07, 0.20, "upper-arm diameter / arm length"),
    "shoulder_to_hip": (0.85, 1.7, "shoulder half-width / hip half-width"),
    "torso_depth_to_width": (0.35, 0.95, "torso depth / torso width at the chest"),
}


def ratios(p: dict[str, float], height: float) -> dict[str, float]:
    """Proportion ratios implied by body parameters (procedural body)."""
    H = height
    j = joints_from_params(p, H)
    crotch = p["leg_frac"] * H
    neck_base = j["neck"][0][2]
    torso = max(neck_base - crotch, 1e-6)
    leg_len = max(crotch, 1e-6)
    arm_len = p["arm_frac"] * H
    head_h = 2 * 0.065 * p["head_scale"] * H
    chest_w = 2 * p["shoulder_half"] * H * 0.93
    chest_d = 2 * 0.066 * H * p["torso_depth"]
    return {
        "heads_tall": H / head_h,
        "leg_to_torso": leg_len / torso,
        "arm_to_leg": arm_len / leg_len,
        "thigh_slenderness": (2 * 0.045 * H * p["leg_thickness"]) / leg_len,
        "arm_slenderness": (2 * 0.027 * H * p["arm_thickness"]) / max(arm_len, 1e-6),
        "shoulder_to_hip": p["shoulder_half"] / max(p["hip_half"], 1e-6),
        "torso_depth_to_width": chest_d / max(chest_w, 1e-6),
    }


def report(p: dict[str, float], height: float) -> dict:
    """{"ratios": {...}, "warnings": [{ratio, value, low, high, side, what}]}."""
    r = ratios(p, height)
    warnings = []
    for name, v in r.items():
        lo, hi, what = RANGES[name]
        if v < lo or v > hi:
            warnings.append({"ratio": name, "value": round(v, 3), "low": lo, "high": hi,
                             "side": "below" if v < lo else "above", "what": what})
    return {"ratios": {k: round(v, 3) for k, v in r.items()}, "warnings": warnings}
